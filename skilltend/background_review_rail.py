# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""BackgroundReviewRail — Jiuwen implementation of Hermes background review.

Priority: 70
  (below existing SkillEvolutionRail at 80 and HealingRail at 90;
   above EvolutionRail base at 60; runs AFTER primary rails have finished)

Trigger logic (mirrors Hermes exactly):
  _user_turn_count   incremented in after_model_call when a non-tool
                     assistant message is observed.
  _tool_iter_count   incremented in after_tool_call on every tool completion.

Counter reset (mirrors Hermes tool_executor.py reset-on-parse behavior):
  When a memory_write / memory tool fires  → _user_turn_count = 0
  When a skill_write / skill_patch / skill_manage fires → _tool_iter_count = 0

  Hermes resets at parse-time; we reset in after_tool_call (same effect).

Session-resume hydration (mirrors Hermes gateway session hydration):
  Call hydrate_from_history(messages) to restore the correct cadence when
  attaching the rail to a resumed/gateway session:
      _user_turn_count = prior_user_turns % memory_nudge_interval
  This prevents an immediate review fire on the first message.

Interrupted guard (mirrors Hermes conversation_loop.py line 4328):
  Background review does NOT fire if the invoke was interrupted.

flush_min_turns guard:
  Background review does NOT fire until the session has seen at least
  flush_min_turns user turns (default 6, matches Hermes config).

User-facing summary (mirrors Hermes _safe_print call):
  On completion, prints  💾 Self-improvement review: <summary>

Background task (mirrors Hermes _spawn_background_review):
  asyncio.create_task(run_background_review(...)) spawned in after_invoke.
  Task is awaited in the NEXT after_invoke if still running (serialisation).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import logging
logger = logging.getLogger(__name__)

from openjiuwen.core.single_agent.rail.base import AgentCallbackContext
from openjiuwen.harness.rails.base import DeepAgentRail

from agent_evolving.online.config import BackgroundReviewConfig
from agent_evolving.online.curator import SkillCurator
from agent_evolving.online.review_executor import run_background_review
from agent_evolving.online.types import ReviewMode, ReviewResult, ReviewTrigger


# ── Tool name sets that trigger counter resets ─────────────────────────────
# Mirrors Hermes tool_executor.py lines 91-94 and 523-526
_MEMORY_TOOL_NAMES: frozenset = frozenset({"memory_write", "memory"})
_SKILL_TOOL_NAMES: frozenset = frozenset({"skill_write", "skill_patch", "skill_manage", "skill_create"})


class BackgroundReviewRail(DeepAgentRail):
    """Hermes-style background review rail for Jiuwen.

    After every N tool iterations or M user turns, spawns an async background
    task that reads the full conversation and uses an LLM to write targeted
    updates to SKILL.md files and memory stores.

    Does NOT touch existing agent_evolving or agent_healing.
    """

    priority: int = 70

    def __init__(self, config: Optional[BackgroundReviewConfig] = None):
        super().__init__()
        self._config: BackgroundReviewConfig = config or BackgroundReviewConfig()
        # Nudge counters (mirrors Hermes _turns_since_memory, _iters_since_skill)
        self._user_turn_count: int = 0
        self._tool_iter_count: int = 0
        # Total user turns ever seen in this rail instance (for flush_min_turns guard)
        self._total_turn_count: int = 0
        self._review_task: Optional[asyncio.Task] = None
        self._last_result: Optional[ReviewResult] = None
        # ── Curator ───────────────────────────────────────────────────────────
        # Tracks idle time between invokes (for curator scheduling gate).
        # Initialized to current monotonic time; idle = now - _invoke_end_at
        # at the START of each after_invoke before updating.
        self._curator: SkillCurator = SkillCurator(self._config)
        self._curator_task: Optional[asyncio.Task] = None
        self._invoke_end_at: float = time.monotonic()

    # ── Session-resume hydration ──────────────────────────────────────────────

    def hydrate_from_history(self, messages: List[Dict[str, Any]]) -> None:
        """Restore correct nudge cadence from a prior conversation history.

        Call this when attaching the rail to a RESUMED session (e.g. gateway
        platforms that create a fresh agent per message).  Mirrors Hermes
        gateway session hydration in conversation_loop.py:

            prior_user_turns = sum(1 for m in history if m["role"] == "user")
            agent._turns_since_memory = prior_user_turns % memory_nudge_interval

        Uses modulo so the counter does NOT immediately fire on the very first
        message of a resumed session — the original cadence is preserved.
        """
        if self._total_turn_count > 0:
            # Already in an active session — don't overwrite
            return
        prior_user_turns = sum(
            1 for m in messages if isinstance(m, dict) and m.get("role") == "user"
        )
        self._total_turn_count = prior_user_turns
        if prior_user_turns > 0 and self._config.memory_nudge_interval > 0:
            self._user_turn_count = prior_user_turns % self._config.memory_nudge_interval
        # Skill iteration counter is NOT hydrated — it tracks current-session tool calls only
        # (matches Hermes: _iters_since_skill is not hydrated from history)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_model(self, ctx: AgentCallbackContext) -> str:
        """Resolve LLM model for review. Falls back to agent model."""
        if self._config.review_model:
            return self._config.review_model
        agent = ctx.agent
        for attr in ("model", "_model", "model_name"):
            val = getattr(agent, attr, None)
            if isinstance(val, str) and val:
                return val
        return "gpt-4o-mini"  # safe fallback

    def _get_session_id(self, ctx: AgentCallbackContext) -> str:
        if ctx.session:
            return ctx.session.get_session_id() or ""
        return ""

    def _capture_messages(self, ctx: AgentCallbackContext) -> List[Dict[str, Any]]:
        """Snapshot the current conversation messages as plain dicts."""
        if ctx.context is None:
            return []
        msgs = ctx.context.get_messages()
        result = []
        for m in msgs:
            role = getattr(m, "role", "unknown")
            content = getattr(m, "content", "")
            result.append({"role": role, "content": content})
        return result

    @staticmethod
    def _extract_tool_name(ctx: AgentCallbackContext) -> str:
        """Best-effort extraction of the called tool's name from the context."""
        for attr_path in (
            ("inputs", "tool_call"),
            ("inputs", "tool"),
            ("inputs", "tool_use"),
            ("outputs", "tool_call"),
        ):
            obj: Any = ctx
            for attr in attr_path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                name = getattr(obj, "name", None) or getattr(obj, "tool_name", None)
                if not name and hasattr(obj, "function"):
                    fn = getattr(obj, "function", None)
                    name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    return str(name)
        return str(getattr(getattr(ctx, "inputs", None), "tool_name", "") or "")

    @staticmethod
    def _is_interrupted(ctx: AgentCallbackContext) -> bool:
        """Check whether the invoke was interrupted (best-effort).

        Mirrors Hermes: review only fires when 'final_response and not interrupted'
        (conversation_loop.py line 4328).
        """
        for src in (ctx, getattr(ctx, "inputs", None), getattr(ctx, "outputs", None)):
            if src is None:
                continue
            if getattr(src, "interrupted", None) is True:
                return True
        return False

    # ── Hooks ─────────────────────────────────────────────────────────────────

    async def after_model_call(self, ctx: AgentCallbackContext) -> None:
        """Increment user-turn counter when assistant produces a text response."""
        if not self._config.enabled:
            return
        msg = getattr(ctx.inputs, "response_message", None) or getattr(
            ctx.inputs, "message", None
        )
        if msg:
            role = getattr(msg, "role", "")
            tool_calls = getattr(msg, "tool_calls", None)
            if role == "assistant" and not tool_calls:
                self._user_turn_count += 1
                self._total_turn_count += 1

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        """Increment/reset counters on every tool completion.

        Counter reset mirrors Hermes tool_executor.py reset-on-parse:
          memory tool called  → _user_turn_count = 0  (memory was actively used)
          skill tool called   → _tool_iter_count = 0  (skill was actively used)

        Then unconditionally increment _tool_iter_count toward nudge threshold.
        """
        if not self._config.enabled:
            return

        tool_name = self._extract_tool_name(ctx)

        if tool_name in _MEMORY_TOOL_NAMES:
            self._user_turn_count = 0
            logger.debug("BackgroundReviewRail: memory tool fired — reset user_turn_count")
        if tool_name in _SKILL_TOOL_NAMES:
            self._tool_iter_count = 0
            logger.debug("BackgroundReviewRail: skill tool fired — reset tool_iter_count")

        if self._config.skill_nudge_interval > 0:
            self._tool_iter_count += 1

    async def after_invoke(self, ctx: AgentCallbackContext) -> None:
        """At the end of each invoke:

        1. Skip if session was interrupted.
        2. Skip if session hasn't reached flush_min_turns yet.
        3. Wait for any in-flight review task (serialisation).
        4. Check trigger conditions.
        5. Snapshot messages and spawn background review task.
        6. Spawn curator task if scheduling conditions are met (always, after review).
        """
        # Compute idle time (gap between previous invoke end and this invoke end).
        # Captured before the method body so all return paths share the same value.
        idle_seconds = time.monotonic() - self._invoke_end_at

        try:
            if not self._config.enabled:
                return

            # ── Guard: skip if interrupted ──────────────────────────────────
            if self._is_interrupted(ctx):
                logger.debug("BackgroundReviewRail: skipping review — invoke was interrupted")
                return

            # ── Guard: minimum session turns before any review fires ────────
            if self._total_turn_count < self._config.flush_min_turns:
                return

            # Serialise: wait for previous review to finish
            if self._review_task and not self._review_task.done():
                try:
                    await asyncio.wait_for(self._review_task, timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    pass  # Best-effort; don't block current invoke

            # Determine what to review
            should_review_memory = (
                self._config.memory_nudge_interval > 0
                and self._user_turn_count >= self._config.memory_nudge_interval
            )
            should_review_skills = (
                self._config.skill_nudge_interval > 0
                and self._tool_iter_count >= self._config.skill_nudge_interval
            )

            if not (should_review_memory or should_review_skills):
                return

            # Determine mode
            if should_review_memory and should_review_skills:
                mode = ReviewMode.COMBINED
            elif should_review_memory:
                mode = ReviewMode.MEMORY_ONLY
            else:
                mode = ReviewMode.SKILLS_ONLY

            # Reset counters (mirrors Hermes reset-on-fire)
            if should_review_memory:
                self._user_turn_count = 0
            if should_review_skills:
                self._tool_iter_count = 0

            messages_snapshot = self._capture_messages(ctx)
            session_id = self._get_session_id(ctx)
            model = self._get_model(ctx)

            trigger = ReviewTrigger(
                mode=mode,
                user_turn_count=self._user_turn_count,
                tool_iter_count=self._tool_iter_count,
                session_id=session_id,
            )

            self._review_task = asyncio.create_task(
                self._run_review(messages_snapshot, trigger, model, session_id)
            )
            logger.debug(
                "BackgroundReviewRail: spawned review task [mode=%s session=%s]",
                mode.value,
                session_id,
            )

        finally:
            # Always update the invoke-end timestamp so idle_seconds is correct
            # on the NEXT after_invoke call.
            self._invoke_end_at = time.monotonic()
            # Spawn curator as a fire-and-forget background task (never blocks user).
            model = self._get_model(ctx)
            self._spawn_curator_if_due(idle_seconds, model)

    # ── Curator background task ───────────────────────────────────────────────

    def _spawn_curator_if_due(self, idle_seconds: float, model: str) -> None:
        """Fire a background curator task if scheduling conditions are met.

        Only one curator task runs at a time (same serialisation as the review task).
        Uses asyncio.create_task so the curator never blocks the calling coroutine.
        """
        if self._curator_task and not self._curator_task.done():
            return  # Previous run still in flight — skip

        self._curator_task = asyncio.create_task(
            self._run_curator_bg(idle_seconds, model)
        )

    async def _run_curator_bg(self, idle_seconds: float, model: str) -> None:
        """Background coroutine: delegate to SkillCurator.maybe_run()."""
        try:
            result = await self._curator.maybe_run(
                idle_seconds=idle_seconds,
                model=model or None,
            )
            if result:
                logger.info(
                    "BackgroundReviewRail curator: %s [%.1fs]",
                    result.summary_line(),
                    result.elapsed_seconds,
                )
                if result.transitions:
                    print(f"\n🗂️  Curator: {result.summary_line()}", flush=True)
        except Exception as exc:
            logger.error("BackgroundReviewRail curator task crashed: %s", exc)

    # ── Background task ───────────────────────────────────────────────────────

    async def _run_review(
        self,
        messages_snapshot: List[Dict[str, Any]],
        trigger: ReviewTrigger,
        model: str,
        session_id: str,
    ) -> None:
        """Background coroutine. Mirrors Hermes _run_review_in_thread().

        On completion, prints a user-facing summary line:
          '💾 Self-improvement review: <summary>'
        (mirrors Hermes background_review.py agent._safe_print() call)
        """
        try:
            result = await run_background_review(
                messages_snapshot=messages_snapshot,
                trigger=trigger,
                config=self._config,
                model=model,
                session_id=session_id,
            )
            self._last_result = result
            if result.actions:
                # User-facing summary line (mirrors Hermes _safe_print)
                print(f"\n💾 Self-improvement review: {result.summary_line}", flush=True)
                logger.info(
                    "BackgroundReview completed: %s [%.1fs]",
                    result.summary_line,
                    result.duration_seconds,
                )
            if result.error:
                logger.warning("BackgroundReview error: %s", result.error)
        except Exception as e:
            logger.error("BackgroundReviewRail background task crashed: %s", e)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def uninit(self, agent) -> None:
        """Cancel any in-flight review or curator task on rail teardown."""
        if self._review_task and not self._review_task.done():
            self._review_task.cancel()
        if self._curator_task and not self._curator_task.done():
            self._curator_task.cancel()

    # ── Public accessors ──────────────────────────────────────────────────────

    def last_review_result(self) -> Optional[ReviewResult]:
        """Return the result of the most recently completed review, or None."""
        return self._last_result

    def pending_counts(self) -> Dict[str, int]:
        """Return current trigger counters (useful for debugging/testing)."""
        return {
            "user_turns_since_review": self._user_turn_count,
            "tool_iters_since_review": self._tool_iter_count,
            "total_turns_seen": self._total_turn_count,
        }
