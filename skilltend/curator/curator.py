# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Skill lifecycle curator — Hermes-style scheduled maintenance.

Mirrors hermes-agent/agent/curator.py two-phase algorithm:

Phase 1 — Rule-based lifecycle transitions (no LLM):
  ACTIVE  → STALE    when unused for curator_stale_after_days   (default 30 d)
  STALE   → ACTIVE   when used again (anchor > stale_cutoff)
  STALE   → ARCHIVED when unused for curator_archive_after_days (default 90 d)

  Pinned skills are immune to all automatic transitions.
  Archived skills are physically moved to .archive/ by skill_archive().

Phase 2 — LLM consolidation pass (optional, disabled by default):
  Reads active skill names + descriptions, asks an LLM to identify overlapping
  groups, and appends suggestions to a JSONL file for human review.
  No automatic PATCH/ARCHIVE is executed in this implementation for safety.

Scheduling (mirrors Hermes maybe_run_curator):
  - Runs at most once every `curator_interval_hours` (default 168 h = 7 days).
  - Only fires when the agent has been idle >= `curator_min_idle_seconds` (2 h).
  - On the very first call, seeds last_run_at = now and defers the actual run
    by one full interval (same deferral as Hermes to prevent running immediately
    on a fresh skill library).

State:
  Persisted as JSON to `curator_state_path`
  (default: skills_root parent dir / "curator_state.json").
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import logging
logger = logging.getLogger(__name__)

from agent_evolving.online.stores.skill.api.skill_archiver import skill_archive
from agent_evolving.online.stores.skill.api.skill_lister import skill_list
from agent_evolving.online.stores.skill.api.skill_reader import skill_read
from agent_evolving.online.stores.skill.skill_finder import _find_skill
from agent_evolving.online.stores.skill.skill_states import (
    SKILL_STATE_ACTIVE,
    SKILL_STATE_ARCHIVED,
    SKILL_STATE_STALE,
)
from agent_evolving.online.stores.skill.usages.usage_reader import _read_usage
from agent_evolving.online.stores.skill.usages.usage_sidecar import UsageSidecar
from agent_evolving.online.stores.skill.usages.usage_writer import _write_usage

if TYPE_CHECKING:
    from agent_evolving.online.config import BackgroundReviewConfig


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class CuratorState:
    """Persisted scheduling state for the curator."""

    last_run_at: Optional[float] = None   # epoch seconds; None on first install
    last_summary: str = ""
    total_runs: int = 0


@dataclass
class LifecycleTransition:
    """One skill state change produced by Phase 1."""

    skill: str
    from_state: str
    to_state: str
    anchor_iso: str     # ISO-8601 of the last-activity timestamp used
    reason: str


@dataclass
class ConsolidationSuggestion:
    """One overlapping-skills suggestion from Phase 2 LLM pass."""

    cluster: List[str]          # skill names that overlap
    keep: str                   # suggested skill to keep
    rationale: str              # why these overlap
    suggested_action: str       # e.g. "merge 'X' into 'Y' then archive 'X'"


@dataclass
class CuratorResult:
    """Full result of one curator run."""

    transitions: List[LifecycleTransition] = field(default_factory=list)
    suggestions: List[ConsolidationSuggestion] = field(default_factory=list)
    skipped_pinned: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    phase1_ran: bool = False
    phase2_ran: bool = False
    error: Optional[str] = None

    def summary_line(self) -> str:
        parts = []
        if self.transitions:
            parts.append(f"{len(self.transitions)} lifecycle transition(s)")
        if self.suggestions:
            parts.append(f"{len(self.suggestions)} consolidation suggestion(s)")
        if self.skipped_pinned:
            parts.append(f"{len(self.skipped_pinned)} pinned (skipped)")
        if not parts:
            parts.append("no changes")
        return "Curator: " + ", ".join(parts)


# ── Internal helpers ────────────────────────────────────────────────────────────


def _anchor_epoch(usage: UsageSidecar, skill_md_path: Path) -> float:
    """Return the best epoch-seconds estimate of when the skill was last active.

    Priority:
    1. usage.last_activity_at (ISO-8601 written by skill API on read/edit)
    2. Modification time of SKILL.md (set on every write)
    3. Now — so a skill with no history is not immediately staled
    """
    if usage.last_activity_at:
        try:
            dt = datetime.fromisoformat(usage.last_activity_at)
            return dt.timestamp()
        except (ValueError, TypeError):
            pass
    if skill_md_path.exists():
        return skill_md_path.stat().st_mtime
    return time.time()


async def _set_skill_state(
    skill_name: str,
    skills_root: Path,
    new_state: str,
) -> bool:
    """Update .usage.json state field in-place without moving the directory."""
    skill_dir = _find_skill(skill_name, skills_root)
    if not skill_dir:
        return False
    usage = _read_usage(skill_dir)
    if usage.state == new_state:
        return False
    usage.state = new_state
    _write_usage(skill_dir, usage)
    return True


# ── State file I/O ─────────────────────────────────────────────────────────────


def _state_path(config: "BackgroundReviewConfig") -> Path:
    if config.curator_state_path:
        return config.curator_state_path
    root = config.skills_root or (Path.home() / ".jiuwen" / "skills")
    return root.parent / "curator_state.json"


def _load_state(config: "BackgroundReviewConfig") -> CuratorState:
    p = _state_path(config)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return CuratorState(
                last_run_at=data.get("last_run_at"),
                last_summary=data.get("last_summary", ""),
                total_runs=data.get("total_runs", 0),
            )
        except (json.JSONDecodeError, OSError):
            pass
    return CuratorState()


def _save_state(config: "BackgroundReviewConfig", state: CuratorState) -> None:
    p = _state_path(config)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Curator: failed to save state to %s — %s", p, exc)


# ── Phase 1: rule-based lifecycle transitions ──────────────────────────────────


async def _run_phase1(
    config: "BackgroundReviewConfig",
) -> tuple[List[LifecycleTransition], List[str]]:
    """Apply rule-based ACTIVE/STALE/ARCHIVE transitions.

    Returns (transitions, skipped_pinned).
    """
    skills_root: Path = config.skills_root or (Path.home() / ".jiuwen" / "skills")
    now = time.time()
    stale_cutoff = now - config.curator_stale_after_days * 86_400
    archive_cutoff = now - config.curator_archive_after_days * 86_400

    transitions: List[LifecycleTransition] = []
    skipped_pinned: List[str] = []

    skill_names = await skill_list(skills_root, include_archived=False)

    for name in skill_names:
        skill_dir = _find_skill(name, skills_root)
        if not skill_dir:
            continue
        usage = _read_usage(skill_dir)

        if usage.pinned:
            skipped_pinned.append(name)
            continue

        skill_md = skill_dir / "SKILL.md"
        anchor_epoch = _anchor_epoch(usage, skill_md)
        anchor_iso = datetime.fromtimestamp(anchor_epoch, tz=timezone.utc).isoformat()
        current_state = usage.state

        if current_state == SKILL_STATE_STALE and anchor_epoch <= archive_cutoff:
            # STALE → ARCHIVED: move to .archive/
            ok, msg = await skill_archive(name, skills_root)
            if ok:
                transitions.append(LifecycleTransition(
                    skill=name,
                    from_state=SKILL_STATE_STALE,
                    to_state=SKILL_STATE_ARCHIVED,
                    anchor_iso=anchor_iso,
                    reason=f"unused for >{config.curator_archive_after_days} days",
                ))
                logger.info("Curator: archived '%s' (%s)", name, msg)
            else:
                logger.warning("Curator: could not archive '%s': %s", name, msg)

        elif current_state == SKILL_STATE_ACTIVE and anchor_epoch <= stale_cutoff:
            # ACTIVE → STALE: only update .usage.json, skill stays in place
            changed = await _set_skill_state(name, skills_root, SKILL_STATE_STALE)
            if changed:
                transitions.append(LifecycleTransition(
                    skill=name,
                    from_state=SKILL_STATE_ACTIVE,
                    to_state=SKILL_STATE_STALE,
                    anchor_iso=anchor_iso,
                    reason=f"unused for >{config.curator_stale_after_days} days",
                ))
                logger.info("Curator: marked '%s' as stale", name)

        elif current_state == SKILL_STATE_STALE and anchor_epoch > stale_cutoff:
            # STALE → ACTIVE: skill was recently used again
            changed = await _set_skill_state(name, skills_root, SKILL_STATE_ACTIVE)
            if changed:
                transitions.append(LifecycleTransition(
                    skill=name,
                    from_state=SKILL_STATE_STALE,
                    to_state=SKILL_STATE_ACTIVE,
                    anchor_iso=anchor_iso,
                    reason="recently used — reactivated",
                ))
                logger.info("Curator: reactivated '%s'", name)

    return transitions, skipped_pinned


# ── Phase 2: LLM consolidation suggestions ────────────────────────────────────


async def _run_phase2(
    config: "BackgroundReviewConfig",
    model: str,
) -> List[ConsolidationSuggestion]:
    """Ask an LLM to identify overlapping skills and suggest consolidations.

    Returns a list of ConsolidationSuggestion objects.  Results are also
    appended to a JSONL file alongside the curator state for human review.
    No automatic modifications are made to the skill library.
    """
    import dspy  # lazy import — not required when consolidation is disabled

    skills_root: Path = config.skills_root or (Path.home() / ".jiuwen" / "skills")
    skill_names = await skill_list(skills_root, include_archived=False)

    if len(skill_names) < 3:
        return []   # too few skills to consolidate

    # Build a compact summary of each skill's name + first 200 chars of description
    skill_summaries: List[Dict] = []
    for name in skill_names:
        result = await skill_read(name, skills_root)
        if not result:
            continue
        raw = result.get("raw", "")
        # Extract description from frontmatter if present
        desc = ""
        in_fm = False
        for line in raw.splitlines():
            if line.strip() == "---":
                in_fm = not in_fm
                continue
            if in_fm and line.startswith("description:"):
                desc = line[len("description:"):].strip().strip('"').strip("'")
                break
        if not desc:
            # Fall back to first non-empty body line
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("---") and ":" not in stripped[:20]:
                    desc = stripped[:200]
                    break
        skill_summaries.append({"name": name, "description": desc[:200]})

    if not skill_summaries:
        return []

    class IdentifyOverlaps(dspy.Signature):
        """Identify groups of skills that overlap in purpose and suggest consolidation.

        Given a list of skill names and short descriptions, find clusters of
        2+ skills that serve similar or overlapping purposes.  For each cluster
        suggest which skill to keep and how to consolidate the others into it.

        Return JSON: a list of objects, each with:
          cluster: [list of skill names]
          keep: "name of skill to keep"
          rationale: "one sentence explaining overlap"
          suggested_action: "one sentence describing what to do"
        """

        skill_list_json: str = dspy.InputField(
            desc="JSON array of {name, description} objects"
        )
        consolidation_json: str = dspy.OutputField(
            desc="JSON array of consolidation suggestions"
        )

    lm = dspy.LM(model)
    with dspy.context(lm=lm):
        result = dspy.ChainOfThought(IdentifyOverlaps)(
            skill_list_json=json.dumps(skill_summaries)
        )

    suggestions: List[ConsolidationSuggestion] = []
    try:
        raw_json = result.consolidation_json
        items = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("cluster"):
                    suggestions.append(ConsolidationSuggestion(
                        cluster=item.get("cluster", []),
                        keep=item.get("keep", ""),
                        rationale=item.get("rationale", ""),
                        suggested_action=item.get("suggested_action", ""),
                    ))
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Curator Phase 2: could not parse LLM consolidation suggestions")
        return []

    # Append suggestions to a JSONL file for human review
    if suggestions:
        suggestions_path = _state_path(config).parent / "curator_suggestions.jsonl"
        try:
            ts = datetime.now(timezone.utc).isoformat()
            with open(suggestions_path, "a", encoding="utf-8") as f:
                for s in suggestions:
                    f.write(json.dumps({"timestamp": ts, **asdict(s)}) + "\n")
        except OSError as exc:
            logger.warning("Curator: could not write suggestions to %s: %s", suggestions_path, exc)

    return suggestions


# ── Main public class ──────────────────────────────────────────────────────────


class SkillCurator:
    """Scheduled skill lifecycle maintenance, mirroring Hermes curator.py.

    Typical usage in BackgroundReviewRail::

        curator = SkillCurator(config)

        # In after_invoke():
        idle_seconds = time.monotonic() - self._last_invoke_at
        result = await curator.maybe_run(idle_seconds=idle_seconds, model=review_model)
        if result:
            logger.info(result.summary_line())
    """

    def __init__(self, config: "BackgroundReviewConfig") -> None:
        self._config = config

    async def maybe_run(
        self,
        idle_seconds: float,
        model: Optional[str] = None,
    ) -> Optional[CuratorResult]:
        """Run the curator if scheduling conditions are met.

        Args:
            idle_seconds: Seconds since the last agent invoke (used for idle gate).
            model:        LLM model for Phase 2 consolidation pass (if enabled).

        Returns:
            CuratorResult if the curator ran, else None.
        """
        if not self._config.curator_enabled:
            return None

        # ── Idle gate ─────────────────────────────────────────────────────────
        if idle_seconds < self._config.curator_min_idle_seconds:
            return None

        state = _load_state(self._config)
        now = time.time()

        # ── First-run deferral ────────────────────────────────────────────────
        if state.last_run_at is None:
            state.last_run_at = now
            state.last_summary = "first install — deferred by one interval"
            _save_state(self._config, state)
            logger.debug("Curator: first install — seeded last_run_at, deferring one interval")
            return None

        # ── Interval gate ─────────────────────────────────────────────────────
        interval_seconds = self._config.curator_interval_hours * 3600.0
        if (now - state.last_run_at) < interval_seconds:
            return None

        # ── Run ───────────────────────────────────────────────────────────────
        t_start = time.monotonic()
        result = CuratorResult()

        try:
            transitions, skipped = await _run_phase1(self._config)
            result.transitions = transitions
            result.skipped_pinned = skipped
            result.phase1_ran = True
        except Exception as exc:
            result.error = f"Phase 1 failed: {exc}"
            logger.error("Curator Phase 1 error: %s", exc, exc_info=True)

        if self._config.curator_llm_consolidation and not result.error:
            effective_model = (
                model
                or self._config.curator_consolidation_model
                or self._config.review_model
                or "openai/gpt-4.1"
            )
            try:
                suggestions = await _run_phase2(self._config, effective_model)
                result.suggestions = suggestions
                result.phase2_ran = True
            except Exception as exc:
                logger.warning("Curator Phase 2 error: %s", exc)

        result.elapsed_seconds = time.monotonic() - t_start

        # ── Persist state ─────────────────────────────────────────────────────
        state.last_run_at = now
        state.last_summary = result.summary_line()
        state.total_runs += 1
        _save_state(self._config, state)

        logger.info("Curator finished in %.1fs: %s", result.elapsed_seconds, result.summary_line())
        return result
