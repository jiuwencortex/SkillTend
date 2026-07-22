# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Core background review execution — orchestrator.

Each step delegates to the matching stage module under stages/:

  stage01_conversation_builder  — resolve paths, create MemoryStore, serialise conversation
  stage02_prompt_selector        — select review prompt + build system prompt
  stage03_llm_caller             — call review LLM with timeout / error handling
  stage04_tool_call_dispatcher   — dispatch tool calls to skill_store / memory_store
  stage05_result_assembler       — compute duration, build summary, return ReviewResult
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from agent_evolving.online.config import BackgroundReviewConfig
from agent_evolving.online.review_executor.stages.stage01_conversation_builder.stage import build_conversation_context
from agent_evolving.online.review_executor.stages.stage02_prompt_selector.stage import build_review_prompts
from agent_evolving.online.review_executor.stages.stage03_llm_caller.stage import call_llm_with_timeout
from agent_evolving.online.review_executor.stages.stage04_tool_call_dispatcher.stage import dispatch_all_tool_calls
from agent_evolving.online.review_executor.stages.stage05_result_assembler.stage import assemble_review_result
from agent_evolving.online.types import ReviewResult, ReviewTrigger


async def run_background_review(
    messages_snapshot: List[Dict[str, Any]],
    trigger: ReviewTrigger,
    config: BackgroundReviewConfig,
    model: str,
    session_id: str,
    parent_session_id: str = "",
) -> ReviewResult:
    """Core review execution. Mirrors Hermes _run_review_in_thread().

    1. Build conversation text from snapshot.
    2. Select review prompt by mode.
    3. Call LLM with tool definitions (skill_write, skill_patch, memory_write).
    4. Dispatch each tool call to skill_store / memory_store.
    5. Return ReviewResult with all actions taken.
    """
    # ── Stage 1: Build conversation context ──────────────────────────────────
    skills_root, memory_store, conversation_text = build_conversation_context(
        messages_snapshot, config,
    )

    # ── Stage 2: Select review prompt ────────────────────────────────────────
    review_prompt, system_prompt = build_review_prompts(trigger)

    # ── Stage 3: Call LLM ────────────────────────────────────────────────────
    t0 = time.monotonic()
    tool_calls, error = await call_llm_with_timeout(
        system_prompt, conversation_text, review_prompt, model, config,
    )

    # ── Stage 4: Dispatch tool calls ─────────────────────────────────────────
    actions = await dispatch_all_tool_calls(
        tool_calls, skills_root, memory_store, config, session_id,
    )

    # ── Stage 5: Assemble and return result ───────────────────────────────────
    return assemble_review_result(trigger, actions, error, t0)
