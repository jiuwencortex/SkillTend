# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""agent_evolving — Hermes-style self-evolution for Jiuwen.

Two independent tracks, one package:

Track — Online (BackgroundReviewRail)
  Mirrors Hermes background review daemon thread.
  Spawns an asyncio task after every N tool-calls or M user turns,
  reads the conversation, and uses an LLM to update SKILL.md files
  and memory entries directly.
"""
from __future__ import annotations

# ── Online track ──────────────────────────────────────────────────────────────
from agent_evolving.online.background_review_rail import BackgroundReviewRail
from agent_evolving.online.review_executor.stages.stage02_prompt_selector.prompts import (
    COMBINED_REVIEW_PROMPT,
    MEMORY_REVIEW_PROMPT,
    SKILL_REVIEW_PROMPT,
    select_prompt,
)
from agent_evolving.online.config import BackgroundReviewConfig
from .stores import MemoryStore
from agent_evolving.online.review_executor.provenance import (
    background_review_context,
    get_write_origin,
    make_write_metadata,
    set_write_origin,
)
from agent_evolving.online.review_executor import run_background_review
from .stores import (
    SKILL_STATE_ACTIVE,
    SKILL_STATE_ARCHIVED,
    SKILL_STATE_STALE,
    UsageSidecar,
    build_skills_system_prompt,
    skill_archive,
    skill_create,
    skill_delete,
    skill_edit,
    skill_get_usage,
    skill_list,
    skill_patch,
    skill_read,
    skill_restore,
    skill_set_pinned,
)
from agent_evolving.online.types import (
    ReviewAction,
    ReviewMode,
    ReviewResult,
    ReviewTrigger,
)


__all__ = [
    # Online track — rail + config
    "BackgroundReviewRail",
    "BackgroundReviewConfig",
    "ReviewMode",
    "ReviewTrigger",
    "ReviewAction",
    "ReviewResult",
    # Memory
    "MemoryStore",
    "run_background_review",
    # Provenance — ContextVar-based write-origin tracking
    "make_write_metadata",
    "get_write_origin",
    "set_write_origin",
    "background_review_context",
    # Prompts
    "select_prompt",
    "MEMORY_REVIEW_PROMPT",
    "SKILL_REVIEW_PROMPT",
    "COMBINED_REVIEW_PROMPT",
    # Skill store — CRUD
    "skill_read",
    "skill_create",
    "skill_edit",
    "skill_patch",
    "skill_delete",
    "skill_list",
    "build_skills_system_prompt",
    # Skill store — lifecycle
    "skill_archive",
    "skill_restore",
    "skill_get_usage",
    "skill_set_pinned",
    "UsageSidecar",
    "SKILL_STATE_ACTIVE",
    "SKILL_STATE_STALE",
    "SKILL_STATE_ARCHIVED",
]
