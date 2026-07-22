# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""skilltend — Hermes-style online skill maintenance for Jiuwen.

Provides two cooperating components:

  Reviewer
    Spawns an asyncio task after every N tool-calls or M user turns,
    reads the conversation, and uses an LLM to update SKILL.md files
    and memory entries directly.

  Curator
    Scheduled background daemon that transitions skills between
    ACTIVE / STALE / ARCHIVED states based on usage age.
"""
from __future__ import annotations

# ── Reviewer + config ─────────────────────────────────────────────────────────
from skilltend.reviewer import Reviewer
from skilltend.pipeline.stages.stage02_prompt_selector.prompts import (
    COMBINED_REVIEW_PROMPT,
    MEMORY_REVIEW_PROMPT,
    SKILL_REVIEW_PROMPT,
    select_prompt,
)
from skilltend.config import ReviewerConfig
from .stores import MemoryStore
from skilltend.pipeline.provenance import (
    background_review_context,
    get_write_origin,
    make_write_metadata,
    set_write_origin,
)
from skilltend.pipeline import run_background_review
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
from skilltend.types import (
    ReviewAction,
    ReviewMode,
    ReviewResult,
    ReviewTrigger,
)


__all__ = [
    # Online track — reviewer + config
    "Reviewer",
    "ReviewerConfig",
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
