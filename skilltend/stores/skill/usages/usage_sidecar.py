# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""SKILL.md filesystem operations.

Mirrors Hermes skill_manager_tool.py internal helpers.
Handles concurrent access with per-skill asyncio locks and atomic writes.

New vs original implementation:
  - UsageSidecar dataclass (.usage.json per skill) with full telemetry
  - Lifecycle states: SKILL_STATE_ACTIVE / STALE / ARCHIVED
  - skill_delete() with absorbed_into consolidation-intent tracking
  - skill_archive() / skill_restore() for reversible deactivation
  - skill_get_usage() / skill_set_pinned() helpers
  - skill_create() records write origin from provenance ContextVar
  - skill_list() filters archived skills by default
"""
from dataclasses import dataclass
from typing import Optional

from agent_evolving.online.stores.skill.skill_states import SKILL_STATE_ACTIVE


# ── Usage sidecar ─────────────────────────────────────────────────────────────


@dataclass
class UsageSidecar:
    """Per-skill telemetry sidecar, mirroring Hermes skill_usage.py.

    Written to ``<skill_dir>/.usage.json``.

    Lifecycle states mirror Hermes:
      active   — in regular use
      stale    — unused for stale_after_days (default 30)
      archived — unused for archive_after_days (default 90); moved to .archive/
    """

    created_by: str = "user"           # "user" | "agent"
    use_count: int = 0
    view_count: int = 0
    patch_count: int = 0
    pinned: bool = False               # pin blocks deletion/archive
    state: str = SKILL_STATE_ACTIVE
    last_activity_at: Optional[str] = None  # ISO-8601; updated on read/edit; curator anchor
    archived_at: Optional[str] = None  # ISO-8601 timestamp when archived
    absorbed_into: Optional[str] = None  # If deleted via consolidation, target skill
