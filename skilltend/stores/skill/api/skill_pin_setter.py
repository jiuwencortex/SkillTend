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


from pathlib import Path
from typing import Tuple

from agent_evolving.online.stores.skill.skill_finder import _find_skill
from agent_evolving.online.stores.skill import _get_lock
from agent_evolving.online.stores.skill.usages.usage_reader import _read_usage
from agent_evolving.online.stores.skill import _write_usage


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_set_pinned(
    name: str,
    skills_root: Path,
    pinned: bool,
) -> Tuple[bool, str]:
    """Pin or unpin a skill.

    Pinned skills cannot be deleted or archived.
    Returns (success, message).
    """
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    async with _get_lock(name):
        usage = _read_usage(skill_dir)
        usage.pinned = pinned
        _write_usage(skill_dir, usage)
    verb = "Pinned" if pinned else "Unpinned"
    return True, f"{verb} skill '{name}'."
