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


import shutil
from pathlib import Path
from typing import Tuple

from agent_evolving.online.stores.skill.skill_finder import _find_skill, _find_archived
from agent_evolving.online.stores.skill import _get_lock
from agent_evolving.online.stores.skill.skill_states import SKILL_STATE_ACTIVE
from agent_evolving.online.stores.skill.usages.usage_reader import _read_usage
from agent_evolving.online.stores.skill import _write_usage


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_restore(
    name: str,
    skills_root: Path,
) -> Tuple[bool, str]:
    """Restore an archived skill back to active.

    Returns (success, message).
    """
    archived_dir = _find_archived(name, skills_root)
    if not archived_dir:
        return False, f"No archived skill '{name}' found."
    if _find_skill(name, skills_root):
        return False, f"Skill '{name}' already exists as an active skill — cannot restore."

    restore_dest = skills_root / name

    async with _get_lock(name):
        restore_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(archived_dir), str(restore_dest))
        usage = _read_usage(restore_dest)
        usage.state = SKILL_STATE_ACTIVE
        usage.archived_at = None
        _write_usage(restore_dest, usage)
    return True, f"Restored skill '{name}' from archive."