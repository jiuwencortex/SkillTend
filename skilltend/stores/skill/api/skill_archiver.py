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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from agent_evolving.online.stores.skill.frontmatter_handler import _is_immutable
from agent_evolving.online.stores.skill.skill_finder import _find_skill, _ARCHIVE_SUBDIR
from agent_evolving.online.stores.skill import _get_lock
from agent_evolving.online.stores.skill.skill_states import SKILL_STATE_ARCHIVED
from agent_evolving.online.stores.skill.usages.usage_reader import _read_usage
from agent_evolving.online.stores.skill import _write_usage


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_archive(
    name: str,
    skills_root: Path,
    protected_names: List[str] = (),
) -> Tuple[bool, str]:
    """Move a skill to the .archive/ subdirectory (reversible).

    Pinned skills cannot be archived.
    Archived skills are hidden from skill_list() by default.

    Returns (success, message).
    """
    if name in protected_names:
        return False, f"Skill '{name}' is protected."
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    existing = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    if _is_immutable(existing):
        return False, f"Skill '{name}' is immutable and cannot be archived."
    usage = _read_usage(skill_dir)
    if usage.pinned:
        return False, f"Skill '{name}' is pinned — unpin before archiving."
    if usage.state == SKILL_STATE_ARCHIVED:
        return False, f"Skill '{name}' is already archived."

    archive_dest = skills_root / _ARCHIVE_SUBDIR / name
    if archive_dest.exists():
        return False, f"Archive collision: {archive_dest} already exists."

    async with _get_lock(name):
        usage.state = SKILL_STATE_ARCHIVED
        usage.archived_at = datetime.now(timezone.utc).isoformat()
        _write_usage(skill_dir, usage)
        archive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(skill_dir), str(archive_dest))
    return True, f"Archived skill '{name}' to {archive_dest}."