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
from typing import List, Tuple

from agent_evolving.online.stores.skill.frontmatter_handler import _is_immutable
from agent_evolving.online.stores.skill.skill_finder import _find_skill
from agent_evolving.online.stores.skill import _get_lock
from agent_evolving.online.stores.skill.usages.usage_reader import _read_usage
from agent_evolving.online.stores.skill import _write_usage


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_delete(
    name: str,
    skills_root: Path,
    protected_names: List[str] = (),
    absorbed_into: str = "",
) -> Tuple[bool, str]:
    """Permanently delete a skill directory.

    ``absorbed_into`` mirrors Hermes skill_manager_tool.py absorbed_into
    parameter: declare which skill absorbed this skill's content (if any).
    This intent is recorded in .usage.json before deletion so audit logs
    can reconstruct consolidation history.

    Pinned skills cannot be deleted — use skill_set_pinned() first.

    Returns (success, message).
    """
    if name in protected_names:
        return False, f"Skill '{name}' is protected."
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    existing = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    if _is_immutable(existing):
        return False, f"Skill '{name}' is immutable and cannot be deleted."
    usage = _read_usage(skill_dir)
    if usage.pinned:
        return False, (
            f"Skill '{name}' is pinned — unpin it with skill_set_pinned() before deleting."
        )

    async with _get_lock(name):
        if absorbed_into:
            usage.absorbed_into = absorbed_into
            _write_usage(skill_dir, usage)
        shutil.rmtree(skill_dir)
    return True, (
        f"Deleted skill '{name}'"
        + (f" (content absorbed into '{absorbed_into}')" if absorbed_into else "")
        + "."
    )

