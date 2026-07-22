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
from typing import List, Tuple

from agent_evolving.online.stores.skill.atomic_writter import _atomic_write
from agent_evolving.online.stores.skill.frontmatter_handler import (_validate_frontmatter,
                                                     _is_immutable)
from agent_evolving.online.stores.skill.skill_finder import _find_skill
from agent_evolving.online.stores.skill import _get_lock
from agent_evolving.online.stores.skill.usages.usage_reader import _read_usage
from agent_evolving.online.stores.skill import _write_usage
from agent_evolving.online.stores.skill.max_skill_size import _MAX_SKILL_SIZE


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_patch(
    name: str,
    old_string: str,
    new_string: str,
    skills_root: Path,
    replace_all: bool = False,
    protected_names: List[str] = (),
) -> Tuple[bool, str]:
    """String-replace patch of SKILL.md. Returns (success, message)."""
    if name in protected_names:
        return False, f"Skill '{name}' is protected."
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    path = skill_dir / "SKILL.md"
    content = path.read_text(encoding="utf-8")
    if _is_immutable(content):
        return False, f"Skill '{name}' is immutable."
    if old_string not in content:
        return False, f"old_string not found in skill '{name}'."

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)

    err = _validate_frontmatter(new_content)
    if err:
        return False, err
    if len(new_content) > _MAX_SKILL_SIZE:
        return False, "Patched content exceeds size limit."
    if len(new_content) > len(content) * 1.20:
        return False, "Patch rejected: would grow skill by > 20%."

    async with _get_lock(name):
        _atomic_write(path, new_content)
        usage = _read_usage(skill_dir)
        usage.patch_count += 1
        _write_usage(skill_dir, usage)
    return True, f"Patched skill '{name}'."
