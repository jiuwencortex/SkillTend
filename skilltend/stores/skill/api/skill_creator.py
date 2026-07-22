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


import re
from pathlib import Path
from typing import List, Optional, Tuple

from agent_evolving.online.stores.skill.atomic_writter import _atomic_write
from agent_evolving.online.stores.skill.frontmatter_handler import _validate_frontmatter
from agent_evolving.online.stores.skill.skill_finder import _find_skill
from agent_evolving.online.stores.skill import _get_lock
from agent_evolving.online.stores.skill.usages.usage_sidecar import UsageSidecar
from agent_evolving.online.stores.skill import _write_usage
from agent_evolving.online.stores.skill.max_skill_size import _MAX_SKILL_SIZE


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_create(
    name: str,
    content: str,
    skills_root: Path,
    category: Optional[str] = None,
    protected_names: List[str] = (),
) -> Tuple[bool, str]:
    """Create a new skill. Returns (success, message).

    Records write origin from the provenance ContextVar:
      - foreground        → created_by='user'    (not curator-managed)
      - background_review → created_by='agent'   (curator-eligible)
    """
    if not re.compile(r"^[a-z0-9][a-z0-9._-]*$").match(name):
        return False, f"Invalid skill name '{name}'."
    err = _validate_frontmatter(content)
    if err:
        return False, err
    if len(content) > _MAX_SKILL_SIZE:
        return False, f"Content exceeds {_MAX_SKILL_SIZE} char limit."
    if _find_skill(name, skills_root):
        return False, f"Skill '{name}' already exists."

    # Import here to avoid circular imports at module level
    from agent_evolving.online.review_executor.provenance import get_write_origin

    origin = get_write_origin()
    created_by = "agent" if origin == "background_review" else "user"

    async with _get_lock(name):
        skill_dir = (
            (skills_root / category / name) if category else (skills_root / name)
        )
        _atomic_write(skill_dir / "SKILL.md", content)
        usage = UsageSidecar(created_by=created_by)
        _write_usage(skill_dir, usage)

    return True, f"Created skill '{name}' at {skill_dir}."
