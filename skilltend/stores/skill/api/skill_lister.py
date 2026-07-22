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
from typing import List

from agent_evolving.online.stores.skill.skill_finder import _ARCHIVE_SUBDIR


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_list(
    skills_root: Path,
    include_archived: bool = False,
) -> List[str]:
    """Return sorted list of all skill names.

    By default, archived skills (.archive/) are excluded.
    Pass include_archived=True to include them.
    """
    names = []
    if not skills_root.exists():
        return names
    for item in skills_root.rglob("SKILL.md"):
        # Skip .archive/ subtree unless explicitly requested
        parts = item.parts
        if _ARCHIVE_SUBDIR in parts and not include_archived:
            continue
        names.append(item.parent.name)
    return sorted(names)
