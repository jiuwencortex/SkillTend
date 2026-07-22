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
from typing import Optional

from agent_evolving.online.stores.skill.skill_finder import _find_skill
from agent_evolving.online.stores.skill.usages.usage_reader import _read_usage
from agent_evolving.online.stores.skill import _write_usage


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_read(name: str, skills_root: Path) -> Optional[str]:
    """Read and return raw SKILL.md content, or None if not found."""
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return None
    usage = _read_usage(skill_dir)
    usage.view_count += 1
    _write_usage(skill_dir, usage)
    return (skill_dir / "SKILL.md").read_text(encoding="utf-8")