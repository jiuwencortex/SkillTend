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


_ARCHIVE_SUBDIR = ".archive"


def _find_skill(name: str, skills_root: Path) -> Optional[Path]:
    """Return the skill directory path if it exists (not archived)."""
    direct = skills_root / name
    if (direct / "SKILL.md").exists():
        return direct
    # Search one level of category subdirectories (skip .archive)
    for category_dir in skills_root.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        candidate = category_dir / name
        if (candidate / "SKILL.md").exists():
            return candidate
    return None


def _find_archived(name: str, skills_root: Path) -> Optional[Path]:
    """Return the archived skill directory path, or None."""
    archive_root = skills_root / _ARCHIVE_SUBDIR
    direct = archive_root / name
    if (direct / "SKILL.md").exists():
        return direct
    for category_dir in archive_root.iterdir() if archive_root.exists() else []:
        if not category_dir.is_dir():
            continue
        candidate = category_dir / name
        if (candidate / "SKILL.md").exists():
            return candidate
    return None
