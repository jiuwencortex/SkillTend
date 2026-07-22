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
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def _parse_frontmatter(content: str) -> Tuple[Optional[dict], str]:
    """Split YAML frontmatter from markdown body.

    Returns (frontmatter_dict, body) or (None, content) if no frontmatter.
    """
    if not content.startswith("---"):
        return None, content
    end = content.find("\n---", 3)
    if end == -1:
        return None, content
    try:
        fm = yaml.safe_load(content[3:end])
        return fm or {}, content[end + 4:].lstrip("\n")
    except yaml.YAMLError:
        return None, content


def _validate_frontmatter(content: str) -> Optional[str]:
    """Return error string or None if frontmatter is valid."""
    fm, _ = _parse_frontmatter(content)
    if fm is None:
        return "SKILL.md must begin with YAML frontmatter (--- ... ---)."
    if not fm.get("name"):
        return "Frontmatter must contain a 'name' field."
    if not fm.get("description"):
        return "Frontmatter must contain a 'description' field."
    if len(fm.get("description", "")) > 1024:
        return "description must be ≤ 1024 characters."
    return None


def _is_immutable(content: str) -> bool:
    fm, _ = _parse_frontmatter(content)
    return bool(fm and fm.get("immutable"))
