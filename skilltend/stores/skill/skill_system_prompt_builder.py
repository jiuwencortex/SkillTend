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
from __future__ import annotations

from pathlib import Path
from typing import List

from agent_evolving.online.stores.skill.api.skill_lister import skill_list
from agent_evolving.online.stores.skill.api.skill_reader import skill_read
from agent_evolving.online.stores.skill.frontmatter_handler import _parse_frontmatter


# ── Public API ────────────────────────────────────────────────────────────────

async def build_skills_system_prompt(skills_root: Path) -> str:
    """Build a compact skills index for injection into the agent system prompt.

    Mirrors Hermes prompt_builder.py build_skills_system_prompt():
    - Iterates all SKILL.md files
    - Extracts name + description from frontmatter
    - Marks bundled/hub-installed skills with [bundled]
    - Returns a markdown-formatted index string

    Returns empty string if no skills exist.

    The returned block is suitable for the STABLE tier of the system prompt
    (it changes only when skills are added/removed/renamed).
    """
    names = await skill_list(skills_root)
    if not names:
        return ""

    lines: List[str] = ["## Available Skills", ""]
    for name in names:
        content = await skill_read(name, skills_root)
        if not content:
            continue
        fm, _ = _parse_frontmatter(content)
        desc = ""
        immutable = False
        if fm and isinstance(fm, dict):
            desc = str(fm.get("description", ""))
            immutable = bool(fm.get("immutable", False))
        tag = " [bundled]" if immutable else ""
        if desc:
            lines.append(f"- **{name}**{tag}: {desc}")
        else:
            lines.append(f"- **{name}**{tag}")

    lines.append("")
    lines.append(
        "Use the skill_view tool to read a skill's full instructions before executing a task."
    )
    return "\n".join(lines)
