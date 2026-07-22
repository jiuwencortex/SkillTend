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
import json
from pathlib import Path

from agent_evolving.online.stores.skill.usages.usage_sidecar import UsageSidecar


def _read_usage(skill_dir: Path) -> UsageSidecar:
    p = skill_dir / ".usage.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {f for f in UsageSidecar.__dataclass_fields__}
            return UsageSidecar(**{k: v for k, v in data.items() if k in known})
        except Exception:
            pass
    return UsageSidecar()



