# coding: utf-8
"""Example 08 — Skill provenance tracking and full lifecycle management.

Demonstrates the Hermes-style features added to skill_store and provenance:

  1. background_review_context() — marks skill writes as 'agent'-created
  2. get_write_origin() — reads the current ContextVar value
  3. UsageSidecar / skill_get_usage() — per-skill telemetry (.usage.json)
  4. skill_set_pinned() — pin protection blocks delete/archive
  5. skill_archive() / skill_restore() — reversible deactivation
  6. skill_list(include_archived=True) — see all skills including archived
  7. skill_delete(absorbed_into=...) — consolidation-intent tracking
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agent_evolving.online import (
    background_review_context,
    get_write_origin,
)
from agent_evolving.online.stores.skill import (
    SKILL_STATE_ACTIVE,
    UsageSidecar,
    skill_archive,
    skill_create,
    skill_delete,
    skill_get_usage,
    skill_list,
    skill_read,
    skill_restore,
    skill_set_pinned,
)

_SKILL_A = """\
---
name: skill-alpha
description: Does important alpha work for the agent.
---
## Instructions
Do alpha work here.
"""

_SKILL_B = """\
---
name: skill-beta
description: Beta skill that consolidates alpha and other things.
---
## Instructions
Do beta work here (superset of alpha).
"""


async def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # ── 1. Default write origin is "foreground" ───────────────────────
        print(f"[1] Default write origin: '{get_write_origin()}'")
        assert get_write_origin() == "foreground"

        # ── 2. skill_create in foreground → created_by='user' ─────────────
        ok, msg = await skill_create("skill-alpha", _SKILL_A, root)
        assert ok, msg
        usage = await skill_get_usage("skill-alpha", root)
        assert isinstance(usage, UsageSidecar)
        print(f"[2] skill-alpha created_by='{usage.created_by}'  (expected: 'user')")
        assert usage.created_by == "user"
        assert usage.state == SKILL_STATE_ACTIVE

        # ── 3. skill_create inside background_review_context → created_by='agent'
        async with background_review_context():
            print(f"[3] Inside context, write origin: '{get_write_origin()}'")
            assert get_write_origin() == "background_review"
            ok, msg = await skill_create("skill-beta", _SKILL_B, root)
            assert ok, msg

        print(f"[3] After context, write origin: '{get_write_origin()}'")
        assert get_write_origin() == "foreground"  # restored

        usage_b = await skill_get_usage("skill-beta", root)
        assert isinstance(usage_b, UsageSidecar)
        print(f"[3] skill-beta created_by='{usage_b.created_by}'  (expected: 'agent')")
        assert usage_b.created_by == "agent"

        # ── 4. view_count increments on skill_read ─────────────────────────
        _ = await skill_read("skill-alpha", root)
        usage_after_read = await skill_get_usage("skill-alpha", root)
        assert isinstance(usage_after_read, UsageSidecar)
        print(f"[4] skill-alpha view_count after 1 read: {usage_after_read.view_count}")
        assert usage_after_read.view_count == 1

        # ── 5. skill_set_pinned blocks delete ─────────────────────────────
        ok, msg = await skill_set_pinned("skill-alpha", root, pinned=True)
        assert ok, msg
        print(f"[5] Pinned skill-alpha")

        ok, msg = await skill_delete("skill-alpha", root)
        print(f"[5] Delete pinned skill: ok={ok}, msg='{msg}'")
        assert not ok  # must fail

        ok, msg = await skill_archive("skill-alpha", root)
        print(f"[5] Archive pinned skill: ok={ok}, msg='{msg}'")
        assert not ok  # must fail

        # Unpin first, then delete succeeds
        await skill_set_pinned("skill-alpha", root, pinned=False)

        # ── 6. skill_archive and skill_restore ────────────────────────────
        ok, msg = await skill_archive("skill-alpha", root)
        print(f"[6] Archive skill-alpha: ok={ok}")
        assert ok, msg

        # Archived skill should not appear in normal listing
        active_names = await skill_list(root, include_archived=False)
        print(f"[6] Active skills after archive: {active_names}")
        assert "skill-alpha" not in active_names
        assert "skill-beta" in active_names

        # But appears with include_archived=True
        all_names = await skill_list(root, include_archived=True)
        print(f"[6] All skills (incl archived): {all_names}")
        assert "skill-alpha" in all_names

        usage_archived = await skill_get_usage("skill-alpha", root)
        # Note: archived skills are in .archive/ — skill_get_usage only checks active
        print(f"[6] skill_get_usage for archived returns: {usage_archived}")

        # Restore
        ok, msg = await skill_restore("skill-alpha", root)
        print(f"[6] Restore skill-alpha: ok={ok}")
        assert ok, msg

        active_after = await skill_list(root)
        print(f"[6] Active after restore: {active_after}")
        assert "skill-alpha" in active_after

        # ── 7. skill_delete with absorbed_into ────────────────────────────
        ok, msg = await skill_delete("skill-alpha", root, absorbed_into="skill-beta")
        print(f"[7] Delete with absorbed_into: ok={ok}, msg='{msg}'")
        assert ok, msg

        remaining = await skill_list(root)
        print(f"[7] Remaining skills: {remaining}")
        assert "skill-alpha" not in remaining
        assert "skill-beta" in remaining

        print("\n✓ All provenance and lifecycle assertions passed.")


if __name__ == "__main__":
    asyncio.run(demo())
