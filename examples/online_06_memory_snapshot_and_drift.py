# coding: utf-8
"""Example 09 — Memory frozen snapshot pattern and drift detection.

Demonstrates the Hermes-style additions to MemoryStore:

  1. load_from_disk() — builds a frozen system-prompt snapshot
  2. get_snapshot_block() — returns the stable frozen state
  3. Snapshot does NOT change when add() is called mid-session
     (ensures byte-identical prefix cache across all turns)
  4. detect_drift() — detects external modification of a memory file
     and creates a .bak.<timestamp> backup

Background: Hermes uses a frozen snapshot to keep the system prompt
byte-identical across all turns within a session, which maximises
prompt-prefix cache hits.  Live mutations (add/replace/remove) update
the disk but NOT the snapshot.  The fresh snapshot is only built once,
at session start (load_from_disk).
"""
from __future__ import annotations

import time
from pathlib import Path

import tempfile

from agent_evolving.online.stores.memory import MemoryStore


def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = MemoryStore(memory_root=root)

        # ── 1. Empty snapshot before any data ─────────────────────────────
        store.load_from_disk()
        snapshot_empty = store.get_snapshot_block()
        print(f"[1] Empty snapshot: {repr(snapshot_empty)}")
        assert snapshot_empty == ""

        # ── 2. Add some entries, then call load_from_disk ──────────────────
        store.add("memory", "User prefers dark mode in all apps.")
        store.add("memory", "User works in the fintech domain.")
        store.add("user", "Name: Alice; timezone: UTC+8.")
        store.load_from_disk()  # snapshot taken HERE

        snapshot_v1 = store.get_snapshot_block()
        print(f"\n[2] Snapshot after load:\n{snapshot_v1}\n")
        assert "<memory-context>" in snapshot_v1
        assert "dark mode" in snapshot_v1
        assert "Alice" in snapshot_v1

        # ── 3. Add more entries mid-session — snapshot must NOT change ─────
        store.add("memory", "User dislikes verbose explanations.")
        store.add("user", "Preferred language: Python.")

        snapshot_v2 = store.get_snapshot_block()
        print(f"[3] Snapshot after mid-session adds (should be UNCHANGED):")
        print(f"    'verbose' in snapshot: {'verbose' in snapshot_v2}")
        assert snapshot_v1 == snapshot_v2, (
            "Snapshot changed mid-session — prefix cache would break!"
        )

        # Live block reflects the new entries
        live_block = store.build_memory_context_block()
        print(f"[3] Live block contains 'verbose': {'verbose' in live_block}")
        assert "verbose" in live_block
        assert "Python" in live_block

        # ── 4. Snapshot after a new load_from_disk captures latest state ───
        store.load_from_disk()
        snapshot_v3 = store.get_snapshot_block()
        print(f"\n[4] Snapshot after 2nd load (should include 'verbose'):")
        print(f"    'verbose' in snapshot: {'verbose' in snapshot_v3}")
        assert "verbose" in snapshot_v3
        assert "Python" in snapshot_v3

        # ── 5. detect_drift — no drift immediately after load ─────────────
        drifted = store.detect_drift("memory")
        print(f"\n[5] Drift immediately after load: {drifted}  (expected: False)")
        assert not drifted

        # ── 6. Simulate external modification ─────────────────────────────
        memory_file = root / "MEMORY.md"
        time.sleep(0.01)  # ensure mtime differs
        memory_file.write_text(
            memory_file.read_text(encoding="utf-8")
            + "\n§\nInjected externally.",
            encoding="utf-8",
        )

        drifted = store.detect_drift("memory")
        print(f"[6] Drift after external write: {drifted}  (expected: True)")
        assert drifted

        # A .bak.<timestamp> backup should have been created
        bak_files = list(root.glob("MEMORY.md.bak.*"))
        print(f"[6] Backup files created: {[f.name for f in bak_files]}")
        assert len(bak_files) == 1, f"Expected 1 backup, got {len(bak_files)}"

        # Second call: mtime already updated, no new drift
        drifted_again = store.detect_drift("memory")
        print(f"[6] Second drift check: {drifted_again}  (expected: False)")
        assert not drifted_again

        # ── 7. char_counts utility ─────────────────────────────────────────
        counts = store.char_counts()
        print(f"\n[7] Character counts: {counts}")
        assert counts["memory"] > 0
        assert counts["user"] > 0

        print("\n✓ All memory snapshot and drift detection assertions passed.")


if __name__ == "__main__":
    demo()
