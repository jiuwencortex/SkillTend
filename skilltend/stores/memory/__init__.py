# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Simple flat-file memory store.

Mirrors Hermes MemoryStore exactly.
Two stores: "memory" (agent observations) and "user" (user profile).
Uses fcntl file locking and atomic writes.

Package layout:
    _io_helpers.py    — module-level I/O helpers (_ENTRY_DELIMITER, _atomic_write, …)
    _locking.py       — _LockMixin  (_acquire / _release)
    _snapshot.py      — _SnapshotMixin  (load_from_disk / get_snapshot_block)
    _drift.py         — _DriftMixin  (detect_drift)
    _mutations.py     — _MutationsMixin  (add / replace / remove)
    _reads.py         — _ReadsMixin  (read_all / build_memory_context_block / char_counts)
    _memory_store.py  — MemoryStore (composes all mixins)
    __init__.py       — re-exports MemoryStore
"""
from agent_evolving.online.stores.memory.memory_store import MemoryStore

__all__ = ["MemoryStore"]
