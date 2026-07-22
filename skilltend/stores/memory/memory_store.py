# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MemoryStore — composes all mixins into the public class."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from agent_evolving.online.stores.memory._io_helpers import _default_memory_root
from agent_evolving.online.stores.memory._locking import _LockMixin
from agent_evolving.online.stores.memory._snapshot import _SnapshotMixin
from agent_evolving.online.stores.memory._drift import _DriftMixin
from agent_evolving.online.stores.memory._mutations import _MutationsMixin
from agent_evolving.online.stores.memory import _ReadsMixin


class MemoryStore(
    _LockMixin,
    _SnapshotMixin,
    _DriftMixin,
    _MutationsMixin,
    _ReadsMixin,
):
    """File-backed key-value memory. Mirrors Hermes MemoryStore.

    Frozen snapshot:
        Call ``load_from_disk()`` once at session start.  The snapshot is
        stored in ``_system_prompt_snapshot`` and never updated mid-session,
        ensuring prefix-cache stability.  Subsequent ``add/replace/remove``
        calls mutate the live disk state but NOT the snapshot.

        Use ``get_snapshot_block()`` to inject the frozen state into the
        system prompt, and ``build_memory_context_block()`` to get the
        *current* live state (e.g. for tool call responses).

    Drift detection:
        Call ``detect_drift(target)`` to check whether a memory file was
        modified externally since it was loaded.  If drift is detected, a
        ``.bak.<timestamp>`` backup is created and the method returns True.
    """

    def __init__(
        self,
        memory_root: Optional[Path] = None,
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
    ):
        self._root = memory_root or _default_memory_root()
        self._limits: Dict[str, int] = {
            "memory": memory_char_limit,
            "user": user_char_limit,
        }
        self._paths: Dict[str, Path] = {
            "memory": self._root / "MEMORY.md",
            "user": self._root / "USER.md",
        }
        self._lock_paths: Dict[str, Path] = {
            "memory": self._root / "MEMORY.md.lock",
            "user": self._root / "USER.md.lock",
        }
        # Frozen snapshot: taken at load_from_disk(), never updated mid-session
        self._system_prompt_snapshot: Optional[str] = None
        # File mtimes at snapshot time, used for drift detection
        self._snapshot_mtimes: Dict[str, Optional[float]] = {
            "memory": None,
            "user": None,
        }
