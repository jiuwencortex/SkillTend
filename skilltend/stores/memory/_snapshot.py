# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Frozen system-prompt snapshot mixin.

Mirrors Hermes memory_tool.py _system_prompt_snapshot design:
load_from_disk() freezes the snapshot once at session start so the
prefix-cache key never changes mid-session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


class _SnapshotMixin:
    """Provides load_from_disk / get_snapshot_block.

    Subclass must provide:
        self._paths: Dict[str, Path]
        self.build_memory_context_block() -> str   (from _ReadsMixin)
    """

    def load_from_disk(self) -> None:
        """Load memory from disk and freeze a system-prompt snapshot.

        Call once at session start (e.g. in gateway hydration).
        The snapshot is byte-identical across the entire session so the
        prefix cache key never changes.

        Subsequent add/replace/remove writes mutate disk but not the snapshot.
        """
        self._snapshot_mtimes: Dict[str, Optional[float]] = {
            t: (self._paths[t].stat().st_mtime if self._paths[t].exists() else None)  # type: ignore[attr-defined]
            for t in ("memory", "user")
        }
        self._system_prompt_snapshot: Optional[str] = self.build_memory_context_block()  # type: ignore[attr-defined]

    def get_snapshot_block(self) -> str:
        """Return the frozen system-prompt snapshot (from load_from_disk()).

        Falls back to the current live block if load_from_disk() was never called.
        """
        if self._system_prompt_snapshot is not None:  # type: ignore[attr-defined]
            return self._system_prompt_snapshot  # type: ignore[attr-defined]
        return self.build_memory_context_block()  # type: ignore[attr-defined]
