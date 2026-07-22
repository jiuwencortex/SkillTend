# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""fcntl-based exclusive file locking mixin."""
from __future__ import annotations

import fcntl
from pathlib import Path
from typing import IO


class _LockMixin:
    """Provides _acquire / _release for exclusive per-target file locks."""

    # Subclass must provide: self._lock_paths: Dict[str, Path]

    def _acquire(self, target: str) -> IO:
        lock_path: Path = self._lock_paths[target]  # type: ignore[attr-defined]
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "w")
        fcntl.flock(fh, fcntl.LOCK_EX)
        return fh

    def _release(self, fh: IO) -> None:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
