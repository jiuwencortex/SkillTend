# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""External-drift detection mixin.

Mirrors Hermes memory_tool.py external-drift detection (lines 516-569):
if a memory file is modified externally between load and the current call,
a .bak.<timestamp> backup is created.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


class _DriftMixin:
    """Provides detect_drift(target).

    Subclass must provide:
        self._paths: Dict[str, Path]
        self._snapshot_mtimes: Dict[str, Optional[float]]
    """

    def detect_drift(self, target: str) -> bool:
        """Check whether target file was modified externally since load.

        If drift is detected:
          1. Creates a .bak.<timestamp> backup of the current file.
          2. Updates the recorded mtime so repeated calls don't re-backup.
          3. Returns True.

        Returns False if no drift, or if load_from_disk() was never called.
        """
        snapshot_mtimes: Dict[str, Optional[float]] = self._snapshot_mtimes  # type: ignore[attr-defined]
        if snapshot_mtimes.get(target) is None:
            return False
        paths: Dict[str, Path] = self._paths  # type: ignore[attr-defined]
        path = paths.get(target)
        if not path or not path.exists():
            return False
        current_mtime = path.stat().st_mtime
        if current_mtime <= snapshot_mtimes[target]:  # type: ignore[operator]
            return False
        # Drift detected — create backup
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = path.parent / f"{path.name}.bak.{ts}"
        try:
            import shutil
            shutil.copy2(str(path), str(backup))
        except OSError:
            pass
        # Update mtime record so we don't keep re-backing-up
        snapshot_mtimes[target] = current_mtime
        return True
