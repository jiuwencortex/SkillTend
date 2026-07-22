# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Low-level I/O helpers for the flat-file memory store."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

_ENTRY_DELIMITER = "\n§\n"


def _default_memory_root() -> Path:
    return Path.home() / ".jiuwen" / "memories"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_entries(path: Path) -> List[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    # Load-time deduplication, preserving order (mirrors Hermes line 158)
    seen: Dict[str, None] = {}
    for e in raw.split(_ENTRY_DELIMITER):
        e = e.strip()
        if e:
            seen[e] = None
    return list(seen.keys())


def _write_entries(path: Path, entries: List[str]) -> None:
    _atomic_write(path, _ENTRY_DELIMITER.join(entries))
