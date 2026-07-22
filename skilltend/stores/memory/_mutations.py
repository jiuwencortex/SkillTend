# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Write-mutation methods (add / replace / remove) mixin."""
from __future__ import annotations

from typing import Tuple

from agent_evolving.online.stores.memory._io_helpers import (
    _ENTRY_DELIMITER,
    _read_entries,
    _write_entries,
)


class _MutationsMixin:
    """Provides add, replace, remove.

    Subclass must provide:
        self._paths: Dict[str, Path]
        self._limits: Dict[str, int]
        self._acquire(target) -> IO
        self._release(fh) -> None
    """

    def add(self, target: str, content: str) -> Tuple[bool, str]:
        content = content.strip()
        if not content:
            return False, "Content must not be empty."
        if target not in self._paths:  # type: ignore[attr-defined]
            return False, f"Unknown target '{target}'. Use 'memory' or 'user'."

        fh = self._acquire(target)  # type: ignore[attr-defined]
        try:
            entries = _read_entries(self._paths[target])  # type: ignore[attr-defined]
            # Exact duplicate rejection (mirrors Hermes memory_tool.py lines 321-323)
            if content in entries:
                return False, "Entry already exists (no duplicate added)."
            new_total = (
                sum(len(e) for e in entries)
                + len(content)
                + len(_ENTRY_DELIMITER)
            )
            if new_total > self._limits[target]:  # type: ignore[attr-defined]
                return False, (
                    f"Would exceed {target} limit of {self._limits[target]} chars. "  # type: ignore[attr-defined]
                    f"Current: {sum(len(e) for e in entries)}, adding: {len(content)}."
                )
            entries.append(content)
            _write_entries(self._paths[target], entries)  # type: ignore[attr-defined]
        finally:
            self._release(fh)  # type: ignore[attr-defined]
        return True, f"Added to {target}."

    def replace(self, target: str, old_text: str, new_content: str) -> Tuple[bool, str]:
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text or not new_content:
            return False, "old_text and new_content must not be empty."
        if target not in self._paths:  # type: ignore[attr-defined]
            return False, f"Unknown target '{target}'."

        fh = self._acquire(target)  # type: ignore[attr-defined]
        try:
            entries = _read_entries(self._paths[target])  # type: ignore[attr-defined]
            indices = [i for i, e in enumerate(entries) if old_text in e]
            if not indices:
                return False, "old_text not found in any entry."
            if len(indices) > 1:
                return False, "old_text matches multiple entries — be more specific."
            entries[indices[0]] = new_content
            _write_entries(self._paths[target], entries)  # type: ignore[attr-defined]
        finally:
            self._release(fh)  # type: ignore[attr-defined]
        return True, f"Replaced entry in {target}."

    def remove(self, target: str, old_text: str) -> Tuple[bool, str]:
        old_text = old_text.strip()
        if not old_text:
            return False, "old_text must not be empty."
        if target not in self._paths:  # type: ignore[attr-defined]
            return False, f"Unknown target '{target}'."

        fh = self._acquire(target)  # type: ignore[attr-defined]
        try:
            entries = _read_entries(self._paths[target])  # type: ignore[attr-defined]
            indices = [i for i, e in enumerate(entries) if old_text in e]
            if not indices:
                return False, "old_text not found."
            entries.pop(indices[0])
            _write_entries(self._paths[target], entries)  # type: ignore[attr-defined]
        finally:
            self._release(fh)  # type: ignore[attr-defined]
        return True, f"Removed entry from {target}."
