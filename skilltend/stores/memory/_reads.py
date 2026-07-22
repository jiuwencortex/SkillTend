# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Read-only access methods mixin."""
from __future__ import annotations

from typing import List

from agent_evolving.online.stores.memory._io_helpers import _read_entries


class _ReadsMixin:
    """Provides read_all, build_memory_context_block, char_counts.

    Subclass must provide:
        self._paths: Dict[str, Path]
    """

    def read_all(self, target: str) -> List[str]:
        if target not in self._paths:  # type: ignore[attr-defined]
            return []
        return _read_entries(self._paths[target])  # type: ignore[attr-defined]

    def build_memory_context_block(self) -> str:
        """Build a <memory-context> fenced block for injection into the system prompt.

        Mirrors Hermes build_memory_context_block() in system_prompt.py:
        both MEMORY.md and USER.md entries are combined and wrapped in a
        <memory-context> ... </memory-context> XML fence.

        This reflects LIVE disk state.  Use get_snapshot_block() for the
        session-start frozen snapshot needed for prefix-cache stability.

        Returns empty string if both stores are empty.
        """
        memory_entries = self.read_all("memory")
        user_entries = self.read_all("user")

        if not memory_entries and not user_entries:
            return ""

        sections: List[str] = []
        if memory_entries:
            sections.append("## Agent Memory")
            sections.extend(memory_entries)
        if user_entries:
            if sections:
                sections.append("")
            sections.append("## User Profile")
            sections.extend(user_entries)

        body = "\n".join(sections).strip()
        return f"<memory-context>\n{body}\n</memory-context>"

    def char_counts(self) -> dict:
        """Return current character usage for each store (useful for monitoring)."""
        return {
            target: sum(len(e) for e in self.read_all(target))
            for target in ("memory", "user")
        }
