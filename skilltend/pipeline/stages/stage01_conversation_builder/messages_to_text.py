from __future__ import annotations

from typing import Any, Dict, List


def messages_to_text(messages: List[Dict[str, Any]]) -> str:
    """Convert conversation snapshot to plain text for the review prompt."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle tool-result message content blocks
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", block.get("content", "")))
                else:
                    parts.append(str(block))
            content = " ".join(parts)
        if content:
            lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)
