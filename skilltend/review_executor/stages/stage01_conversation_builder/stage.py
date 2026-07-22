from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .messages_to_text import messages_to_text
from ....config import BackgroundReviewConfig
from ....stores.memory import MemoryStore


def build_conversation_context(
    messages_snapshot: List[Dict[str, Any]],
    config: BackgroundReviewConfig,
) -> Tuple[Path, MemoryStore, str]:
    """Resolve storage roots, create MemoryStore, serialise conversation to text.

    Returns (skills_root, memory_store, conversation_text).
    """
    skills_root = config.skills_root or (Path.home() / ".jiuwen" / "skills")
    memory_root = config.memory_root or (Path.home() / ".jiuwen" / "memories")
    memory_store = MemoryStore(
        memory_root=memory_root,
        memory_char_limit=config.memory_char_limit,
        user_char_limit=config.user_char_limit,
    )
    conversation_text = messages_to_text(messages_snapshot)
    return skills_root, memory_store, conversation_text
