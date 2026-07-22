from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import logging
logger = logging.getLogger(__name__)

from .tool_call_dispatcher import dispatch_tool_call
from ....config import BackgroundReviewConfig
from ....stores.memory import MemoryStore
from ....types import ReviewAction


async def dispatch_all_tool_calls(
    tool_calls: List[Dict[str, Any]],
    skills_root: Path,
    memory_store: MemoryStore,
    config: BackgroundReviewConfig,
    session_id: str,
) -> List[ReviewAction]:
    """Dispatch every tool call to skill_store / memory_store and collect actions.

    Successful calls append a ReviewAction; failed calls are logged at DEBUG.
    Returns the list of ReviewAction objects for all successful writes.
    """
    actions: List[ReviewAction] = []
    for tc in tool_calls:
        ok, msg, action = await dispatch_tool_call(
            tool_name=tc["tool"],
            args=tc["args"],
            skill_store_root=skills_root,
            memory_store=memory_store,
            config=config,
            session_id=session_id,
        )
        if ok and action:
            actions.append(action)
        elif not ok:
            logger.debug(
                "BackgroundReview tool call failed: %s — %s", tc["tool"], msg
            )
    return actions
