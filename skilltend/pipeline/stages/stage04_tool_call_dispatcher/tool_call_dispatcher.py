from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from ....config import BackgroundReviewConfig
from ....stores import MemoryStore
from ....stores import skill_create, skill_edit, skill_patch
from ....types import ReviewAction


async def dispatch_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    skill_store_root: Path,
    memory_store: MemoryStore,
    config: BackgroundReviewConfig,
    session_id: str,
) -> Tuple[bool, str, Optional[ReviewAction]]:
    """Execute one tool call. Returns (success, message, action_or_none)."""

    if tool_name == "skill_write":
        action_type = args.get("action", "edit")
        name = args.get("name", "")
        content = args.get("content", "")
        category = args.get("category")
        if action_type == "create":
            ok, msg = await skill_create(
                name,
                content,
                skill_store_root,
                category=category,
                protected_names=config.protected_skill_names,
            )
        else:
            ok, msg = await skill_edit(
                name,
                content,
                skill_store_root,
                protected_names=config.protected_skill_names,
            )
        if ok:
            return True, msg, ReviewAction(
                action_type=f"skill_{action_type}",
                target_name=name,
                summary=f"{'Created' if action_type == 'create' else 'Edited'} skill '{name}'",
                session_id=session_id,
            )
        return False, msg, None

    if tool_name == "skill_patch":
        name = args.get("name", "")
        old_s = args.get("old_string", "")
        new_s = args.get("new_string", "")
        replace_all = args.get("replace_all", False)
        ok, msg = await skill_patch(
            name,
            old_s,
            new_s,
            skill_store_root,
            replace_all=replace_all,
            protected_names=config.protected_skill_names,
        )
        if ok:
            return True, msg, ReviewAction(
                action_type="skill_patch",
                target_name=name,
                summary=f"Patched skill '{name}'",
                session_id=session_id,
            )
        return False, msg, None

    if tool_name == "memory_write":
        mem_action = args.get("action", "add")
        target = args.get("target", "memory")
        content = args.get("content", "")
        old_text = args.get("old_text", "")
        if mem_action == "add":
            ok, msg = memory_store.add(target, content)
        elif mem_action == "replace":
            ok, msg = memory_store.replace(target, old_text, content)
        elif mem_action == "remove":
            ok, msg = memory_store.remove(target, old_text)
        else:
            return False, f"Unknown memory action '{mem_action}'", None
        if ok:
            return True, msg, ReviewAction(
                action_type=f"memory_{mem_action}",
                target_name=target,
                summary=f"{'Memory' if target == 'memory' else 'User profile'} {mem_action}",
                session_id=session_id,
            )
        return False, msg, None

    return False, f"Unknown tool '{tool_name}'", None
