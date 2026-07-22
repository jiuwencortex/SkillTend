# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Write provenance / audit metadata for skill and memory writes.

Two systems in one module:

1. make_write_metadata() — explicit metadata dict for skill/memory writes.
   Mirrors Hermes build_memory_write_metadata().

2. ContextVar-based write-origin tracking — implicit per-task origin tag.
   Mirrors Hermes skill_provenance.py _write_origin ContextVar:
     _write_origin == "foreground"         → user-directed (never curator-managed)
     _write_origin == "background_review"  → agent-self-improvement (curator-eligible)

   Usage:
     # In a background review task:
     async with background_review_context():
         await skill_create(name, content, skills_root)  # tagged as agent-created

     # Anywhere:
     origin = get_write_origin()  # "foreground" | "background_review"
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from typing import Any, AsyncGenerator, Dict

# ── ContextVar-based origin tracking ─────────────────────────────────────────

_WRITE_ORIGIN: ContextVar[str] = ContextVar("_write_origin", default="foreground")


def get_write_origin() -> str:
    """Return the current write origin: 'foreground' or 'background_review'."""
    return _WRITE_ORIGIN.get()


def set_write_origin(origin: str) -> Token:
    """Set the write origin for the current task.

    Returns the ContextVar token — call _WRITE_ORIGIN.reset(token) to restore.
    Prefer the ``background_review_context()`` async context manager instead.
    """
    return _WRITE_ORIGIN.set(origin)


@asynccontextmanager
async def background_review_context() -> AsyncGenerator[None, None]:
    """Async context manager: mark all writes within as 'background_review'.

    Mirrors Hermes background_review.py which sets _write_origin before
    forking the review agent. Skills created within this context are
    marked ``created_by='agent'`` and become curator-eligible.

    Usage::

        async with background_review_context():
            ok, msg = await skill_create(name, content, skills_root)
    """
    token = _WRITE_ORIGIN.set("background_review")
    try:
        yield
    finally:
        _WRITE_ORIGIN.reset(token)


# ── Explicit metadata dict ────────────────────────────────────────────────────


def make_write_metadata(
    *,
    write_origin: str = "",
    execution_context: str = "background_review",
    session_id: str = "",
    parent_session_id: str = "",
    platform: str = "",
) -> Dict[str, Any]:
    """Build provenance metadata dict for skill/memory writes.

    ``write_origin`` defaults to the current ContextVar value so callers
    do not need to pass it explicitly inside background_review_context().
    """
    platform = platform or os.environ.get("JIUWEN_SESSION_SOURCE", "sdk")
    origin = write_origin or get_write_origin()
    result: Dict[str, Any] = {
        "write_origin": origin,
        "execution_context": execution_context,
        "tool_name": "background_review",
    }
    if session_id:
        result["session_id"] = session_id
    if parent_session_id:
        result["parent_session_id"] = parent_session_id
    if platform:
        result["platform"] = platform
    return result
