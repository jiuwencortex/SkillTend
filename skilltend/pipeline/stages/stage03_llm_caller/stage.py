# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Core background review execution.

Hermes pattern: forks AIAgent, restricts to skill_manage + memory tools,
runs review prompt over conversation snapshot, parses tool call outputs
into ReviewAction list.

Jiuwen mapping:
  - No forked AIAgent available; instead, make a direct LLM call with
    the conversation + review prompt and parse the JSON tool-call outputs.
  - Uses litellm (same dependency Jiuwen uses for other LLM calls).
  - Tool definitions for skill_write, skill_patch, skill_create,
    memory_write are defined here as JSON schemas and validated locally.
  - Actual writes are executed by skill_store.py and memory_store.py.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

from .review_llm_caller import call_review_llm
from ....config import BackgroundReviewConfig


async def call_llm_with_timeout(
    system_prompt: str,
    conversation_text: str,
    review_prompt: str,
    model: str,
    config: BackgroundReviewConfig,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Call the review LLM with a hard timeout, absorbing errors into the return value.

    Returns (tool_calls, error) where error is None on success.
    On timeout or exception, returns ([], error_message).
    """
    try:
        tool_calls = await asyncio.wait_for(
            call_review_llm(
                system_prompt=system_prompt,
                conversation_text=conversation_text,
                review_prompt=review_prompt,
                model=model,
                max_iterations=config.review_max_iterations,
            ),
            timeout=config.review_timeout_seconds,
        )
        return tool_calls, None
    except asyncio.TimeoutError:
        error = f"Background review timed out after {config.review_timeout_seconds}s"
        logger.warning(error)
        return [], error
    except Exception as e:
        error = f"Background review LLM call failed: {e}"
        logger.warning(error)
        return [], error
