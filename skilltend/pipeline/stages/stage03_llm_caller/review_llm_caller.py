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

import json

from typing import Any, Dict, List
import logging
logger = logging.getLogger(__name__)

from .tools._review_tools import REVIEW_TOOLS


async def call_review_llm(
    system_prompt: str,
    conversation_text: str,
    review_prompt: str,
    model: str,
    max_iterations: int,
) -> List[Dict[str, Any]]:
    """Run a tool-enabled LLM conversation for background review.

    Uses litellm (same dependency Jiuwen uses for LLM calls in other places).
    Returns list of tool call dicts: [{"tool": name, "args": {...}}, ...]
    """
    try:
        import litellm
    except ImportError:
        logger.error("litellm not installed; cannot run background review LLM call.")
        return []

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation_text},
        {"role": "user", "content": review_prompt},
    ]
    tool_calls_made = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                tools=[{"type": "function", "function": t} for t in REVIEW_TOOLS],
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            logger.warning("BackgroundReview LLM call failed: %s", e)
            break

        choice = response.choices[0]
        message = choice.message

        # Collect any tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls_made.append(
                    {"tool": tc.function.name, "id": tc.id, "args": args}
                )
                # Append assistant + synthetic tool result to continue the loop
                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": [tc]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"queued": True}),
                    }
                )
        else:
            # No more tool calls — model is done
            break

        if choice.finish_reason == "stop":
            break

    return tool_calls_made
