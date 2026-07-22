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


MEMORY_WRITE_TOOL = {
    "name": "memory_write",
    "description": "Add, replace, or remove an entry in the agent memory or user profile store.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"]},
            "target": {"type": "string", "enum": ["memory", "user"]},
            "content": {
                "type": "string",
                "description": "New content (required for add/replace).",
            },
            "old_text": {
                "type": "string",
                "description": "Text to find (required for replace/remove).",
            },
        },
        "required": ["action", "target"],
    },
}