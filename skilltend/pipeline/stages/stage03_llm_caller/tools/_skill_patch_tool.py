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


SKILL_PATCH_TOOL = {
    "name": "skill_patch",
    "description": (
        "Apply a targeted string-replacement patch to an existing SKILL.md. "
        "Prefer this over skill_write for small targeted changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace.",
            },
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "default": False,
                "description": "Replace all occurrences (default: first only).",
            },
        },
        "required": ["name", "old_string", "new_string"],
    },
}

