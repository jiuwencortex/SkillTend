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


SKILL_WRITE_TOOL = {
    "name": "skill_write",
    "description": (
        "Create a new skill or fully replace an existing skill's SKILL.md. "
        "Use skill_patch for small targeted changes. "
        "The content must begin with YAML frontmatter (--- name: ... description: ... ---)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "edit"],
                "description": "'create' for new skills, 'edit' to replace an existing skill.",
            },
            "name": {
                "type": "string",
                "description": "Skill directory name (lowercase, hyphens ok).",
            },
            "content": {
                "type": "string",
                "description": "Full SKILL.md text including frontmatter.",
            },
            "category": {
                "type": "string",
                "description": "Optional category subdirectory (only used with action='create').",
            },
        },
        "required": ["action", "name", "content"],
    },
}
