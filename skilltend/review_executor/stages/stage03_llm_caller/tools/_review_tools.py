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

from agent_evolving.online.review_executor.stages.stage03_llm_caller.tools._memory_write_tool import MEMORY_WRITE_TOOL
from agent_evolving.online.review_executor.stages.stage03_llm_caller.tools._skill_patch_tool import SKILL_PATCH_TOOL
from agent_evolving.online.review_executor.stages.stage03_llm_caller.tools._skill_write_tool import SKILL_WRITE_TOOL

# ── Tool schemas exposed to the review LLM ───────────────────────────────────

REVIEW_TOOLS = [SKILL_WRITE_TOOL, SKILL_PATCH_TOOL, MEMORY_WRITE_TOOL]