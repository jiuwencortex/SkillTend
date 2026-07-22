# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Background review prompt texts.

Faithful translations of Hermes's three review prompts into the Jiuwen context.
"""
from __future__ import annotations

from agent_evolving.online.review_executor.stages.stage02_prompt_selector.prompts.combined import COMBINED_REVIEW_PROMPT
from agent_evolving.online.review_executor.stages.stage02_prompt_selector.prompts.memory import MEMORY_REVIEW_PROMPT
from agent_evolving.online.review_executor.stages.stage02_prompt_selector.prompts.skill import SKILL_REVIEW_PROMPT
from agent_evolving.online.types import ReviewMode


def select_prompt(mode: "ReviewMode") -> str:  # noqa: F821
    """Return the correct prompt string for the given ReviewMode."""
    if mode == ReviewMode.MEMORY_ONLY:
        return MEMORY_REVIEW_PROMPT
    if mode == ReviewMode.SKILLS_ONLY:
        return SKILL_REVIEW_PROMPT
    return COMBINED_REVIEW_PROMPT
