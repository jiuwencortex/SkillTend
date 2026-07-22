# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""agent_evolving — Hermes-style self-evolution for Jiuwen."""

from skilltend.review_executor.stages.stage02_prompt_selector.prompts.skill import SKILL_REVIEW_PROMPT
from skilltend.review_executor.stages.stage02_prompt_selector.prompts.memory import MEMORY_REVIEW_PROMPT
from skilltend.review_executor.stages.stage02_prompt_selector.prompts.combined import COMBINED_REVIEW_PROMPT
from skilltend.review_executor.stages.stage02_prompt_selector.prompts.selector import select_prompt
from skilltend.review_executor.stages.stage02_prompt_selector.prompts.system import SYSTEM_PROMPT
