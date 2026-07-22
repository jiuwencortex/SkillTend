# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Background review prompt texts.

Faithful translations of Hermes's three review prompts into the Jiuwen context.
"""
from agent_evolving.online.review_executor.stages.stage02_prompt_selector.prompts.skill import SKILL_REVIEW_PROMPT


COMBINED_REVIEW_PROMPT = """\
Review the conversation above for two things: memory-worthy user information \
and skill improvements.

━━━  MEMORY  ━━━
Save a memory entry if the user revealed durable preferences, personal details,
working style, or explicit expectations about your behaviour.
Use memory_write(target="memory", ...) or memory_write(target="user", ...).

━━━  SKILLS  ━━━
""" + SKILL_REVIEW_PROMPT.split("━━━  SIGNALS TO LOOK FOR")[1].split("━━━  OUTPUT FORMAT")[0] + """
━━━  OUTPUT FORMAT  ━━━
Call memory_write and/or skill_write/skill_patch/skill_create as needed.
If nothing needs updating, say 'No changes needed.' and stop.
"""
