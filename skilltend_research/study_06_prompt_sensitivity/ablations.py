# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Prompt variant definitions for Study 06 ablation experiments.

Each variant is a dict with the same keys used by the production prompt
system.  Ablations are defined by removing or replacing individual
structural elements of the full prompt.

Prompt elements (from the research plan):
  1. task_framing   — "You are a skill maintenance agent…"
  2. instruction_list — what to look for
  3. format_constraints — tool call output format
  4. negative_constraints — "Do not…"
  5. few_shot — example (skill, conversation, output) pairs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptVariant:
    """One prompt configuration to test in the ablation sweep."""

    name: str
    """Identifier for this variant."""

    task_framing: str
    """Opening framing paragraph."""

    instruction_list: str
    """The list of things the LLM should look for."""

    format_constraints: str
    """Output format / tool call instructions."""

    negative_constraints: str
    """Things the LLM must NOT do."""

    few_shot: str
    """Few-shot examples (empty string = zero-shot)."""

    ablated_elements: List[str] = field(default_factory=list)
    """Which elements were removed vs. the FULL variant (for reporting)."""

    def to_system_prompt(self) -> str:
        """Combine all elements into a single system prompt string."""
        parts = []
        if self.task_framing:
            parts.append(self.task_framing)
        if self.instruction_list:
            parts.append(self.instruction_list)
        if self.format_constraints:
            parts.append(self.format_constraints)
        if self.negative_constraints:
            parts.append(self.negative_constraints)
        if self.few_shot:
            parts.append(self.few_shot)
        return "\n\n".join(parts)


# ── Element text pools ────────────────────────────────────────────────────────

_TASK_FRAMING = """\
You are a skill maintenance agent. Your job is to review the agent's recent
conversation and improve the skill library and memory store to help the agent
perform better in future sessions."""

_INSTRUCTION_LIST = """\
When reviewing the conversation:
- Identify new techniques, patterns, or domain knowledge that should be captured.
- Find existing skills that are incomplete, outdated, or incorrect.
- Update agent memory with key facts about the user's environment or preferences.
- Prioritise high-signal information over noise."""

_FORMAT_CONSTRAINTS = """\
Use the available tools to make changes:
- skill_write(name, content) — create a new skill
- skill_patch(name, find, replace) — update part of an existing skill
- memory_write(key, value) — add or update a memory entry
Only call tools when you have identified a concrete improvement."""

_NEGATIVE_CONSTRAINTS = """\
Do NOT:
- Make changes that are not supported by the conversation.
- Duplicate information already present in the skill library.
- Write overly generic or boilerplate content.
- Modify skills unrelated to the conversation topics."""

_FEW_SHOT_EXAMPLE = """\
Example:
CONVERSATION: User asked how to fix a KeyError. Agent explained dict.get().
GOOD ACTION: skill_patch("python_debugging", "KeyError", "Use dict.get(key, default) to avoid KeyError.")
BAD ACTION: skill_write("python", "Python is a programming language.")"""


# ── Variant definitions ───────────────────────────────────────────────────────

FULL_PROMPT = PromptVariant(
    name="full",
    task_framing=_TASK_FRAMING,
    instruction_list=_INSTRUCTION_LIST,
    format_constraints=_FORMAT_CONSTRAINTS,
    negative_constraints=_NEGATIVE_CONSTRAINTS,
    few_shot="",
    ablated_elements=[],
)

FULL_WITH_FEW_SHOT = PromptVariant(
    name="full_few_shot",
    task_framing=_TASK_FRAMING,
    instruction_list=_INSTRUCTION_LIST,
    format_constraints=_FORMAT_CONSTRAINTS,
    negative_constraints=_NEGATIVE_CONSTRAINTS,
    few_shot=_FEW_SHOT_EXAMPLE,
    ablated_elements=[],
)

NO_TASK_FRAMING = PromptVariant(
    name="no_task_framing",
    task_framing="",
    instruction_list=_INSTRUCTION_LIST,
    format_constraints=_FORMAT_CONSTRAINTS,
    negative_constraints=_NEGATIVE_CONSTRAINTS,
    few_shot="",
    ablated_elements=["task_framing"],
)

NO_INSTRUCTION_LIST = PromptVariant(
    name="no_instruction_list",
    task_framing=_TASK_FRAMING,
    instruction_list="",
    format_constraints=_FORMAT_CONSTRAINTS,
    negative_constraints=_NEGATIVE_CONSTRAINTS,
    few_shot="",
    ablated_elements=["instruction_list"],
)

NO_FORMAT_CONSTRAINTS = PromptVariant(
    name="no_format_constraints",
    task_framing=_TASK_FRAMING,
    instruction_list=_INSTRUCTION_LIST,
    format_constraints="",
    negative_constraints=_NEGATIVE_CONSTRAINTS,
    few_shot="",
    ablated_elements=["format_constraints"],
)

NO_NEGATIVE_CONSTRAINTS = PromptVariant(
    name="no_negative_constraints",
    task_framing=_TASK_FRAMING,
    instruction_list=_INSTRUCTION_LIST,
    format_constraints=_FORMAT_CONSTRAINTS,
    negative_constraints="",
    few_shot="",
    ablated_elements=["negative_constraints"],
)

MINIMAL_PROMPT = PromptVariant(
    name="minimal",
    task_framing="Review the conversation and update skills and memory.",
    instruction_list="",
    format_constraints=_FORMAT_CONSTRAINTS,
    negative_constraints="",
    few_shot="",
    ablated_elements=["task_framing", "instruction_list", "negative_constraints"],
)

ALL_VARIANTS: List[PromptVariant] = [
    FULL_PROMPT,
    FULL_WITH_FEW_SHOT,
    NO_TASK_FRAMING,
    NO_INSTRUCTION_LIST,
    NO_FORMAT_CONSTRAINTS,
    NO_NEGATIVE_CONSTRAINTS,
    MINIMAL_PROMPT,
]


# ── Context truncation strategies ─────────────────────────────────────────────

@dataclass
class ContextStrategy:
    """How to truncate the conversation before passing to the review LLM."""

    name: str
    max_turns: Optional[int]  # None = full conversation

    def apply(self, messages: List[Dict]) -> List[Dict]:
        from typing import Dict as D
        if self.max_turns is None:
            return messages
        # Keep last max_turns pairs (user + assistant)
        user_turns = [m for m in messages if m.get("role") == "user"]
        cutoff = user_turns[-self.max_turns]["content"] if len(user_turns) >= self.max_turns else None
        if cutoff is None:
            return messages
        for i, m in enumerate(messages):
            if m.get("role") == "user" and m.get("content") == cutoff:
                return messages[i:]
        return messages


CONTEXT_STRATEGIES = [
    ContextStrategy("full", None),
    ContextStrategy("last_5_turns", 5),
    ContextStrategy("last_10_turns", 10),
    ContextStrategy("last_20_turns", 20),
]
