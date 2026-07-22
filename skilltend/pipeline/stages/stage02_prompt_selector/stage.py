from __future__ import annotations

from typing import Tuple

from .prompts import select_prompt
from .prompts import SYSTEM_PROMPT
from ....types import ReviewTrigger


def build_review_prompts(trigger: ReviewTrigger) -> Tuple[str, str]:
    """Select the review prompt for the trigger mode and return the system prompt.

    Returns (review_prompt, system_prompt).
    """
    review_prompt = select_prompt(trigger.mode)
    return review_prompt, SYSTEM_PROMPT
