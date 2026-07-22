# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared data types for skilltend.

Contains enums and dataclasses used by both the online
(Reviewer) and offline (GEPA skill evolver) tracks.
"""
from typing import List, Optional

from dataclasses import dataclass, field

from skilltend.types.review_action import ReviewAction
from skilltend.types.review_trigger import ReviewTrigger


@dataclass
class ReviewResult:
    """Result returned by a completed background review pass."""
    trigger: ReviewTrigger
    actions: List[ReviewAction] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    summary_line: str = ""
    """Human-readable summary, e.g. 'Updated git-review · Added memory entry'."""
