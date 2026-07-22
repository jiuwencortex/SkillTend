# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared data types for skilltend.

Contains enums and dataclasses used by both the online
(Reviewer) and offline (GEPA skill evolver) tracks.
"""
from dataclasses import dataclass

from skilltend.types.review_mode import ReviewMode


@dataclass
class ReviewTrigger:
    """Snapshot of what triggered a background review pass."""
    mode: ReviewMode
    user_turn_count: int
    tool_iter_count: int
    session_id: str
