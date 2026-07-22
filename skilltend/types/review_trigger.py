# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared data types for agent_evolving_hermes.

Contains enums and dataclasses used by both the online
(BackgroundReviewRail) and offline (GEPA skill evolver) tracks.
"""
from dataclasses import dataclass

from agent_evolving.online.types.review_mode import ReviewMode


@dataclass
class ReviewTrigger:
    """Snapshot of what triggered a background review pass."""
    mode: ReviewMode
    user_turn_count: int
    tool_iter_count: int
    session_id: str
