# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared data types for agent_evolving_hermes.

Contains enums and dataclasses used by both the online
(BackgroundReviewRail) and offline (GEPA skill evolver) tracks.
"""
from enum import Enum


class ReviewMode(str, Enum):
    """Which stores the background review pass should update."""
    MEMORY_ONLY = "memory_only"
    SKILLS_ONLY = "skills_only"
    COMBINED = "combined"
