# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared data types for agent_evolving_hermes.

Contains enums and dataclasses used by both the online
(BackgroundReviewRail) and offline (GEPA skill evolver) tracks.
"""
from dataclasses import dataclass


@dataclass
class ReviewAction:
    """One write action successfully emitted by background review."""
    action_type: str
    """One of: skill_create, skill_edit, skill_patch, skill_delete,
    memory_add, memory_replace, memory_remove."""

    target_name: str
    """Skill name, or 'memory' / 'user' for memory actions."""

    summary: str
    """Human-readable one-line description, e.g. 'Updated git-review'."""

    write_origin: str = "background_review"
    execution_context: str = "background_review"
    session_id: str = ""
    parent_session_id: str = ""
