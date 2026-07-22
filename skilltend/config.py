# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Configuration dataclasses for online (BackgroundReviewRail) track"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BackgroundReviewConfig:
    """All tunable parameters for the background review rail.

    Mirrors Hermes cli-config.yaml [memory] and [skills] sections.
    """
    enabled: bool = True

    # ── Trigger thresholds ──────────────────────────────────────────────────
    skill_nudge_interval: int = 10
    """Trigger skill review after this many tool-call completions (0 = disabled)."""

    memory_nudge_interval: int = 10
    """Trigger memory review after this many user turns (0 = disabled)."""

    # ── Memory limits ────────────────────────────────────────────────────────
    memory_char_limit: int = 2200
    """Maximum character count for agent memory store (~800 tokens)."""

    user_char_limit: int = 1375
    """Maximum character count for user profile store (~500 tokens)."""

    # ── Skill constraints ────────────────────────────────────────────────────
    max_skill_size: int = 100_000
    """Maximum characters allowed in a single SKILL.md file."""

    max_skill_growth_ratio: float = 0.20
    """Background review edits may not grow a skill by more than 20%."""

    # ── Execution budget ─────────────────────────────────────────────────────
    review_max_iterations: int = 16
    """Maximum LLM iterations inside one background review run."""

    review_timeout_seconds: float = 120.0
    """Hard wall-clock timeout for a background review task."""

    # ── Skill storage ─────────────────────────────────────────────────────────
    skills_root: Optional[Path] = None
    """Root directory for SKILL.md files. Defaults to ~/.jiuwen/skills/"""

    memory_root: Optional[Path] = None
    """Root directory for memory files. Defaults to ~/.jiuwen/memories/"""

    # ── Model ────────────────────────────────────────────────────────────────
    review_model: Optional[str] = None
    """LLM model for background review. None = inherit from parent agent."""

    # ── Guards ───────────────────────────────────────────────────────────────
    protected_skill_names: List[str] = field(default_factory=list)
    """Skill names that background review must never modify (immutable skills)."""

    flush_min_turns: int = 6
    """Only trigger a review on exit if session had at least this many user turns."""

    # ── Curator (skill lifecycle maintenance) ─────────────────────────────────
    curator_enabled: bool = True
    """Enable the skill lifecycle curator (ACTIVE → STALE → ARCHIVED transitions)."""

    curator_interval_hours: float = 168.0
    """Minimum hours between curator runs (default 7 days, mirrors Hermes)."""

    curator_min_idle_seconds: float = 7200.0
    """Curator only runs when the agent has been idle at least this long (default 2 h)."""

    curator_stale_after_days: int = 30
    """Mark a skill STALE when unused for this many days (mirrors Hermes)."""

    curator_archive_after_days: int = 90
    """Archive a skill when unused for this many days (must be > stale_after_days)."""

    curator_state_path: Optional[Path] = None
    """Path for the JSON file that persists curator scheduling state.
    Defaults to skills_root.parent / 'curator_state.json' when None."""

    curator_llm_consolidation: bool = False
    """Run the optional LLM consolidation pass (identify and merge overlapping skills).
    Disabled by default — expensive; enable only when reviewing a mature skill library."""

    curator_consolidation_model: Optional[str] = None
    """LLM model for the consolidation pass.  None = inherit review_model."""
