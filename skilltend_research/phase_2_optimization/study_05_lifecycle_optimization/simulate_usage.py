# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Skill usage pattern simulation — standalone module.

Generates synthetic per-skill usage histories (lists of days on which
a skill was accessed) over a configurable time horizon.

Skills follow a Poisson-like usage model with heterogeneous per-skill
rates, simulating the realistic mix of popular and rarely-used skills
observed in production agent libraries.

This module has no dependency on the runner or any async code.
It is imported by both threshold_sweep.py and runner.py.

Usage::

    from phase_2_optimization.study_05_lifecycle_optimization.simulate_usage import (
        simulate_library_usage,
        SkillUsageProfile,
    )

    profiles = simulate_library_usage(
        n_skills=50,
        n_days=180,
        seed=42,
    )
    for p in profiles:
        print(p.skill_name, len(p.use_days), "uses over 180 days")
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SkillUsageProfile:
    """Synthetic usage history for one skill over a simulated period."""

    skill_name: str
    """Unique skill identifier."""

    n_days: int
    """Length of the simulation window in days."""

    use_days: List[int]
    """Sorted list of day indices (0-indexed) on which the skill was accessed."""

    rate_uses_per_week: float
    """The Poisson rate parameter used to generate this history."""

    created_by: str = "agent"
    """'user' or 'agent' — mirrors the .usage.json field."""

    patch_count: int = 0
    """Number of times the skill was edited during the simulation window."""

    skill_char_len: int = 500
    """Approximate content length in characters."""

    def use_count_before(self, day: int) -> int:
        return sum(1 for d in self.use_days if d < day)

    def last_use_before(self, day: int) -> Optional[int]:
        past = [d for d in self.use_days if d < day]
        return max(past) if past else None

    def days_since_last_use(self, observation_day: int) -> float:
        last = self.last_use_before(observation_day)
        return float(observation_day - last) if last is not None else float(observation_day)

    def will_be_used_after(self, day: int, window: int = 30) -> bool:
        return any(day <= d < day + window for d in self.use_days)

    def avg_days_between_uses(self, observation_day: int) -> float:
        past = [d for d in self.use_days if d < observation_day]
        if len(past) < 2:
            return float(observation_day)
        return (max(past) - min(past)) / (len(past) - 1)


def _generate_use_days(
    n_days: int,
    rate_per_week: float,
    rng: random.Random,
) -> List[int]:
    """Sample days on which a skill is used using a Bernoulli trial per day.

    P(use on day d) = 1 - exp(-rate/7) for Poisson rate r uses/week.
    """
    prob = 1.0 - math.exp(-rate_per_week / 7.0)
    return [d for d in range(n_days) if rng.random() < prob]


def simulate_library_usage(
    n_skills: int = 50,
    n_days: int = 180,
    seed: int = 42,
    min_rate: float = 0.05,   # min uses/week (very rare skill)
    max_rate: float = 5.0,    # max uses/week (heavily used skill)
    user_skill_fraction: float = 0.3,  # fraction of skills created by "user"
    mean_skill_char_len: int = 600,
) -> List[SkillUsageProfile]:
    """Generate a synthetic skill library with heterogeneous usage rates.

    The rate distribution follows a log-uniform distribution, which
    produces the heavy-tailed usage pattern typically seen in real skill
    libraries (most skills are rarely used; a few are used frequently).

    Args:
        n_skills:             Number of skills to simulate.
        n_days:               Duration of the simulation window in days.
        seed:                 Random seed for full reproducibility.
        min_rate / max_rate:  Bounds for the log-uniform rate distribution.
        user_skill_fraction:  Fraction of skills whose `created_by` = "user".
        mean_skill_char_len:  Mean content length; individual skills vary ±50%.

    Returns:
        List of SkillUsageProfile objects, one per skill.
    """
    rng = random.Random(seed)

    log_min = math.log(min_rate)
    log_max = math.log(max_rate)

    profiles: List[SkillUsageProfile] = []
    for i in range(n_skills):
        rate = math.exp(rng.uniform(log_min, log_max))
        use_days = sorted(_generate_use_days(n_days, rate, rng))

        # Vary content length ±50%
        char_len = int(mean_skill_char_len * rng.uniform(0.5, 1.5))

        # Patch count: skills with higher use tend to be patched more often
        patch_count = max(0, int(len(use_days) * rng.uniform(0.05, 0.25)))

        created_by = "user" if rng.random() < user_skill_fraction else "agent"

        profiles.append(SkillUsageProfile(
            skill_name=f"skill_{i:03d}",
            n_days=n_days,
            use_days=use_days,
            rate_uses_per_week=round(rate, 4),
            created_by=created_by,
            patch_count=patch_count,
            skill_char_len=char_len,
        ))

    return profiles


def survival_analysis(
    profiles: List[SkillUsageProfile],
    observation_day: int,
    stale_threshold_days: int = 30,
) -> dict:
    """Compute survival statistics: how many skills remain useful at each recency band.

    A skill is "alive" (useful) if it will be used in the next 30 days from
    the observation point.

    Returns a dict with recency-band breakdowns useful for setting
    curator_stale_after_days.
    """
    bands = [7, 14, 21, 30, 45, 60, 90, 120, 180]
    results = {}
    for band in bands:
        in_band = [
            p for p in profiles
            if p.days_since_last_use(observation_day) <= band
        ]
        if not in_band:
            results[f"last_used_within_{band}d"] = {"n": 0, "survival_rate": None}
            continue
        still_useful = [p for p in in_band if p.will_be_used_after(observation_day, window=30)]
        results[f"last_used_within_{band}d"] = {
            "n": len(in_band),
            "survival_rate": round(len(still_useful) / len(in_band), 4),
        }

    # Also report overall: what fraction of all skills are useful?
    useful = [p for p in profiles if p.will_be_used_after(observation_day, window=30)]
    results["overall_useful_fraction"] = round(len(useful) / len(profiles), 4) if profiles else 0.0
    results["n_skills"] = len(profiles)
    results["observation_day"] = observation_day

    return results
