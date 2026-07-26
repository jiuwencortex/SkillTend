# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Skill interference experiment — importable trial execution module.

Separates the core trial logic from the CLI runner so it can be:
  - Imported by other studies
  - Called from notebooks
  - Unit-tested in isolation

A trial measures whether background review firing at turn T+Δ helps or
hurts downstream task success (the remainder of the session after the patch).

Design
------
For each (domain, delta, seed) combination we run two sessions in parallel:

  Treatment  — review fires at turn `split + delta`; we score skill quality
               from the treatment library at session end.

  Control    — no review fires; we score skill quality from the unmodified
               seed library.

The effect size is Cohen's d over the Q-score distributions.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from shared.session_simulator import SessionSimulator
from shared.skill_quality_scorer import SkillQualityScorer
from shared.task_bank import TASK_BANK


@dataclass
class TrialResult:
    """Outcome of one (domain, delta, seed) interference trial."""

    domain: str
    delta: int          # turns between first skill use and review fire
    seed: int
    q_treatment: float  # Q score after review-patched library
    q_control: float    # Q score from unpatched control library
    q_delta: float      # q_treatment - q_control  (positive = review helped)
    review_fired: bool
    review_latency_s: float


@dataclass
class DeltaSummary:
    """Aggregated results across all seeds for one (domain, delta) pair."""

    domain: str
    delta: int
    n_trials: int
    q_treatment_mean: float
    q_control_mean: float
    q_delta_mean: float
    cohens_d: float
    direction: str       # "beneficial" | "harmful" | "neutral"
    review_fired_fraction: float


def _cohens_d(treatment: List[float], control: List[float]) -> float:
    """Compute Cohen's d between two groups."""
    all_vals = treatment + control
    n = len(all_vals)
    if n < 2:
        return 0.0
    grand_mean = sum(all_vals) / n
    pooled_var = sum((v - grand_mean) ** 2 for v in all_vals) / (n - 1)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    mean_t = sum(treatment) / len(treatment)
    mean_c = sum(control) / len(control)
    return (mean_t - mean_c) / pooled_std


def _direction(cohens_d: float) -> str:
    if cohens_d > 0.1:
        return "beneficial"
    if cohens_d < -0.1:
        return "harmful"
    return "neutral"


async def run_trial(
    domain: str,
    delta: int,
    seed: int,
    sim: SessionSimulator,
    scorer: SkillQualityScorer,
    model: str,
) -> TrialResult:
    """Run one interference trial and return a TrialResult.

    Args:
        domain:  Task domain (e.g. 'code_debug').
        delta:   Number of turns between skill use and review fire.
        seed:    Random seed for reproducibility.
        sim:     Initialised SessionSimulator (may be dry_run).
        scorer:  Initialised SkillQualityScorer (may be dry_run).
        model:   LLM model name for the review call.
    """
    from skilltend.config import ReviewerConfig

    task_ids = [t.id for t in TASK_BANK.by_domain(domain)[:3]]
    if not task_ids:
        task_ids = TASK_BANK.ids()[:3]

    full_messages = sim.generate_multi_task_session(task_ids, seed=seed)
    split = max(2, len(full_messages) // 2)

    # Filler turns representing the Δ gap
    rng = random.Random(seed + 9000)
    filler = sim._filler_turns(domain, delta, rng) if delta > 0 else []

    tasks_for_scoring = TASK_BANK.by_domain(domain)[:3] or TASK_BANK.all()[:3]

    # ── Treatment ──────────────────────────────────────────────────────────
    skills_root_t = sim.build_skill_library(domains=[domain])
    config_t = ReviewerConfig(skills_root=skills_root_t)
    review_messages = full_messages[:split] + filler

    t0 = time.monotonic()
    try:
        await sim.replay_review(review_messages, config_t, model=model)
        review_fired = True
    except Exception:
        review_fired = False
    latency = time.monotonic() - t0

    skill_paths_t = list(skills_root_t.rglob("SKILL.md"))
    q_t = 0.5
    if skill_paths_t:
        try:
            sc = await scorer.score(skill_paths_t[0].read_text(encoding="utf-8"), tasks_for_scoring)
            q_t = sc.q
        except Exception:
            pass
    sim.cleanup_skill_library(skills_root_t)

    # ── Control (no review) ────────────────────────────────────────────────
    skills_root_c = sim.build_skill_library(domains=[domain])
    skill_paths_c = list(skills_root_c.rglob("SKILL.md"))
    q_c = 0.5
    if skill_paths_c:
        try:
            sc = await scorer.score(skill_paths_c[0].read_text(encoding="utf-8"), tasks_for_scoring)
            q_c = sc.q
        except Exception:
            pass
    sim.cleanup_skill_library(skills_root_c)

    return TrialResult(
        domain=domain,
        delta=delta,
        seed=seed,
        q_treatment=round(q_t, 4),
        q_control=round(q_c, 4),
        q_delta=round(q_t - q_c, 4),
        review_fired=review_fired,
        review_latency_s=round(latency, 3),
    )


async def run_delta_trials(
    domain: str,
    delta: int,
    n_seeds: int,
    sim: SessionSimulator,
    scorer: SkillQualityScorer,
    model: str,
    concurrency: int = 4,
) -> DeltaSummary:
    """Run `n_seeds` trials for one (domain, delta) pair and summarise."""
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(seed: int) -> TrialResult:
        async with sem:
            return await run_trial(domain, delta, seed, sim, scorer, model)

    trial_results = await asyncio.gather(*[_bounded(s) for s in range(n_seeds)])

    q_t_vals = [r.q_treatment for r in trial_results]
    q_c_vals = [r.q_control   for r in trial_results]
    q_d_vals = [r.q_delta     for r in trial_results]
    fired_frac = sum(1 for r in trial_results if r.review_fired) / len(trial_results)

    d = _cohens_d(q_t_vals, q_c_vals)

    return DeltaSummary(
        domain=domain,
        delta=delta,
        n_trials=len(trial_results),
        q_treatment_mean=round(sum(q_t_vals) / len(q_t_vals), 4),
        q_control_mean=round(sum(q_c_vals)   / len(q_c_vals), 4),
        q_delta_mean=round(sum(q_d_vals)     / len(q_d_vals), 4),
        cohens_d=round(d, 4),
        direction=_direction(d),
        review_fired_fraction=round(fired_frac, 3),
    )


def warm_up_turn(
    ps_by_turn: List[Tuple[int, float]],
    stability_threshold: float = 0.85,
) -> Optional[int]:
    """Return the first turn at which Patch Stability exceeds the threshold.

    Args:
        ps_by_turn: List of (turn_number, ps_score) pairs sorted by turn.
        stability_threshold: PS value considered "stable".

    Returns:
        The turn number, or None if stability is never reached.
    """
    for turn, ps in sorted(ps_by_turn):
        if ps >= stability_threshold:
            return turn
    return None
