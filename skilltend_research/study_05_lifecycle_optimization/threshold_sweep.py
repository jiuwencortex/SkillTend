# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Threshold grid search — importable module for study_05.

Evaluates every (stale_days, archive_days) pair and computes false-positive
and false-negative rates relative to the ground-truth "useful" label from
the simulated usage data.

This module contains only pure functions (no async, no CLI) so it can be
imported and called from both runner.py and from interactive notebooks.

Usage::

    from study_05_lifecycle_optimization.simulate_usage import simulate_library_usage
    from study_05_lifecycle_optimization.threshold_sweep import (
        sweep_thresholds,
        best_threshold_pair,
        format_sweep_table,
    )

    profiles = simulate_library_usage(n_skills=50, n_days=180, seed=42)
    results  = sweep_thresholds(profiles, observation_day=90)
    best     = best_threshold_pair(results)
    print(format_sweep_table(results))
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from study_05_lifecycle_optimization.simulate_usage import SkillUsageProfile

# Default grid from the research plan
DEFAULT_STALE_DAYS   = [7, 14, 21, 30, 45, 60]
DEFAULT_ARCHIVE_DAYS = [30, 60, 90, 120, 180]

# False-positive / false-negative cost weights
FP_WEIGHT = 0.7   # prefer not to lose useful skills
FN_WEIGHT = 0.3   # tolerate keeping some stale skills


@dataclass
class ThresholdResult:
    """Outcome of evaluating one (stale_days, archive_days) pair."""

    stale_days: int
    archive_days: int
    fp_rate: float      # fraction of useful skills incorrectly archived
    fn_rate: float      # fraction of stale skills kept active
    f_score: float      # FP_WEIGHT * fp_rate + FN_WEIGHT * fn_rate (lower is better)
    fp_count: int
    fn_count: int
    n_useful: int       # skills that will be used in the next 30 days
    n_genuinely_stale: int  # skills that will NOT be used in the next 30 days
    observation_day: int


def _evaluate_pair(
    profiles: List[SkillUsageProfile],
    stale_days: int,
    archive_days: int,
    observation_day: int,
    predict_window_days: int = 30,
) -> ThresholdResult:
    """Evaluate one (stale_days, archive_days) pair against the ground truth."""
    fp_count = 0
    fn_count = 0
    n_useful = 0
    n_stale = 0

    for p in profiles:
        days_since = p.days_since_last_use(observation_day)
        is_useful = p.will_be_used_after(observation_day, window=predict_window_days)

        if is_useful:
            n_useful += 1
        else:
            n_stale += 1

        would_archive = days_since >= archive_days
        # False positive: we would archive a skill that is still useful
        if would_archive and is_useful:
            fp_count += 1
        # False negative: skill is genuinely stale but we keep it active
        if not would_archive and not is_useful:
            fn_count += 1

    fp_rate = fp_count / max(1, n_useful)
    fn_rate = fn_count / max(1, n_stale)
    f_score = FP_WEIGHT * fp_rate + FN_WEIGHT * fn_rate

    return ThresholdResult(
        stale_days=stale_days,
        archive_days=archive_days,
        fp_rate=round(fp_rate, 4),
        fn_rate=round(fn_rate, 4),
        f_score=round(f_score, 4),
        fp_count=fp_count,
        fn_count=fn_count,
        n_useful=n_useful,
        n_genuinely_stale=n_stale,
        observation_day=observation_day,
    )


def sweep_thresholds(
    profiles: List[SkillUsageProfile],
    observation_day: int,
    stale_values: Optional[List[int]] = None,
    archive_values: Optional[List[int]] = None,
    predict_window_days: int = 30,
) -> List[ThresholdResult]:
    """Evaluate all valid (stale, archive) combinations.

    Invalid pairs where archive_days <= stale_days are skipped.

    Args:
        profiles:           Simulated skill usage profiles.
        observation_day:    Day at which to evaluate the thresholds.
        stale_values:       stale_days values to test.
        archive_values:     archive_days values to test.
        predict_window_days: How many days ahead to predict usage.

    Returns:
        List of ThresholdResult objects sorted by f_score ascending
        (best first, i.e. lowest combined error rate first).
    """
    stale_vals   = stale_values   or DEFAULT_STALE_DAYS
    archive_vals = archive_values or DEFAULT_ARCHIVE_DAYS

    results: List[ThresholdResult] = []
    for stale in stale_vals:
        for archive in archive_vals:
            if archive <= stale:
                continue
            result = _evaluate_pair(profiles, stale, archive, observation_day, predict_window_days)
            results.append(result)

    results.sort(key=lambda r: r.f_score)
    return results


def best_threshold_pair(results: List[ThresholdResult]) -> Optional[ThresholdResult]:
    """Return the ThresholdResult with the lowest f_score (fewest combined errors)."""
    return results[0] if results else None


def pareto_threshold_pairs(results: List[ThresholdResult]) -> List[ThresholdResult]:
    """Return pairs on the FP-rate vs. FN-rate Pareto frontier.

    Pareto-optimal means no other pair has both a lower FP rate AND
    a lower FN rate simultaneously.
    """
    frontier: List[ThresholdResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            if other.fp_rate <= r.fp_rate and other.fn_rate <= r.fn_rate:
                if other.fp_rate < r.fp_rate or other.fn_rate < r.fn_rate:
                    dominated = True
                    break
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda r: r.fp_rate)
    return frontier


def format_sweep_table(results: List[ThresholdResult], top_n: int = 15) -> str:
    """Return a human-readable table of the top N results."""
    header = (
        f"{'stale':>6}  {'archive':>7}  {'fp_rate':>7}  {'fn_rate':>7}  {'f_score':>7}"
    )
    divider = "-" * len(header)
    rows = [header, divider]
    for r in results[:top_n]:
        rows.append(
            f"{r.stale_days:>6}  {r.archive_days:>7}  "
            f"{r.fp_rate:>7.4f}  {r.fn_rate:>7.4f}  {r.f_score:>7.4f}"
        )
    return "\n".join(rows)


def survival_informed_recommendation(
    results: List[ThresholdResult],
    profiles: List[SkillUsageProfile],
    observation_day: int,
) -> dict:
    """Combine threshold sweep with survival analysis to produce a recommendation.

    Returns a dict with:
      stale_days, archive_days, rationale, survival_stats
    """
    from study_05_lifecycle_optimization.simulate_usage import survival_analysis

    best = best_threshold_pair(results)
    survival = survival_analysis(profiles, observation_day)

    # Find the recency band at which survival rate drops below 0.5
    stale_signal = None
    for band in [7, 14, 21, 30, 45, 60, 90]:
        key = f"last_used_within_{band}d"
        if key in survival:
            rate = survival[key].get("survival_rate")
            if rate is not None and rate < 0.50:
                stale_signal = band
                break

    return {
        "recommended_stale_days": best.stale_days if best else 30,
        "recommended_archive_days": best.archive_days if best else 90,
        "f_score": best.f_score if best else None,
        "fp_rate": best.fp_rate if best else None,
        "fn_rate": best.fn_rate if best else None,
        "survival_stale_signal_days": stale_signal,
        "survival_stats": survival,
        "rationale": (
            f"Optimal thresholds minimize F = {FP_WEIGHT}*FP + {FN_WEIGHT}*FN. "
            f"Survival analysis suggests stale signal at ~{stale_signal} days "
            f"(survival rate drops below 50%)."
        ) if best else "Insufficient data.",
    }
