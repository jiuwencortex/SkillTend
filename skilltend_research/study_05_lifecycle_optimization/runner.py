# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 05 runner — lifecycle threshold optimization.

Usage::

    python -m study_05_lifecycle_optimization.runner \
        --simulated-days 180 \
        --output-dir results/

    python -m study_05_lifecycle_optimization.runner --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from shared.result_logger import ResultLogger
from shared.session_simulator import SessionSimulator
from shared.task_bank import TASK_BANK
from study_05_lifecycle_optimization.lifecycle_predictor import (
    LifecyclePredictor,
    SkillSnapshot,
    snapshot_from_files,
)

# Threshold grid from research plan
STALE_THRESHOLDS = [7, 14, 21, 30, 45, 60]
ARCHIVE_THRESHOLDS = [30, 60, 90, 120, 180]

# False-positive vs false-negative weight (alpha = prefer not to lose useful skills)
FP_WEIGHT = 0.7
FN_WEIGHT = 0.3


def _simulate_usage_pattern(n_days: int, skill_name: str, seed: int) -> List[int]:
    """Generate a synthetic list of days on which a skill is used.

    Skills follow a Poisson-like usage pattern with varying rates.
    """
    rng = random.Random(seed)
    # Each skill has a base usage rate (uses per week)
    rate = rng.uniform(0.1, 3.0)  # some skills are popular, some rarely used
    days_used: List[int] = []
    for day in range(n_days):
        # Poisson: P(use on day) = 1 - exp(-rate/7)
        import math
        prob = 1 - math.exp(-rate / 7.0)
        if rng.random() < prob:
            days_used.append(day)
    return days_used


def _evaluate_threshold_pair(
    stale_days: int,
    archive_days: int,
    usage_histories: Dict[str, List[int]],
    observation_day: int,
    skill_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    """Compute FP and FN rates for a given (stale_days, archive_days) pair."""
    # A "useful" skill is one that will be used in the next 30 days from observation_day
    fp_count = 0  # archived a skill that would have been used
    fn_count = 0  # kept a skill that won't be used (missed archive)
    total_useful = 0
    total_stale = 0

    for name, history in usage_histories.items():
        past = [d for d in history if d < observation_day]
        future = [d for d in history if observation_day <= d < observation_day + 30]

        is_useful = len(future) > 0
        if is_useful:
            total_useful += 1

        days_since = (observation_day - max(past)) if past else observation_day

        would_archive = days_since >= archive_days
        would_stale = days_since >= stale_days and not would_archive

        if would_archive and is_useful:
            fp_count += 1  # false positive: archived a useful skill
        if not would_archive and not is_useful:
            fn_count += 1  # false negative: kept a useless skill
            total_stale += 1

    n = max(1, len(usage_histories))
    fp_rate = fp_count / max(1, total_useful) if total_useful > 0 else 0.0
    fn_rate = fn_count / max(1, total_stale) if total_stale > 0 else 0.0
    f_score = FP_WEIGHT * fp_rate + FN_WEIGHT * fn_rate

    return {
        "stale_days": stale_days,
        "archive_days": archive_days,
        "fp_rate": round(fp_rate, 4),
        "fn_rate": round(fn_rate, 4),
        "f_score": round(f_score, 4),
        "fp_count": fp_count,
        "fn_count": fn_count,
        "n_skills": n,
        "observation_day": observation_day,
    }


async def _main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    sim = SessionSimulator(TASK_BANK, dry_run=args.dry_run)
    logger = ResultLogger(output_dir / "study_05_threshold_sweep.jsonl", study="05")

    n_days = args.simulated_days if not args.dry_run else 30
    n_skills = 20 if not args.dry_run else 5

    print("=" * 60)
    print("Study 05 — Lifecycle Threshold Optimization")
    print(f"Simulated days: {n_days}  |  Skills: {n_skills}")
    print("=" * 60)

    # Build skill library
    skills_root = sim.build_skill_library(domains=list(TASK_BANK.domains))
    skill_dirs = {p.parent.name: p.parent for p in skills_root.rglob("SKILL.md")}

    # Add synthetic skills to get to n_skills
    for i in range(n_skills - len(skill_dirs)):
        domain = list(TASK_BANK.domains)[i % len(TASK_BANK.domains)]
        extra_dir = skills_root / f"synthetic_{i:02d}"
        extra_dir.mkdir(exist_ok=True)
        (extra_dir / "SKILL.md").write_text(f"---\ntitle: Synthetic skill {i}\n---\n\nContent {i}.")
        SessionSimulator._write_usage_sidecar(extra_dir, f"synthetic_{i:02d}")
        skill_dirs[f"synthetic_{i:02d}"] = extra_dir

    # Simulate usage histories
    usage_histories = {
        name: _simulate_usage_pattern(n_days, name, seed=hash(name) % 10000)
        for name in skill_dirs
    }

    # Phase A: Threshold grid search
    print("\nPhase A: Threshold grid search")
    observation_day = n_days // 2  # evaluate at the midpoint
    for stale in (STALE_THRESHOLDS if not args.dry_run else [14, 30]):
        for archive in (ARCHIVE_THRESHOLDS if not args.dry_run else [60, 90]):
            if archive <= stale:
                continue
            result = _evaluate_threshold_pair(stale, archive, usage_histories, observation_day, skill_dirs)
            logger.log(
                run_id=f"stale{stale}_arch{archive}_{uuid.uuid4().hex[:6]}",
                config={"stale_days": stale, "archive_days": archive},
                metrics={"fp_rate": result["fp_rate"], "fn_rate": result["fn_rate"], "f_score": result["f_score"]},
                meta={"n_skills": result["n_skills"], "observation_day": observation_day},
            )
            print(f"  stale={stale:3d}  archive={archive:3d}  F={result['f_score']:.4f}  "
                  f"FP={result['fp_rate']:.3f}  FN={result['fn_rate']:.3f}")

    # Phase B: Feature-enriched predictor
    print("\nPhase B: Feature-enriched lifecycle predictor")
    snapshots: List[SkillSnapshot] = []
    for name, skill_dir in skill_dirs.items():
        history = usage_histories[name]
        for obs_day in [n_days // 4, n_days // 2, 3 * n_days // 4]:
            try:
                snap = snapshot_from_files(skill_dir, obs_day, history, n_days)
                snapshots.append(snap)
            except Exception:
                pass

    if snapshots:
        predictor = LifecyclePredictor()
        cv_result = predictor.fit(snapshots)
        (output_dir / "study_05_predictor_cv.json").write_text(json.dumps(cv_result, indent=2))
        print(f"  Predictor CV AUC = {cv_result.get('cv_auc_mean', 'N/A')}")
        print(f"  Feature importances: {cv_result.get('feature_importances', {})}")
        if not args.dry_run:
            predictor.save(output_dir / "study_05_lifecycle_predictor.pkl")

    # Best threshold recommendation
    records = ResultLogger.read_metrics(output_dir / "study_05_threshold_sweep.jsonl")
    if records:
        best = min(records, key=lambda r: r.get("f_score", 1.0))
        print(f"\nBest thresholds:")
        print(f"  curator_stale_after_days   = {best.get('stale_days')}")
        print(f"  curator_archive_after_days = {best.get('archive_days')}")
        print(f"  F-score = {best.get('f_score')} (lower is better)")

    sim.cleanup_skill_library(skills_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 05: Lifecycle threshold optimization")
    parser.add_argument("--simulated-days", type=int, default=180)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
