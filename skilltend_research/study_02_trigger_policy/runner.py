# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 02 runner — trigger policy optimization.

Runs the N×M grid sweep and then compares the best fixed policy against
AdaptiveTrigger on the same task distribution.

Usage::

    # Full sweep (expensive — use --sessions-per-config 5 for a quick run)
    python -m study_02_trigger_policy.runner \
        --model openai/gpt-4o-mini \
        --sessions-per-config 20 \
        --output-dir results/

    # Quick smoke test
    python -m study_02_trigger_policy.runner --dry-run --sessions-per-config 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import List

from shared.result_logger import ResultLogger
from shared.session_simulator import SessionSimulator
from shared.skill_quality_scorer import SkillQualityScorer
from shared.task_bank import TASK_BANK
from study_02_trigger_policy.adaptive_trigger import AdaptiveTrigger
from study_02_trigger_policy.sweep import run_sweep


# ── Adaptive trigger benchmark ────────────────────────────────────────────────


async def _run_adaptive_benchmark(
    output_path: Path,
    model: str,
    sessions: int,
    thresholds: List[float],
    dry_run: bool,
    concurrency: int,
) -> None:
    """Benchmark AdaptiveTrigger against a fixed (N=10, M=10) baseline."""
    from skilltend.config import ReviewerConfig

    sim = SessionSimulator(TASK_BANK, dry_run=dry_run)
    scorer = SkillQualityScorer(dry_run=dry_run, judge_model=model)
    logger = ResultLogger(output_path, study="02_adaptive")
    sem = asyncio.Semaphore(concurrency)

    async def _one_run(threshold: float, seed: int) -> None:
        async with sem:
            trigger = AdaptiveTrigger(
                distance_threshold=threshold,
                dry_run=dry_run,
            )

            task_ids = [t.id for t in TASK_BANK.by_domain("code_debug")[:2]]
            messages = sim.generate_multi_task_session(task_ids, seed=seed)
            skills_root = sim.build_skill_library(domains=["code_debug"])
            config = ReviewerConfig(skills_root=skills_root)

            # Check trigger at each user-turn boundary
            review_count = 0
            total_time = 0.0
            for i in range(1, len(messages), 2):  # every user-turn
                should_fire, reason = await trigger.check(messages[:i])
                if should_fire:
                    t0 = time.monotonic()
                    await sim.replay_review(messages[:i], config, model=model)
                    total_time += time.monotonic() - t0
                    review_count += 1
                    trigger.reset()

            # Score final skill library
            skill_paths = list(skills_root.rglob("SKILL.md"))
            q = 0.5
            if skill_paths:
                tasks = TASK_BANK.by_domain("code_debug")[:3]
                try:
                    scores = await scorer.score(skill_paths[0].read_text(), tasks)
                    q = scores.q
                except Exception:
                    pass

            logger.log(
                run_id=f"adaptive_t{threshold}_s{seed}_{uuid.uuid4().hex[:6]}",
                config={"trigger_type": "adaptive", "distance_threshold": threshold},
                metrics={"q_mean": round(q, 4), "review_count": review_count, "review_time_s": round(total_time, 3)},
                meta={"seed": seed},
            )
            sim.cleanup_skill_library(skills_root)

    tasks = [_one_run(t, s) for t in thresholds for s in range(sessions)]
    await asyncio.gather(*tasks)
    print(f"Adaptive benchmark: {logger.count()} records written to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def _main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    sweep_path = output_dir / "study_02_sweep.jsonl"
    adaptive_path = output_dir / "study_02_adaptive.jsonl"

    print("=" * 60)
    print("Study 02 — Trigger Policy Optimization")
    print("=" * 60)

    # Phase A: Grid sweep
    print("\nPhase A: N×M grid sweep")
    await run_sweep(
        output_path=sweep_path,
        model=args.model,
        sessions_per_config=args.sessions_per_config,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        skill_intervals=[3, 5, 10, 15, 30, 999] if args.dry_run else None,
        memory_intervals=[3, 5, 10, 15, 30, 999] if args.dry_run else None,
    )

    # Phase B: Adaptive trigger comparison
    print("\nPhase B: Adaptive trigger comparison")
    await _run_adaptive_benchmark(
        output_path=adaptive_path,
        model=args.model,
        sessions=max(2, args.sessions_per_config // 4),
        thresholds=[0.10, 0.20, 0.25, 0.35, 0.50],
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    )

    # Phase C: Analyze
    print("\nPhase C: Analysis")
    from study_02_trigger_policy.analyze import analyze
    findings = analyze(sweep_path, output_dir / "study_02_pareto.png", output_dir / "study_02_findings.json")
    knee = findings.get("knee_point")
    if knee:
        print(f"\nFinal recommendation:")
        print(f"  skill_nudge_interval  = {knee.get('skill_nudge_interval')}")
        print(f"  memory_nudge_interval = {knee.get('memory_nudge_interval')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 02: Trigger policy optimization")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--sessions-per-config", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
