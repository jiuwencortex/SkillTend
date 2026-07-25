# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 08 runner — skill library dynamics and coverage.

Usage::

    python -m study_08_library_dynamics.runner \
        --model openai/gpt-4o-mini \
        --sessions 200 \
        --output-dir results/

    python -m study_08_library_dynamics.runner --dry-run --sessions 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from shared.result_logger import ResultLogger
from shared.session_simulator import SessionSimulator
from shared.task_bank import TASK_BANK
from study_08_library_dynamics.coverage_metric import compute_library_coverage
from study_08_library_dynamics.coverage_gap_detector import CoverageGapDetector

# Library sizes at which to evaluate Phase C (LLM consolidation ROI)
CONSOLIDATION_EVAL_SIZES = [10, 25, 50, 100]


async def _run_growth_simulation(
    total_sessions: int,
    sim: SessionSimulator,
    model: str,
    dry_run: bool,
    logger: ResultLogger,
    task_queries: List[str],
    measurement_interval: int = 10,
) -> Path:
    """Simulate N sequential sessions and track library growth metrics."""
    from skilltend.config import ReviewerConfig

    skills_root = sim.build_skill_library(domains=[])  # start empty
    config = ReviewerConfig(skills_root=skills_root)

    # Cycle through all available tasks
    all_task_ids = TASK_BANK.ids()
    session_histories: List[List[Dict[str, Any]]] = []

    for session_idx in range(total_sessions):
        task_id = all_task_ids[session_idx % len(all_task_ids)]
        messages = sim.generate_session(task_id, seed=session_idx)
        session_histories.append(messages)

        # Replay review
        try:
            await sim.replay_review(messages, config, model=model)
        except Exception:
            pass

        # Measure every measurement_interval sessions
        if (session_idx + 1) % measurement_interval == 0 or session_idx == total_sessions - 1:
            coverage = await compute_library_coverage(
                skills_root=skills_root,
                task_queries=task_queries[:20],  # use first 20 task queries
                dry_run=dry_run,
            )
            logger.log(
                run_id=f"growth_s{session_idx + 1}_{uuid.uuid4().hex[:6]}",
                config={"phase": "A_growth", "session": session_idx + 1},
                metrics={
                    "lc": coverage["lc"],
                    "mean_pairwise_sim": coverage["mean_pairwise_sim"],
                    "gini": coverage["gini"],
                    "n_skills": coverage["n_skills"],
                },
            )
            print(f"  session={session_idx + 1:4d}  LC={coverage['lc']:.3f}  "
                  f"redundancy={coverage['mean_pairwise_sim']:.3f}  "
                  f"skills={coverage['n_skills']}")

    return skills_root


async def _run_consolidation_roi(
    skills_root: Path,
    target_sizes: List[int],
    model: str,
    dry_run: bool,
    task_queries: List[str],
    logger: ResultLogger,
) -> None:
    """Measure LLM consolidation ROI at different library sizes."""
    from skilltend.config import ReviewerConfig
    from skilltend.curator.curator import _run_phase2

    for size in target_sizes:
        skill_paths = list(skills_root.rglob("SKILL.md"))
        if len(skill_paths) < size:
            print(f"  Skipping size={size}: only {len(skill_paths)} skills in library")
            continue

        # Measure Q before consolidation
        coverage_before = await compute_library_coverage(skills_root, task_queries[:10], dry_run=dry_run)

        # Run Phase 2 consolidation
        suggestions_count = 0
        if not dry_run:
            config = ReviewerConfig(
                skills_root=skills_root,
                curator_llm_consolidation=True,
                curator_consolidation_model=model,
            )
            try:
                suggestions = await _run_phase2(config, model)
                suggestions_count = len(suggestions)
            except Exception as exc:
                print(f"  Phase 2 error: {exc}")

        coverage_after = await compute_library_coverage(skills_root, task_queries[:10], dry_run=dry_run)

        logger.log(
            run_id=f"consolidation_size{size}_{uuid.uuid4().hex[:6]}",
            config={"phase": "C_consolidation", "target_size": size},
            metrics={
                "lc_before": coverage_before["lc"],
                "lc_after": coverage_after["lc"],
                "lc_delta": round(coverage_after["lc"] - coverage_before["lc"], 4),
                "redundancy_before": coverage_before["mean_pairwise_sim"],
                "redundancy_after": coverage_after["mean_pairwise_sim"],
                "suggestions_count": suggestions_count,
                "n_skills": len(skill_paths),
            },
        )
        print(f"  size≈{size}  LC: {coverage_before['lc']:.3f}→{coverage_after['lc']:.3f}  "
              f"suggestions={suggestions_count}")


async def _main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_path = output_dir / "study_08_dynamics.jsonl"

    print("=" * 60)
    print("Study 08 — Skill Library Dynamics & Coverage")
    print(f"Sessions: {args.sessions}")
    print("=" * 60)

    sim = SessionSimulator(TASK_BANK, dry_run=args.dry_run)
    logger = ResultLogger(output_path, study="08")

    # Collect all task queries for coverage measurement
    task_queries = [t.user_query for t in TASK_BANK.all()]
    n_sessions = args.sessions if not args.dry_run else 20
    measurement_interval = max(1, n_sessions // 20)

    # Phase A+B: Growth curve simulation
    print("\nPhase A+B: Library growth simulation")
    skills_root = await _run_growth_simulation(
        total_sessions=n_sessions,
        sim=sim,
        model=args.model,
        dry_run=args.dry_run,
        logger=logger,
        task_queries=task_queries,
        measurement_interval=measurement_interval,
    )

    # Phase C: Consolidation ROI
    print("\nPhase C: LLM consolidation ROI")
    target_sizes = (CONSOLIDATION_EVAL_SIZES if not args.dry_run
                    else [s for s in CONSOLIDATION_EVAL_SIZES if s <= 10])
    await _run_consolidation_roi(
        skills_root=skills_root,
        target_sizes=target_sizes,
        model=args.model,
        dry_run=args.dry_run,
        task_queries=task_queries,
        logger=logger,
    )

    # Phase D: Coverage gap detection
    print("\nPhase D: Coverage gap detection")
    all_sessions = [
        sim.generate_session(t.id, seed=i)
        for i, t in enumerate(TASK_BANK.all()[:20])
    ]
    detector = CoverageGapDetector(skills_root, dry_run=args.dry_run)
    gaps = await detector.detect(all_sessions[-20:])  # use last 20 sessions
    print(detector.report(gaps))

    # Save gap report
    gap_report = [
        {"domain": g.domain, "tool_call_count": g.tool_call_count,
         "max_rr": g.max_rr, "severity": g.severity,
         "sample_query": g.representative_query}
        for g in gaps
    ]
    (output_dir / "study_08_coverage_gaps.json").write_text(
        json.dumps(gap_report, indent=2), encoding="utf-8"
    )

    # Analysis
    records = ResultLogger.read_metrics(output_path)
    growth = [r for r in records if r.get("phase") == "A_growth"]
    consol = [r for r in records if r.get("phase") == "C_consolidation"]

    if growth:
        # LC saturation: find session where marginal gain < 0.005
        lc_series = [(r.get("session", 0), r.get("lc", 0)) for r in sorted(growth, key=lambda r: r.get("session", 0))]
        saturation_session = None
        for i in range(1, len(lc_series)):
            gain = lc_series[i][1] - lc_series[i - 1][1]
            if gain < 0.005:
                saturation_session = lc_series[i][0]
                break
        print(f"\nLC saturation at session ≈ {saturation_session}")

    if consol:
        min_roi_size = None
        for r in sorted(consol, key=lambda r: r.get("target_size", 0)):
            if r.get("lc_delta", 0) > 0:
                min_roi_size = r.get("target_size")
                break
        if min_roi_size:
            print(f"Enable curator_llm_consolidation when library >= {min_roi_size} skills")

    sim.cleanup_skill_library(skills_root)
    print(f"\nDone. {logger.count()} records written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 08: Skill library dynamics")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--sessions", type=int, default=200)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
