# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 03 runner — review model calibration.

Tests 8 LLM models across 4 price tiers for background review quality.
For each model, runs the review pipeline on 40 sessions and measures Q score
and estimated token cost.

Usage::

    python -m phase_2_optimization.study_03_review_model_calibration.runner \
        --output-dir results/ \
        --sessions 40

    python -m phase_2_optimization.study_03_review_model_calibration.runner --dry-run --sessions 5
"""
from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from shared.result_logger import ResultLogger
from shared.session_simulator import SessionSimulator
from shared.skill_quality_scorer import SkillQualityScorer
from shared.task_bank import TASK_BANK

# Model grid: (model_id, tier, cost_per_1M_input_tokens_usd, cost_per_1M_output_tokens_usd)
MODEL_GRID = [
    # Frontier
    ("openai/gpt-4.1",               "frontier", 2.00, 8.00),
    ("anthropic/claude-opus-4",      "frontier", 15.0, 75.0),
    # Mid
    ("openai/gpt-4.1-mini",          "mid",      0.40, 1.60),
    ("anthropic/claude-sonnet-4",    "mid",      3.00, 15.0),
    # Small
    ("openai/gpt-4.1-nano",          "small",    0.10, 0.40),
    ("anthropic/claude-haiku-3-5",   "small",    0.80, 4.00),
    # Open (hosted via litellm)
    ("openai/llama-3.3-70b",         "open",     0.59, 0.79),
    ("openai/qwen2.5-72b",           "open",     0.35, 0.40),
]

# Tasks per difficulty × 4 domains = 40 sessions
EVAL_TASKS = (
    [t.id for t in TASK_BANK.by_difficulty("easy")[:4]]
    + [t.id for t in TASK_BANK.by_difficulty("medium")[:4]]
    + [t.id for t in TASK_BANK.by_difficulty("hard")[:2]]
)


async def _benchmark_one_model(
    model_id: str,
    tier: str,
    cost_in: float,
    cost_out: float,
    sessions: int,
    scorer: SkillQualityScorer,
    logger: ResultLogger,
    sim: SessionSimulator,
    concurrency: int,
) -> None:
    from skilltend.config import ReviewerConfig

    sem = asyncio.Semaphore(concurrency)
    q_scores: List[float] = []
    review_times: List[float] = []

    async def _one(task_id: str, seed: int) -> None:
        async with sem:
            messages = sim.generate_session(task_id, seed=seed)
            skills_root = sim.build_skill_library(domains=[task_id.split("_")[0]])
            config = ReviewerConfig(skills_root=skills_root, review_model=model_id)

            t0 = time.monotonic()
            try:
                await sim.replay_review(messages, config, model=model_id)
            except Exception as exc:
                print(f"    [{model_id}] {task_id}: ERROR — {exc}")
                sim.cleanup_skill_library(skills_root)
                return
            elapsed = time.monotonic() - t0
            review_times.append(elapsed)

            skill_paths = list(skills_root.rglob("SKILL.md"))
            if skill_paths:
                tasks = TASK_BANK.by_domain(task_id.split("_")[0])[:3]
                try:
                    scores = await scorer.score(skill_paths[0].read_text(), tasks)
                    q_scores.append(scores.q)
                except Exception:
                    q_scores.append(0.5)

            sim.cleanup_skill_library(skills_root)

    task_cycle = (EVAL_TASKS * (sessions // len(EVAL_TASKS) + 1))[:sessions]
    await asyncio.gather(*[_one(t, s) for s, t in enumerate(task_cycle)])

    if not q_scores:
        return

    mean_q = sum(q_scores) / len(q_scores)
    mean_time = sum(review_times) / len(review_times) if review_times else 0

    # Estimated cost: rough token estimate per review (input ~2000, output ~300)
    est_input_tokens = 2000
    est_output_tokens = 300
    est_cost_per_review = (est_input_tokens * cost_in + est_output_tokens * cost_out) / 1_000_000

    logger.log(
        run_id=f"{model_id.replace('/', '_')}_{uuid.uuid4().hex[:6]}",
        config={"model": model_id, "tier": tier},
        metrics={
            "q_mean": round(mean_q, 4),
            "review_time_mean_s": round(mean_time, 3),
            "est_cost_per_review_usd": round(est_cost_per_review, 6),
            "est_cost_per_1M_reviews_usd": round(est_cost_per_review * 1_000_000, 2),
        },
        meta={"sessions": len(q_scores), "cost_in_per_1M": cost_in, "cost_out_per_1M": cost_out},
    )
    print(f"  [{tier:8s}] {model_id:40s}  Q={mean_q:.3f}  cost/review=${est_cost_per_review:.5f}")


async def _run_failure_taxonomy(
    output_path: Path,
    frontier_model: str,
    cheap_model: str,
    sessions: int,
    dry_run: bool,
) -> None:
    """Compare frontier vs cheap model outputs and categorize failure modes."""
    # Placeholder: in a real run, this compares tool_calls from each model
    # and categorizes discrepancies into a taxonomy
    taxonomy_path = output_path.parent / "study_03_failure_taxonomy.json"
    taxonomy = {
        "frontier_model": frontier_model,
        "cheap_model": cheap_model,
        "categories": [
            "wrong_tool_called",
            "correct_tool_wrong_arguments",
            "hallucinated_skill_name",
            "missed_patch_opportunity",
            "over_patching",
        ],
        "note": "Populate by diffing tool_calls from frontier vs cheap model outputs",
    }
    taxonomy_path.write_text(__import__("json").dumps(taxonomy, indent=2))
    print(f"Failure taxonomy template written to {taxonomy_path}")


async def _main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_path = output_dir / "study_03_benchmark.jsonl"

    print("=" * 60)
    print("Study 03 — Review Model Calibration")
    print(f"Sessions per model: {args.sessions}")
    print("=" * 60)

    sim = SessionSimulator(TASK_BANK, dry_run=args.dry_run)
    scorer = SkillQualityScorer(dry_run=args.dry_run)
    logger = ResultLogger(output_path, study="03")

    models = MODEL_GRID if not args.dry_run else MODEL_GRID[:3]
    for model_id, tier, cost_in, cost_out in models:
        print(f"\nBenchmarking: {model_id}")
        await _benchmark_one_model(
            model_id=model_id,
            tier=tier,
            cost_in=cost_in,
            cost_out=cost_out,
            sessions=args.sessions,
            scorer=scorer,
            logger=logger,
            sim=sim,
            concurrency=args.concurrency,
        )

    # Analyze
    from phase_2_optimization.study_03_review_model_calibration.analyze import analyze
    analyze(output_path, output_dir / "study_03_cost_quality.png", output_dir / "study_03_findings.json")

    await _run_failure_taxonomy(output_dir, "openai/gpt-4.1", "openai/gpt-4.1-nano", args.sessions, args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 03: Review model calibration")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
