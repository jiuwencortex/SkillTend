# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 01 runner — compute Q sub-metrics for a set of skills.

This script computes RR, ID, TSR, and PS for each SKILL.md in a given
directory, writing one JSONL record per skill per measurement round.
Multiple rounds (--rounds N) allow stability analysis (CV computation).

After this script completes:
  1. Run label_tool.py to collect human quality labels.
  2. Run compute_validity.py to check metric validity.
  3. Run calibrate_weights.py to fit the final Q weights.

Usage::

    python -m study_01_skill_quality_metrics.runner \
        --skills-dir ~/.jiuwen/skills \
        --model gpt-4o-mini \
        --embedding-model openai/text-embedding-3-small \
        --rounds 5 \
        --output results/study_01_metrics.jsonl

    # Dry-run (no LLM calls, useful for CI):
    python -m study_01_skill_quality_metrics.runner --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from shared.result_logger import ResultLogger
from shared.skill_quality_scorer import SkillQualityScorer
from shared.task_bank import TASK_BANK


async def _score_one_skill(
    skill_path: Path,
    scorer: SkillQualityScorer,
    logger: ResultLogger,
    round_idx: int,
) -> None:
    skill_name = skill_path.parent.name
    skill_text = skill_path.read_text(encoding="utf-8")

    # Select tasks related to this skill by name-matching domain
    domain_guess = skill_name.split("_")[0]
    tasks = TASK_BANK.by_domain(domain_guess)
    if not tasks:
        tasks = TASK_BANK.all()[:5]

    try:
        scores = await scorer.score(skill_text, tasks)
        logger.log(
            run_id=f"{skill_name}_r{round_idx}_{uuid.uuid4().hex[:6]}",
            config={"round": round_idx},
            metrics=scores.as_dict(),
            meta={"skill_name": skill_name, "skill_path": str(skill_path)},
        )
        print(f"  [{round_idx}] {skill_name}: Q={scores.q:.3f}  RR={scores.rr:.3f}  "
              f"ID={scores.id_score:.3f}  TSR={scores.tsr:.3f}")
    except Exception as exc:
        logger.log_error(
            run_id=f"{skill_name}_r{round_idx}",
            config={"round": round_idx},
            error=str(exc),
            meta={"skill_name": skill_name},
        )
        print(f"  [{round_idx}] {skill_name}: ERROR — {exc}")


async def run(
    skills_dir: Path,
    output_path: Path,
    model: str,
    embedding_model: str,
    rounds: int,
    dry_run: bool,
    concurrency: int,
) -> None:
    skill_paths = sorted(skills_dir.rglob("SKILL.md"))
    if not skill_paths:
        print(f"No SKILL.md files found in {skills_dir}")
        return

    print(f"Study 01 — Skill Quality Metrics")
    print(f"Skills: {len(skill_paths)}  |  Rounds: {rounds}  |  Model: {model}")
    print(f"Output: {output_path}\n")

    scorer = SkillQualityScorer(
        embedding_model=embedding_model,
        judge_model=model,
        dry_run=dry_run,
    )
    logger = ResultLogger(output_path, study="01")
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(path: Path, r: int) -> None:
        async with sem:
            await _score_one_skill(path, scorer, logger, r)

    for round_idx in range(rounds):
        print(f"Round {round_idx + 1}/{rounds}:")
        tasks = [_bounded(p, round_idx) for p in skill_paths]
        await asyncio.gather(*tasks)

    print(f"\nDone. {logger.count()} records written to {output_path}")
    print("Next steps:")
    print("  1. python -m study_01_skill_quality_metrics.label_tool --skills-dir ... --rater-id you")
    print("  2. python -m study_01_skill_quality_metrics.compute_validity")
    print("  3. python -m study_01_skill_quality_metrics.calibrate_weights")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 01: Compute Q sub-metrics for skills")
    parser.add_argument("--skills-dir", default=Path.home() / ".jiuwen/skills", type=Path)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    parser.add_argument("--rounds", type=int, default=5, help="Repetitions for stability analysis")
    parser.add_argument("--output", default="results/study_01_metrics.jsonl", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(run(
        skills_dir=args.skills_dir,
        output_path=args.output,
        model=args.model,
        embedding_model=args.embedding_model,
        rounds=args.rounds,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
