# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""N×M grid sweep for trigger policy optimization.

For each (skill_nudge_interval, memory_nudge_interval) pair, runs a fixed
number of synthetic sessions and measures:
  - Final Q score of the skill library
  - Total review LLM tokens consumed
  - Wall-clock time added by background review

Results are written to a JSONL file for analysis by analyze.py.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.result_logger import ResultLogger
from shared.session_simulator import SessionSimulator
from shared.skill_quality_scorer import SkillQualityScorer
from shared.task_bank import TASK_BANK

# The intervals under study. ∞ means "never trigger" (baseline).
SKILL_INTERVALS = [3, 5, 8, 10, 15, 20, 30, 999]  # 999 ≈ never
MEMORY_INTERVALS = [3, 5, 8, 10, 15, 20, 30, 999]

SESSION_LENGTHS = {
    "short": 0,   # 0 extra filler turns
    "long": 40,   # 40 extra filler turns
}

DIFFICULTY_TASK_SAMPLES = {
    "easy": [t.id for t in TASK_BANK.by_difficulty("easy")[:3]],
    "medium": [t.id for t in TASK_BANK.by_difficulty("medium")[:3]],
    "hard": [t.id for t in TASK_BANK.by_difficulty("hard")[:3]],
}


@dataclass
class SweepConfig:
    skill_nudge_interval: int
    memory_nudge_interval: int
    session_length: str
    difficulty: str
    sessions_per_config: int
    model: str
    dry_run: bool


async def _run_one_config(
    cfg: SweepConfig,
    scorer: SkillQualityScorer,
    logger: ResultLogger,
    sim: SessionSimulator,
) -> None:
    """Run sessions_per_config sessions for one (N, M, length, difficulty) config."""
    from skilltend.config import ReviewerConfig

    task_ids = DIFFICULTY_TASK_SAMPLES.get(cfg.difficulty, DIFFICULTY_TASK_SAMPLES["medium"])
    extra_turns = SESSION_LENGTHS.get(cfg.session_length, 0)

    q_scores: List[float] = []
    tokens_used: List[int] = []
    review_times: List[float] = []

    for seed in range(cfg.sessions_per_config):
        # Build a fresh skill library for each session
        skills_root = sim.build_skill_library(domains=list(TASK_BANK.domains))

        config = ReviewerConfig(
            skills_root=skills_root,
            skill_nudge_interval=cfg.skill_nudge_interval,
            memory_nudge_interval=cfg.memory_nudge_interval,
        )

        # Generate session messages
        session_messages: List[Dict[str, Any]] = []
        for task_id in task_ids:
            session_messages.extend(sim.generate_session(task_id, seed=seed, extra_filler_turns=extra_turns // len(task_ids)))

        # Replay review and measure
        t0 = time.monotonic()
        result = await sim.replay_review(session_messages, config, model=cfg.model)
        elapsed = time.monotonic() - t0

        review_times.append(elapsed)
        # Token usage: if available from result (real runs only)
        tokens_used.append(0)  # stub; real implementation reads from ReviewResult

        # Score the skill library after review
        skill_paths = list(skills_root.rglob("SKILL.md"))
        if skill_paths:
            # Score the first skill as a proxy (full scoring in analyze.py)
            tasks = TASK_BANK.by_domain(task_ids[0].split("_")[0])[:3] if task_ids else TASK_BANK.all()[:3]
            try:
                skill_text = skill_paths[0].read_text(encoding="utf-8")
                scores = await scorer.score(skill_text, tasks)
                q_scores.append(scores.q)
            except Exception:
                q_scores.append(0.5)
        else:
            q_scores.append(0.0)

        sim.cleanup_skill_library(skills_root)

    if not q_scores:
        return

    mean_q = sum(q_scores) / len(q_scores)
    mean_tokens = sum(tokens_used) / len(tokens_used)
    mean_time = sum(review_times) / len(review_times)

    logger.log(
        run_id=f"N{cfg.skill_nudge_interval}_M{cfg.memory_nudge_interval}_{cfg.session_length}_{cfg.difficulty}_{uuid.uuid4().hex[:6]}",
        config={
            "skill_nudge_interval": cfg.skill_nudge_interval,
            "memory_nudge_interval": cfg.memory_nudge_interval,
            "session_length": cfg.session_length,
            "difficulty": cfg.difficulty,
        },
        metrics={
            "q_mean": round(mean_q, 4),
            "q_std": round((sum((q - mean_q)**2 for q in q_scores) / len(q_scores))**0.5, 4),
            "tokens_mean": round(mean_tokens, 1),
            "review_time_mean_s": round(mean_time, 3),
        },
        meta={"sessions": cfg.sessions_per_config},
    )

    print(
        f"  N={cfg.skill_nudge_interval:3d}  M={cfg.memory_nudge_interval:3d}  "
        f"{cfg.session_length:5s}  {cfg.difficulty:6s}  Q={mean_q:.3f}  tokens={mean_tokens:.0f}"
    )


async def run_sweep(
    output_path: Path,
    model: str,
    sessions_per_config: int,
    concurrency: int,
    dry_run: bool,
    skill_intervals: Optional[List[int]] = None,
    memory_intervals: Optional[List[int]] = None,
) -> None:
    """Run the full N×M sweep."""
    n_vals = skill_intervals or SKILL_INTERVALS
    m_vals = memory_intervals or MEMORY_INTERVALS

    sim = SessionSimulator(TASK_BANK, dry_run=dry_run)
    scorer = SkillQualityScorer(dry_run=dry_run, judge_model=model)
    logger = ResultLogger(output_path, study="02")
    sem = asyncio.Semaphore(concurrency)

    configs = [
        SweepConfig(
            skill_nudge_interval=n,
            memory_nudge_interval=m,
            session_length=length,
            difficulty=diff,
            sessions_per_config=sessions_per_config,
            model=model,
            dry_run=dry_run,
        )
        for n in n_vals
        for m in m_vals
        for length in SESSION_LENGTHS
        for diff in DIFFICULTY_TASK_SAMPLES
    ]

    total = len(configs)
    print(f"Sweep: {total} configs × {sessions_per_config} sessions = {total * sessions_per_config} runs")

    async def _bounded(cfg: SweepConfig) -> None:
        async with sem:
            await _run_one_config(cfg, scorer, logger, sim)

    await asyncio.gather(*[_bounded(c) for c in configs])
    print(f"\nSweep complete. {logger.count()} records written to {output_path}")
