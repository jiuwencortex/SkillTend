# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Standalone growth simulation module for Study 08.

Extracts the sequential session simulation loop from runner.py so it can be
imported and called programmatically without the CLI.

Usage::

    from phase_4_dynamics.study_08_library_dynamics.growth_simulation import run_growth_simulation

    skills_root, records = await run_growth_simulation(
        total_sessions=200,
        sim=simulator,
        model="openai/gpt-4o-mini",
        dry_run=False,
        task_queries=queries,
        measurement_interval=10,
    )
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.result_logger import ResultLogger
from shared.session_simulator import SessionSimulator
from shared.task_bank import TASK_BANK
from phase_4_dynamics.study_08_library_dynamics.coverage_metric import compute_library_coverage


@dataclass
class GrowthRecord:
    """Metrics captured at one measurement checkpoint during library growth."""

    session: int
    """Session index (1-based) at which this measurement was taken."""

    lc: float
    """Library Coverage at this checkpoint."""

    mean_pairwise_sim: float
    """Mean pairwise cosine similarity between all skill embeddings (redundancy proxy)."""

    gini: float
    """Gini coefficient of skill use_count distribution."""

    n_skills: int
    """Number of skills in the library at this checkpoint."""

    marginal_lc_gain: float = 0.0
    """LC gain since the previous checkpoint (0 for the first record)."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class GrowthSimulationResult:
    """Full output of a growth simulation run."""

    skills_root: Path
    """Temporary skill library root (caller is responsible for cleanup)."""

    records: List[GrowthRecord]
    """All measurement records in session order."""

    saturation_session: Optional[int]
    """First session where marginal LC gain < ``saturation_threshold``; None if not reached."""

    peak_lc: float
    """Highest LC observed across all checkpoints."""

    peak_redundancy: float
    """Highest mean_pairwise_sim observed across all checkpoints."""


async def run_growth_simulation(
    total_sessions: int,
    sim: SessionSimulator,
    model: str,
    dry_run: bool,
    task_queries: List[str],
    measurement_interval: int = 10,
    saturation_threshold: float = 0.005,
    logger: Optional[ResultLogger] = None,
) -> GrowthSimulationResult:
    """Simulate *total_sessions* sequential sessions and track library growth.

    For every session:
      1. Generate a task message list from ``TASK_BANK``.
      2. Replay the background review via ``sim.replay_review()``.
      3. Every ``measurement_interval`` sessions compute LC, redundancy, and Gini.

    Args:
        total_sessions:       Number of sessions to simulate.
        sim:                  Configured ``SessionSimulator`` instance.
        model:                litellm model string for the reviewer LLM.
        dry_run:              If True, skip real LLM calls and return stub metrics.
        task_queries:         List of task query strings used for LC evaluation.
        measurement_interval: How often (in sessions) to take a measurement.
        saturation_threshold: Marginal LC gain below which we call the library saturated.
        logger:               Optional ``ResultLogger``; if provided, records are logged.

    Returns:
        ``GrowthSimulationResult`` with skills_root, all records, saturation info, and peaks.
    """
    from skilltend.config import ReviewerConfig  # type: ignore[import]

    skills_root = sim.build_skill_library(domains=[])  # start empty
    config = ReviewerConfig(skills_root=skills_root)

    all_task_ids = TASK_BANK.ids()
    records: List[GrowthRecord] = []
    prev_lc: float = 0.0

    for session_idx in range(total_sessions):
        task_id = all_task_ids[session_idx % len(all_task_ids)]
        messages = sim.generate_session(task_id, seed=session_idx)

        try:
            await sim.replay_review(messages, config, model=model)
        except Exception:
            pass

        take_measurement = (
            (session_idx + 1) % measurement_interval == 0
            or session_idx == total_sessions - 1
        )
        if not take_measurement:
            continue

        coverage = await compute_library_coverage(
            skills_root=skills_root,
            task_queries=task_queries[:20],
            dry_run=dry_run,
        )

        marginal_gain = round(coverage["lc"] - prev_lc, 4)
        prev_lc = coverage["lc"]

        run_id = f"growth_s{session_idx + 1}_{uuid.uuid4().hex[:6]}"
        rec = GrowthRecord(
            session=session_idx + 1,
            lc=coverage["lc"],
            mean_pairwise_sim=coverage["mean_pairwise_sim"],
            gini=coverage["gini"],
            n_skills=coverage["n_skills"],
            marginal_lc_gain=marginal_gain,
            run_id=run_id,
        )
        records.append(rec)

        if logger is not None:
            logger.log(
                run_id=run_id,
                config={"phase": "A_growth", "session": session_idx + 1},
                metrics={
                    "lc": rec.lc,
                    "mean_pairwise_sim": rec.mean_pairwise_sim,
                    "gini": rec.gini,
                    "n_skills": rec.n_skills,
                    "marginal_lc_gain": rec.marginal_lc_gain,
                },
            )

        print(
            f"  session={session_idx + 1:4d}  LC={rec.lc:.3f}  "
            f"redundancy={rec.mean_pairwise_sim:.3f}  "
            f"gini={rec.gini:.3f}  skills={rec.n_skills}"
        )

    # Determine saturation session
    saturation_session: Optional[int] = None
    for rec in records[1:]:
        if rec.marginal_lc_gain < saturation_threshold:
            saturation_session = rec.session
            break

    peak_lc = max((r.lc for r in records), default=0.0)
    peak_redundancy = max((r.mean_pairwise_sim for r in records), default=0.0)

    return GrowthSimulationResult(
        skills_root=skills_root,
        records=records,
        saturation_session=saturation_session,
        peak_lc=peak_lc,
        peak_redundancy=peak_redundancy,
    )


def lc_growth_series(records: List[GrowthRecord]) -> List[Tuple[int, float]]:
    """Return (session, lc) pairs for plotting."""
    return [(r.session, r.lc) for r in records]


def redundancy_series(records: List[GrowthRecord]) -> List[Tuple[int, float]]:
    """Return (session, mean_pairwise_sim) pairs for plotting."""
    return [(r.session, r.mean_pairwise_sim) for r in records]


def gini_series(records: List[GrowthRecord]) -> List[Tuple[int, float]]:
    """Return (session, gini) pairs for plotting."""
    return [(r.session, r.gini) for r in records]
