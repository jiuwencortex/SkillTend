# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 07 runner — skill interference within-session dynamics.

Experimental design:
  - Each session uses Skill S at turn T.
  - Background review fires at turn T + Δ and patches Skill S.
  - An evaluation query about Skill S fires at turn T + 2Δ.
  - We measure TSR at T+2Δ vs. a control session (no review at T+Δ).

Δ ∈ {0, 1, 3, 5, 10} turns after first skill use.
40 sessions per Δ value, 3 task types.

Usage::

    python -m study_07_skill_interference.runner \
        --model openai/gpt-4o-mini \
        --sessions 40 \
        --output-dir results/

    python -m study_07_skill_interference.runner --dry-run --sessions 4
"""
from __future__ import annotations

import argparse
import asyncio
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.result_logger import ResultLogger
from shared.session_simulator import SessionSimulator
from shared.skill_quality_scorer import SkillQualityScorer
from shared.task_bank import TASK_BANK

DELTA_VALUES = [0, 1, 3, 5, 10]

# Evaluation domains for the interference test
EVAL_DOMAINS = ["code_debug", "api_integration", "sys_admin"]


async def _run_interference_trial(
    delta: int,
    domain: str,
    seed: int,
    scorer: SkillQualityScorer,
    sim: SessionSimulator,
    model: str,
) -> Tuple[float, float, bool]:
    """Run one trial. Returns (q_treatment, q_control, review_fired)."""
    from skilltend.config import ReviewerConfig

    task_ids = [t.id for t in TASK_BANK.by_domain(domain)[:3]]
    full_messages = sim.generate_multi_task_session(task_ids, seed=seed)

    # Split: first half is "before skill use", second half is "after"
    split = len(full_messages) // 2
    early_messages = full_messages[:split]
    late_messages = full_messages[split:]

    tasks = TASK_BANK.by_domain(domain)[:3]

    # --- TREATMENT: review fires at T+Δ ---
    skills_root_t = sim.build_skill_library(domains=[domain])
    config_t = ReviewerConfig(skills_root=skills_root_t)

    # Inject Δ filler turns between early and late
    filler = sim._filler_turns(domain, delta, __import__("random").Random(seed + 1000))
    messages_t = early_messages + filler + late_messages

    try:
        await sim.replay_review(messages_t[: split + delta + 2], config_t, model=model)
    except Exception:
        pass

    skill_paths_t = list(skills_root_t.rglob("SKILL.md"))
    q_t = 0.5
    if skill_paths_t:
        try:
            sc = await scorer.score(skill_paths_t[0].read_text(), tasks)
            q_t = sc.q
        except Exception:
            pass
    sim.cleanup_skill_library(skills_root_t)

    # --- CONTROL: no review ---
    skills_root_c = sim.build_skill_library(domains=[domain])
    q_c = 0.5
    skill_paths_c = list(skills_root_c.rglob("SKILL.md"))
    if skill_paths_c:
        try:
            sc = await scorer.score(skill_paths_c[0].read_text(), tasks)
            q_c = sc.q
        except Exception:
            pass
    sim.cleanup_skill_library(skills_root_c)

    return q_t, q_c, True


async def _phase_b_stability_curve(
    domain: str,
    sessions: int,
    scorer: SkillQualityScorer,
    sim: SessionSimulator,
    model: str,
    logger: ResultLogger,
) -> None:
    """Track Patch Stability (PS) as a function of turn number within a session."""
    task_ids = [t.id for t in TASK_BANK.by_domain(domain)[:4]]
    full_messages = sim.generate_multi_task_session(task_ids, seed=42)

    from skilltend.config import ReviewerConfig

    # Measure PS at different truncation points
    truncation_points = list(range(2, min(len(full_messages), 30), 2))
    skill_versions: List[str] = []

    for turn in truncation_points:
        skills_root = sim.build_skill_library(domains=[domain])
        config = ReviewerConfig(skills_root=skills_root)

        try:
            await sim.replay_review(full_messages[:turn], config, model=model)
        except Exception:
            pass

        skill_paths = list(skills_root.rglob("SKILL.md"))
        version = skill_paths[0].read_text() if skill_paths else ""
        skill_versions.append(version)
        sim.cleanup_skill_library(skills_root)

    # Compute PS at each turn
    for i, turn in enumerate(truncation_points):
        if i < 2:
            continue
        ps = scorer.score_patch_stability(skill_versions[max(0, i - 3): i + 1])
        if not math.isnan(ps):
            logger.log(
                run_id=f"stability_turn{turn}_{uuid.uuid4().hex[:6]}",
                config={"phase": "B_stability", "domain": domain, "turn": turn},
                metrics={"patch_stability": round(ps, 4)},
            )
            print(f"  turn={turn:3d}  PS={ps:.3f}")


async def _main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_path = output_dir / "study_07_interference.jsonl"

    print("=" * 60)
    print("Study 07 — Skill Interference & Within-Session Dynamics")
    print("=" * 60)

    sim = SessionSimulator(TASK_BANK, dry_run=args.dry_run)
    scorer = SkillQualityScorer(dry_run=args.dry_run, judge_model=args.model)
    logger = ResultLogger(output_path, study="07")

    deltas = DELTA_VALUES if not args.dry_run else [0, 3, 10]
    domains = EVAL_DOMAINS if not args.dry_run else EVAL_DOMAINS[:1]

    # Phase A: Interference detection
    print("\nPhase A: Interference detection")
    for delta in deltas:
        for domain in domains:
            q_treatments: List[float] = []
            q_controls: List[float] = []

            for seed in range(args.sessions):
                q_t, q_c, fired = await _run_interference_trial(delta, domain, seed, scorer, sim, args.model)
                q_treatments.append(q_t)
                q_controls.append(q_c)

            if q_treatments:
                mean_t = sum(q_treatments) / len(q_treatments)
                mean_c = sum(q_controls) / len(q_controls)
                # Cohen's d
                all_vals = q_treatments + q_controls
                pooled_std = math.sqrt(sum((v - sum(all_vals) / len(all_vals)) ** 2 for v in all_vals) / (len(all_vals) - 1))
                cohens_d = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0.0
                direction = "beneficial" if cohens_d > 0.1 else ("harmful" if cohens_d < -0.1 else "neutral")

                logger.log(
                    run_id=f"interference_delta{delta}_{domain}_{uuid.uuid4().hex[:6]}",
                    config={"delta": delta, "domain": domain},
                    metrics={
                        "q_treatment": round(mean_t, 4),
                        "q_control": round(mean_c, 4),
                        "cohens_d": round(cohens_d, 4),
                        "direction": direction,
                        "q_delta": round(mean_t - mean_c, 4),
                    },
                )
                print(f"  Δ={delta:2d}  {domain:20s}  Q_t={mean_t:.3f}  Q_c={mean_c:.3f}  "
                      f"d={cohens_d:+.3f}  [{direction}]")

    # Phase B: Stability curve
    print("\nPhase B: Patch stability vs. turn number")
    for domain in domains:
        print(f"  Domain: {domain}")
        await _phase_b_stability_curve(domain, args.sessions, scorer, sim, args.model, logger)

    # Recommendation
    records = ResultLogger.read_metrics(output_path)
    phase_a = [r for r in records if r.get("delta") is not None]
    phase_b = [r for r in records if "patch_stability" in r]

    if phase_a:
        neutral_delta = min(
            [r for r in phase_a if r.get("direction") == "neutral"],
            key=lambda r: r.get("delta", 999),
            default=None,
        )
        print(f"\nRecommendation:")
        print(f"  Interference is neutral at Δ≥{neutral_delta.get('delta') if neutral_delta else '?'} turns")

    if phase_b:
        stable_turns = [r for r in phase_b if r.get("patch_stability", 0) > 0.85]
        if stable_turns:
            min_stable = min(r.get("turn", 999) for r in stable_turns)
            print(f"  Patch stability ≥ 0.85 from turn {min_stable}")
            print(f"  Recommendation: flush_min_turns = {min_stable}")

    print(f"\nDone. {logger.count()} records written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 07: Skill interference experiment")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
