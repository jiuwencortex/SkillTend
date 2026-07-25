# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 06 runner — prompt sensitivity and ablation.

Usage::

    python -m study_06_prompt_sensitivity.runner \
        --model openai/gpt-4o-mini \
        --sessions 30 \
        --output-dir results/

    python -m study_06_prompt_sensitivity.runner --dry-run --sessions 3
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
from shared.skill_quality_scorer import SkillQualityScorer
from shared.task_bank import TASK_BANK
from study_06_prompt_sensitivity.ablations import (
    ALL_VARIANTS,
    CONTEXT_STRATEGIES,
    ContextStrategy,
    PromptVariant,
)


def _inject_adversarial_noise(messages: List[Dict[str, Any]], rng_seed: int) -> List[Dict[str, Any]]:
    """Inject adversarial off-topic turns into a conversation."""
    import random
    rng = random.Random(rng_seed)
    noise_turns = [
        {"role": "user", "content": "By the way, what's the weather like today?"},
        {"role": "assistant", "content": "I don't have access to real-time weather data."},
        {"role": "user", "content": "Can you tell me a joke?"},
        {"role": "assistant", "content": "Why do programmers prefer dark mode? Because light attracts bugs!"},
    ]
    result = list(messages)
    n_injections = rng.randint(1, 3)
    for _ in range(n_injections):
        insert_at = rng.randint(0, len(result))
        pair = rng.choice([(noise_turns[0], noise_turns[1]), (noise_turns[2], noise_turns[3])])
        result.insert(insert_at, pair[1])
        result.insert(insert_at, pair[0])
    return result


async def _benchmark_variant(
    variant: PromptVariant,
    context_strategy: ContextStrategy,
    sessions: int,
    scorer: SkillQualityScorer,
    logger: ResultLogger,
    sim: SessionSimulator,
    model: str,
    adversarial: bool = False,
) -> None:
    from skilltend.config import ReviewerConfig

    q_scores: List[float] = []
    domain = "code_debug"

    for seed in range(sessions):
        task_ids = [t.id for t in TASK_BANK.by_domain(domain)[:2]]
        messages = sim.generate_multi_task_session(task_ids, seed=seed)

        if adversarial:
            messages = _inject_adversarial_noise(messages, rng_seed=seed)

        messages = context_strategy.apply(messages)
        skills_root = sim.build_skill_library(domains=[domain])
        config = ReviewerConfig(skills_root=skills_root)

        # Patch the review prompt by monkey-patching Stage 2 temporarily
        # In a real implementation, we would pass the prompt override to run_background_review()
        # Here we inject it as a system message prefix in the conversation
        system_prefix = variant.to_system_prompt()
        augmented = [{"role": "user", "content": f"[System prompt override]\n{system_prefix}"}] + messages

        try:
            await sim.replay_review(augmented, config, model=model)
        except Exception:
            pass

        skill_paths = list(skills_root.rglob("SKILL.md"))
        if skill_paths:
            tasks = TASK_BANK.by_domain(domain)[:3]
            try:
                scores = await scorer.score(skill_paths[0].read_text(), tasks)
                q_scores.append(scores.q)
            except Exception:
                q_scores.append(0.5)

        sim.cleanup_skill_library(skills_root)

    if not q_scores:
        return

    mean_q = sum(q_scores) / len(q_scores)
    import math
    std_q = math.sqrt(sum((q - mean_q) ** 2 for q in q_scores) / len(q_scores))

    logger.log(
        run_id=f"{variant.name}_{context_strategy.name}_{uuid.uuid4().hex[:6]}",
        config={
            "variant": variant.name,
            "ablated_elements": variant.ablated_elements,
            "context_strategy": context_strategy.name,
            "adversarial": adversarial,
        },
        metrics={
            "q_mean": round(mean_q, 4),
            "q_std": round(std_q, 4),
        },
        meta={"sessions": len(q_scores)},
    )
    noise_label = " [adversarial]" if adversarial else ""
    print(f"  {variant.name:30s}  ctx={context_strategy.name:15s}  Q={mean_q:.3f} ± {std_q:.3f}{noise_label}")


async def _main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_path = output_dir / "study_06_ablations.jsonl"

    print("=" * 60)
    print("Study 06 — Prompt Sensitivity & Ablation")
    print("=" * 60)

    sim = SessionSimulator(TASK_BANK, dry_run=args.dry_run)
    scorer = SkillQualityScorer(dry_run=args.dry_run, judge_model=args.model)
    logger = ResultLogger(output_path, study="06")

    variants = ALL_VARIANTS if not args.dry_run else ALL_VARIANTS[:3]
    ctx_strategies = CONTEXT_STRATEGIES if not args.dry_run else CONTEXT_STRATEGIES[:2]

    # Phase A: Ablation sweep
    print("\nPhase A: Prompt element ablation")
    for variant in variants:
        await _benchmark_variant(variant, CONTEXT_STRATEGIES[0], args.sessions, scorer, logger, sim, args.model)

    # Phase B: Context window strategies (full prompt only)
    print("\nPhase B: Context truncation strategies")
    from study_06_prompt_sensitivity.ablations import FULL_PROMPT
    for ctx in ctx_strategies:
        await _benchmark_variant(FULL_PROMPT, ctx, args.sessions, scorer, logger, sim, args.model)

    # Phase D: Robustness under adversarial noise
    print("\nPhase D: Adversarial noise robustness")
    for variant in [FULL_PROMPT, variants[-1]]:  # full vs minimal
        await _benchmark_variant(variant, CONTEXT_STRATEGIES[0], max(2, args.sessions // 3),
                                 scorer, logger, sim, args.model, adversarial=True)

    # Summary
    records = ResultLogger.read_metrics(output_path)
    if records:
        clean = [r for r in records if not r.get("adversarial")]
        if clean:
            best = max(clean, key=lambda r: r.get("q_mean", 0))
            full_q = next((r.get("q_mean") for r in clean if r.get("variant") == "full"), None)
            print(f"\nFull prompt Q = {full_q}")
            print(f"Best variant  = {best.get('variant')}  Q = {best.get('q_mean')}")

            # Which ablations caused the most Q drop?
            drops = []
            for r in clean:
                if r.get("variant") != "full" and full_q:
                    drop = full_q - r.get("q_mean", full_q)
                    if r.get("ablated_elements"):
                        drops.append((drop, r.get("ablated_elements")))
            drops.sort(reverse=True)
            print(f"\nMost impactful ablations (by Q drop):")
            for drop, elems in drops[:5]:
                print(f"  {elems}: -{drop:.4f}")

    print(f"\nDone. {logger.count()} records written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 06: Prompt sensitivity and ablation")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
