# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 04 runner — memory abstraction strategy comparison.

Usage::

    python -m study_04_memory_abstraction.runner \
        --model openai/gpt-4o-mini \
        --sessions 50 \
        --output-dir results/

    python -m study_04_memory_abstraction.runner --dry-run --sessions 4
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
from study_04_memory_abstraction.strategies import ALL_STRATEGIES, MemoryStrategy

# Memory size ablation values (chars)
MEMORY_CHAR_LIMITS = [500, 1000, 2200, 4000, 8000]


def _build_synthetic_memory(domain: str, n_entries: int = 15) -> str:
    """Generate synthetic MEMORY.md content for a given domain."""
    templates = {
        "code_debug": [
            "User often works with Python async code and asyncio.",
            "Prefers concise error messages with specific line numbers.",
            "Uses pytest for testing; fixtures are function-scoped by default.",
            "Frequently debugs SQLAlchemy session issues.",
            "Prefers type annotations and uses mypy for checking.",
        ],
        "api_integration": [
            "Uses httpx for all HTTP calls (prefers async).",
            "All API keys are loaded from environment variables.",
            "Implements retry with tenacity library.",
            "Prefers pydantic for response validation.",
            "Works primarily with REST APIs returning JSON.",
        ],
        "doc_draft": [
            "Prefers concise documentation with working code examples.",
            "Uses Markdown for all documentation.",
            "Audience is senior engineers — no need to explain basics.",
            "Always includes a Quick Start section first.",
            "Uses present tense and active voice.",
        ],
    }
    defaults = [
        "User prefers explicit over implicit code.",
        "Error messages should always include the cause and a fix.",
        "Prefers async patterns for I/O-heavy work.",
        "Uses Python 3.11+.",
        "Tests must be deterministic and isolated.",
    ]
    entries = templates.get(domain, defaults)
    # Repeat to hit n_entries
    full = (entries * (n_entries // len(entries) + 1))[:n_entries]
    return "\n\n".join(full)


async def _benchmark_strategy(
    strategy: MemoryStrategy,
    char_limit: int,
    sessions: int,
    scorer: SkillQualityScorer,
    logger: ResultLogger,
    sim: SessionSimulator,
    model: str,
    dry_run: bool,
) -> None:
    from skilltend.config import ReviewerConfig

    domain = "code_debug"
    memory_text = _build_synthetic_memory(domain, n_entries=15)

    q_scores: List[float] = []
    context_chars: List[int] = []

    for seed in range(sessions):
        # Inject memory into the session via strategy
        injected, chars_used = await strategy.inject(memory_text, "user query", char_limit)
        context_chars.append(chars_used)

        task_ids = [t.id for t in TASK_BANK.by_domain(domain)[:2]]
        messages = sim.generate_multi_task_session(task_ids, seed=seed)

        # Prepend memory as a system-like user turn for the review LLM to see
        augmented = [{"role": "user", "content": f"[Memory context]\n{injected}"}] + messages

        skills_root = sim.build_skill_library(domains=[domain])
        config = ReviewerConfig(skills_root=skills_root, memory_char_limit=char_limit)

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

    mean_q = sum(q_scores) / len(q_scores) if q_scores else 0.5
    mean_chars = sum(context_chars) / len(context_chars) if context_chars else 0

    logger.log(
        run_id=f"{strategy.name}_lim{char_limit}_{uuid.uuid4().hex[:6]}",
        config={"strategy": strategy.name, "char_limit": char_limit},
        metrics={
            "q_mean": round(mean_q, 4),
            "mean_chars_injected": round(mean_chars, 1),
            "sessions": len(q_scores),
        },
        meta={"domain": domain},
    )
    print(f"  {strategy.name:25s}  limit={char_limit:5d}  Q={mean_q:.3f}  chars={mean_chars:.0f}")


async def _main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_path = output_dir / "study_04_strategies.jsonl"

    print("=" * 60)
    print("Study 04 — Memory Abstraction Strategies")
    print("=" * 60)

    sim = SessionSimulator(TASK_BANK, dry_run=args.dry_run)
    scorer = SkillQualityScorer(dry_run=args.dry_run, judge_model=args.model)
    logger = ResultLogger(output_path, study="04")

    # Strategy × char_limit grid
    strategies = ALL_STRATEGIES if not args.dry_run else ALL_STRATEGIES[:3]
    limits = MEMORY_CHAR_LIMITS if not args.dry_run else [1000, 2200]

    for strategy in strategies:
        for limit in limits:
            print(f"\n{strategy.name}  limit={limit}")
            await _benchmark_strategy(
                strategy=strategy,
                char_limit=limit,
                sessions=args.sessions,
                scorer=scorer,
                logger=logger,
                sim=sim,
                model=args.model,
                dry_run=args.dry_run,
            )

    print(f"\nDone. {logger.count()} records written to {output_path}")

    # Quick summary
    records = ResultLogger.read_metrics(output_path)
    if records:
        best = max(records, key=lambda r: r.get("q_mean", 0))
        print(f"\nBest: strategy={best.get('strategy')}  limit={best.get('char_limit')}  Q={best.get('q_mean')}")
        print(f"Recommendation: update memory_char_limit to {best.get('char_limit')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 04: Memory abstraction strategy comparison")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--sessions", type=int, default=50)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
