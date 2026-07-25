# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 04 analysis — memory abstraction strategy results.

Reads the JSONL produced by runner.py and produces:
  - Efficiency frontier plot: Q vs. context chars for each strategy
  - char_limit saturation curve (where marginal Q gain drops below threshold)
  - findings.json with recommended strategy and memory_char_limit

Usage::

    python -m study_04_memory_abstraction.analyze \
        --input results/study_04_strategies.jsonl \
        --plot  results/study_04_efficiency.png \
        --output results/study_04_findings.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.result_logger import ResultLogger


# ── Data helpers ──────────────────────────────────────────────────────────────


def _load(path: Path) -> List[Dict[str, Any]]:
    records = ResultLogger.read_metrics(path)
    return [r for r in records if "q_mean" in r and not r.get("_failed")]


def _by_strategy(records: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r.get("strategy", "unknown")].append(r)
    return groups


def _saturation_limit(records: List[Dict], strategy: str, threshold: float = 0.005) -> Optional[int]:
    """Return the char_limit at which marginal Q gain drops below `threshold`."""
    relevant = sorted(
        [r for r in records if r.get("strategy") == strategy],
        key=lambda r: r.get("char_limit", 0),
    )
    for i in range(1, len(relevant)):
        gain = relevant[i].get("q_mean", 0) - relevant[i - 1].get("q_mean", 0)
        if gain < threshold:
            return relevant[i].get("char_limit")
    return relevant[-1].get("char_limit") if relevant else None


# ── Plot ──────────────────────────────────────────────────────────────────────


def _plot(records: List[Dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    by_strategy = _by_strategy(records)
    colors = cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Q vs. context chars (efficiency frontier)
    ax = axes[0]
    for i, (strategy, recs) in enumerate(by_strategy.items()):
        xs = [r.get("mean_chars_injected", 0) for r in recs]
        ys = [r.get("q_mean", 0) for r in recs]
        ax.scatter(xs, ys, label=strategy, color=colors[i % len(colors)], s=60)
        # Connect same strategy across char_limits
        paired = sorted(zip(xs, ys))
        if len(paired) > 1:
            ax.plot([p[0] for p in paired], [p[1] for p in paired],
                    color=colors[i % len(colors)], alpha=0.4, linewidth=1)
    ax.set_xlabel("Mean chars injected into system prompt")
    ax.set_ylabel("Q score")
    ax.set_title("Study 04 — Q vs. Context Characters")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Q vs. char_limit per strategy (saturation curve)
    ax = axes[1]
    for i, (strategy, recs) in enumerate(by_strategy.items()):
        by_limit = defaultdict(list)
        for r in recs:
            by_limit[r.get("char_limit", 0)].append(r.get("q_mean", 0))
        limits = sorted(by_limit)
        qs = [sum(by_limit[l]) / len(by_limit[l]) for l in limits]
        ax.plot(limits, qs, marker="o", label=strategy, color=colors[i % len(colors)])
    ax.set_xlabel("memory_char_limit")
    ax.set_ylabel("Q score")
    ax.set_title("Study 04 — Q vs. char_limit (saturation curve)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


# ── Main analysis ─────────────────────────────────────────────────────────────


def analyze(
    input_path: Path,
    plot_path: Optional[Path],
    output_path: Path,
) -> Dict[str, Any]:
    records = _load(input_path)
    if not records:
        return {"error": f"No valid records in {input_path}"}

    by_strategy = _by_strategy(records)

    # Best configuration overall
    best = max(records, key=lambda r: r.get("q_mean", 0))

    # Baseline (inject_all) Q for relative comparison
    baseline_recs = by_strategy.get("inject_all", [])
    baseline_q = (
        sum(r.get("q_mean", 0) for r in baseline_recs) / len(baseline_recs)
        if baseline_recs else 0.0
    )

    # Per-strategy summary
    strategy_summary: Dict[str, Dict] = {}
    for strategy, recs in by_strategy.items():
        qs = [r.get("q_mean", 0) for r in recs]
        chars = [r.get("mean_chars_injected", 0) for r in recs]
        best_rec = max(recs, key=lambda r: r.get("q_mean", 0))
        strategy_summary[strategy] = {
            "q_mean_overall": round(sum(qs) / len(qs), 4),
            "q_max": round(max(qs), 4),
            "q_vs_baseline": round(sum(qs) / len(qs) - baseline_q, 4),
            "mean_chars_injected": round(sum(chars) / len(chars), 1),
            "best_char_limit": best_rec.get("char_limit"),
            "saturation_char_limit": _saturation_limit(records, strategy),
        }

    # Recommendation
    best_strategy = max(strategy_summary, key=lambda s: strategy_summary[s]["q_max"])
    sat_limit = strategy_summary[best_strategy].get("saturation_char_limit")

    recommendation = {
        "strategy": best_strategy,
        "memory_char_limit": sat_limit,
        "q_improvement_vs_baseline": round(
            strategy_summary[best_strategy]["q_max"] - baseline_q, 4
        ),
        "adopt_non_baseline": (
            strategy_summary[best_strategy]["q_max"] - baseline_q > 0.05
        ),
        "rationale": (
            f"{best_strategy} achieves Q={strategy_summary[best_strategy]['q_max']:.4f} "
            f"(baseline={baseline_q:.4f}) at char_limit={sat_limit}."
        ),
    }

    findings = {
        "n_records": len(records),
        "strategies_tested": list(by_strategy.keys()),
        "baseline_q_inject_all": round(baseline_q, 4),
        "best_config": best,
        "strategy_summary": strategy_summary,
        "recommendation": recommendation,
    }

    if plot_path:
        _plot(records, plot_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    print(f"\nRecommendation:")
    print(f"  strategy         = {recommendation['strategy']}")
    print(f"  memory_char_limit = {recommendation['memory_char_limit']}")
    print(f"  Q vs baseline    = {recommendation['q_improvement_vs_baseline']:+.4f}")
    print(f"  Adopt?           = {recommendation['adopt_non_baseline']}")

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 04: Memory abstraction analysis")
    parser.add_argument("--input",  default="results/study_04_strategies.jsonl", type=Path)
    parser.add_argument("--plot",   default="results/study_04_efficiency.png",   type=Path)
    parser.add_argument("--output", default="results/study_04_findings.json",    type=Path)
    args = parser.parse_args()
    analyze(args.input, args.plot, args.output)


if __name__ == "__main__":
    main()
