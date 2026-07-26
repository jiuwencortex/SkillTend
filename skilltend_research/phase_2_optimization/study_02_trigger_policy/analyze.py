# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 02 analysis — Pareto frontier and recommendations.

Usage::

    python -m phase_2_optimization.study_02_trigger_policy.analyze \
        --input results/study_02_sweep.jsonl \
        --plot results/study_02_pareto.png \
        --output results/study_02_findings.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.result_logger import ResultLogger


def _load_sweep_results(path: Path) -> List[Dict[str, Any]]:
    records = ResultLogger.read_metrics(path)
    return [r for r in records if "q_mean" in r and "tokens_mean" in r]


def _aggregate_by_config(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Average Q and tokens across session_length and difficulty for each (N, M) pair."""
    groups: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for r in records:
        key = (r.get("skill_nudge_interval", 0), r.get("memory_nudge_interval", 0))
        groups[key].append(r)

    aggregated = []
    for (n, m), recs in groups.items():
        q_vals = [r["q_mean"] for r in recs if r["q_mean"] is not None]
        t_vals = [r["tokens_mean"] for r in recs if r["tokens_mean"] is not None]
        if not q_vals:
            continue
        aggregated.append({
            "skill_nudge_interval": n,
            "memory_nudge_interval": m,
            "q_mean": sum(q_vals) / len(q_vals),
            "tokens_mean": sum(t_vals) / len(t_vals) if t_vals else 0,
            "n_records": len(recs),
        })
    return sorted(aggregated, key=lambda r: r["q_mean"], reverse=True)


def _pareto_frontier(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the Pareto-optimal points (max Q for any given token budget)."""
    # Sort by tokens ascending
    sorted_pts = sorted(points, key=lambda p: p["tokens_mean"])
    frontier: List[Dict] = []
    best_q = -1.0
    for p in sorted_pts:
        if p["q_mean"] > best_q:
            best_q = p["q_mean"]
            frontier.append(p)
    return frontier


def _knee_point(frontier: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the knee of the Pareto frontier (max Q / token efficiency)."""
    if len(frontier) < 2:
        return frontier[0] if frontier else None
    # Normalize both axes
    min_t = min(p["tokens_mean"] for p in frontier)
    max_t = max(p["tokens_mean"] for p in frontier)
    min_q = min(p["q_mean"] for p in frontier)
    max_q = max(p["q_mean"] for p in frontier)

    range_t = max_t - min_t or 1.0
    range_q = max_q - min_q or 1.0

    # Distance from the lower-left origin on normalized axes
    # The knee is the point farthest from the line connecting (0,0) to (1,1)
    best = None
    best_dist = -1.0
    for p in frontier:
        nt = (p["tokens_mean"] - min_t) / range_t
        nq = (p["q_mean"] - min_q) / range_q
        # Perpendicular distance from diagonal line y = x
        dist = abs(nq - nt) / math.sqrt(2)
        if nq > nt and dist > best_dist:  # only consider points above the diagonal
            best_dist = dist
            best = p
    return best or frontier[-1]


def _plot_pareto(points: List[Dict], frontier: List[Dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # All points
    ax.scatter(
        [p["tokens_mean"] for p in points],
        [p["q_mean"] for p in points],
        alpha=0.4, s=40, color="steelblue", label="All configs",
    )

    # Pareto frontier
    if frontier:
        ax.plot(
            [p["tokens_mean"] for p in frontier],
            [p["q_mean"] for p in frontier],
            "r-o", linewidth=2, markersize=6, label="Pareto frontier",
        )

    # Knee point
    knee = _knee_point(frontier)
    if knee:
        ax.scatter([knee["tokens_mean"]], [knee["q_mean"]], color="gold", s=150,
                   zorder=5, marker="*",
                   label=f"Knee: N={knee['skill_nudge_interval']} M={knee['memory_nudge_interval']}")

    ax.set_xlabel("Mean tokens per session")
    ax.set_ylabel("Mean Q score")
    ax.set_title("Study 02 — Trigger Policy: Q vs. Token Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


def analyze(input_path: Path, plot_path: Optional[Path], output_path: Path) -> Dict[str, Any]:
    records = _load_sweep_results(input_path)
    if not records:
        return {"error": f"No valid records in {input_path}"}

    aggregated = _aggregate_by_config(records)
    frontier = _pareto_frontier(aggregated)
    knee = _knee_point(frontier)

    findings: Dict[str, Any] = {
        "total_configs": len(aggregated),
        "total_records": len(records),
        "best_q_config": aggregated[0] if aggregated else None,
        "pareto_frontier": frontier,
        "knee_point": knee,
        "recommendations": {
            "cost_sensitive": next(
                (p for p in frontier if p["tokens_mean"] < 500), frontier[0] if frontier else None
            ),
            "balanced": knee,
            "quality_first": frontier[-1] if frontier else None,
        },
        "top_10_by_q": aggregated[:10],
    }

    if plot_path:
        _plot_pareto(aggregated, frontier, plot_path)

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 02: Pareto frontier analysis")
    parser.add_argument("--input", default="results/study_02_sweep.jsonl", type=Path)
    parser.add_argument("--plot", default="results/study_02_pareto.png", type=Path)
    parser.add_argument("--output", default="results/study_02_findings.json", type=Path)
    args = parser.parse_args()

    findings = analyze(args.input, args.plot, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    knee = findings.get("knee_point")
    if knee:
        print(f"\nRecommendation (balanced mode):")
        print(f"  skill_nudge_interval  = {knee['skill_nudge_interval']}")
        print(f"  memory_nudge_interval = {knee['memory_nudge_interval']}")
        print(f"  Q = {knee['q_mean']:.4f}  |  tokens/session = {knee['tokens_mean']:.0f}")


if __name__ == "__main__":
    main()
