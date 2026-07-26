# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 03 analysis — cost-quality curve and model recommendation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.result_logger import ResultLogger


def analyze(input_path: Path, plot_path: Optional[Path], output_path: Path) -> Dict[str, Any]:
    records = ResultLogger.read_metrics(input_path)
    if not records:
        return {"error": f"No records in {input_path}"}

    # Normalize Q relative to the best model
    max_q = max(r.get("q_mean", 0) for r in records)
    for r in records:
        r["q_relative"] = r.get("q_mean", 0) / max_q if max_q > 0 else 0

    sorted_by_cost = sorted(records, key=lambda r: r.get("est_cost_per_review_usd", 0))

    # Find 90%/95%/99% quality at minimum cost
    thresholds: Dict[str, Optional[Dict]] = {"90pct": None, "95pct": None, "99pct": None}
    for r in sorted_by_cost:
        qr = r.get("q_relative", 0)
        if thresholds["90pct"] is None and qr >= 0.90:
            thresholds["90pct"] = r
        if thresholds["95pct"] is None and qr >= 0.95:
            thresholds["95pct"] = r
        if thresholds["99pct"] is None and qr >= 0.99:
            thresholds["99pct"] = r

    # ANOVA: model tier vs review mode (placeholder — needs per-mode data)
    # Real implementation would filter records by review_mode and run F-test

    findings = {
        "n_models": len(records),
        "best_q_model": max(records, key=lambda r: r.get("q_mean", 0), default=None),
        "cheapest_model": sorted_by_cost[0] if sorted_by_cost else None,
        "minimum_cost_for_quality": {
            "90pct_of_frontier": {
                "model": thresholds["90pct"].get("model") if thresholds["90pct"] else None,
                "cost_per_review_usd": thresholds["90pct"].get("est_cost_per_review_usd") if thresholds["90pct"] else None,
            },
            "95pct_of_frontier": {
                "model": thresholds["95pct"].get("model") if thresholds["95pct"] else None,
                "cost_per_review_usd": thresholds["95pct"].get("est_cost_per_review_usd") if thresholds["95pct"] else None,
            },
        },
        "all_models_by_tier": {
            tier: [r for r in records if r.get("tier") == tier]
            for tier in ["frontier", "mid", "small", "open"]
        },
    }

    if plot_path:
        _plot_cost_quality(records, plot_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(findings, indent=2, default=str))

    rec = findings["minimum_cost_for_quality"]["95pct_of_frontier"]
    print(f"\nRecommendation: {rec['model']} achieves 95% of frontier quality at ${rec['cost_per_review_usd']:.5f}/review")
    return findings


def _plot_cost_quality(records: List[Dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tier_colors = {"frontier": "red", "mid": "orange", "small": "green", "open": "blue"}
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in records:
        color = tier_colors.get(r.get("tier", ""), "grey")
        ax.scatter(r.get("est_cost_per_review_usd", 0), r.get("q_mean", 0), color=color, s=80, zorder=3)
        ax.annotate(r.get("model", "").split("/")[-1], (r.get("est_cost_per_review_usd", 0), r.get("q_mean", 0)),
                    fontsize=8, ha="left", va="bottom", xytext=(4, 4), textcoords="offset points")

    for tier, color in tier_colors.items():
        ax.scatter([], [], color=color, label=tier, s=60)

    ax.set_xscale("log")
    ax.set_xlabel("Estimated cost per review (USD, log scale)")
    ax.set_ylabel("Q score")
    ax.set_title("Study 03 — Review Model: Quality vs. Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/study_03_benchmark.jsonl", type=Path)
    parser.add_argument("--plot", default="results/study_03_cost_quality.png", type=Path)
    parser.add_argument("--output", default="results/study_03_findings.json", type=Path)
    args = parser.parse_args()
    analyze(args.input, args.plot, args.output)


if __name__ == "__main__":
    main()
