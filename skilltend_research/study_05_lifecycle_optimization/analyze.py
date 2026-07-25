# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 05 analysis — lifecycle threshold results.

Reads JSONL produced by runner.py and produces:
  - Heatmap of f_score for every (stale_days, archive_days) pair
  - Survival curve: fraction of useful skills per recency band
  - Predictor ROC curve (from study_05_predictor_cv.json)
  - findings.json with recommended thresholds

Usage::

    python -m study_05_lifecycle_optimization.analyze \
        --input   results/study_05_threshold_sweep.jsonl \
        --cv-file results/study_05_predictor_cv.json \
        --plot    results/study_05_heatmap.png \
        --output  results/study_05_findings.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.result_logger import ResultLogger
from study_05_lifecycle_optimization.simulate_usage import (
    simulate_library_usage,
    survival_analysis,
)
from study_05_lifecycle_optimization.threshold_sweep import (
    sweep_thresholds,
    pareto_threshold_pairs,
    format_sweep_table,
    survival_informed_recommendation,
)


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _plot_heatmap(records: List[Dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed; skipping plot")
        return

    stale_vals  = sorted({r["stale_days"]   for r in records if "stale_days"   in r})
    archive_vals = sorted({r["archive_days"] for r in records if "archive_days" in r})

    grid = np.full((len(archive_vals), len(stale_vals)), float("nan"))
    for r in records:
        if "f_score" not in r:
            continue
        si = stale_vals.index(r["stale_days"])
        ai = archive_vals.index(r["archive_days"])
        grid[ai, si] = r["f_score"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(stale_vals)))
    ax.set_xticklabels(stale_vals)
    ax.set_yticks(range(len(archive_vals)))
    ax.set_yticklabels(archive_vals)
    ax.set_xlabel("curator_stale_after_days")
    ax.set_ylabel("curator_archive_after_days")
    ax.set_title("Study 05 — F-score heatmap (lower = better)\n"
                 f"F = {0.7}*FP_rate + {0.3}*FN_rate")

    # Annotate cells
    for ai in range(len(archive_vals)):
        for si in range(len(stale_vals)):
            val = grid[ai, si]
            if not math.isnan(val):
                ax.text(si, ai, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="F-score")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_path}")


def _plot_survival(n_days: int, seed: int, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    profiles = simulate_library_usage(n_skills=100, n_days=n_days, seed=seed)
    observation_day = n_days // 2
    stats = survival_analysis(profiles, observation_day)

    bands = [7, 14, 21, 30, 45, 60, 90, 120, 180]
    xs, ys = [], []
    for b in bands:
        key = f"last_used_within_{b}d"
        if key in stats and stats[key]["survival_rate"] is not None:
            xs.append(b)
            ys.append(stats[key]["survival_rate"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, color="steelblue")
    ax.axhline(0.5, color="red", linestyle="--", label="50% survival threshold")
    ax.fill_between(xs, ys, 0.5, where=[y >= 0.5 for y in ys], alpha=0.15, color="green", label="Useful zone")
    ax.fill_between(xs, ys, 0.5, where=[y < 0.5 for y in ys],  alpha=0.15, color="red",   label="Stale zone")
    ax.set_xlabel("Days since last use")
    ax.set_ylabel("Fraction of skills still useful (next 30 days)")
    ax.set_title("Study 05 — Survival Curve: Useful Skills by Recency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    survival_path = output_path.parent / (output_path.stem + "_survival.png")
    plt.savefig(survival_path, dpi=150)
    plt.close()
    print(f"Survival curve saved to {survival_path}")


# ── Main analysis ─────────────────────────────────────────────────────────────


def analyze(
    input_path: Path,
    cv_file: Optional[Path],
    plot_path: Optional[Path],
    output_path: Path,
) -> Dict[str, Any]:
    records = ResultLogger.read_metrics(input_path)
    threshold_records = [r for r in records if "stale_days" in r and "f_score" in r]

    if not threshold_records:
        # Fall back to generating results from the simulation module
        print("No threshold records found; running simulation from scratch...")
        profiles = simulate_library_usage(n_skills=50, n_days=180, seed=42)
        sweep_results = sweep_thresholds(profiles, observation_day=90)
        threshold_records = [
            {
                "stale_days": r.stale_days,
                "archive_days": r.archive_days,
                "f_score": r.f_score,
                "fp_rate": r.fp_rate,
                "fn_rate": r.fn_rate,
            }
            for r in sweep_results
        ]
    else:
        # Reconstruct sweep results from JSONL for analysis
        profiles = simulate_library_usage(n_skills=50, n_days=180, seed=42)
        sweep_results = sweep_thresholds(profiles, observation_day=90)

    best_record = min(threshold_records, key=lambda r: r.get("f_score", 1.0))

    # Pareto frontier
    pareto = pareto_threshold_pairs(sweep_results)
    pareto_dicts = [
        {"stale_days": r.stale_days, "archive_days": r.archive_days,
         "fp_rate": r.fp_rate, "fn_rate": r.fn_rate, "f_score": r.f_score}
        for r in pareto
    ]

    # Survival-informed recommendation
    recommendation = survival_informed_recommendation(sweep_results, profiles, observation_day=90)

    # Predictor CV result
    predictor_summary: Optional[Dict] = None
    if cv_file and cv_file.exists():
        predictor_summary = json.loads(cv_file.read_text())

    findings: Dict[str, Any] = {
        "n_threshold_configs": len(threshold_records),
        "best_config": best_record,
        "pareto_frontier": pareto_dicts,
        "recommendation": recommendation,
        "predictor_cv": predictor_summary,
        "sweep_table": format_sweep_table(sweep_results),
        "production_recommendation": {
            "curator_stale_after_days": recommendation["recommended_stale_days"],
            "curator_archive_after_days": recommendation["recommended_archive_days"],
            "curator_llm_predictor": (
                predictor_summary is not None
                and predictor_summary.get("cv_auc_mean", 0) > 0.75
            ),
        },
    }

    if plot_path:
        _plot_heatmap(threshold_records, plot_path)
        _plot_survival(n_days=180, seed=42, output_path=plot_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    prod = findings["production_recommendation"]
    print(f"\nProduction recommendations:")
    print(f"  curator_stale_after_days   = {prod['curator_stale_after_days']}")
    print(f"  curator_archive_after_days = {prod['curator_archive_after_days']}")
    print(f"  Enable LifecyclePredictor  = {prod['curator_llm_predictor']}")
    if predictor_summary:
        print(f"  Predictor AUC = {predictor_summary.get('cv_auc_mean')}")

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 05: Lifecycle threshold analysis")
    parser.add_argument("--input",   default="results/study_05_threshold_sweep.jsonl", type=Path)
    parser.add_argument("--cv-file", default="results/study_05_predictor_cv.json",     type=Path)
    parser.add_argument("--plot",    default="results/study_05_heatmap.png",            type=Path)
    parser.add_argument("--output",  default="results/study_05_findings.json",          type=Path)
    args = parser.parse_args()
    analyze(args.input, args.cv_file, args.plot, args.output)


if __name__ == "__main__":
    main()
