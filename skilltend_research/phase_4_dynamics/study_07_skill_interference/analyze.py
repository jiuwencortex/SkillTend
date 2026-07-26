# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 07 analysis — skill interference results.

Produces:
  - Cohen's d vs. Δ line plot (one line per domain)
  - Patch Stability vs. turn-number curve (warm-up period)
  - findings.json with flush_min_turns recommendation and COMBINED-mode guidance

Usage::

    python -m phase_4_dynamics.study_07_skill_interference.analyze \
        --input  results/study_07_interference.jsonl \
        --plot   results/study_07_cohens_d.png \
        --output results/study_07_findings.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.result_logger import ResultLogger
from phase_4_dynamics.study_07_skill_interference.interference_experiment import warm_up_turn


# ── Data helpers ──────────────────────────────────────────────────────────────


def _load(path: Path) -> List[Dict[str, Any]]:
    return [r for r in ResultLogger.read_metrics(path) if not r.get("_failed")]


def _interference_records(records: List[Dict]) -> List[Dict]:
    return [r for r in records if "delta" in r and "cohens_d" in r]


def _stability_records(records: List[Dict]) -> List[Tuple[int, float]]:
    """Return (turn, patch_stability) pairs from Phase B records."""
    return [
        (r["turn"], r["patch_stability"])
        for r in records
        if "turn" in r and "patch_stability" in r
    ]


# ── Plots ─────────────────────────────────────────────────────────────────────


def _plot_cohens_d(interference: List[Dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    by_domain: Dict[str, Dict[int, float]] = defaultdict(dict)
    for r in interference:
        by_domain[r.get("domain", "all")][r.get("delta", 0)] = r.get("cohens_d", 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Cohen's d vs. Δ
    ax = axes[0]
    import matplotlib.cm as cm
    colors = cm.tab10.colors
    for i, (domain, d_by_delta) in enumerate(by_domain.items()):
        deltas = sorted(d_by_delta)
        ds = [d_by_delta[d] for d in deltas]
        ax.plot(deltas, ds, marker="o", label=domain, color=colors[i % len(colors)])

    ax.axhline(0.1,  color="green", linestyle="--", alpha=0.6, label="beneficial threshold")
    ax.axhline(-0.1, color="red",   linestyle="--", alpha=0.6, label="harmful threshold")
    ax.axhline(0,    color="black", linewidth=0.8)
    ax.set_xlabel("Δ (turns between skill use and review fire)")
    ax.set_ylabel("Cohen's d  (positive = review helped)")
    ax.set_title("Study 07 — Interference Effect Size vs. Δ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Q delta vs. Δ (absolute)
    ax = axes[1]
    by_domain_qdelta: Dict[str, Dict[int, float]] = defaultdict(dict)
    for r in interference:
        by_domain_qdelta[r.get("domain", "all")][r.get("delta", 0)] = r.get("q_delta", 0)

    for i, (domain, q_by_delta) in enumerate(by_domain_qdelta.items()):
        deltas = sorted(q_by_delta)
        qs = [q_by_delta[d] for d in deltas]
        ax.plot(deltas, qs, marker="s", label=domain, color=colors[i % len(colors)])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ (turns)")
    ax.set_ylabel("Q_treatment − Q_control")
    ax.set_title("Study 07 — Q Delta vs. Δ (absolute benefit)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Cohen's d plot saved to {output_path}")


def _plot_stability_curve(stability_pts: List[Tuple[int, float]], output_path: Path) -> None:
    if not stability_pts:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    pts = sorted(stability_pts)
    turns = [t for t, _ in pts]
    ps    = [p for _, p in pts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(turns, ps, marker="o", linewidth=2, color="steelblue", label="Patch Stability")
    ax.axhline(0.85, color="green", linestyle="--", label="stability threshold (0.85)")

    # Mark warm-up turn
    wt = warm_up_turn(pts)
    if wt is not None:
        ax.axvline(wt, color="orange", linestyle=":", linewidth=2, label=f"warm-up complete (turn {wt})")

    ax.set_xlabel("Turn number in session")
    ax.set_ylabel("Patch Stability (PS)")
    ax.set_title("Study 07 — Patch Stability vs. Turn Number\n(warm-up period detection)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    stab_path = output_path.parent / (output_path.stem + "_stability.png")
    plt.tight_layout()
    plt.savefig(stab_path, dpi=150)
    plt.close()
    print(f"Stability curve saved to {stab_path}")


# ── Main analysis ─────────────────────────────────────────────────────────────


def analyze(
    input_path: Path,
    plot_path: Optional[Path],
    output_path: Path,
) -> Dict[str, Any]:
    records = _load(input_path)
    if not records:
        return {"error": f"No valid records in {input_path}"}

    interference = _interference_records(records)
    stability_pts = _stability_records(records)

    # Warm-up turn from stability curve
    warm_up = warm_up_turn(stability_pts, stability_threshold=0.85)

    # Per-Δ summary across all domains
    by_delta: Dict[int, List[Dict]] = defaultdict(list)
    for r in interference:
        by_delta[r.get("delta", 0)].append(r)

    delta_summary = []
    for delta, recs in sorted(by_delta.items()):
        ds = [r.get("cohens_d", 0) for r in recs]
        qs = [r.get("q_delta", 0) for r in recs]
        mean_d = sum(ds) / len(ds)
        direction = "beneficial" if mean_d > 0.1 else ("harmful" if mean_d < -0.1 else "neutral")
        delta_summary.append({
            "delta": delta,
            "cohens_d_mean": round(mean_d, 4),
            "q_delta_mean": round(sum(qs) / len(qs), 4),
            "direction": direction,
            "n_records": len(recs),
        })

    # Minimum Δ at which effect is neutral or beneficial
    safe_delta = next(
        (d["delta"] for d in sorted(delta_summary, key=lambda d: d["delta"])
         if d["direction"] in ("neutral", "beneficial")),
        None,
    )

    # COMBINED mode: check Phase C records if present
    combined_recs = [r for r in records if r.get("phase") == "C_combined"]
    combined_summary = None
    if combined_recs:
        qs = [r.get("q_mean", 0) for r in combined_recs]
        combined_summary = {
            "q_mean": round(sum(qs) / len(qs), 4),
            "n_records": len(combined_recs),
        }

    findings: Dict[str, Any] = {
        "n_records": len(records),
        "interference_records": len(interference),
        "stability_points": len(stability_pts),
        "warm_up_turn": warm_up,
        "delta_summary": delta_summary,
        "safe_delta": safe_delta,
        "combined_mode_summary": combined_summary,
        "recommendations": {
            "flush_min_turns": warm_up if warm_up is not None else 6,
            "minimum_delta_before_review": safe_delta,
            "use_sequential_over_combined": (
                combined_summary is not None
                and any(r.get("direction") == "harmful" for r in delta_summary if r.get("delta", 0) == 0)
            ),
        },
    }

    if plot_path:
        _plot_cohens_d(interference, plot_path)
        _plot_stability_curve(stability_pts, plot_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    recs = findings["recommendations"]
    print(f"\nRecommendations:")
    print(f"  flush_min_turns                = {recs['flush_min_turns']}")
    print(f"  minimum_delta_before_review    = {recs['minimum_delta_before_review']}")
    print(f"  sequential over COMBINED mode  = {recs['use_sequential_over_combined']}")

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 07: Interference analysis")
    parser.add_argument("--input",  default="results/study_07_interference.jsonl", type=Path)
    parser.add_argument("--plot",   default="results/study_07_cohens_d.png",       type=Path)
    parser.add_argument("--output", default="results/study_07_findings.json",      type=Path)
    args = parser.parse_args()
    analyze(args.input, args.plot, args.output)


if __name__ == "__main__":
    main()
