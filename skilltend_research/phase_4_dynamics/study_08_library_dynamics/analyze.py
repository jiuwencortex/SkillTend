# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 08 analysis — skill library dynamics and coverage.

Produces:
  - LC growth curve plot (LC vs. sessions, with saturation annotation)
  - Redundancy accumulation plot (mean_pairwise_sim vs. sessions)
  - Gini coefficient over sessions plot
  - Consolidation ROI bar chart (ΔLC per library-size checkpoint)
  - study_08_findings.json with curator_llm_consolidation enable threshold
    and LC saturation session recommendation

Usage::

    python -m phase_4_dynamics.study_08_library_dynamics.analyze \\
        --input  results/study_08_dynamics.jsonl \\
        --plot   results/study_08_growth.png \\
        --output results/study_08_findings.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.result_logger import ResultLogger


# ── Data helpers ──────────────────────────────────────────────────────────────


def _load(path: Path) -> List[Dict[str, Any]]:
    return [r for r in ResultLogger.read_metrics(path) if not r.get("_failed")]


def _growth_records(records: List[Dict]) -> List[Dict]:
    """Phase A growth records, sorted by session number."""
    gr = [r for r in records if r.get("phase") == "A_growth"]
    return sorted(gr, key=lambda r: r.get("session", 0))


def _consolidation_records(records: List[Dict]) -> List[Dict]:
    """Phase C consolidation records, sorted by target_size."""
    cr = [r for r in records if r.get("phase") == "C_consolidation"]
    return sorted(cr, key=lambda r: r.get("target_size", 0))


# ── Saturation detection ───────────────────────────────────────────────────────


def _saturation_session(
    growth: List[Dict], threshold: float = 0.005
) -> Optional[int]:
    """First session where marginal LC gain falls below *threshold*."""
    lc_series = [(r.get("session", 0), r.get("lc", 0.0)) for r in growth]
    for i in range(1, len(lc_series)):
        gain = lc_series[i][1] - lc_series[i - 1][1]
        if gain < threshold:
            return lc_series[i][0]
    return None


def _minimum_beneficial_consolidation_size(consol: List[Dict]) -> Optional[int]:
    """Smallest target_size where lc_delta > 0 (consolidation helped)."""
    for r in consol:
        if r.get("lc_delta", 0.0) > 0:
            return r.get("target_size")
    return None


# ── Plots ─────────────────────────────────────────────────────────────────────


def _plot_growth_curves(
    growth: List[Dict],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping growth-curve plot")
        return

    if not growth:
        return

    sessions = [r.get("session", 0) for r in growth]
    lc       = [r.get("lc", 0.0)              for r in growth]
    red      = [r.get("mean_pairwise_sim", 0.0) for r in growth]
    gini     = [r.get("gini", 0.0)             for r in growth]
    n_skills = [r.get("n_skills", 0)           for r in growth]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: LC growth curve
    ax = axes[0][0]
    ax.plot(sessions, lc, marker="o", linewidth=2, color="steelblue", label="LC")
    sat = _saturation_session(growth)
    if sat is not None:
        ax.axvline(sat, color="orange", linestyle=":", linewidth=2,
                   label=f"saturation ≈ session {sat}")
    ax.set_xlabel("Session number")
    ax.set_ylabel("Library Coverage (LC)")
    ax.set_title("Study 08 — Library Coverage vs. Sessions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Redundancy accumulation
    ax = axes[0][1]
    ax.plot(sessions, red, marker="s", linewidth=2, color="tomato", label="Mean pairwise sim")
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="high redundancy (0.5)")
    ax.set_xlabel("Session number")
    ax.set_ylabel("Mean Pairwise Cosine Similarity")
    ax.set_title("Study 08 — Redundancy Accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Gini coefficient over sessions
    ax = axes[1][0]
    ax.plot(sessions, gini, marker="^", linewidth=2, color="forestgreen", label="Gini (use_count)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Session number")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Study 08 — Usage Inequality (Gini) vs. Sessions")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Library size
    ax = axes[1][1]
    ax.plot(sessions, n_skills, marker="D", linewidth=2, color="mediumpurple", label="# skills")
    ax.set_xlabel("Session number")
    ax.set_ylabel("Number of skills")
    ax.set_title("Study 08 — Library Size Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Growth-curve plot saved to {output_path}")


def _plot_consolidation_roi(
    consol: List[Dict],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not consol:
        return

    labels = [str(r.get("target_size", "?")) for r in consol]
    lc_deltas = [r.get("lc_delta", 0.0) for r in consol]
    colors = ["green" if d > 0 else "salmon" for d in lc_deltas]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, lc_deltas, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.005, color="green", linestyle="--", alpha=0.6, label="benefit threshold (0.005)")

    for bar, val in zip(bars, lc_deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.0015 if val >= 0 else -0.003),
            f"{val:+.3f}",
            ha="center", va="bottom" if val >= 0 else "top", fontsize=9,
        )

    ax.set_xlabel("Approximate library size at consolidation")
    ax.set_ylabel("ΔLC (LC after − LC before)")
    ax.set_title("Study 08 — LLM Consolidation ROI by Library Size")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    roi_path = output_path.parent / (output_path.stem + "_consolidation_roi.png")
    plt.tight_layout()
    plt.savefig(roi_path, dpi=150)
    plt.close()
    print(f"Consolidation ROI plot saved to {roi_path}")


# ── Main analysis ─────────────────────────────────────────────────────────────


def analyze(
    input_path: Path,
    plot_path: Optional[Path],
    output_path: Path,
) -> Dict[str, Any]:
    records = _load(input_path)
    if not records:
        return {"error": f"No valid records in {input_path}"}

    growth = _growth_records(records)
    consol = _consolidation_records(records)

    # Saturation
    sat_session = _saturation_session(growth)

    # LC summary
    lc_series = [(r.get("session", 0), r.get("lc", 0.0)) for r in growth]
    peak_lc = max((lc for _, lc in lc_series), default=0.0)
    peak_redundancy = max((r.get("mean_pairwise_sim", 0.0) for r in growth), default=0.0)
    peak_gini = max((r.get("gini", 0.0) for r in growth), default=0.0)

    # Consolidation ROI summary
    min_beneficial_size = _minimum_beneficial_consolidation_size(consol)
    consol_summary = [
        {
            "target_size": r.get("target_size"),
            "lc_before": r.get("lc_before"),
            "lc_after": r.get("lc_after"),
            "lc_delta": r.get("lc_delta"),
            "suggestions_count": r.get("suggestions_count"),
            "is_beneficial": (r.get("lc_delta", 0.0) or 0.0) > 0,
        }
        for r in consol
    ]

    # Recommendations
    enable_consolidation = min_beneficial_size is not None
    recommendations: Dict[str, Any] = {
        "lc_saturation_session": sat_session,
        "enable_curator_llm_consolidation": enable_consolidation,
        "curator_llm_consolidation_min_skills": min_beneficial_size,
        "notes": (
            f"LC plateaus around session {sat_session}; "
            "adding more sessions beyond this yields diminishing returns. "
            + (
                f"Enable curator_llm_consolidation when the library has ≥ {min_beneficial_size} skills."
                if enable_consolidation else
                "Consolidation showed no LC benefit at any tested library size."
            )
        ),
    }

    findings: Dict[str, Any] = {
        "n_records": len(records),
        "growth_records": len(growth),
        "consolidation_records": len(consol),
        "peak_lc": round(peak_lc, 4),
        "peak_redundancy": round(peak_redundancy, 4),
        "peak_gini": round(peak_gini, 4),
        "saturation_session": sat_session,
        "consolidation_summary": consol_summary,
        "minimum_beneficial_consolidation_size": min_beneficial_size,
        "recommendations": recommendations,
    }

    if plot_path:
        _plot_growth_curves(growth, plot_path)
        _plot_consolidation_roi(consol, plot_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    print(f"\nStudy 08 Recommendations:")
    print(f"  LC saturation session                    = {sat_session}")
    print(f"  enable_curator_llm_consolidation         = {enable_consolidation}")
    print(f"  curator_llm_consolidation_min_skills     = {min_beneficial_size}")

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 08: Library dynamics analysis")
    parser.add_argument("--input",  default="results/study_08_dynamics.jsonl", type=Path)
    parser.add_argument("--plot",   default="results/study_08_growth.png",     type=Path)
    parser.add_argument("--output", default="results/study_08_findings.json",  type=Path)
    args = parser.parse_args()
    analyze(args.input, args.plot, args.output)


if __name__ == "__main__":
    main()
