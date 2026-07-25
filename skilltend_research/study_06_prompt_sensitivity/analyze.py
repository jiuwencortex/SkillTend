# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 06 analysis — prompt sensitivity results.

Produces:
  - Bar chart: Q drop per ablated element vs. full prompt
  - Line chart: Q vs. context window size
  - Scatter: Q mean vs. Q std (robustness under adversarial noise)
  - findings.json with revised prompt element priorities and context recommendation

Usage::

    python -m study_06_prompt_sensitivity.analyze \
        --input  results/study_06_ablations.jsonl \
        --plot   results/study_06_ablation_bars.png \
        --output results/study_06_findings.json
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
    return [r for r in ResultLogger.read_metrics(path) if "q_mean" in r]


def _full_prompt_q(records: List[Dict]) -> Optional[float]:
    """Return the Q score for the 'full' variant at 'full' context strategy."""
    candidates = [
        r for r in records
        if r.get("variant") == "full"
        and r.get("context_strategy") == "full"
        and not r.get("adversarial")
    ]
    if not candidates:
        return None
    return sum(r["q_mean"] for r in candidates) / len(candidates)


def _ablation_drops(records: List[Dict], full_q: float) -> List[Dict[str, Any]]:
    """Compute Q drop for each ablated variant vs. the full prompt."""
    drops = []
    ablation_variants = [r for r in records
                         if r.get("ablated_elements") and not r.get("adversarial")]
    by_variant: Dict[str, List[float]] = defaultdict(list)
    for r in ablation_variants:
        by_variant[r.get("variant", "")].append(r.get("q_mean", 0))

    for variant, qs in by_variant.items():
        mean_q = sum(qs) / len(qs)
        drops.append({
            "variant": variant,
            "q_mean": round(mean_q, 4),
            "q_drop": round(full_q - mean_q, 4),
            "q_drop_pct": round((full_q - mean_q) / full_q * 100, 2) if full_q else 0,
        })
    return sorted(drops, key=lambda d: d["q_drop"], reverse=True)


def _context_curve(records: List[Dict]) -> List[Dict]:
    """Q vs. context strategy for the full prompt variant."""
    ctx_recs = [r for r in records
                if r.get("variant") == "full" and not r.get("adversarial")]
    by_ctx: Dict[str, List[float]] = defaultdict(list)
    for r in ctx_recs:
        by_ctx[r.get("context_strategy", "")].append(r.get("q_mean", 0))

    # Order by number of turns (full last)
    order = ["last_5_turns", "last_10_turns", "last_20_turns", "full"]
    result = []
    for ctx in order:
        if ctx in by_ctx:
            qs = by_ctx[ctx]
            result.append({
                "context_strategy": ctx,
                "q_mean": round(sum(qs) / len(qs), 4),
            })
    return result


def _adversarial_robustness(records: List[Dict], full_q: Optional[float]) -> Dict:
    """Compare Q std under adversarial noise for full vs. minimal prompt."""
    adv = [r for r in records if r.get("adversarial")]
    by_variant: Dict[str, Dict] = {}
    for r in adv:
        v = r.get("variant", "")
        if v not in by_variant:
            by_variant[v] = {"q_vals": [], "q_std_vals": []}
        by_variant[v]["q_vals"].append(r.get("q_mean", 0))
        by_variant[v]["q_std_vals"].append(r.get("q_std", 0))

    result = {}
    for v, data in by_variant.items():
        qs = data["q_vals"]
        stds = data["q_std_vals"]
        result[v] = {
            "q_mean_adversarial": round(sum(qs) / len(qs), 4),
            "q_std_adversarial": round(sum(stds) / len(stds), 4),
            "q_drop_vs_clean": round((full_q or 0) - sum(qs) / len(qs), 4) if full_q else None,
        }
    return result


# ── Plot ──────────────────────────────────────────────────────────────────────


def _plot(records: List[Dict], drops: List[Dict], ctx_curve: List[Dict],
          output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Q drop bar chart
    ax = axes[0]
    labels = [d["variant"].replace("no_", "−").replace("_", " ") for d in drops]
    values = [d["q_drop"] for d in drops]
    colors = ["#d62728" if v > 0.02 else "#ff7f0e" if v > 0.005 else "#2ca02c"
              for v in values]
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Q drop vs. full prompt (positive = worse without this element)")
    ax.set_title("Study 06 — Prompt Element Ablation\n(which elements are load-bearing?)")
    ax.grid(True, alpha=0.3, axis="x")

    # Right: context window curve
    ax = axes[1]
    if ctx_curve:
        ctx_labels = [c["context_strategy"] for c in ctx_curve]
        ctx_qs = [c["q_mean"] for c in ctx_curve]
        ax.plot(ctx_labels, ctx_qs, marker="o", linewidth=2, color="steelblue")
        ax.set_ylabel("Q score")
        ax.set_xlabel("Context window strategy")
        ax.set_title("Study 06 — Q vs. Context Window Strategy")
        ax.grid(True, alpha=0.3)
        # Annotate each point
        for x, y in zip(ctx_labels, ctx_qs):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

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

    full_q = _full_prompt_q(records)
    drops = _ablation_drops(records, full_q or 0.0)
    ctx_curve = _context_curve(records)
    robustness = _adversarial_robustness(records, full_q)

    # Few-shot benefit
    few_shot_recs = [r for r in records if r.get("variant") == "full_few_shot"
                     and not r.get("adversarial")]
    few_shot_q = (
        sum(r["q_mean"] for r in few_shot_recs) / len(few_shot_recs)
        if few_shot_recs else None
    )
    few_shot_improvement = round(few_shot_q - (full_q or 0), 4) if few_shot_q and full_q else None

    # Identify load-bearing elements (drop > 0.02 = critical)
    critical = [d["variant"].replace("no_", "") for d in drops if d["q_drop"] > 0.02]
    dispensable = [d["variant"].replace("no_", "") for d in drops if d["q_drop"] < 0.005]

    # Best context strategy
    best_ctx = max(ctx_curve, key=lambda c: c["q_mean"]) if ctx_curve else None

    findings: Dict[str, Any] = {
        "n_records": len(records),
        "full_prompt_q": full_q,
        "ablation_drops": drops,
        "critical_elements": critical,
        "dispensable_elements": dispensable,
        "few_shot_improvement": few_shot_improvement,
        "adopt_few_shot": (few_shot_improvement is not None and few_shot_improvement > 0.03),
        "context_curve": ctx_curve,
        "best_context_strategy": best_ctx,
        "adversarial_robustness": robustness,
        "recommendations": {
            "keep_in_prompt": critical,
            "can_simplify": dispensable,
            "context_strategy": best_ctx.get("context_strategy") if best_ctx else "full",
            "add_few_shot_examples": (few_shot_improvement or 0) > 0.03,
        },
    }

    if plot_path:
        _plot(records, drops, ctx_curve, plot_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    print(f"\nLoad-bearing elements (drop > 0.02): {critical}")
    print(f"Dispensable elements (drop < 0.005): {dispensable}")
    print(f"Few-shot improvement:                 {few_shot_improvement}")
    print(f"Best context strategy:                {best_ctx.get('context_strategy') if best_ctx else 'N/A'}")

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 06: Prompt sensitivity analysis")
    parser.add_argument("--input",  default="results/study_06_ablations.jsonl", type=Path)
    parser.add_argument("--plot",   default="results/study_06_ablation_bars.png", type=Path)
    parser.add_argument("--output", default="results/study_06_findings.json",    type=Path)
    args = parser.parse_args()
    analyze(args.input, args.plot, args.output)


if __name__ == "__main__":
    main()
