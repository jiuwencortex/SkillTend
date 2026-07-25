# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Validity analysis for Q sub-metrics vs human labels.

Reads the labels JSONL from label_tool.py and the computed metrics JSONL
from runner.py, then computes:

  - Spearman ρ between each metric and the consensus human label
  - Coefficient of variation (CV) for metric stability
  - Pairwise Pearson r between metrics (redundancy check)
  - Fleiss κ for inter-rater agreement (when ≥ 2 raters)

Usage::

    python -m study_01_skill_quality_metrics.compute_validity \
        --labels results/study_01_labels.jsonl \
        --metrics results/study_01_metrics.jsonl \
        --output results/study_01_validity_report.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Statistics helpers ────────────────────────────────────────────────────────


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _std(xs: List[float]) -> float:
    return math.sqrt(_var(xs))


def _rank(xs: List[float]) -> List[float]:
    """Return rank list (1-indexed) with average ranks for ties."""
    sorted_vals = sorted(enumerate(xs), key=lambda iv: iv[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    rx = _rank(xs)
    ry = _rank(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6 * d2) / (n * (n ** 2 - 1))


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den > 0 else 0.0


def _fleiss_kappa(ratings: List[List[int]], n_categories: int = 3) -> float:
    """Compute Fleiss κ for multiple raters.

    ratings: list of lists; each inner list is one item's ratings from all raters.
    n_categories: number of possible categories (here 1/2/3).
    """
    n_items = len(ratings)
    n_raters = max(len(r) for r in ratings)
    if n_items == 0 or n_raters < 2:
        return float("nan")

    # Count matrix: P[i][k] = fraction of raters who assigned item i to category k
    P_i = []
    for item_ratings in ratings:
        counts = [0] * n_categories
        for r in item_ratings:
            if 1 <= r <= n_categories:
                counts[r - 1] += 1
        n_r = len(item_ratings)
        P_ij = [c / n_r for c in counts]
        # Observed agreement for item i
        p_i = (sum(c * (c - 1) for c in counts)) / (n_r * (n_r - 1)) if n_r > 1 else 0.0
        P_i.append(p_i)

    P_bar = _mean(P_i)

    # Expected agreement
    all_ratings_flat = [r for item in ratings for r in item]
    p_k = [all_ratings_flat.count(k + 1) / len(all_ratings_flat) for k in range(n_categories)]
    P_e = sum(p ** 2 for p in p_k)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_labels(path: Path) -> Dict[str, Dict[str, int]]:
    """Return {skill_name: {rater_id: label}} from labels JSONL."""
    result: Dict[str, Dict[str, int]] = defaultdict(dict)
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            result[rec["skill_name"]][rec["rater_id"]] = rec["label"]
        except (json.JSONDecodeError, KeyError):
            pass
    return result


def _consensus_labels(labels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Return {skill_name: mean_label} as the consensus across raters."""
    return {
        name: _mean(list(raters.values()))
        for name, raters in labels.items()
        if raters
    }


def _load_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    """Return {skill_name: {metric_name: value}} from metrics JSONL."""
    result: Dict[str, Dict[str, float]] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            name = rec.get("skill_name")
            if name:
                result[name] = rec.get("metrics", {})
        except (json.JSONDecodeError, KeyError):
            pass
    return result


# ── Validity report ───────────────────────────────────────────────────────────


_METRIC_NAMES = ["rr", "id_score", "tsr", "ps"]
_VALIDITY_THRESHOLD_SPEARMAN = 0.50
_STABILITY_CV_THRESHOLD = {"rr": 0.08, "id_score": 0.08, "tsr": 0.15, "ps": 0.08}
_REDUNDANCY_THRESHOLD_PEARSON = 0.92


def compute_validity_report(
    labels_path: Path,
    metrics_path: Path,
) -> Dict[str, Any]:
    """Compute the full validity report and return it as a dict."""
    labels_by_rater = _load_labels(labels_path)
    consensus = _consensus_labels(labels_by_rater)
    metrics_by_skill = _load_metrics(metrics_path)

    # Align on skills that have both labels and metrics
    common_skills = sorted(set(consensus) & set(metrics_by_skill))
    if len(common_skills) < 5:
        return {"error": f"Only {len(common_skills)} skills have both labels and metrics. Need at least 5."}

    human_scores = [consensus[s] for s in common_skills]

    # ── Spearman correlations ─────────────────────────────────────────────────
    spearman: Dict[str, float] = {}
    metric_vectors: Dict[str, List[float]] = {m: [] for m in _METRIC_NAMES}
    for s in common_skills:
        for m in _METRIC_NAMES:
            val = metrics_by_skill[s].get(m, float("nan"))
            metric_vectors[m].append(val)

    for m in _METRIC_NAMES:
        vals = metric_vectors[m]
        valid_pairs = [(h, v) for h, v in zip(human_scores, vals) if not math.isnan(v)]
        if len(valid_pairs) < 5:
            spearman[m] = float("nan")
        else:
            h_clean, v_clean = zip(*valid_pairs)
            spearman[m] = _spearman(list(h_clean), list(v_clean))

    # ── Fleiss κ ─────────────────────────────────────────────────────────────
    ratings_matrix = [
        list(labels_by_rater.get(s, {}).values())
        for s in common_skills
    ]
    kappa = _fleiss_kappa(ratings_matrix)

    # ── Pairwise Pearson (redundancy) ─────────────────────────────────────────
    pairwise: Dict[str, float] = {}
    for i, m1 in enumerate(_METRIC_NAMES):
        for m2 in _METRIC_NAMES[i + 1:]:
            v1 = metric_vectors[m1]
            v2 = metric_vectors[m2]
            valid = [(a, b) for a, b in zip(v1, v2) if not math.isnan(a) and not math.isnan(b)]
            if len(valid) >= 3:
                a_vals, b_vals = zip(*valid)
                pairwise[f"{m1}_vs_{m2}"] = _pearson(list(a_vals), list(b_vals))
            else:
                pairwise[f"{m1}_vs_{m2}"] = float("nan")

    # ── Validity decisions ────────────────────────────────────────────────────
    metric_pass = {
        m: (not math.isnan(spearman[m]) and spearman[m] >= _VALIDITY_THRESHOLD_SPEARMAN)
        for m in _METRIC_NAMES
    }
    redundancy_flags = {
        pair: (not math.isnan(r) and abs(r) >= _REDUNDANCY_THRESHOLD_PEARSON)
        for pair, r in pairwise.items()
    }

    return {
        "n_skills": len(common_skills),
        "n_raters": len({r for raters in labels_by_rater.values() for r in raters}),
        "fleiss_kappa": round(kappa, 4) if not math.isnan(kappa) else None,
        "kappa_ok": (not math.isnan(kappa) and kappa >= 0.65),
        "spearman": {m: round(v, 4) if not math.isnan(v) else None for m, v in spearman.items()},
        "metric_validity": metric_pass,
        "pairwise_pearson": {k: round(v, 4) if not math.isnan(v) else None for k, v in pairwise.items()},
        "redundancy_flags": redundancy_flags,
        "passing_metrics": [m for m, ok in metric_pass.items() if ok],
        "failing_metrics": [m for m, ok in metric_pass.items() if not ok],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 01: Metric validity analysis")
    parser.add_argument("--labels", default="results/study_01_labels.jsonl", type=Path)
    parser.add_argument("--metrics", default="results/study_01_metrics.jsonl", type=Path)
    parser.add_argument("--output", default="results/study_01_validity_report.json", type=Path)
    args = parser.parse_args()

    report = compute_validity_report(args.labels, args.metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if "error" not in report:
        print(f"\n{'=' * 60}")
        print(f"Fleiss κ = {report['fleiss_kappa']}  (pass: {report['kappa_ok']})")
        for m in _METRIC_NAMES:
            status = "PASS" if report["metric_validity"].get(m) else "FAIL"
            print(f"  {m:12s}  ρ = {report['spearman'].get(m)}  [{status}]")
        print(f"\nPassing metrics: {report['passing_metrics']}")
        print(f"Failing metrics: {report['failing_metrics']}")
        if any(report["redundancy_flags"].values()):
            print(f"\nWARNING: Redundant metric pairs: "
                  f"{[k for k, v in report['redundancy_flags'].items() if v]}")


if __name__ == "__main__":
    main()
