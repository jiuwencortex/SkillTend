# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fit Q metric weights using the human-labelled dataset.

Uses ridge regression with leave-one-out cross-validation to find the
weight vector (w_rr, w_id_score, w_tsr, w_ps) that best predicts the
consensus human label from the four sub-metrics.

The resulting weights replace the default 0.30/0.20/0.40/0.10 split in
QWeights and should be written back to shared/skill_quality_scorer.py
as the new DEFAULT_WEIGHTS after Study 01 completes.

Usage::

    python -m study_01_skill_quality_metrics.calibrate_weights \
        --labels results/study_01_labels.jsonl \
        --metrics results/study_01_metrics.jsonl \
        --output results/study_01_calibrated_weights.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from study_01_skill_quality_metrics.compute_validity import (
    _load_labels,
    _load_metrics,
    _consensus_labels,
)

_METRIC_NAMES = ["rr", "id_score", "tsr", "ps"]


def _build_dataset(
    labels_path: Path,
    metrics_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, skill_names) aligned on skills with both labels and metrics."""
    consensus = _consensus_labels(_load_labels(labels_path))
    metrics = _load_metrics(metrics_path)

    common = sorted(set(consensus) & set(metrics))
    rows: List[List[float]] = []
    y_vals: List[float] = []
    valid_skills: List[str] = []

    for s in common:
        row = [metrics[s].get(m, float("nan")) for m in _METRIC_NAMES]
        if any(math.isnan(v) for v in row):
            continue
        rows.append(row)
        y_vals.append(consensus[s])
        valid_skills.append(s)

    return np.array(rows), np.array(y_vals), valid_skills


def calibrate(
    labels_path: Path,
    metrics_path: Path,
) -> Dict:
    """Run ridge regression and return calibrated weights + LOO-CV stats."""
    X, y, skills = _build_dataset(labels_path, metrics_path)

    if len(skills) < 6:
        return {"error": f"Need at least 6 labelled skills with all metrics; got {len(skills)}"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge with LOO-CV to select regularisation strength
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=LeaveOneOut(), scoring="r2", fit_intercept=True)
    model.fit(X_scaled, y)

    # Raw coefficients (in scaled space)
    raw_coefs = model.coef_

    # Map back to original scale (divide by feature std to get importance per unit)
    feature_stds = scaler.scale_
    unscaled_coefs = raw_coefs / feature_stds

    # Normalise to sum to 1 (clamp negatives to 0 to keep weights interpretable)
    clamped = np.maximum(unscaled_coefs, 0)
    total = clamped.sum()
    if total < 1e-9:
        normalised = np.ones(len(_METRIC_NAMES)) / len(_METRIC_NAMES)
    else:
        normalised = clamped / total

    weights = {m: round(float(w), 4) for m, w in zip(_METRIC_NAMES, normalised)}

    # LOO-CV R² score
    loo = LeaveOneOut()
    residuals: List[float] = []
    for train_idx, test_idx in loo.split(X_scaled):
        m = RidgeCV(alphas=alphas, cv=3, fit_intercept=True)
        m.fit(X_scaled[train_idx], y[train_idx])
        pred = m.predict(X_scaled[test_idx])[0]
        residuals.append((pred - y[test_idx[0]]) ** 2)

    loo_rmse = math.sqrt(sum(residuals) / len(residuals))
    y_mean = float(y.mean())
    ss_tot = float(sum((yi - y_mean) ** 2 for yi in y))
    loo_r2 = 1 - (sum(residuals) / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "n_skills": len(skills),
        "alpha_selected": float(model.alpha_),
        "raw_ridge_coefs": {m: round(float(c), 6) for m, c in zip(_METRIC_NAMES, raw_coefs)},
        "calibrated_weights": weights,
        "loo_cv_rmse": round(loo_rmse, 4),
        "loo_cv_r2": round(loo_r2, 4) if not math.isnan(loo_r2) else None,
        "recommendation": (
            f"Update QWeights in shared/skill_quality_scorer.py to: "
            + ", ".join(f"{m}={w}" for m, w in weights.items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 01: Calibrate Q metric weights")
    parser.add_argument("--labels", default="results/study_01_labels.jsonl", type=Path)
    parser.add_argument("--metrics", default="results/study_01_metrics.jsonl", type=Path)
    parser.add_argument("--output", default="results/study_01_calibrated_weights.json", type=Path)
    args = parser.parse_args()

    result = calibrate(args.labels, args.metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
