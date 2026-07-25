# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Feature-enriched XGBoost lifecycle predictor.

Predicts whether a skill will be used in the next 30 days given a set of
observable features from its .usage.json sidecar and SKILL.md content.

Features:
  - days_since_last_use       — primary recency signal
  - total_use_count           — cumulative usage
  - patch_count               — number of times edited
  - skill_char_len            — content length
  - information_density       — zstd-based ID score
  - days_since_created        — age of skill
  - avg_days_between_uses     — inverse of use rate
  - is_user_created           — 1 if created_by == "user"

Training data comes from simulate_usage.py (180-day simulation).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zstandard as zstd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


@dataclass
class SkillSnapshot:
    """Feature vector for one skill at one point in time."""

    skill_name: str
    days_since_last_use: float
    total_use_count: int
    patch_count: int
    skill_char_len: int
    information_density: float
    days_since_created: float
    avg_days_between_uses: float
    is_user_created: int

    # Target: will this skill be used in the next 30 days?
    used_in_next_30_days: Optional[int] = None

    def to_feature_array(self) -> np.ndarray:
        return np.array([
            self.days_since_last_use,
            self.total_use_count,
            self.patch_count,
            self.skill_char_len,
            self.information_density,
            self.days_since_created,
            self.avg_days_between_uses,
            self.is_user_created,
        ], dtype=float)

    FEATURE_NAMES = [
        "days_since_last_use",
        "total_use_count",
        "patch_count",
        "skill_char_len",
        "information_density",
        "days_since_created",
        "avg_days_between_uses",
        "is_user_created",
    ]


_COMPRESSOR = zstd.ZstdCompressor(level=3)


def _information_density(text: str) -> float:
    encoded = text.encode("utf-8")
    if len(encoded) < 64:
        return 0.0
    compressed = _COMPRESSOR.compress(encoded)
    return max(0.0, min(1.0, 1.0 - len(compressed) / len(encoded)))


def snapshot_from_files(
    skill_dir: Path,
    observation_day: int,
    usage_history: List[int],  # days on which skill was used
    total_days: int = 180,
) -> SkillSnapshot:
    """Build a SkillSnapshot from a skill directory and its synthetic usage history."""
    skill_md = skill_dir / "SKILL.md"
    usage_json = skill_dir / ".usage.json"

    text = skill_md.read_text(encoding="utf-8") if skill_md.exists() else ""
    usage = json.loads(usage_json.read_text()) if usage_json.exists() else {}

    past_uses = [d for d in usage_history if d < observation_day]
    future_uses = [d for d in usage_history if observation_day <= d < observation_day + 30]

    days_since_last = (observation_day - max(past_uses)) if past_uses else observation_day
    avg_days_between = (
        (max(past_uses) - min(past_uses)) / max(1, len(past_uses) - 1)
        if len(past_uses) >= 2 else float(observation_day)
    )

    return SkillSnapshot(
        skill_name=skill_dir.name,
        days_since_last_use=days_since_last,
        total_use_count=len(past_uses),
        patch_count=usage.get("patch_count", 0),
        skill_char_len=len(text),
        information_density=_information_density(text),
        days_since_created=observation_day,
        avg_days_between_uses=avg_days_between,
        is_user_created=1 if usage.get("created_by") == "user" else 0,
        used_in_next_30_days=1 if future_uses else 0,
    )


class LifecyclePredictor:
    """XGBoost-based skill lifecycle predictor."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 4) -> None:
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        self._is_fitted = False

    def fit(self, snapshots: List[SkillSnapshot]) -> Dict[str, Any]:
        """Train on a list of SkillSnapshots.  Returns CV AUC stats."""
        labelled = [s for s in snapshots if s.used_in_next_30_days is not None]
        if len(labelled) < 20:
            return {"error": f"Need ≥ 20 labelled snapshots; got {len(labelled)}"}

        X = np.array([s.to_feature_array() for s in labelled])
        y = np.array([s.used_in_next_30_days for s in labelled])

        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores: List[float] = []
        for train_idx, val_idx in skf.split(X, y):
            m = XGBClassifier(
                n_estimators=self._model.n_estimators,
                max_depth=self._model.max_depth,
                learning_rate=self._model.learning_rate,
                subsample=self._model.subsample,
                colsample_bytree=self._model.colsample_bytree,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            m.fit(X[train_idx], y[train_idx])
            proba = m.predict_proba(X[val_idx])[:, 1]
            auc = roc_auc_score(y[val_idx], proba) if len(set(y[val_idx])) > 1 else 0.5
            auc_scores.append(auc)

        # Fit final model on all data
        self._model.fit(X, y)
        self._is_fitted = True

        importances = dict(zip(SkillSnapshot.FEATURE_NAMES, self._model.feature_importances_.tolist()))

        return {
            "cv_auc_mean": round(float(np.mean(auc_scores)), 4),
            "cv_auc_std": round(float(np.std(auc_scores)), 4),
            "feature_importances": {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])},
            "n_samples": len(labelled),
            "recommendation": "Replace fixed FSM if CV AUC > 0.75",
        }

    def predict_proba(self, snapshot: SkillSnapshot) -> float:
        """Return probability that the skill will be used in the next 30 days."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba()")
        X = snapshot.to_feature_array().reshape(1, -1)
        return float(self._model.predict_proba(X)[0, 1])

    def should_keep_active(self, snapshot: SkillSnapshot, threshold: float = 0.3) -> bool:
        """Return True if the skill should remain ACTIVE (not be stalened)."""
        return self.predict_proba(snapshot) >= threshold

    def save(self, path: Path) -> None:
        import pickle
        path.write_bytes(pickle.dumps(self._model))

    @classmethod
    def load(cls, path: Path) -> "LifecyclePredictor":
        import pickle
        predictor = cls()
        predictor._model = pickle.loads(path.read_bytes())
        predictor._is_fitted = True
        return predictor
