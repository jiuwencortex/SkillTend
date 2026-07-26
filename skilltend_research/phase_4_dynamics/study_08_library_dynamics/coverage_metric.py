# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Library Coverage (LC) metric and redundancy measures.

Library Coverage (LC):
  Given a task distribution P(task) and a skill library S,
  LC = P(task is solvable using at least one skill in S with RR >= 0.6)

In practice, we approximate this as:
  LC = fraction of evaluation tasks for which at least one skill has
       cosine similarity >= 0.6 with the task query embedding.

Redundancy proxy:
  Mean pairwise cosine similarity between all skill embeddings.
  Higher = more redundant library.

Gini coefficient of use_count:
  Measures inequality in skill usage.  Gini = 0 means all skills used equally;
  Gini = 1 means one skill accounts for all use.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RR_THRESHOLD = 0.6  # minimum RR for a skill to "cover" a task


async def compute_library_coverage(
    skills_root: Path,
    task_queries: List[str],
    embedding_model: str = "openai/text-embedding-3-small",
    dry_run: bool = False,
) -> Dict[str, float]:
    """Compute LC, mean pairwise similarity, and Gini coefficient.

    Args:
        skills_root:     Root directory containing SKILL.md files.
        task_queries:    List of evaluation task user queries.
        embedding_model: litellm embedding model.
        dry_run:         If True, return stub values without LLM calls.

    Returns:
        Dict with keys: lc, mean_pairwise_sim, gini, n_skills.
    """
    skill_paths = list(skills_root.rglob("SKILL.md"))
    n_skills = len(skill_paths)

    if dry_run or not skill_paths or not task_queries:
        return {
            "lc": 0.5,
            "mean_pairwise_sim": 0.5,
            "gini": 0.5,
            "n_skills": n_skills,
        }

    skill_texts = [p.read_text(encoding="utf-8")[:4000] for p in skill_paths]

    # Embed all skills and tasks
    skill_embeddings = [await _embed(t, embedding_model) for t in skill_texts]
    task_embeddings = [await _embed(q, embedding_model) for q in task_queries]

    # LC: for each task, check if any skill has sim >= threshold
    covered = 0
    for task_vec in task_embeddings:
        for skill_vec in skill_embeddings:
            if _cosine(skill_vec, task_vec) >= RR_THRESHOLD:
                covered += 1
                break
    lc = covered / len(task_queries) if task_queries else 0.0

    # Mean pairwise similarity
    mean_pw = _mean_pairwise_similarity(skill_embeddings)

    # Gini coefficient of use_count
    use_counts = _read_use_counts(skills_root, skill_paths)
    gini = _gini_coefficient(use_counts)

    return {
        "lc": round(lc, 4),
        "mean_pairwise_sim": round(mean_pw, 4),
        "gini": round(gini, 4),
        "n_skills": n_skills,
    }


def _mean_pairwise_similarity(embeddings: List[List[float]]) -> float:
    """Compute mean cosine similarity over all pairs."""
    if len(embeddings) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            total += _cosine(embeddings[i], embeddings[j])
            count += 1
    return total / count if count > 0 else 0.0


def _read_use_counts(skills_root: Path, skill_paths: List[Path]) -> List[float]:
    """Read use_count from .usage.json sidecars."""
    counts: List[float] = []
    for p in skill_paths:
        usage_file = p.parent / ".usage.json"
        if usage_file.exists():
            try:
                usage = json.loads(usage_file.read_text())
                counts.append(float(usage.get("use_count", 1)))
            except (json.JSONDecodeError, TypeError):
                counts.append(1.0)
        else:
            counts.append(1.0)
    return counts


def _gini_coefficient(values: List[float]) -> float:
    """Gini coefficient of a list of non-negative values."""
    if not values:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    return (2 * cumsum) / (n * total) - (n + 1) / n


async def _embed(text: str, model: str) -> List[float]:
    import litellm
    response = await litellm.aembedding(model=model, input=[text])
    return response.data[0]["embedding"]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(y * y for y in b))
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)
