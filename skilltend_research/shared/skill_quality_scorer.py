# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Canonical skill quality metric — Q score.

Four sub-metrics, each in [0, 1]:

  RR  — Retrieval Relevance
        Cosine similarity between the skill text embedding and the task query
        embeddings for the tasks the skill was derived from.  High RR means
        a downstream retrieval system would surface this skill for relevant queries.

  ID  — Information Density
        1 - (compressed_size / uncompressed_size) using zstd.  High ID means the
        skill content is not redundant.  A skill full of boilerplate or repeated
        phrases will compress well → low ID.

  TSR — Task Success Rate
        Fraction of tasks where an LLM agent equipped with the skill produces a
        better answer than the same agent without the skill (as judged by a
        separate LLM judge).  This is the most expensive metric to compute.

  PS  — Patch Stability
        1 - mean(normalized_edit_distance) across consecutive versions of a skill.
        Computed when multiple historical versions are available.  High PS means
        the skill has converged; low PS means it is still churning.

Combined score:
  Q = w_rr * RR + w_id * ID + w_tsr * TSR + w_ps * PS

Default weights are from the research plan (prior to Study 01 calibration):
  w_rr=0.30, w_id=0.20, w_tsr=0.40, w_ps=0.10
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, TYPE_CHECKING

import zstandard as zstd

if TYPE_CHECKING:
    from shared.task_bank import Task


# ── Weight configuration ──────────────────────────────────────────────────────


@dataclass
class QWeights:
    """Weights for the combined Q score. Must sum to 1.0."""

    rr: float = 0.30
    id_score: float = 0.20
    tsr: float = 0.40
    ps: float = 0.10

    def __post_init__(self) -> None:
        total = self.rr + self.id_score + self.tsr + self.ps
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"QWeights must sum to 1.0, got {total:.6f}")


DEFAULT_WEIGHTS = QWeights()


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class QScores:
    """Full quality score for one skill snapshot."""

    rr: float
    """Retrieval Relevance in [0, 1]."""

    id_score: float
    """Information Density in [0, 1]."""

    tsr: float
    """Task Success Rate in [0, 1]."""

    ps: float
    """Patch Stability in [0, 1]. NaN when only one version is available."""

    q: float
    """Combined weighted score in [0, 1]."""

    weights: QWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)

    def as_dict(self) -> dict:
        return {
            "rr": round(self.rr, 4),
            "id_score": round(self.id_score, 4),
            "tsr": round(self.tsr, 4),
            "ps": round(self.ps, 4) if not math.isnan(self.ps) else None,
            "q": round(self.q, 4),
        }


# ── Scorer ────────────────────────────────────────────────────────────────────


class SkillQualityScorer:
    """Computes the four Q sub-metrics for a skill snapshot.

    Args:
        embedding_model:  litellm-compatible model name for text embeddings.
                          e.g. "openai/text-embedding-3-small"
        judge_model:      litellm-compatible model name for the LLM judge used
                          in TSR scoring.  e.g. "openai/gpt-4o-mini"
        weights:          Q weight configuration.
        dry_run:          If True, RR and TSR return fixed stubs (0.5) without
                          making any LLM API calls.  Useful for structural tests.
    """

    def __init__(
        self,
        embedding_model: str = "openai/text-embedding-3-small",
        judge_model: str = "openai/gpt-4o-mini",
        weights: Optional[QWeights] = None,
        dry_run: bool = False,
    ) -> None:
        self._embedding_model = embedding_model
        self._judge_model = judge_model
        self._weights = weights or DEFAULT_WEIGHTS
        self._dry_run = dry_run
        self._zstd_compressor = zstd.ZstdCompressor(level=3)

    # ── Public API ────────────────────────────────────────────────────────────

    async def score(
        self,
        skill_text: str,
        tasks: List["Task"],
        skill_history: Optional[List[str]] = None,
    ) -> QScores:
        """Compute all four metrics and return a QScores object.

        Args:
            skill_text:    Current text content of the SKILL.md.
            tasks:         Tasks the skill should be relevant for (used by RR and TSR).
            skill_history: Previous versions of skill_text (oldest first, current last).
                           Required for PS > stub. If None, PS returns NaN.
        """
        rr = await self.score_retrieval_relevance(skill_text, [t.user_query for t in tasks])
        id_score = self.score_information_density(skill_text)
        tsr = await self.score_task_success_rate(skill_text, tasks)
        ps = self.score_patch_stability(skill_history) if skill_history else float("nan")
        q = self._combine(rr, id_score, tsr, ps)
        return QScores(rr=rr, id_score=id_score, tsr=tsr, ps=ps, q=q, weights=self._weights)

    async def score_retrieval_relevance(
        self,
        skill_text: str,
        queries: List[str],
    ) -> float:
        """Compute mean cosine similarity between skill embedding and query embeddings.

        Returns a float in [0, 1].  When dry_run=True returns 0.5.
        """
        if self._dry_run or not queries:
            return 0.5

        skill_vec = await self._embed(skill_text)
        similarities: List[float] = []
        for query in queries:
            query_vec = await self._embed(query)
            similarities.append(_cosine(skill_vec, query_vec))

        # Cosine similarity is in [-1, 1]; shift to [0, 1]
        raw = sum(similarities) / len(similarities)
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    def score_information_density(self, skill_text: str) -> float:
        """Compute information density as 1 - (compressed / uncompressed) ratio.

        Returns a float in [0, 1].  Empty or very short texts return 0.0.
        """
        encoded = skill_text.encode("utf-8")
        if len(encoded) < 64:
            return 0.0
        compressed = self._zstd_compressor.compress(encoded)
        ratio = len(compressed) / len(encoded)
        # High compression ratio → low density. Invert and clamp.
        return max(0.0, min(1.0, 1.0 - ratio))

    async def score_task_success_rate(
        self,
        skill_text: str,
        tasks: List["Task"],
        trials_per_task: int = 1,
    ) -> float:
        """Estimate TSR by comparing agent answers with and without the skill.

        For each task:
          1. Build a system prompt WITH the skill injected.
          2. Build a system prompt WITHOUT the skill.
          3. Ask the judge LLM which answer is better.
          4. TSR = fraction of tasks where the skill version wins.

        Args:
            skill_text:      Skill content to inject.
            tasks:           Tasks to evaluate.
            trials_per_task: Number of judge calls per task (averaging reduces noise).
        """
        if self._dry_run or not tasks:
            return 0.5

        wins = 0
        total = 0
        for task in tasks:
            for _ in range(trials_per_task):
                won = await self._judge_task(skill_text, task)
                if won:
                    wins += 1
                total += 1

        return wins / total if total > 0 else 0.5

    def score_patch_stability(self, versions: List[str]) -> float:
        """Compute stability across consecutive skill versions.

        Returns 1 - mean(normalised_edit_distance) over consecutive pairs.
        Returns NaN when fewer than 2 versions are provided.
        """
        if len(versions) < 2:
            return float("nan")

        distances: List[float] = []
        for a, b in zip(versions[:-1], versions[1:]):
            distances.append(_normalised_edit_distance(a, b))

        mean_dist = sum(distances) / len(distances)
        return max(0.0, min(1.0, 1.0 - mean_dist))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _combine(self, rr: float, id_score: float, tsr: float, ps: float) -> float:
        """Weighted sum, treating NaN ps as 0 (penalty for no history)."""
        ps_val = 0.0 if math.isnan(ps) else ps
        return (
            self._weights.rr * rr
            + self._weights.id_score * id_score
            + self._weights.tsr * tsr
            + self._weights.ps * ps_val
        )

    async def _embed(self, text: str) -> List[float]:
        """Return an embedding vector for the given text via litellm."""
        import litellm

        response = await litellm.aembedding(
            model=self._embedding_model,
            input=[text[:8000]],  # truncate to avoid token limit
        )
        return response.data[0]["embedding"]

    async def _judge_task(self, skill_text: str, task: "Task") -> bool:
        """Return True if the skill-equipped agent answer is preferred by the judge."""
        import litellm

        # Generate answer without skill
        answer_no_skill = await self._agent_answer(task, skill_injection=None)
        # Generate answer with skill
        answer_with_skill = await self._agent_answer(task, skill_injection=skill_text)

        judge_prompt = _build_judge_prompt(task, answer_no_skill, answer_with_skill)
        response = await litellm.acompletion(
            model=self._judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return verdict.startswith("B")  # "B" = with-skill answer is better

    async def _agent_answer(
        self,
        task: "Task",
        skill_injection: Optional[str],
    ) -> str:
        """Generate a short agent answer for the task, optionally with skill injected."""
        import litellm

        system = (
            "You are a helpful technical assistant. Answer the user's question concisely."
        )
        if skill_injection:
            system += f"\n\n---\nReference skill:\n{skill_injection[:2000]}\n---"

        response = await litellm.acompletion(
            model=self._judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": task.user_query},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _normalised_edit_distance(a: str, b: str) -> float:
    """Normalised Levenshtein distance (character level), result in [0, 1].

    Uses a word-level approximation for long strings to keep it fast.
    """
    # For long texts use word-level to keep O(n^2) tractable
    tokens_a = a.split()
    tokens_b = b.split()

    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 1.0

    n, m = len(tokens_a), len(tokens_b)
    # Cap for performance
    max_tokens = 500
    if n > max_tokens or m > max_tokens:
        tokens_a = tokens_a[:max_tokens]
        tokens_b = tokens_b[:max_tokens]
        n, m = len(tokens_a), len(tokens_b)

    # DP table
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])

    return dp[m] / max(n, m)


def _build_judge_prompt(task: "Task", answer_a: str, answer_b: str) -> str:
    """Build a prompt for the LLM judge to compare two answers.

    Returns "A" or "B" depending on which answer is better.
    """
    return f"""You are an expert technical judge. Compare these two answers to the following question.

QUESTION:
{task.user_query}

GROUND TRUTH (use as reference):
{task.ground_truth}

ANSWER A (without skill reference):
{answer_a}

ANSWER B (with skill reference):
{answer_b}

Which answer is more accurate, complete, and aligned with the ground truth?
Respond with only a single letter: A or B."""
