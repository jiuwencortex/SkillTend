# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""AdaptiveTrigger — conversation-distance-based review trigger.

Instead of firing every fixed N tool-calls or M user-turns, AdaptiveTrigger
monitors the cosine distance between successive conversation embeddings and
fires a review when the conversation has changed meaningfully since the last review.

The key insight: if the agent is doing repetitive tool calls on the same topic,
firing a skill review every 10 calls wastes tokens.  If the conversation jumps
to a new domain rapidly, waiting 10 calls may be too slow to capture the learning.

Usage::

    trigger = AdaptiveTrigger(
        embedding_model="openai/text-embedding-3-small",
        distance_threshold=0.25,   # cosine distance; tune in the sweep
    )

    # After each user turn or tool call:
    should_review, reason = await trigger.check(messages_snapshot)
    if should_review:
        # fire run_background_review()
        trigger.reset()
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import litellm


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Cosine distance in [0, 2]."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    similarity = dot / (mag_a * mag_b)
    return 1.0 - similarity  # distance in [0, 2], typically [0, 1]


def _extract_text(messages: List[Dict[str, Any]]) -> str:
    """Concatenate the last N assistant+user messages into a single string."""
    parts: List[str] = []
    for m in messages[-20:]:  # only consider the most recent window
        role = m.get("role", "")
        content = m.get("content") or ""
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return " ".join(parts)[:4000]  # cap to avoid large embedding costs


class AdaptiveTrigger:
    """Fires a review when the conversation has drifted by more than a threshold.

    Comparison between studies: this is evaluated against fixed (N, M) configs
    in study_02 to determine whether it achieves better Q/token efficiency.
    """

    def __init__(
        self,
        embedding_model: str = "openai/text-embedding-3-small",
        distance_threshold: float = 0.25,
        min_turns_between_reviews: int = 3,
        dry_run: bool = False,
    ) -> None:
        self._model = embedding_model
        self._threshold = distance_threshold
        self._min_turns = min_turns_between_reviews
        self._dry_run = dry_run

        self._last_review_embedding: Optional[List[float]] = None
        self._turns_since_review: int = 0

    async def check(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Return (should_fire, reason) for the current messages snapshot.

        Args:
            messages: Current conversation messages snapshot.

        Returns:
            Tuple of (bool, reason_string).
        """
        self._turns_since_review += 1

        # Enforce minimum turns between reviews
        if self._turns_since_review < self._min_turns:
            return False, f"min_turns not reached ({self._turns_since_review}/{self._min_turns})"

        if self._dry_run:
            # Stub: fire every min_turns turns in dry-run mode
            if self._turns_since_review >= self._min_turns:
                return True, "dry-run: firing at min_turns"
            return False, "dry-run: waiting"

        text = _extract_text(messages)
        if not text:
            return False, "no conversation text yet"

        current_embedding = await self._embed(text)

        if self._last_review_embedding is None:
            # First check: store embedding but don't fire yet
            self._last_review_embedding = current_embedding
            return False, "first embedding stored; no prior to compare"

        distance = _cosine_distance(self._last_review_embedding, current_embedding)

        if distance >= self._threshold:
            return True, f"conversation distance {distance:.3f} >= threshold {self._threshold}"
        return False, f"conversation distance {distance:.3f} < threshold {self._threshold}"

    def reset(self, current_embedding: Optional[List[float]] = None) -> None:
        """Reset the trigger after a review fires.

        Optionally provide the embedding at review time to use as the new baseline.
        """
        self._last_review_embedding = current_embedding
        self._turns_since_review = 0

    async def _embed(self, text: str) -> List[float]:
        response = await litellm.aembedding(model=self._model, input=[text])
        return response.data[0]["embedding"]
