# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Coverage gap detector — finds task domains with repeated tool use but no skill creation.

A "coverage gap" is a domain where:
  1. The agent repeatedly calls tools (high tool-call count in recent sessions)
  2. No skill exists that covers that domain (low RR for that domain's queries)

These gaps represent missed opportunities for skill creation and should be
surfaced to the Curator or the agent for proactive skill generation.

Usage::

    detector = CoverageGapDetector(skills_root, embedding_model=...)
    gaps = await detector.detect(recent_sessions, threshold=0.6)
    for gap in gaps:
        print(f"Gap: {gap.domain}  tool_calls={gap.tool_call_count}  max_rr={gap.max_rr:.3f}")
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

RR_COVERAGE_THRESHOLD = 0.6
MIN_TOOL_CALLS_FOR_GAP = 5  # domain must have at least this many tool calls to qualify


@dataclass
class CoverageGap:
    """One identified coverage gap."""

    domain: str
    """Task domain with the gap."""

    tool_call_count: int
    """Number of tool calls in recent sessions for this domain."""

    max_rr: float
    """Maximum RR between any skill and this domain's queries (lower = bigger gap)."""

    representative_query: str
    """A sample query from this domain that is not covered."""

    severity: str
    """high / medium / low based on tool_call_count and max_rr."""


class CoverageGapDetector:
    """Detects domains where the agent works heavily but has no covering skill.

    Integrates with the Curator — can be called after each curator run to
    generate gap reports for human review or automated skill seeding.
    """

    def __init__(
        self,
        skills_root: Path,
        embedding_model: str = "openai/text-embedding-3-small",
        dry_run: bool = False,
    ) -> None:
        self._skills_root = skills_root
        self._embedding_model = embedding_model
        self._dry_run = dry_run

    async def detect(
        self,
        recent_sessions: List[List[Dict[str, Any]]],
        coverage_threshold: float = RR_COVERAGE_THRESHOLD,
        min_tool_calls: int = MIN_TOOL_CALLS_FOR_GAP,
    ) -> List[CoverageGap]:
        """Analyse recent session messages and return identified coverage gaps.

        Args:
            recent_sessions:   List of recent message lists (OpenAI format).
            coverage_threshold: Minimum RR for a skill to "cover" a query.
            min_tool_calls:     Minimum tool-call count before a domain qualifies as a gap.

        Returns:
            List of CoverageGap objects sorted by severity (high → low).
        """
        # Count tool calls per detected domain
        domain_tool_counts = self._count_tool_calls_by_domain(recent_sessions)
        if not domain_tool_counts:
            return []

        # Load skills from the library
        skill_paths = list(self._skills_root.rglob("SKILL.md"))
        if not skill_paths:
            # No skills at all — every domain with tool calls is a gap
            return [
                CoverageGap(
                    domain=d,
                    tool_call_count=count,
                    max_rr=0.0,
                    representative_query=f"Common task in domain: {d}",
                    severity=self._severity(count, 0.0),
                )
                for d, count in domain_tool_counts.items()
                if count >= min_tool_calls
            ]

        if self._dry_run:
            # Stub: report first domain as gap
            domain, count = next(iter(domain_tool_counts.items()))
            return [CoverageGap(domain=domain, tool_call_count=count, max_rr=0.2,
                                representative_query="stub query", severity="medium")]

        skill_texts = [p.read_text(encoding="utf-8")[:3000] for p in skill_paths]
        skill_embeddings = [await self._embed(text) for text in skill_texts]

        gaps: List[CoverageGap] = []
        for domain, tool_count in domain_tool_counts.items():
            if tool_count < min_tool_calls:
                continue

            # Representative query for this domain (use first detected query)
            representative = self._representative_query(domain, recent_sessions)
            if not representative:
                continue

            # Check maximum RR across all skills
            query_vec = await self._embed(representative)
            max_rr = max((_cosine(skill_vec, query_vec) for skill_vec in skill_embeddings), default=0.0)
            # Map cosine similarity to [0,1] (it's already in [-1,1] typically positive)
            max_rr_01 = max(0.0, min(1.0, (max_rr + 1.0) / 2.0))

            if max_rr_01 < coverage_threshold:
                gaps.append(CoverageGap(
                    domain=domain,
                    tool_call_count=tool_count,
                    max_rr=round(max_rr_01, 3),
                    representative_query=representative,
                    severity=self._severity(tool_count, max_rr_01),
                ))

        gaps.sort(key=lambda g: {"high": 0, "medium": 1, "low": 2}[g.severity])
        return gaps

    def report(self, gaps: List[CoverageGap]) -> str:
        """Format gaps as a human-readable report string."""
        if not gaps:
            return "No coverage gaps detected."
        lines = [f"Coverage gaps detected ({len(gaps)} domains):"]
        for g in gaps:
            lines.append(
                f"  [{g.severity.upper():6s}] {g.domain:20s}  "
                f"tool_calls={g.tool_call_count:4d}  max_rr={g.max_rr:.3f}"
            )
            lines.append(f"    Sample query: {g.representative_query[:80]}")
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _count_tool_calls_by_domain(
        self,
        sessions: List[List[Dict[str, Any]]],
    ) -> Dict[str, int]:
        """Count tool calls per domain by scanning messages for tool_call content."""
        # Heuristic: classify tool calls by keywords in the user turns preceding them
        domain_keywords: Dict[str, List[str]] = {
            "code_debug":       ["debug", "error", "crash", "fix", "KeyError", "exception", "traceback"],
            "api_integration":  ["API", "endpoint", "request", "response", "webhook", "token", "OAuth"],
            "sys_admin":        ["server", "service", "nginx", "systemd", "cron", "ssh", "permission"],
            "data_analysis":    ["DataFrame", "pandas", "aggregate", "plot", "CSV", "column"],
            "doc_draft":        ["write", "document", "README", "runbook", "ADR"],
            "qa":               ["explain", "difference", "compare", "concept", "pattern"],
        }

        counts: Counter = Counter()
        for messages in sessions:
            for msg in messages:
                if msg.get("role") == "tool" or (msg.get("role") == "assistant" and msg.get("tool_calls")):
                    # Find the closest preceding user message
                    content = self._preceding_user_content(msg, messages)
                    for domain, keywords in domain_keywords.items():
                        if any(kw.lower() in content.lower() for kw in keywords):
                            counts[domain] += 1

        return dict(counts)

    def _preceding_user_content(self, target_msg: Dict, messages: List[Dict]) -> str:
        idx = messages.index(target_msg) if target_msg in messages else -1
        for i in range(idx - 1, -1, -1):
            if messages[i].get("role") == "user":
                return messages[i].get("content") or ""
        return ""

    def _representative_query(self, domain: str, sessions: List[List[Dict]]) -> str:
        """Find a user turn from the given domain in recent sessions."""
        domain_keywords: Dict[str, List[str]] = {
            "code_debug":       ["debug", "error", "fix"],
            "api_integration":  ["API", "endpoint", "request"],
            "sys_admin":        ["server", "service", "ssh"],
            "data_analysis":    ["DataFrame", "pandas", "data"],
            "doc_draft":        ["write", "document"],
            "qa":               ["explain", "difference", "compare"],
        }
        keywords = domain_keywords.get(domain, [domain])
        for messages in sessions:
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content") or ""
                    if any(kw.lower() in content.lower() for kw in keywords):
                        return content[:200]
        return f"How do I work with {domain}?"

    @staticmethod
    def _severity(tool_count: int, max_rr: float) -> str:
        if tool_count >= 20 and max_rr < 0.3:
            return "high"
        if tool_count >= 10 or max_rr < 0.4:
            return "medium"
        return "low"

    async def _embed(self, text: str) -> List[float]:
        import litellm
        response = await litellm.aembedding(model=self._embedding_model, input=[text[:4000]])
        return response.data[0]["embedding"]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(y * y for y in b))
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)
