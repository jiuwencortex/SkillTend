# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Synthetic agent session generator.

SessionSimulator converts Task objects into realistic OpenAI-format message
lists and drives the real `run_background_review()` pipeline against them.
All LLM calls in the review pipeline are real unless `dry_run=True`.

Conversation format (OpenAI messages API):
  {"role": "user",      "content": "..."}
  {"role": "assistant", "content": "...", "tool_calls": [...]}   (optional)
  {"role": "tool",      "content": "...", "tool_call_id": "..."}  (when tool called)
  {"role": "assistant", "content": "..."}
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.task_bank import Task, TaskBank
    from skilltend.config import ReviewerConfig
    from skilltend.types import ReviewResult

# Domain-level assistant opening lines (before tool use)
_DOMAIN_OPENINGS: Dict[str, str] = {
    "code_debug": "Let me analyse the problem and identify the root cause.",
    "doc_draft": "I'll draft this document for you now.",
    "data_analysis": "Let me walk through the analysis step by step.",
    "api_integration": "I'll show you how to implement this correctly.",
    "sys_admin": "Let me diagnose this systematically.",
    "qa": "Great question. Here is a thorough explanation.",
}

_DOMAIN_TOOL_RESULTS: Dict[str, str] = {
    "code_debug":       "Tool executed successfully. Code updated.",
    "doc_draft":        "File written successfully.",
    "data_analysis":    "Script executed. Output captured.",
    "api_integration":  "Code file written. Ready for review.",
    "sys_admin":        "Command ran on remote host.",
    "qa":               "",  # Q&A tasks rarely call tools
}

# Initial SKILL.md content templates (seeded into the temp library)
_SKILL_TEMPLATES: Dict[str, str] = {
    "code_debug": """\
---
title: Python Debugging Techniques
description: Common patterns for debugging Python code defects
tags: [python, debugging, code]
---

# Python Debugging Techniques

When debugging Python code, start by understanding the error type and traceback.

## Key strategies

- Read the full traceback from bottom to top — the last line is the immediate error.
- Use `print()` or `logging.debug()` to inspect intermediate state.
- Isolate the failure in the smallest possible reproduction case.
- Use `pdb` or `breakpoint()` for interactive inspection.

## Common patterns

- **KeyError**: key not present in dict — use `.get()` with a default.
- **AttributeError**: attribute not present on object — check type before access.
- **TypeError**: wrong type passed to function — add type checking.
""",
    "doc_draft": """\
---
title: Technical Writing Best Practices
description: Structure and style guide for technical documentation
tags: [writing, documentation, markdown]
---

# Technical Writing Best Practices

Good technical documentation is accurate, concise, and example-driven.

## Structure

1. Start with a one-sentence summary of what the thing does.
2. Show a minimal working example as early as possible.
3. Follow with reference details (parameters, options, caveats).

## Style

- Write in active voice and present tense.
- Use short sentences. Break long explanations into numbered steps.
- Every code block should be copy-pasteable and correct.
""",
    "data_analysis": """\
---
title: Pandas Data Manipulation Patterns
description: Common pandas idioms for data cleaning and aggregation
tags: [pandas, python, data]
---

# Pandas Data Manipulation Patterns

## Loading and inspection

```python
df = pd.read_csv('data.csv')
df.info()          # dtypes and null counts
df.describe()      # numeric stats
```

## Filtering

```python
# Boolean indexing with multiple conditions
filtered = df[(df['col_a'] > 0) & (df['col_b'] == 'active')]
```

## Aggregation

```python
df.groupby('category').agg(total=('value', 'sum'), count=('value', 'count'))
```
""",
    "api_integration": """\
---
title: HTTP Client Patterns with httpx
description: Best practices for async HTTP API calls using httpx
tags: [httpx, api, async, python]
---

# HTTP Client Patterns with httpx

## Basic async client

```python
import httpx

async with httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {token}"},
    timeout=30.0,
) as client:
    response = await client.get("/users")
    response.raise_for_status()
    return response.json()
```

## Retry on transient errors

Retry on 429/503 with exponential backoff. Read `Retry-After` header on 429.
""",
    "sys_admin": """\
---
title: Linux Service Diagnostics
description: Step-by-step approach to diagnosing Linux service issues
tags: [linux, systemd, diagnostics, bash]
---

# Linux Service Diagnostics

## Check service status

```bash
systemctl status myservice
journalctl -u myservice -n 100 --no-pager
```

## Common causes of service failure

- Missing environment variables → EnvironmentFile not set or file not readable.
- Permission denied → check file ownership and mode with `ls -la`.
- Port already in use → `ss -tlnp | grep :8000`.
- Dependency service not started → check After= and Requires= in unit file.
""",
    "qa": """\
---
title: Software Architecture Concepts
description: Key concepts in distributed systems and software design
tags: [architecture, design, concepts]
---

# Software Architecture Concepts

## Key trade-offs

Every architectural decision involves trade-offs. Common dimensions:
- Consistency vs. Availability (CAP theorem)
- Latency vs. Throughput
- Simplicity vs. Flexibility
- Cost vs. Reliability

## Communication patterns

- **Synchronous RPC**: simple, but tight coupling.
- **Async messaging**: decoupled, but harder to debug.
- **Event sourcing**: complete audit trail, but complex queries.
""",
}


class SessionSimulator:
    """Generates and replays synthetic agent sessions for SkillTend research.

    Usage::

        from shared.task_bank import TASK_BANK
        from shared.session_simulator import SessionSimulator

        sim = SessionSimulator(TASK_BANK)

        # Generate a synthetic conversation
        messages = sim.generate_session("code_debug_01", seed=42)

        # Build a skill library
        skills_root = sim.build_skill_library(["code_debug", "sys_admin"])

        # Replay the review pipeline (real LLM call unless dry_run=True)
        result = await sim.replay_review(messages, config, model="gpt-4o-mini")
    """

    def __init__(self, task_bank: "TaskBank", dry_run: bool = False) -> None:
        self._bank = task_bank
        self._dry_run = dry_run

    # ── Conversation generation ───────────────────────────────────────────────

    def generate_session(
        self,
        task_id: str,
        seed: int = 0,
        extra_filler_turns: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return a list of OpenAI-format message dicts for the given task.

        Args:
            task_id:            ID of the task to simulate.
            seed:               Random seed for reproducibility (affects filler turns).
            extra_filler_turns: Additional off-topic turns injected before the main task.
                                Used to simulate longer sessions in trigger-policy experiments.
        """
        rng = random.Random(seed)
        task = self._bank.get(task_id)
        messages: List[Dict[str, Any]] = []

        # Optional filler turns (simulate context the agent already handled)
        if extra_filler_turns > 0:
            messages.extend(self._filler_turns(task.domain, extra_filler_turns, rng))

        # Main task conversation
        messages.append({"role": "user", "content": task.user_query})

        opening = _DOMAIN_OPENINGS.get(task.domain, "Let me help you with that.")
        tool_result_template = _DOMAIN_TOOL_RESULTS.get(task.domain, "")

        for i, step in enumerate(task.solution_steps):
            is_last = i == len(task.solution_steps) - 1
            has_tool = bool(task.tool_names) and not is_last

            if has_tool:
                tool_name = rng.choice(task.tool_names)
                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                # Assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": f"{opening}\n\n{step}",
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps({"input": step[:120]}),
                        },
                    }],
                })
                # Tool result
                messages.append({
                    "role": "tool",
                    "content": tool_result_template or f"Completed: {tool_name}",
                    "tool_call_id": tool_call_id,
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": step,
                })

        return messages

    def generate_multi_task_session(
        self,
        task_ids: List[str],
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Concatenate conversations for multiple tasks into one session.

        Useful for simulating long sessions where the agent handles several tasks.
        """
        messages: List[Dict[str, Any]] = []
        for i, task_id in enumerate(task_ids):
            messages.extend(self.generate_session(task_id, seed=seed + i))
        return messages

    # ── Skill library construction ────────────────────────────────────────────

    def build_skill_library(
        self,
        domains: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
        seed: int = 0,
        base_dir: Optional[Path] = None,
    ) -> Path:
        """Create a temporary skill library on disk and return the skills_root path.

        Either `domains` or `task_ids` may be specified.  If both are None, all
        domain seed skills are created (one per domain).

        The caller is responsible for cleaning up the directory when done.
        Use `shutil.rmtree(skills_root)` or `build_skill_library_ctx()` for
        automatic cleanup.

        Args:
            domains:  List of domain names; one seed skill is created per domain.
            task_ids: List of task IDs; skills are derived from those tasks.
            seed:     Not currently used; reserved for future randomisation.
            base_dir: Parent directory for the skills_root. If None, a system
                      temp dir is used.
        """
        parent = base_dir or Path(tempfile.mkdtemp(prefix="skilltend_research_"))
        skills_root = parent / "skills"
        skills_root.mkdir(parents=True, exist_ok=True)

        if domains is not None:
            for domain in domains:
                self._write_domain_seed_skill(domain, skills_root)
        elif task_ids is not None:
            for task_id in task_ids:
                task = self._bank.get(task_id)
                self._write_task_skill(task, skills_root)
        else:
            # Default: one seed skill per domain
            for domain in self._bank.domains:
                self._write_domain_seed_skill(domain, skills_root)

        return skills_root

    def cleanup_skill_library(self, skills_root: Path) -> None:
        """Remove a skill library created by build_skill_library()."""
        parent = skills_root.parent
        if parent.exists() and "skilltend_research_" in parent.name:
            shutil.rmtree(parent, ignore_errors=True)
        elif skills_root.exists():
            shutil.rmtree(skills_root, ignore_errors=True)

    # ── Review replay ─────────────────────────────────────────────────────────

    async def replay_review(
        self,
        messages: List[Dict[str, Any]],
        config: "ReviewerConfig",
        model: str,
        session_id: Optional[str] = None,
    ) -> "ReviewResult":
        """Run the real SkillTend review pipeline on the given messages.

        In dry-run mode the review pipeline is bypassed and a stub ReviewResult
        is returned immediately (useful for CI and structural testing).
        """
        if self._dry_run:
            return self._stub_result(messages)

        from skilltend.pipeline.runner import run_background_review
        from skilltend.types import ReviewTrigger, ReviewMode

        trigger = ReviewTrigger(
            mode=ReviewMode.COMBINED,
            tool_iter_count=len([m for m in messages if m.get("role") == "tool"]),
            user_turn_count=len([m for m in messages if m.get("role") == "user"]),
        )

        return await run_background_review(
            messages_snapshot=messages,
            trigger=trigger,
            config=config,
            model=model,
            session_id=session_id or uuid.uuid4().hex,
        )

    async def replay_reviews_batch(
        self,
        sessions: List[List[Dict[str, Any]]],
        config: "ReviewerConfig",
        model: str,
        concurrency: int = 4,
    ) -> List["ReviewResult"]:
        """Run review on a batch of sessions with bounded concurrency.

        Args:
            sessions:    List of message lists (one per session).
            config:      ReviewerConfig to use for all replays.
            model:       LLM model name.
            concurrency: Max simultaneous review tasks.
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(msgs: List[Dict[str, Any]]) -> "ReviewResult":
            async with semaphore:
                return await self.replay_review(msgs, config, model)

        return list(await asyncio.gather(*[_bounded(s) for s in sessions]))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _filler_turns(
        self,
        domain: str,
        n: int,
        rng: random.Random,
    ) -> List[Dict[str, Any]]:
        """Generate n pairs of (user, assistant) turns unrelated to the main task."""
        fillers = [
            ("Can you give me a quick recap of what we did earlier?",
             "Sure — earlier we worked through the setup steps and verified everything was in order."),
            ("What was the command we used to check the logs?",
             "We used `journalctl -u myservice -n 100` to inspect the service logs."),
            ("Just to confirm — is this approach compatible with Python 3.10?",
             "Yes, this works with Python 3.10 and above."),
            ("What's the best way to test this locally before pushing?",
             "Run the full test suite with `pytest -x` and verify the integration test passes."),
            ("Can you remind me of the error message we saw earlier?",
             "The error was: `ValueError: expected float, got str` on line 47."),
        ]
        msgs: List[Dict[str, Any]] = []
        for pair in rng.choices(fillers, k=n):
            msgs.append({"role": "user", "content": pair[0]})
            msgs.append({"role": "assistant", "content": pair[1]})
        return msgs

    def _write_domain_seed_skill(self, domain: str, skills_root: Path) -> None:
        template = _SKILL_TEMPLATES.get(domain)
        if not template:
            return
        skill_name = f"{domain}_fundamentals"
        skill_dir = skills_root / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(template, encoding="utf-8")
        self._write_usage_sidecar(skill_dir, skill_name)

    def _write_task_skill(self, task: "Task", skills_root: Path) -> None:
        """Create a SKILL.md for a specific task in the skill library."""
        skill_name = task.id.replace("_", "-")
        skill_dir = skills_root / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        content = (
            f"---\n"
            f"title: {task.title}\n"
            f"description: {task.ground_truth[:120]}\n"
            f"tags: [{task.domain}, {', '.join(task.skill_keywords[:3])}]\n"
            f"---\n\n"
            f"# {task.title}\n\n"
            f"{task.ground_truth}\n\n"
            f"## Key concepts\n\n"
            + "\n".join(f"- {kw}" for kw in task.skill_keywords)
        )
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        self._write_usage_sidecar(skill_dir, skill_name)

    @staticmethod
    def _write_usage_sidecar(skill_dir: Path, skill_name: str) -> None:
        """Write a minimal .usage.json sidecar so the SkillStore can find the skill."""
        import time
        from datetime import datetime, timezone

        usage = {
            "state": "ACTIVE",
            "created_by": "user",
            "pinned": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity_at": datetime.now(timezone.utc).isoformat(),
            "last_used_at": datetime.now(timezone.utc).isoformat(),
            "use_count": 1,
            "patch_count": 0,
            "view_count": 1,
        }
        (skill_dir / ".usage.json").write_text(
            json.dumps(usage, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _stub_result(messages: List[Dict[str, Any]]) -> "ReviewResult":
        """Return a dummy ReviewResult for dry-run mode."""
        # Import here to avoid hard dependency when dry_run is not used
        from skilltend.types import ReviewResult, ReviewTrigger, ReviewMode

        trigger = ReviewTrigger(
            mode=ReviewMode.COMBINED,
            tool_iter_count=0,
            user_turn_count=len([m for m in messages if m.get("role") == "user"]),
        )
        return ReviewResult(
            trigger=trigger,
            actions=[],
            error="[dry-run] LLM call skipped",
            duration_seconds=0.0,
            summary_line="dry-run: no review performed",
        )
