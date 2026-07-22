# SkillTend Architecture

## Overview

SkillTend implements the **online track** of Jiuwen's skill-management system.
It runs inside a live agent session and continuously improves skills and memory
without blocking the user.

The design mirrors Hermes's background review mechanism, adapted to Jiuwen's
`DeepAgentRail` hook system.

---

## Two Components

```
┌─────────────────────────────────────────────────────────┐
│                  Reviewer                    │
│  (priority 70 — runs after primary rails finish)        │
│                                                         │
│  ┌──────────────┐          ┌───────────────────────┐   │
│  │   Reviewer   │          │       Curator          │   │
│  │              │          │                        │   │
│  │ fires every  │          │ fires opportunistically │   │
│  │ N tool calls │          │ during idle time        │   │
│  │ M user turns │          │ (idle ≥ 2h, interval   │   │
│  └──────┬───────┘          │  ≥ 7 days)             │   │
│         │                  └───────────┬────────────┘   │
└─────────┼──────────────────────────────┼────────────────┘
          │ asyncio.create_task          │ asyncio.create_task
          ▼                              ▼
   run_background_review()       Curator.maybe_run()
   (pipeline/runner.py)          (curator/curator.py)
```

### Reviewer — `Reviewer`

Implemented as a `DeepAgentRail` with three hooks:

| Hook | What it does |
|---|---|
| `after_model_call` | Increments `_user_turn_count` on non-tool assistant messages |
| `after_tool_call` | Increments `_tool_iter_count`; resets counters if a skill/memory tool fired |
| `after_invoke` | Checks thresholds; spawns background review task; spawns curator task |

**Counter reset logic** (mirrors Hermes tool_executor.py):
- `memory_write` / `memory` tool called → `_user_turn_count = 0`
- `skill_write` / `skill_patch` / `skill_manage` / `skill_create` called → `_tool_iter_count = 0`

**Guards** (mirrors Hermes conversation_loop.py):
- Skips if invoke was interrupted
- Skips until `flush_min_turns` user turns have been seen (default 6)
- Serialises: waits up to 5 s for previous review to finish before spawning a new one

### Curator — `Curator`

Scheduled daemon that manages skill lifecycle states:

```
ACTIVE  ──(unused > stale_days)──►  STALE
STALE   ──(unused > archive_days)── ARCHIVED
ARCHIVED ──(skill_restore())──────► ACTIVE
```

Gating conditions before a curator run:
1. Curator is enabled in config
2. Idle time since last invoke ≥ `min_idle_seconds` (default 2h)
3. Wall-clock interval since last curator run ≥ `min_interval_days` (default 7d)

---

## Review Pipeline

`run_background_review()` in `pipeline/runner.py` orchestrates 5 sequential stages:

```
messages_snapshot
       │
       ▼
┌─────────────────────────────┐
│  Stage 1                    │
│  conversation_builder       │  Resolves skills_root, creates MemoryStore,
│                             │  serialises messages to plain text
└────────────┬────────────────┘
             │ (skills_root, memory_store, conversation_text)
             ▼
┌─────────────────────────────┐
│  Stage 2                    │
│  prompt_selector            │  Picks MEMORY_ONLY / SKILLS_ONLY / COMBINED
│                             │  review prompt + system prompt
└────────────┬────────────────┘
             │ (review_prompt, system_prompt)
             ▼
┌─────────────────────────────┐
│  Stage 3                    │
│  llm_caller                 │  Calls litellm with tool schemas:
│                             │  skill_write, skill_patch, memory_write
│                             │  (timeout from config)
└────────────┬────────────────┘
             │ (tool_calls, error)
             ▼
┌─────────────────────────────┐
│  Stage 4                    │
│  tool_call_dispatcher       │  Dispatches each tool call to
│                             │  skill_store / memory_store
└────────────┬────────────────┘
             │ (actions: List[ReviewAction])
             ▼
┌─────────────────────────────┐
│  Stage 5                    │
│  result_assembler           │  Computes duration, builds summary_line,
│                             │  returns ReviewResult
└─────────────────────────────┘
```

---

## Review Modes

| Mode | Trigger | Prompt |
|---|---|---|
| `MEMORY_ONLY` | `_user_turn_count ≥ memory_nudge_interval` | `MEMORY_REVIEW_PROMPT` |
| `SKILLS_ONLY` | `_tool_iter_count ≥ skill_nudge_interval` | `SKILL_REVIEW_PROMPT` |
| `COMBINED` | Both thresholds reached simultaneously | `COMBINED_REVIEW_PROMPT` |

---

## Provenance Tracking

`pipeline/provenance.py` exposes a `ContextVar`-based write-origin mechanism.
When the review pipeline is running, `background_review_context()` sets the
origin to `"background_review"`.

In `skill_creator.py`, this origin is read at write time:

```python
origin = get_write_origin()
created_by = "agent" if origin == "background_review" else "user"
```

Skills created by the agent are curator-eligible; skills created by the user
are protected from automatic lifecycle transitions.

---

## Skill Store

Each skill lives at `<skills_root>/<category>/<name>/SKILL.md` (or
`<skills_root>/<name>/SKILL.md` without a category).

Alongside each SKILL.md is a `.usage.json` sidecar (`UsageSidecar`) that
records:

- `state`: `ACTIVE` / `STALE` / `ARCHIVED`
- `created_by`: `"user"` or `"agent"`
- `pinned`: bool — pinned skills are exempt from curation
- `last_used_at`, `use_count`, `patch_count`, `created_at`

Concurrent access is serialised per-skill with per-name `asyncio.Lock` objects.
Writes use `atomic_writter.py` (write-to-temp + rename) to prevent torn files.

---

## Wiring into an Agent

`Reviewer` is a standard `DeepAgentRail` and must be registered
with the agent adapter (not with `DeepAgent` itself):

```python
# In interface_deep.py or interface_code.py
from skilltend import Reviewer, ReviewerConfig

config = ReviewerConfig(skills_root=..., ...)
rail = Reviewer(config=config)
await agent.register_rail(rail)
```

For gateway platforms that create a fresh agent per message, call
`rail.hydrate_from_history(prior_messages)` after registration to restore
the correct review cadence without firing immediately.

---

## Configuration (`ReviewerConfig`)

| Field | Default | Description |
|---|---|---|
| `enabled` | `True` | Master switch |
| `skills_root` | `~/.jiuwen/skills` | Root dir for SKILL.md files |
| `memory_nudge_interval` | `5` | User turns between memory reviews |
| `skill_nudge_interval` | `10` | Tool iterations between skill reviews |
| `flush_min_turns` | `6` | Minimum turns before any review fires |
| `review_model` | `None` | LLM for reviews (falls back to agent model) |
| `review_timeout_seconds` | `60` | LLM call timeout |
| `curator_enabled` | `True` | Enable the Curator |
| `curator_min_idle_seconds` | `7200` | Min idle time (2h) before curator runs |
| `curator_min_interval_days` | `7` | Min wall-clock days between curator runs |
| `stale_days` | `30` | Days unused before ACTIVE → STALE |
| `archive_days` | `90` | Days unused before STALE → ARCHIVED |
