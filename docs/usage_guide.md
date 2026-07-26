# SkillTend Usage Guide

A developer's guide to integrating, configuring, and operating SkillTend in
production. For internal architecture and pipeline details see `architecture.md`.

---

## What SkillTend does

SkillTend runs inside a live agent session and continuously improves the agent's
skill library and memory without interrupting the user. Two components handle
this:

- **Reviewer** — a `DeepAgentRail` that fires after every N tool calls or M
  user turns. It takes a conversation snapshot, calls a review LLM, and applies
  skill patches and memory writes in the background.
- **Curator** — a scheduled daemon that transitions skills through lifecycle
  states (`ACTIVE → STALE → ARCHIVED`) based on usage age. Runs during idle
  time between agent invocations.

Both are zero-latency from the user's perspective: they run as `asyncio`
background tasks and never block a response.

---

## Installation

```bash
pip install -e .
# openjiuwen (agent-core) must be installed separately:
pip install -e ../agent-core
```

---

## Basic usage — wiring the rail

```python
from skilltend import Reviewer, ReviewerConfig

config = ReviewerConfig(
    skill_nudge_interval=10,   # review skills every 10 tool calls
    memory_nudge_interval=10,  # review memory every 10 user turns
    review_model="gpt-4o-mini",
)
rail = Reviewer(config=config)

# Register on your agent adapter (interface_deep.py / interface_code.py)
await agent.register_rail(rail)
```

The rail wires three hooks automatically:

| Hook | What it does |
|---|---|
| `after_model_call` | Increments the user-turn counter |
| `after_tool_call` | Increments the tool-call counter; resets if a skill/memory tool fired |
| `after_invoke` | Checks thresholds; spawns the background review task |

See `examples/online_01_review_basic.py` for a runnable demo.

---

## Configuration

All options live in `ReviewerConfig`. Full reference:

**Review triggering**

| Field | Default | Description |
|---|---|---|
| `enabled` | `True` | Master switch. Set `False` to disable all reviews. |
| `skill_nudge_interval` | `10` | Tool-call count between skill reviews. |
| `memory_nudge_interval` | `10` | User-turn count between memory reviews. |
| `flush_min_turns` | `6` | Minimum user turns in session before any review fires. Prevents noisy reviews at session start. |
| `protected_skill_names` | `[]` | Skill names the review LLM cannot edit or delete. Use for bundled or system skills. |

**LLM**

| Field | Default | Description |
|---|---|---|
| `review_model` | `None` | Model for the review LLM. Falls back to the parent agent's model if `None`. |
| `review_timeout_seconds` | `60.0` | Hard timeout on the LLM call. |
| `review_max_iterations` | `10` | Maximum tool calls in one review cycle. |

**Storage**

| Field | Default | Description |
|---|---|---|
| `skills_root` | `~/.jiuwen/skills` | Root directory for SKILL.md files. |
| `memory_root` | `~/.jiuwen/memories` | Root directory for MEMORY.md and USER.md. |

**Curator**

| Field | Default | Description |
|---|---|---|
| `curator_enabled` | `True` | Enable the Curator lifecycle daemon. |
| `curator_min_idle_seconds` | `7200` | Agent must have been idle for this long before Curator runs (default 2 hours). |
| `curator_min_interval_days` | `7` | Minimum wall-clock days between Curator runs. |
| `stale_days` | `30` | Days since last use before ACTIVE → STALE. |
| `archive_days` | `90` | Days since last use before STALE → ARCHIVED. |

---

## Custom config and protected skills

```python
from skilltend import Reviewer, ReviewerConfig

config = ReviewerConfig(
    skill_nudge_interval=3,       # fire quickly (useful in testing)
    memory_nudge_interval=3,
    review_timeout_seconds=30.0,
    review_max_iterations=8,
    protected_skill_names=[       # these skills will never be auto-edited
        "bundled-core",
        "system-instructions",
    ],
)
rail = Reviewer(config=config)
```

Protected skill names are matched exactly. The review LLM's `skill_patch` and
`skill_write` tool calls are silently rejected for any skill in this list.

See `examples/online_02_review_custom_config.py` for a full demo including
calling `run_background_review()` directly for testing.

---

## Directly calling the review pipeline

You can invoke the review pipeline without a live agent — useful for integration
tests, one-off review runs, or studying its behavior:

```python
from skilltend import run_background_review, ReviewMode, ReviewTrigger

trigger = ReviewTrigger(
    mode=ReviewMode.COMBINED,    # review both memory AND skills
    user_turn_count=2,
    tool_iter_count=0,
    session_id="test-session-001",
)

result = await run_background_review(
    messages_snapshot=messages,  # List[Dict] in OpenAI message format
    trigger=trigger,
    config=config,
    model="gpt-4o-mini",
    session_id="test-session-001",
)

print(result.summary_line)
print(result.duration_seconds)
for action in result.actions:
    print(f"[{action.action_type}] {action.target_name} — {action.summary}")
```

**Review modes:**

| Mode | When it fires | What it reviews |
|---|---|---|
| `SKILLS_ONLY` | `tool_iter_count ≥ skill_nudge_interval` | Skill library only |
| `MEMORY_ONLY` | `user_turn_count ≥ memory_nudge_interval` | Memory and user profile only |
| `COMBINED` | Both thresholds reached simultaneously | Both |

**`ReviewResult` fields:**

| Field | Type | Description |
|---|---|---|
| `summary_line` | `str` | One-line human-readable summary of what changed |
| `duration_seconds` | `float` | Wall-clock time for the full review |
| `actions` | `List[ReviewAction]` | Each write/patch that was applied |
| `error` | `str \| None` | Error message if the review LLM call failed |
| `trigger` | `ReviewTrigger` | The trigger that initiated this review |

**`ReviewAction` fields:**

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | `"skill_write"`, `"skill_patch"`, or `"memory_write"` |
| `target_name` | `str` | Skill name or memory section name |
| `summary` | `str` | Short description of the change |

---

## Gateway and resumed sessions — counter hydration

On stateless gateway platforms (Telegram, Discord, web chat), the agent creates
a fresh instance for every incoming message. A fresh `Reviewer` starts both
counters at zero, so if the session already has 9 turns with a threshold of 10,
the review would never fire.

Fix: call `hydrate_from_history()` immediately after creating the rail:

```python
rail = Reviewer(config=config)
rail.hydrate_from_history(prior_messages)
# prior_messages is the existing List[Dict] conversation history

# Verify the restored counter
counts = rail.pending_counts()
# {"user_turns_since_review": N, "tool_iters_since_review": M}
```

`hydrate_from_history()` uses modulo arithmetic to restore the correct position
in the review cycle without immediately triggering a review. Calling it twice
is safe — the second call is a no-op once the session is active.

```python
# Example: 9 prior user turns, memory_nudge_interval=10
# → user_turns_since_review = 9 % 10 = 9  (one turn away from firing)

# Example: 25 prior turns
# → user_turns_since_review = 25 % 10 = 5  (mid-cycle, no immediate fire)
```

See `examples/online_03_session_resume_hydration.py` for all four scenarios
including the idempotency guarantee.

---

## Injecting memory and skills into the system prompt

SkillTend can build the memory and skill-index blocks that are injected into
the agent's system prompt at session start. This is the mechanism that gives
the agent access to its accumulated knowledge.

### Memory context block

```python
from skilltend import MemoryStore

store = MemoryStore(memory_root=Path("~/.jiuwen/memories"))

# Build the <memory-context>…</memory-context> block for the system prompt
block = store.build_memory_context_block()
# Returns "" if memory is empty — safe to inject unconditionally

# Check how much character space memory is consuming
counts = store.char_counts()
# {"memory": 843, "user": 210}
```

The block is fenced in `<memory-context>` tags and combines entries from
`MEMORY.md` (agent observations) and `USER.md` (user profile facts).

### Skill index block

```python
from skilltend import build_skills_system_prompt

prompt = await build_skills_system_prompt(skills_root=Path("~/.jiuwen/skills"))
# Returns "" if no skills exist — safe to inject unconditionally
```

This builds a compact index of all ACTIVE skills, including their name and
description from the YAML frontmatter of each SKILL.md file. Inject it into
the STABLE tier of the agent system prompt.

### Typical system prompt assembly

```python
memory_block = store.build_memory_context_block()
skills_block = await build_skills_system_prompt(skills_root)

system_prompt = base_instructions
if memory_block:
    system_prompt += "\n\n" + memory_block
if skills_block:
    system_prompt += "\n\n" + skills_block
```

See `examples/online_04_memory_context_and_skills_prompt.py` for a complete
runnable demo.

---

## Skill lifecycle — provenance, pinning, archiving

### Provenance

Every skill records whether it was created by the user or by the agent's
background review. Skills created by the user are protected from automatic
lifecycle transitions by the Curator.

```python
from skilltend import background_review_context, get_write_origin
from skilltend.stores.skill import skill_create, skill_get_usage

# Outside the context: writes are attributed to "user"
print(get_write_origin())  # "foreground"
await skill_create("my-skill", content, skills_root)
usage = await skill_get_usage("my-skill", skills_root)
print(usage.created_by)  # "user"

# Inside the context: writes are attributed to "agent"
async with background_review_context():
    await skill_create("agent-skill", content, skills_root)
usage = await skill_get_usage("agent-skill", skills_root)
print(usage.created_by)  # "agent"
```

The `background_review_context()` sets a `ContextVar` that is read at write
time. It is restored to `"foreground"` when the block exits.

### Usage telemetry

Each skill's `.usage.json` sidecar tracks:

```python
usage = await skill_get_usage("my-skill", skills_root)
print(usage.state)           # "ACTIVE" / "STALE" / "ARCHIVED"
print(usage.created_by)      # "user" or "agent"
print(usage.pinned)          # True / False
print(usage.view_count)      # increments on skill_read()
print(usage.patch_count)     # increments on skill_patch()
print(usage.last_used_at)    # ISO timestamp
print(usage.use_count)       # total invocations
```

### Pinning

A pinned skill is exempt from all Curator lifecycle transitions and cannot be
deleted or archived by any code path:

```python
from skilltend.stores.skill import skill_set_pinned

await skill_set_pinned("my-skill", skills_root, pinned=True)

# These will both return ok=False while the skill is pinned
ok, msg = await skill_delete("my-skill", skills_root)
ok, msg = await skill_archive("my-skill", skills_root)
```

### Archiving and restoring

```python
from skilltend.stores.skill import skill_archive, skill_restore, skill_list

# Archive a skill (ACTIVE → ARCHIVED)
ok, msg = await skill_archive("old-skill", skills_root)

# Archived skills are hidden from the normal listing
active = await skill_list(skills_root)                     # excludes archived
all_   = await skill_list(skills_root, include_archived=True)  # includes archived

# Restore it back to ACTIVE
ok, msg = await skill_restore("old-skill", skills_root)
```

### Deleting with consolidation intent

When deleting a skill because its content was absorbed into another, record
the intent for audit purposes:

```python
from skilltend.stores.skill import skill_delete

ok, msg = await skill_delete(
    "old-skill", skills_root,
    absorbed_into="new-combined-skill",  # optional, for traceability
)
```

See `examples/online_05_skill_provenance_and_lifecycle.py` for all seven
provenance and lifecycle operations in sequence.

---

## Memory snapshots and drift detection

### Frozen snapshot for prefix-cache stability

When the agent system prompt is assembled at session start, the memory block
should stay byte-identical across all turns within the session. This maximises
prompt-prefix cache hits on the LLM provider side.

SkillTend handles this with a frozen snapshot:

```python
from skilltend.stores.memory import MemoryStore

store = MemoryStore(memory_root=memory_root)
store.load_from_disk()  # take the snapshot here, at session start

# Use get_snapshot_block() for the system prompt — this never changes
# within the session even if add() is called later
system_block = store.get_snapshot_block()

# Use build_memory_context_block() if you want the live, up-to-date block
live_block = store.build_memory_context_block()
```

`load_from_disk()` freezes the current on-disk state. After that:
- `add()` / `replace()` / `remove()` write to disk but do **not** update the snapshot.
- `get_snapshot_block()` always returns the state at the last `load_from_disk()`.
- A second call to `load_from_disk()` refreshes the snapshot to the latest disk state.

### Drift detection

Detects whether a memory file was modified externally (e.g. by another process
or a manual edit) since it was last loaded:

```python
drifted = store.detect_drift("memory")  # checks MEMORY.md
# Returns True if mtime has changed since load_from_disk()
# Creates a .bak.<timestamp> backup of the drifted file
# Updates the stored mtime so subsequent calls return False
```

Useful in multi-process or gateway deployments where the memory file may be
written by a background process while the agent is running.

See `examples/online_06_memory_snapshot_and_drift.py` for the full snapshot
lifecycle including idempotent drift detection.

---

## Inspecting the rail state at runtime

```python
# Current counter positions
counts = rail.pending_counts()
# {"user_turns_since_review": 7, "tool_iters_since_review": 3}

# Result of the most recent review (None if no review has fired yet)
result = rail.last_review_result()
if result:
    print(result.summary_line)
    print(f"{len(result.actions)} action(s) applied")
```

---

## Examples index

| File | What it demonstrates |
|---|---|
| `online_01_review_basic.py` | Wiring the rail, `pending_counts()`, `last_review_result()` |
| `online_02_review_custom_config.py` | Custom thresholds, protected skills, calling `run_background_review()` directly, inspecting `ReviewResult` |
| `online_03_session_resume_hydration.py` | `hydrate_from_history()` for gateway/stateless deployments |
| `online_04_memory_context_and_skills_prompt.py` | `build_memory_context_block()`, `build_skills_system_prompt()`, `char_counts()` |
| `online_05_skill_provenance_and_lifecycle.py` | `background_review_context()`, `skill_set_pinned()`, `skill_archive()`, `skill_restore()`, `skill_delete(absorbed_into=...)` |
| `online_06_memory_snapshot_and_drift.py` | `load_from_disk()`, `get_snapshot_block()` for prefix-cache stability, `detect_drift()` |
