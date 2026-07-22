# SkillTend

**Online skill maintenance for Jiuwen** — Hermes-style background review and lifecycle curation.

SkillTend keeps a Jiuwen agent's skill library and memory up to date while the agent is running,
without blocking the user. It is one half of the skill-management system; the offline evolutionary
optimizer lives in the companion **SkillForge** project.

---

## Components

### Reviewer (`Reviewer`)

A `DeepAgentRail` that fires after every N tool-calls or M user turns.
It snapshots the conversation, calls an LLM with three focused tools
(`skill_write`, `skill_patch`, `memory_write`), and applies the changes
directly to the SKILL.md files and memory store — all in the background,
serialised so at most one review runs at a time.

### Curator (`Curator`)

A scheduled background daemon that transitions skills between lifecycle states
(`ACTIVE → STALE → ARCHIVED`) based on usage age. Runs opportunistically
during idle time between invokes; never blocks the user.

---

## Quick Start

```python
from skilltend import Reviewer, ReviewerConfig

config = ReviewerConfig(
    skills_root="~/.jiuwen/skills",
    memory_nudge_interval=5,  # review memory every 5 user turns
    skill_nudge_interval=10,  # review skills every 10 tool calls
    flush_min_turns=6,  # don't fire until 6 turns into the session
)
rail = Reviewer(config=config)

# Wire into your agent adapter:
await agent.register_rail(rail)
```

For resumed/gateway sessions, hydrate the counters from history so the
first message doesn't immediately trigger a review:

```python
rail.hydrate_from_history(prior_messages)
```

---

## Install

```bash
pip install -e .
# openjiuwen (agent-core) must be installed separately:
pip install -e ../agent-core
```

---

## Project Layout

```
skilltend/
  reviewer.py             # Reviewer — the hook-based trigger
  curator/                # Curator — lifecycle state machine
  config.py               # ReviewerConfig dataclass
  types/                  # ReviewMode, ReviewTrigger, ReviewResult, ReviewAction
  pipeline/               # 5-stage review execution pipeline
    provenance.py         # ContextVar write-origin tracking
    runner.py             # Orchestrator: build → prompt → LLM → dispatch → assemble
    stages/
      stage01_conversation_builder/
      stage02_prompt_selector/
      stage03_llm_caller/
      stage04_tool_call_dispatcher/
      stage05_result_assembler/
  stores/
    skill/                # SKILL.md CRUD, UsageSidecar, lifecycle API
    memory/               # MemoryStore read/write/snapshot/drift
examples/
  online_01_review_basic.py
  online_02_review_custom_config.py
  online_03_session_resume_hydration.py
  online_04_memory_context_and_skills_prompt.py
  online_05_skill_provenance_and_lifecycle.py
  online_06_memory_snapshot_and_drift.py
```

---

## Relationship to SkillForge

| | SkillTend (this project) | SkillForge |
|---|---|---|
| Track | Online — runs during live sessions | Offline — runs between sessions |
| Mechanism | LLM-driven review of recent conversation | GEPA evolutionary optimizer |
| Writes | SKILL.md patches + memory entries | New / rewritten SKILL.md files |
| Shared | `~/.jiuwen/skills/` (SKILL.md files) | same |
| State aware | Yes — reads `.usage.json`, respects ACTIVE/STALE/ARCHIVED | No |
