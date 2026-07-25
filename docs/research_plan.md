# SkillTend Research Plan

**Companion folder:** `skilltend_research/` (sibling of `skilltend/`)

This document defines the research agenda for the empirical track of SkillTend.
The production system (`skilltend/`) encodes many design decisions as constants:
trigger intervals, memory size limits, lifecycle thresholds, review prompts, model
choices. None of these constants have been measured against outcomes. This research
agenda exists to replace those constants with findings.

---

## Guiding Principle

Every study in `skilltend_research/` must produce a **concrete, actionable
recommendation** for a parameter or architectural choice in the production system.
Studies that do not connect back to a specific `ReviewerConfig` field or design
decision are out of scope.

---

## Folder Layout

```
skilltend_research/
├── shared/
│   ├── session_simulator.py       # Synthetic session generator
│   ├── skill_quality_scorer.py    # Canonical quality metrics
│   ├── task_bank.py               # Curated evaluation task sets
│   └── result_logger.py           # JSONL experiment output
│
├── study_01_skill_quality_metrics/
├── study_02_trigger_policy/
├── study_03_review_model_calibration/
├── study_04_memory_abstraction/
├── study_05_lifecycle_optimization/
├── study_06_prompt_sensitivity/
├── study_07_skill_interference/
└── study_08_library_dynamics/
```

Studies must be run sequentially in order: Study 01 defines the metrics that
all subsequent studies depend on.

---

## Shared Infrastructure (`shared/`)

### `session_simulator.py`

Generates synthetic agent sessions without requiring a live agent.

A session is a list of `(role, content, tool_calls)` tuples drawn from
`task_bank.py`. The simulator replays these against the real `run_background_review()`
pipeline so that review execution is real, but user interaction is deterministic.

Key methods:
- `generate_session(task_id, seed)` → `List[Message]`
- `replay_review(messages, config, model)` → `ReviewResult`
- `build_skill_library(task_ids, seed)` → `Path` (temp skills_root)

### `skill_quality_scorer.py`

Defines the four canonical metrics used across all studies:

| Metric | What it measures | How computed |
|---|---|---|
| **Retrieval Relevance (RR)** | Does the skill contain the information a downstream agent would need to solve the task it was written for? | Cosine similarity between skill embedding and task query embedding; averaged over a fixed task bank. |
| **Information Density (ID)** | Bits of useful content per character. | Compress skill with zstd; ID = uncompressed / compressed ratio. High ratio = high redundancy = low density. |
| **Task Success Rate (TSR)** | Does an agent equipped with this skill complete the task better than without it? | Agent solves N tasks with and without skill injected. TSR = fraction of tasks where skill version wins. Requires LLM judge. |
| **Patch Stability (PS)** | Does the skill converge, or does each review cycle keep changing it? | Track character-level diff between consecutive review outputs. PS = 1 - mean(normalized_edit_distance). |

All four metrics return floats in [0, 1]. Combined score: `Q = 0.4*TSR + 0.3*RR + 0.2*ID + 0.1*PS`.

### `task_bank.py`

A curated set of 60 tasks across 6 domains (10 tasks per domain):
code debugging, document drafting, data analysis, API integration,
system administration, and open-domain Q&A. Each task has:
- A natural language description
- A reference conversation (pre-recorded tool-call trace)
- A ground-truth answer for LLM judging
- Domain tag + difficulty label (easy/medium/hard)

### `result_logger.py`

Writes JSONL. Every experiment appends:
```json
{"study": "02", "run_id": "...", "config": {...}, "metrics": {...}, "timestamp": "..."}
```

---

## Study 01 — Skill Quality Metrics Validation

**Research question:** Are the four proposed metrics (RR, ID, TSR, PS)
valid proxies for real skill utility? Do they agree with each other?
Are they stable across repeated measurements?

**Why this comes first:** Every downstream study uses these metrics.
If they are noisy or uncorrelated with actual utility, the other results
are meaningless.

### Method

1. **Construct a ground-truth dataset.** Manually label 30 skills as
   `high / medium / low` quality based on human expert judgment. Use 3
   independent raters; compute inter-rater agreement (Fleiss κ). Only
   proceed if κ > 0.65.

2. **Measure all four metrics on the 30 skills.**

3. **Validity check.** Correlate each metric with the human labels
   (Spearman ρ). A metric passes if ρ > 0.5. Failing metrics are either
   reformulated or dropped.

4. **Stability check.** Measure each metric 5 times on the same skill
   (different random seeds in the LLM judge). Report coefficient of
   variation (CV). Accept if CV < 0.08 for RR, ID, PS and < 0.15 for TSR.

5. **Redundancy check.** Compute pairwise correlations between metrics.
   If two metrics correlate > 0.92, drop the less interpretable one.

6. **Calibrate the combined Q formula.** Fit weights using the human-labeled
   dataset (ridge regression; leave-one-out CV). Replace the default
   `0.4 / 0.3 / 0.2 / 0.1` with the learned weights.

### Deliverable

`study_01_skill_quality_metrics/validated_scorer.py` — a drop-in replacement
for the prototype in `shared/skill_quality_scorer.py`, with calibrated weights
and documented validity bounds.

---

## Study 02 — Trigger Policy Optimization

**Research question:** What values of `skill_nudge_interval` (N) and
`memory_nudge_interval` (M) maximize skill quality per LLM token spent?

**Current state:** N = M = 10 (copied from Hermes; no empirical basis).

### Method

**Phase A — Sweep.**

Generate 20 synthetic sessions per configuration. Sweep:
- N ∈ {3, 5, 8, 10, 15, 20, 30, ∞}
- M ∈ {3, 5, 8, 10, 15, 20, 30, ∞}
- 3 difficulty levels (easy / medium / hard) from `task_bank.py`
- 2 session lengths (short: 20 turns, long: 60 turns)

Total: 8 × 8 × 3 × 2 × 20 = 7,680 simulated sessions.
For each session, record: final skill Q score, total review LLM tokens used,
wall-clock time added (asyncio measured).

**Phase B — Efficiency frontier.**

Plot Q vs. tokens for all (N, M) pairs. Fit a Pareto frontier. Identify
the knee point (maximum Q improvement per additional 1,000 tokens).

**Phase C — Adaptive trigger baseline.**

Implement `AdaptiveTrigger`: instead of fixed N/M, trigger when the running
cosine distance between consecutive conversation embeddings exceeds a threshold
θ (i.e., when the conversation has changed meaningfully since the last review).
Compare adaptive trigger to the best fixed (N, M) on the Pareto frontier.

### Deliverable

Recommended `skill_nudge_interval` and `memory_nudge_interval` values for
three operating modes: cost-sensitive, balanced, quality-first.
If AdaptiveTrigger beats fixed by > 5% Q at equal token budget, provide a
reference implementation for the production system.

---

## Study 03 — Review Model Calibration

**Research question:** What is the minimum-cost model that achieves
≥ 95% of the review quality of the best available model?

**Current state:** `review_model` defaults to the parent agent's model,
which may be an expensive frontier model. There is no evidence that a
cheaper model cannot do background review just as well.

### Method

**Phase A — Model grid.**

Test 8 models across 3 tiers:
- Frontier: `gpt-4.1`, `claude-opus-4`
- Mid: `gpt-4.1-mini`, `claude-sonnet-4`
- Small: `gpt-4.1-nano`, `claude-haiku-3-5`
- Open: `llama-3.3-70b`, `qwen2.5-72b`

For each model: run the full `run_background_review()` pipeline on 40 sessions
drawn from `task_bank.py` (10 per difficulty × 4 task types). Measure Q score
and per-review cost (input + output tokens × published price).

**Phase B — Quality regression.**

Normalize Q scores relative to the frontier model (= 1.0). Plot the
quality-cost curve. Fit a power-law regression to estimate the cost required
to reach 90% / 95% / 99% of frontier quality.

**Phase C — Skill type interaction.**

Test whether the optimal model tier varies by review mode: does
`SKILLS_ONLY` review require a stronger model than `MEMORY_ONLY`? Run
a 2-way ANOVA (model tier × review mode).

**Phase D — Error characterization.**

For sessions where cheaper models produce lower Q than the frontier, categorize
the failure: wrong tool call, correct tool wrong arguments, hallucinated skill
name, missed opportunity to patch, over-patching. Build a failure taxonomy.

### Deliverable

A model-selection table: for each operating budget ($ per 1M reviews), the
recommended model. The failure taxonomy feeds directly into improved review
prompts (Study 06).

---

## Study 04 — Memory Abstraction Strategies

**Research question:** Is the current "inject-all" memory strategy (entire
MEMORY.md prepended to system prompt) optimal, or do retrieval-based or
hierarchical strategies yield better downstream task success?

**Current state:** `memory_char_limit` = 2200 chars. All memory is injected
verbatim. There is no retrieval or abstraction layer.

### Method

**Strategy A — Inject-all (baseline):** current implementation.

**Strategy B — Top-K semantic retrieval:** at each turn, embed the user's
message and retrieve the K most similar memory entries (cosine similarity
over chunk embeddings). Inject only those K entries. K ∈ {3, 5, 10}.

**Strategy C — Hierarchical consolidation:** episodic memories (raw entries
from individual sessions) are periodically consolidated into semantic summaries
(recurring patterns) by a small LLM. Inject semantic summaries by default;
fall back to episodic entries on low similarity.

**Strategy D — Recency-weighted hybrid:** inject all memories from the past
7 days verbatim, plus a condensed summary of older memories. Condense using
extractive summarization (no LLM cost).

**Evaluation protocol:**
- Generate 50 sessions per strategy across 5 task types.
- Primary metric: TSR (task success with memory injected vs. without).
- Secondary metric: context tokens consumed (lower = better for latency/cost).
- Plot TSR vs. context tokens to identify the efficiency frontier.
- Measure memory update latency (how quickly new facts appear in injected context).

**Ablations:**
- What happens at `memory_char_limit` ∈ {500, 1000, 2200, 4000, 8000}?
  Find the saturation point where more characters yield no TSR improvement.

### Deliverable

Recommended memory strategy and `memory_char_limit` value. If retrieval-based
beats inject-all by > 5% TSR, provide `MemoryRetriever` as an optional
module with a clean interface for the production pipeline.

---

## Study 05 — Lifecycle Threshold Optimization

**Research question:** Are 30-day stale and 90-day archive cutoffs the right
thresholds? Should lifecycle transitions depend only on recency, or also on
usage frequency and skill content features?

**Current state:** Purely recency-based FSM. All skills treated identically.
Thresholds are guesses (copied from Hermes).

### Method

**Phase A — Usage pattern analysis.**

Using the `task_bank.py` simulator, generate skill libraries and simulate 180
days of agent operation (time-compressed: 1 simulated day = 100 sessions with
deterministic task draws). Record `use_count` and `last_activity_at` for each
skill. Fit a survival model (Kaplan-Meier) to estimate: how long after last use
does a skill's TSR drop below 0.5 (i.e., when does it become effectively stale)?

**Phase B — Threshold grid search.**

Sweep `curator_stale_after_days` ∈ {7, 14, 21, 30, 45, 60} and
`curator_archive_after_days` ∈ {30, 60, 90, 120, 180}. For each combination:
measure the fraction of "still-useful" skills incorrectly archived (false positives)
and the fraction of "genuinely stale" skills kept ACTIVE (false negatives).
Minimize F = α × FP_rate + (1-α) × FN_rate with α = 0.7 (prefer not to lose
useful skills).

**Phase C — Feature-enriched lifecycle model.**

Train a lightweight gradient-boosted classifier (XGBoost, ≤ 50 features) to
predict whether a skill will be used in the next 30 days, given:
- Days since last use
- Total use count
- Patch count (edited frequently = likely maintained)
- Skill text length
- Information density score (from Study 01)
- Skill domain category (extracted from frontmatter)

Evaluate: AUC-ROC on held-out simulated data. If AUC > 0.75, this model can
replace the fixed-threshold FSM in Phase 1 of the Curator.

**Phase D — Reactivation patterns.**

Currently, STALE → ACTIVE transition fires only when a skill is used again.
Study whether "related skill accessed" (cosine sim > 0.85) is a better
reactivation signal than direct use.

### Deliverable

Empirically grounded `curator_stale_after_days` and `curator_archive_after_days`
values. If the feature-enriched model achieves AUC > 0.75, a reference
`LifecyclePredictor` class for optional use in the Curator.

---

## Study 06 — Prompt Sensitivity & Ablation

**Research question:** How sensitive is review quality to the specific
wording of the review prompt? Which elements are load-bearing, and which
can be simplified without quality loss?

**Current state:** The production review prompts are fixed strings inherited
from Hermes. No ablation has been performed.

### Method

**Phase A — Ablation.**

Identify the structural elements in the current `SKILL_REVIEW_PROMPT`,
`MEMORY_REVIEW_PROMPT`, and `COMBINED_REVIEW_PROMPT`:
1. Task framing ("You are a skill maintenance agent…")
2. Instruction list (what to look for)
3. Output format constraints (tool call format)
4. Negative constraints ("Do not…")
5. Examples (few-shot, if any)

For each element, create an ablated variant (element removed or replaced with
minimal text). Run 30 sessions per variant (10 per review mode). Measure Q
drop vs. full prompt.

**Phase B — Few-shot injection.**

Add 2 high-quality (skill, conversation, output) examples to the review
prompt as few-shot demonstrations. Measure Q improvement vs. zero-shot.
Test whether examples from the same domain outperform cross-domain examples.

**Phase C — Conversation context window.**

Currently the full conversation is serialized and passed to the review LLM.
Test truncation strategies:
- Full conversation (baseline)
- Last N turns only: N ∈ {5, 10, 20}
- TF-IDF extractive summary of conversation
- Semantic chunking: retain only turns containing novel named entities or tool outputs

Measure Q score vs. context tokens for each strategy.

**Phase D — Prompt robustness.**

Inject adversarial noise into the conversation (typos, off-topic turns,
contradictory instructions) and measure Q degradation per prompt variant.
The most robust prompt variant is the one with lowest Q variance under noise.

### Deliverable

Revised production prompts with ablation-backed justification for every
element. Recommended conversation context strategy for the
`stage01_conversation_builder`. If few-shot improves Q by > 3%, a curated
example library for inclusion in Stage 2.

---

## Study 07 — Skill Interference & Within-Session Dynamics

**Research question:** When background review modifies a skill during a
session where that skill is actively being used, does the modification
help or hurt the remainder of that session?

**Why this matters:** The current system allows background review to patch a
skill at any point in a session. If the agent has already loaded (or cached)
the pre-patch version of the skill, it may operate on stale information for
the rest of the session. Alternatively, the patch may be beneficial.
This is entirely unknown.

### Method

**Phase A — Interference detection.**

Construct sessions where:
- The agent uses a specific skill at turn T.
- A background review fires at turn T + Δ (for Δ ∈ {0, 1, 3, 5, 10}).
- The review patches the skill.
- The agent uses the same skill again at turn T + 2Δ.

Measure: does post-patch task success (TSR for turns after T + Δ) increase
or decrease relative to a control session (no review fired)?

Run 40 sessions per Δ value, 3 task types. Report the interference effect
size (Cohen's d) and direction (positive = helpful, negative = harmful).

**Phase B — Stability vs. drift.**

Track the Patch Stability (PS) metric within a single session as a function
of turn number. Is there a "warm-up period" (early reviews are noisy, late
reviews are stable)? Identify the turn threshold after which PS stabilizes.
This directly informs the `flush_min_turns` parameter.

**Phase C — Concurrent skill and memory update.**

Test what happens when background review fires a `skill_patch` and a
`memory_write` in the same review cycle (COMBINED mode). Does doing both
simultaneously degrade quality vs. doing them in separate cycles?

### Deliverable

Recommendation on `flush_min_turns` (based on Phase B warm-up curve).
Recommendation on whether COMBINED review mode should be replaced by
sequential MEMORY_ONLY → SKILLS_ONLY reviews in separate cycles.
A documented safe-update protocol: minimum Δ between skill write and
next agent read of that skill.

---

## Study 08 — Skill Library Dynamics & Coverage

**Research question:** Does the skill library grow toward coverage of the
agent's task distribution, or does it accumulate redundant and overlapping
skills? When does the LLM consolidation pass (Phase 2 of the Curator)
actually help?

**Current state:** Phase 2 consolidation is disabled by default. There is no
measurement of library coverage or redundancy.

### Method

**Phase A — Coverage metric definition.**

Define **Library Coverage (LC)**: given the `task_bank.py` task distribution
P(task), and a skill library S, LC = P(task is solvable using at least one
skill in S with RR ≥ 0.6). Measure LC as a function of library size
(number of skills).

**Phase B — Growth curve analysis.**

Simulate 200 sessions in sequence. After each session, record:
- Library size (active skill count)
- LC score
- Mean pairwise cosine similarity between all skill embeddings (redundancy proxy)
- Gini coefficient of `use_count` distribution (inequality of skill usage)

Plot these four quantities vs. session number. Identify:
- The session at which LC saturates (marginal coverage gain < 0.5% per session)
- The session at which mean pairwise similarity begins rising (redundancy accumulation)

**Phase C — Consolidation effectiveness.**

Run Phase 2 consolidation (LLM overlap detection) at library sizes of
10, 25, 50, 100 skills. Measure:
- Precision of consolidation suggestions (fraction of suggested merges that
  actually reduce redundancy when executed)
- Recall (fraction of redundant pairs correctly identified)
- Q score before and after executing suggested consolidations

Determine the minimum library size at which Phase 2 has positive ROI (quality
improvement > cost of LLM consolidation call).

**Phase D — Category coverage gaps.**

Implement a `CoverageGapDetector`: given the last 20 sessions, find task types
where the agent repeatedly called tools but never created a skill. These are
"skill gaps." Report gap frequency by task domain.

### Deliverable

Empirically grounded recommendation on when to enable `curator_llm_consolidation`
(minimum library size). A `CoverageGapDetector` module for optional integration
into the Curator. Updated LC metric for monitoring library health in production.

---

## Execution Order & Dependencies

```
Study 01 (Metrics Validation)
    │
    ├──► Study 02 (Trigger Policy)
    ├──► Study 03 (Review Model)       ──► Study 06 (Prompt Sensitivity)
    ├──► Study 04 (Memory Abstraction)
    ├──► Study 05 (Lifecycle)
    └──► Study 07 (Interference)       ──► Study 08 (Library Dynamics)
```

Studies 02, 03, 04, 05, 07 can run in parallel after Study 01 completes.
Study 06 depends on Study 03 (model selection must be fixed before prompt
sensitivity testing, to avoid confounding). Study 08 depends on Study 07
(the safe-update protocol must be established before library dynamics
simulation uses live skill modifications).

---

## What Each Study Returns to Production

| Study | Parameter / Design Decision Affected |
|---|---|
| 01 | Shared Q metric used internally; no direct production change |
| 02 | `skill_nudge_interval`, `memory_nudge_interval`; optional `AdaptiveTrigger` |
| 03 | `review_model` default; documentation on cost tiers |
| 04 | `memory_char_limit`; optional retrieval layer in Stage 1 |
| 05 | `curator_stale_after_days`, `curator_archive_after_days`; optional `LifecyclePredictor` |
| 06 | All three review prompts; conversation truncation strategy in Stage 1 |
| 07 | `flush_min_turns`; COMBINED vs. sequential review mode recommendation |
| 08 | `curator_llm_consolidation` enable threshold; `CoverageGapDetector` |

---

## Non-Goals

The following are explicitly out of scope for `skilltend_research/`:

- Evaluating the parent agent's general task performance (SkillTend is not responsible for the agent).
- Studying SkillForge (offline evolutionary optimizer) — that is a separate research track.
- Any study that does not connect to a production parameter or design decision.
- Live user studies (all evaluation uses the synthetic `task_bank.py` task set).
