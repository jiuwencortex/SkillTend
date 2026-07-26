# SkillTend Research — Guide

This is the primary reference for understanding, running, and interpreting the
`skilltend_research/` package. It supersedes `research_plan.md`, which was the
pre-implementation design document and is now historical reference only.

---

## What this research is

`skilltend/` (the production system) encodes every tunable decision — trigger
intervals, memory size limits, lifecycle thresholds, review prompts, model
choices — as hard-coded constants inherited from earlier systems. None of those
constants have been measured against outcomes.

`skilltend_research/` exists to replace each constant with an empirically
grounded recommendation. Every study targets one or more named fields in
`ReviewerConfig` and produces a concrete value to use instead of the current
default.

The complete mapping:

| Study | `ReviewerConfig` fields targeted |
|---|---|
| 01 — Metrics Validation | internal Q metric (weights calibrated for all other studies) |
| 02 — Trigger Policy | `skill_nudge_interval`, `memory_nudge_interval` |
| 03 — Model Selection | `review_model` default |
| 04 — Memory Abstraction | `memory_char_limit`, optional retrieval layer |
| 05 — Lifecycle | `curator_stale_after_days`, `curator_archive_after_days` |
| 06 — Prompt Sensitivity | all three review prompts, conversation context strategy |
| 07 — Skill Interference | `flush_min_turns`, COMBINED vs. sequential mode |
| 08 — Library Dynamics | `curator_llm_consolidation` enable threshold |

---

## Project layout

```
skilltend_research/
│
├── shared/                        Core simulation and evaluation infrastructure
│   ├── task_bank.py               60 evaluation tasks across 6 domains
│   ├── session_simulator.py       Generates synthetic agent sessions
│   ├── skill_quality_scorer.py    Computes Q, RR, ID, TSR, PS metrics
│   └── result_logger.py           Thread-safe JSONL experiment writer
│
├── phase_1_foundations/           Must run first; validates measurement tools
│   └── study_01_skill_quality_metrics/
│
├── phase_2_optimization/          Run in parallel after phase 1
│   ├── study_02_trigger_policy/
│   ├── study_03_review_model_calibration/
│   ├── study_04_memory_abstraction/
│   └── study_05_lifecycle_optimization/
│
├── phase_3_refinement/            Depends on study 03 (model choice must be fixed)
│   └── study_06_prompt_sensitivity/
│
├── phase_4_dynamics/              Run last; depends on phase 1 + study 07 → 08
│   ├── study_07_skill_interference/
│   └── study_08_library_dynamics/
│
├── results/                       All experiment output (gitignored except findings)
└── docs/                          This guide and supporting references
    ├── guide.md                   ← you are here
    ├── environment.md             Setup, API keys, cost estimates
    ├── metrics_reference.md       Precise definitions of every metric
    ├── results_schema.md          JSONL and findings JSON field reference
    └── research_plan.md           Historical: pre-implementation design notes
```

---

## Dependency order

```
Study 01  (Metrics Validation)
    │
    ├──► Study 02  (Trigger Policy)        ─┐
    ├──► Study 03  (Model Selection)        │ run in parallel
    ├──► Study 04  (Memory Abstraction)     │
    ├──► Study 05  (Lifecycle)             ─┘
    │         │
    │         └──► Study 06  (Prompt Sensitivity)   [needs 03 first]
    │
    └──► Study 07  (Skill Interference)
              │
              └──► Study 08  (Library Dynamics)
```

---

## Quick start

```bash
# 1. Set up the environment (see docs/environment.md for full details)
pip install -e ../           # production package
pip install -e ".[dev]"      # research package
export OPENAI_API_KEY=sk-...

# 2. Verify everything works without LLM calls
python -m phase_1_foundations.study_01_skill_quality_metrics.runner --dry-run --rounds 1

# 3. Run Study 01 (must complete before anything else)
python -m phase_1_foundations.study_01_skill_quality_metrics.runner \
    --model openai/gpt-4o-mini \
    --rounds 5 \
    --output results/study_01_metrics.jsonl

# 4. After Study 01 completes, run Phase 2 studies in parallel
parallel python -m {}.runner --model openai/gpt-4o-mini --output-dir results/ ::: \
    phase_2_optimization.study_02_trigger_policy \
    phase_2_optimization.study_03_review_model_calibration \
    phase_2_optimization.study_04_memory_abstraction \
    phase_2_optimization.study_05_lifecycle_optimization

# 5. Then Phase 3 (after study 03 completes)
python -m phase_3_refinement.study_06_prompt_sensitivity.runner \
    --model openai/gpt-4o-mini --output-dir results/

# 6. Then Phase 4
python -m phase_4_dynamics.study_07_skill_interference.runner \
    --model openai/gpt-4o-mini --output-dir results/
python -m phase_4_dynamics.study_08_library_dynamics.runner \
    --model openai/gpt-4o-mini --output-dir results/
```

---

## Phase 1 — Foundations

### Study 01: Skill Quality Metrics Validation

**Purpose:** Validates the four Q sub-metrics (RR, ID, TSR, PS) before any
other study uses them. Also calibrates the linear combination weights
`0.30 / 0.20 / 0.40 / 0.10` using human-labelled ground truth.

**Why first:** All other studies report Q scores. If Q is noisy or uncorrelated
with actual skill utility, every downstream result is meaningless.

**Files:**

| File | Purpose |
|---|---|
| `runner.py` | Measures all four metrics on a set of skill files |
| `label_tool.py` | Interactive CLI for human raters to label skill quality |
| `compute_validity.py` | Computes Spearman ρ, Fleiss κ, CV for each metric |
| `calibrate_weights.py` | Fits Q weights via ridge regression with LOO-CV |

**Running:**

```bash
# Step 1 — measure metrics on your skill library
python -m phase_1_foundations.study_01_skill_quality_metrics.runner \
    --skills-dir /path/to/your/skills \
    --model openai/gpt-4o-mini \
    --rounds 5 \
    --output results/study_01_metrics.jsonl

# Step 2 — human labelling (run once per rater)
python -m phase_1_foundations.study_01_skill_quality_metrics.label_tool \
    --skills-dir /path/to/your/skills \
    --rater-id yourname \
    --output results/study_01_labels.jsonl

# Step 3 — compute validity and inter-rater agreement
python -m phase_1_foundations.study_01_skill_quality_metrics.compute_validity \
    --labels results/study_01_labels.jsonl \
    --metrics results/study_01_metrics.jsonl

# Step 4 — calibrate Q weights
python -m phase_1_foundations.study_01_skill_quality_metrics.calibrate_weights \
    --labels results/study_01_labels.jsonl \
    --metrics results/study_01_metrics.jsonl \
    --output results/study_01_calibrated_weights.json
```

**Gate:** Do not proceed to Phase 2 until inter-rater κ > 0.65 (printed by
`compute_validity`). If κ ≤ 0.65, the raters disagreed too much — re-label
with clearer guidelines.

**Output:** `results/study_01_calibrated_weights.json` with the new Q weights
and LOO-CV R² score. Update `shared/skill_quality_scorer.py` `DEFAULT_WEIGHTS`
before running Phase 2.

---

## Phase 2 — Optimization

All four studies in this phase are independent and can run in parallel.

---

### Study 02: Trigger Policy Optimization

**Purpose:** Answers "how often should background review fire?" Current defaults
`skill_nudge_interval = 10` and `memory_nudge_interval = 10` were never measured.

**What it does:**
- **Phase A** — sweeps N × M combinations (N, M ∈ {3, 5, 8, 10, 15, 20, 30, ∞})
  across 7,680 simulated sessions, recording Q score and review token cost.
- **Phase B** — benchmarks `AdaptiveTrigger` (fires when conversation embedding
  distance exceeds threshold θ) against the best fixed policy from Phase A.
- **Phase C** — plots the Q-vs-cost Pareto frontier and identifies the knee point.

**Run:**

```bash
python -m phase_2_optimization.study_02_trigger_policy.runner \
    --model openai/gpt-4o-mini \
    --sessions-per-config 20 \
    --output-dir results/
```

**Cheap exploratory run** (use `--sessions-per-config 5` — 15× faster, still directional).

**Output:** `results/study_02_findings.json` with recommended intervals for
three operating modes: cost-sensitive, balanced, quality-first. Also a Pareto
plot at `results/study_02_pareto.png`.

---

### Study 03: Review Model Calibration

**Purpose:** Finds the minimum-cost LLM that achieves ≥ 95% of frontier-model
review quality. Current default is the parent agent's model (potentially expensive).

**What it does:** Benchmarks 8 models across 4 tiers (frontier, mid, small,
open) on 40 sessions. Computes Q score and cost-per-review for each. Fits a
quality-vs-cost curve and finds the 90/95/99% thresholds.

**Run:**

```bash
python -m phase_2_optimization.study_03_review_model_calibration.runner \
    --sessions 40 \
    --output-dir results/
```

**Output:** `results/study_03_findings.json` with a model-selection table
(budget → recommended model). **This result must be used before running
Study 06**, because the review model must be fixed before testing prompt
sensitivity — otherwise model quality and prompt quality are confounded.

---

### Study 04: Memory Abstraction Strategies

**Purpose:** Determines whether the current inject-all strategy (entire
MEMORY.md verbatim into every prompt) is optimal, and finds the right
`memory_char_limit`.

**What it does:** Compares four strategies:
- **Inject-all** (baseline) — current behavior
- **Top-K retrieval** — embed user query, inject only K most similar entries
- **Hierarchical** — LLM consolidates episodic entries into semantic summaries
- **Recency-weighted hybrid** — recent entries verbatim + extractive summary of older

Each strategy is tested at 5 char limits: 500, 1000, 2200, 4000, 8000.
Metric: TSR (task success with memory vs. without) vs. context tokens injected.

**Run:**

```bash
python -m phase_2_optimization.study_04_memory_abstraction.runner \
    --model openai/gpt-4o-mini \
    --sessions 50 \
    --output-dir results/
```

**Output:** `results/study_04_findings.json` with recommended strategy,
K value, and `memory_char_limit`. Includes saturation point (char limit beyond
which TSR stops improving).

---

### Study 05: Lifecycle Threshold Optimization

**Purpose:** Empirically tunes `curator_stale_after_days` (currently 30) and
`curator_archive_after_days` (currently 90). Also trains a feature-based
predictor that can replace the fixed-threshold FSM.

**What it does:**
- **Phase A** — simulates 180 days of skill usage (Poisson-like rates) and runs
  a Kaplan-Meier-style survival analysis to find when skills go stale.
- **Phase B** — sweeps all (stale, archive) threshold pairs, computing false-positive
  rate (useful skills archived) and false-negative rate (stale skills kept).
  Minimises F = 0.7 × FP + 0.3 × FN (asymmetric: losing useful skills hurts more).
- **Phase C** — trains an XGBoost classifier on 8 features (recency, use count,
  patch count, etc.) to predict usage in the next 30 days. If AUC > 0.75, the
  model should replace the fixed threshold.

**Run:**

```bash
python -m phase_2_optimization.study_05_lifecycle_optimization.runner \
    --simulated-days 180 \
    --output-dir results/
```

**Output:** `results/study_05_findings.json` with optimal threshold pair and
AUC of the predictor. `results/study_05_lifecycle_predictor.pkl` — the trained
predictor, loadable by the production Curator.

---

## Phase 3 — Refinement

### Study 06: Prompt Sensitivity & Ablation

**Prerequisite:** Study 03 must complete first. Set `--model` to the model
recommended by Study 03 to avoid confounding model quality with prompt quality.

**Purpose:** Determines which elements of the review prompts are load-bearing,
finds the optimal conversation context window, and tests few-shot injection.

**What it does:**
- **Phase A** — ablates 7 structural elements of the review prompt (task framing,
  instruction list, format constraints, negative constraints, examples) one at a
  time. Measures Q drop for each removal.
- **Phase B** — tests 4 context window strategies: full conversation, last 5 turns,
  last 10 turns, last 20 turns. Plots Q vs. context tokens.
- **Phase D** — injects adversarial noise (off-topic turns, typos) and measures
  Q variance for each prompt variant. Most robust = lowest variance.

**Run:**

```bash
python -m phase_3_refinement.study_06_prompt_sensitivity.runner \
    --model openai/gpt-4o-mini \   # use model from Study 03 findings
    --sessions 30 \
    --output-dir results/
```

**Output:** `results/study_06_findings.json` with:
- `critical_elements` — must keep in the prompt (Q drop ≥ 0.02 if removed)
- `dispensable_elements` — can simplify (Q drop < 0.005)
- `adopt_few_shot` — true if adding examples improves Q by > 3%
- `recommended_context_strategy` — best truncation strategy

---

## Phase 4 — Dynamics

### Study 07: Skill Interference & Within-Session Dynamics

**Purpose:** Measures what happens when background review modifies a skill
during the session where that skill is actively being used. Also determines
the "warm-up period" before reviews stabilise — directly calibrating `flush_min_turns`.

**What it does:**
- **Phase A** — constructs sessions where a skill is used at turn T, reviewed
  at T + Δ (Δ ∈ {0, 1, 3, 5, 10}), and used again at T + 2Δ. Compares
  post-patch TSR against a no-review control. Reports Cohen's d and direction.
- **Phase B** — tracks Patch Stability (PS) vs. turn number within a single
  session. Finds the "warm-up turn" — first turn where PS stabilises above 0.85.

**Run:**

```bash
python -m phase_4_dynamics.study_07_skill_interference.runner \
    --model openai/gpt-4o-mini \
    --output-dir results/
```

**Output:** `results/study_07_findings.json` with:
- `flush_min_turns` — recommended value to replace the current default (6)
- `safe_delta` — minimum turns between skill use and next review fire
- `use_sequential_over_combined` — true if COMBINED mode shows harmful interference

---

### Study 08: Skill Library Dynamics & Coverage

**Prerequisite:** Study 07 must complete first (safe-update protocol must be
established before simulating live skill modifications at scale).

**Purpose:** Measures how the skill library evolves over hundreds of sessions,
when it saturates, when redundancy becomes a problem, and at what library size
LLM consolidation has positive ROI.

**What it does:**
- **Phase A+B** — simulates 200 sequential sessions. Every 10 sessions records:
  Library Coverage (LC), mean pairwise skill similarity (redundancy), Gini
  coefficient of use_count distribution, and library size.
- **Phase C** — runs Curator Phase 2 (LLM merge/split consolidation) at library
  sizes of 10, 25, 50, 100. Measures ΔLC and suggestion count.
- **Phase D** — runs `CoverageGapDetector` on recent sessions to find task domains
  with high tool-call frequency but no covering skill.

**Run:**

```bash
python -m phase_4_dynamics.study_08_library_dynamics.runner \
    --model openai/gpt-4o-mini \
    --sessions 200 \
    --output-dir results/
```

**Output:**
- `results/study_08_findings.json` — LC saturation session, minimum library
  size for enabling `curator_llm_consolidation`
- `results/study_08_coverage_gaps.json` — list of skill gaps by domain and severity
- `results/study_08_growth.png` — four-panel growth curve plot

---

## How to read results

Every study produces two outputs:

**`results/study_NN_*.jsonl`** — incremental experiment log. Safe to tail
during a run. Each line is one trial record. See `docs/results_schema.md`
for full field definitions.

**`results/study_NN_findings.json`** — extracted recommendation, written
by the study's `analyze.py` after the run completes. Top-level keys directly
correspond to `ReviewerConfig` field names. This is what you update the
production config with.

To re-run analysis without re-running the full experiment:

```bash
# Re-analyze any study from its JSONL
python -m phase_2_optimization.study_02_trigger_policy.analyze \
    --input results/study_02_sweep.jsonl \
    --plot  results/study_02_pareto.png
```

---

## Shared infrastructure

### `shared/task_bank.py`

60 evaluation tasks across 6 domains (10 each): `code_debug`, `doc_draft`,
`data_analysis`, `api_integration`, `sys_admin`, `qa`. Each task has a user
query, solution steps, ground truth answer, difficulty label, and tool names.

Access via the `TASK_BANK` singleton:

```python
from shared.task_bank import TASK_BANK

tasks = TASK_BANK.by_domain("code_debug")      # 10 tasks
easy  = TASK_BANK.by_difficulty("easy")        # 20 tasks across domains
all_  = TASK_BANK.all()                        # all 60
```

### `shared/session_simulator.py`

Generates synthetic agent sessions deterministically from `TASK_BANK`, then
replays them through the real `run_background_review()` pipeline (so review
execution is real, but user interaction is reproducible).

```python
from shared.session_simulator import SessionSimulator
from shared.task_bank import TASK_BANK

sim = SessionSimulator(TASK_BANK, dry_run=False)

messages    = sim.generate_session("code_debug_001", seed=42)
skills_root = sim.build_skill_library(domains=["code_debug"])
result      = await sim.replay_review(messages, config, model="openai/gpt-4o-mini")
sim.cleanup_skill_library(skills_root)
```

Pass `dry_run=True` to bypass all LLM calls (returns stub results instantly).

### `shared/skill_quality_scorer.py`

Computes the four Q sub-metrics and the combined Q score. Can be used
independently of any study runner:

```python
from shared.skill_quality_scorer import SkillQualityScorer
from shared.task_bank import TASK_BANK

scorer = SkillQualityScorer(dry_run=False, judge_model="openai/gpt-4o-mini")
scores = await scorer.score(skill_text, tasks=TASK_BANK.by_domain("code_debug")[:5])
print(scores.q, scores.rr, scores.tsr, scores.ps)
```

After Study 01 completes, update the default weights:

```python
from shared.skill_quality_scorer import QWeights
# paste calibrated_weights from study_01_calibrated_weights.json
weights = QWeights(rr=0.28, id_score=0.18, tsr=0.44, ps=0.10)
scorer  = SkillQualityScorer(weights=weights)
```

### `shared/result_logger.py`

Thread-safe JSONL writer used by all runners. To read results from a
finished run:

```python
from shared.result_logger import ResultLogger
from pathlib import Path

records = ResultLogger.read_metrics(Path("results/study_02_sweep.jsonl"))
# returns List[Dict] of metrics dicts, with failed records filtered out
```

---

## Adding a new study

1. Create a folder under the appropriate phase (e.g. `phase_2_optimization/study_09_xxx/`).
2. Add `__init__.py` with a one-paragraph docstring describing the study.
3. Create `runner.py` with a `main()` that accepts `--model`, `--output-dir`,
   and `--dry-run` flags.
4. Create `analyze.py` with a `main()` that reads from `results/study_09_*.jsonl`
   and writes `results/study_09_findings.json`.
5. The `findings.json` must have top-level keys matching `ReviewerConfig` field names.
6. Update this guide with a section for the new study.

---

## Dry-run mode

Every runner accepts `--dry-run`. In this mode:
- `SessionSimulator` skips `replay_review()` and returns a stub `ReviewResult`.
- `SkillQualityScorer` returns Q = 0.5, RR = 0.5, etc. for all inputs.
- No LLM API calls are made. No result files are written.
- All CLI argument parsing and code paths are exercised.

Use dry-run to verify your environment is correctly set up before committing
to a full experiment run.

```bash
# Verify all 8 studies parse and import correctly
for study in \
    "phase_1_foundations.study_01_skill_quality_metrics" \
    "phase_2_optimization.study_02_trigger_policy" \
    "phase_2_optimization.study_03_review_model_calibration" \
    "phase_2_optimization.study_04_memory_abstraction" \
    "phase_2_optimization.study_05_lifecycle_optimization" \
    "phase_3_refinement.study_06_prompt_sensitivity" \
    "phase_4_dynamics.study_07_skill_interference" \
    "phase_4_dynamics.study_08_library_dynamics"; do
    python -m ${study}.runner --dry-run && echo "OK: $study"
done
```
