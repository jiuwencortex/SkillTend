# Results Schema

Every study runner writes output to `results/` in two formats:

1. **JSONL experiment log** — one JSON object per line, written incrementally
   during the run. Safe to tail in real time.
2. **Findings JSON** — a single JSON object written by `analyze.py` after the
   run completes. Contains the actionable recommendation extracted from the log.

---

## Common JSONL envelope

Every line written by `ResultLogger` has this outer structure:

```json
{
  "study":     "02",
  "run_group": "sweep_phase_A",
  "run_id":    "a3f82c",
  "config":    { ... },
  "metrics":   { ... },
  "meta":      { ... },
  "timestamp": "2025-08-01T14:22:03.441Z",
  "_failed":   false
}
```

| Field | Type | Description |
|---|---|---|
| `study` | string | Study number as two-digit string, e.g. `"02"` |
| `run_group` | string | Logical grouping within a study (e.g. `"phase_A"`, `"adaptive"`) |
| `run_id` | string | Short hex ID unique within the file |
| `config` | object | The configuration that produced this result (see per-study sections) |
| `metrics` | object | Measured outcomes (see per-study sections) |
| `meta` | object | Auxiliary information (seed, session length, domain, etc.) |
| `timestamp` | string | ISO-8601 UTC timestamp |
| `_failed` | boolean | `true` if the run threw an exception; `metrics` may be incomplete |

Failed records are skipped by all `analyze.py` scripts.

---

## Study 01 — Skill Quality Metrics

**Output file:** `results/study_01_metrics.jsonl`

```json
{
  "config": {
    "skill_name": "python_debugging_101",
    "round": 3,
    "model": "openai/gpt-4o-mini",
    "embedding_model": "openai/text-embedding-3-small"
  },
  "metrics": {
    "rr":  0.712,
    "id":  0.638,
    "tsr": 0.700,
    "ps":  0.881,
    "q":   0.703
  },
  "meta": {
    "skill_char_len": 1240,
    "n_task_queries": 10,
    "tsr_trials_per_task": 3
  }
}
```

**Label file:** `results/study_01_labels.jsonl`
Written by `label_tool.py`, one record per human rating:

```json
{
  "rater_id":   "r1",
  "skill_name": "python_debugging_101",
  "label":      2,
  "justification": "Clear structure but misses edge cases.",
  "timestamp":  "2025-08-01T10:00:00Z"
}
```

Labels: `1` = low quality, `2` = medium, `3` = high.

---

## Study 02 — Trigger Policy

**Output file:** `results/study_02_trigger.jsonl`

Phase A (grid sweep):

```json
{
  "config": {
    "phase": "A_sweep",
    "skill_nudge_interval": 10,
    "memory_nudge_interval": 5,
    "session_length": "long",
    "difficulty": "medium"
  },
  "metrics": {
    "q":             0.651,
    "review_tokens": 4200,
    "review_count":  6
  },
  "meta": {
    "session_id": "s042",
    "seed": 42
  }
}
```

Phase B (adaptive trigger):

```json
{
  "config": {
    "phase": "B_adaptive",
    "distance_threshold": 0.25,
    "min_turns_between_reviews": 3
  },
  "metrics": {
    "q":             0.673,
    "review_tokens": 3800,
    "review_count":  4,
    "trigger_distances": [0.18, 0.31, 0.27, 0.29]
  }
}
```

**Findings file:** `results/findings/study_02_findings.json`

```json
{
  "cost_sensitive":  { "skill_nudge_interval": 20, "memory_nudge_interval": 20, "q": 0.61 },
  "balanced":        { "skill_nudge_interval": 10, "memory_nudge_interval": 10, "q": 0.65 },
  "quality_first":   { "skill_nudge_interval":  5, "memory_nudge_interval":  5, "q": 0.68 },
  "adaptive_trigger_recommendation": true,
  "adaptive_trigger_q_gain": 0.032,
  "pareto_knee": { "skill_nudge_interval": 8, "memory_nudge_interval": 8 }
}
```

---

## Study 03 — Review Model Calibration

**Output file:** `results/study_03_models.jsonl`

```json
{
  "config": {
    "model":       "openai/gpt-4o-mini",
    "tier":        "mid",
    "review_mode": "SKILLS_ONLY"
  },
  "metrics": {
    "q":                   0.643,
    "cost_per_review_usd": 0.000042,
    "q_normalised":        0.91
  },
  "meta": {
    "session_id": "s010",
    "input_tokens": 1987,
    "output_tokens": 312
  }
}
```

**Findings file:** `results/findings/study_03_findings.json`

```json
{
  "recommended_model": "openai/gpt-4o-mini",
  "budget_table": [
    { "budget_usd_per_1M_reviews": 100, "model": "openai/gpt-4o-mini", "q_normalised": 0.93 },
    { "budget_usd_per_1M_reviews": 500, "model": "openai/gpt-4.1",     "q_normalised": 1.00 }
  ],
  "skills_only_best_model":  "openai/gpt-4o-mini",
  "memory_only_best_model":  "openai/gpt-4o-mini",
  "combined_best_model":     "openai/gpt-4.1-mini"
}
```

---

## Study 04 — Memory Abstraction

**Output file:** `results/study_04_memory.jsonl`

```json
{
  "config": {
    "strategy":         "TopKRetrievalStrategy",
    "k":                5,
    "memory_char_limit": 2200
  },
  "metrics": {
    "tsr":            0.690,
    "context_chars":  1820,
    "q":              0.672
  },
  "meta": {
    "domain": "code_debug",
    "session_id": "s017"
  }
}
```

**Findings file:** `results/findings/study_04_findings.json`

```json
{
  "recommended_strategy":    "TopKRetrievalStrategy",
  "recommended_k":           5,
  "recommended_char_limit":  2200,
  "saturation_char_limit":   4000,
  "adopt_non_baseline":      true,
  "tsr_gain_vs_baseline":    0.062
}
```

---

## Study 05 — Lifecycle Optimization

**Output file:** `results/study_05_lifecycle.jsonl`

Threshold sweep records:

```json
{
  "config": {
    "phase": "B_threshold_sweep",
    "stale_after_days":   21,
    "archive_after_days": 60
  },
  "metrics": {
    "fp_rate":   0.12,
    "fn_rate":   0.31,
    "f_score":   0.177,
    "n_skills_evaluated": 80
  }
}
```

Lifecycle predictor evaluation records:

```json
{
  "config": {
    "phase":           "C_predictor",
    "observation_day": 60
  },
  "metrics": {
    "cv_auc_mean":    0.81,
    "cv_auc_std":     0.04,
    "feature_importances": {
      "days_since_last_use": 0.42,
      "use_count":           0.23,
      "patch_count":         0.14,
      "char_len":            0.08,
      "information_density": 0.07,
      "days_since_created":  0.03,
      "avg_days_between_uses": 0.02,
      "is_user_created":     0.01
    }
  }
}
```

**Findings file:** `results/findings/study_05_findings.json`

```json
{
  "curator_stale_after_days":   21,
  "curator_archive_after_days": 60,
  "f_score_at_optimum":         0.177,
  "curator_llm_predictor":      true,
  "predictor_auc":              0.81,
  "survival_50pct_day":         18
}
```

**Predictor artifact:** `results/study_05_lifecycle_predictor.pkl`
Serialised `LifecyclePredictor` instance. Load with:

```python
from phase_2_optimization.study_05_lifecycle_optimization.lifecycle_predictor import LifecyclePredictor

predictor = LifecyclePredictor.load("results/study_05_lifecycle_predictor.pkl")
```

---

## Study 06 — Prompt Sensitivity

**Output file:** `results/study_06_prompts.jsonl`

Ablation records:

```json
{
  "config": {
    "phase":          "A_ablation",
    "variant_name":   "NO_INSTRUCTION_LIST",
    "review_mode":    "SKILLS_ONLY",
    "session_id":     "s012"
  },
  "metrics": {
    "q":      0.601,
    "q_drop": 0.052
  }
}
```

Context window records:

```json
{
  "config": {
    "phase":          "B_context",
    "context_strategy": "last_10_turns",
    "context_chars":  3200
  },
  "metrics": {
    "q":              0.647,
    "context_tokens": 800
  }
}
```

**Findings file:** `results/findings/study_06_findings.json`

```json
{
  "critical_elements":    ["task_framing", "instruction_list", "format_constraints"],
  "dispensable_elements": ["negative_constraints"],
  "adopt_few_shot":       true,
  "few_shot_q_gain":      0.038,
  "recommended_context_strategy": "last_10_turns",
  "adversarial_robustness": {
    "full_prompt":    0.031,
    "minimal_prompt": 0.089
  }
}
```

---

## Study 07 — Skill Interference

**Output file:** `results/study_07_interference.jsonl`

Phase A (interference detection):

```json
{
  "config": {
    "phase":   "A_interference",
    "delta":   3,
    "domain":  "code_debug",
    "seed":    7
  },
  "metrics": {
    "q_treatment": 0.661,
    "q_control":   0.643,
    "cohens_d":    0.24,
    "direction":   "beneficial"
  }
}
```

Phase B (stability vs. turn number):

```json
{
  "config": {
    "phase":  "B_stability",
    "domain": "api_integration",
    "turn":   12
  },
  "metrics": {
    "ps":       0.832,
    "is_stable": true
  }
}
```

**Findings file:** `results/findings/study_07_findings.json`

```json
{
  "flush_min_turns":      8,
  "safe_delta":           3,
  "combined_mode_recommendation": "sequential",
  "interference_by_delta": [
    { "delta": 0, "cohens_d": -0.31, "direction": "harmful" },
    { "delta": 1, "cohens_d": -0.12, "direction": "negligible" },
    { "delta": 3, "cohens_d":  0.24, "direction": "beneficial" },
    { "delta": 5, "cohens_d":  0.29, "direction": "beneficial" },
    { "delta": 10,"cohens_d":  0.27, "direction": "beneficial" }
  ]
}
```

---

## Study 08 — Library Dynamics

**Output file:** `results/study_08_dynamics.jsonl`

Phase A growth records:

```json
{
  "config": {
    "phase":   "A_growth",
    "session": 40
  },
  "metrics": {
    "lc":                 0.58,
    "mean_pairwise_sim":  0.31,
    "gini":               0.44,
    "n_skills":           22,
    "marginal_lc_gain":   0.018
  }
}
```

Phase C consolidation records:

```json
{
  "config": {
    "phase":       "C_consolidation",
    "target_size": 50
  },
  "metrics": {
    "lc_before":          0.71,
    "lc_after":           0.74,
    "lc_delta":           0.03,
    "redundancy_before":  0.52,
    "redundancy_after":   0.38,
    "suggestions_count":  6,
    "n_skills":           51
  }
}
```

Phase D gap detection records:

```json
{
  "config": { "phase": "D_gaps" },
  "metrics": {
    "gaps": [
      {
        "domain":      "data_analysis",
        "tool_calls":  34,
        "max_rr":      0.21,
        "severity":    "high"
      }
    ]
  }
}
```

**Findings file:** `results/findings/study_08_findings.json`

```json
{
  "saturation_session":                    80,
  "enable_curator_llm_consolidation":      true,
  "curator_llm_consolidation_min_skills":  25,
  "peak_lc":                               0.82,
  "peak_redundancy":                       0.61,
  "peak_gini":                             0.53,
  "recommendations": {
    "lc_saturation_session":               80,
    "enable_curator_llm_consolidation":    true,
    "curator_llm_consolidation_min_skills": 25,
    "notes": "LC plateaus around session 80; enable consolidation at ≥ 25 skills."
  }
}
```

---

## Reading results programmatically

```python
from shared.result_logger import ResultLogger
from pathlib import Path

# Read all records from a study
records = ResultLogger.read(Path("results/study_02_trigger.jsonl"))

# Read only the metrics dicts (filtered for non-failed runs)
metrics = ResultLogger.read_metrics(Path("results/study_02_trigger.jsonl"))

# Stream records one at a time (memory-efficient for large files)
for record in ResultLogger.stream(Path("results/study_08_dynamics.jsonl")):
    print(record["metrics"]["lc"])
```

---

## Findings JSON conventions

All `findings/*.json` files follow these conventions:

- Top-level keys that directly correspond to `ReviewerConfig` fields use
  the exact field name (e.g. `"curator_stale_after_days"`, `"flush_min_turns"`).
- Boolean flags named `"enable_*"` or `"adopt_*"` indicate whether a feature
  should be turned on based on the study results.
- A `"recommendations"` sub-object is always present and contains a `"notes"`
  string with a one-paragraph human-readable summary.
- Numeric values are rounded to 4 decimal places.
