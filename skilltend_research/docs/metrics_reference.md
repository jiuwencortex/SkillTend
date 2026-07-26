# Metrics Reference

Authoritative definitions for every metric used across the eight studies.
All metrics are computed by code in `shared/` or within each study folder;
this document is the single source of truth for their mathematical definitions,
value ranges, and interpretation.

---

## Part 1 — Skill Quality Metrics (Q and sub-metrics)

These four metrics constitute the canonical quality score used in every study
that evaluates a skill. They are implemented in `shared/skill_quality_scorer.py`.

---

### Retrieval Relevance (RR)

**What it measures:** Does the skill contain the information a downstream agent
would need to solve the task it was written for?

**Computation:**

```
embeddings_skill = embed(skill_text)                # single vector
embeddings_tasks = [embed(q) for q in task_queries] # one per evaluation query

cosine_sim(a, b) = dot(a, b) / (|a| * |b|)

RR = mean( cosine_sim(embeddings_skill, e) for e in embeddings_tasks )
```

Embeddings use the `text-embedding-3-small` model by default (configurable
via `--embedding-model`). Task queries come from `shared/task_bank.py`.

**Range:** [0, 1]
**Interpretation:** 0 = no semantic overlap with any task; 1 = perfect match.
**Acceptable value:** ≥ 0.4 for a skill to be considered relevant.
**Stability:** Low variance (CV < 0.05); deterministic given fixed model.

---

### Information Density (ID)

**What it measures:** Bits of useful content per character. High redundancy
and filler text compress well, so a low compression ratio indicates high density.

**Computation:**

```
compressed   = zstd.compress(skill_text.encode("utf-8"))
ID           = 1 - (len(compressed) / len(skill_text.encode("utf-8")))
```

No LLM call required. Fully deterministic.

**Range:** [0, 1]
**Interpretation:** 0 = compresses perfectly (all redundancy); 1 = incompressible
(maximum density — rare in natural language, typically 0.5–0.8 for well-written text).
**Acceptable value:** ≥ 0.5.
**Stability:** CV = 0 (deterministic).

---

### Task Success Rate (TSR)

**What it measures:** Does an agent equipped with this skill complete tasks
better than an agent without it?

**Computation:**

```
for task in task_sample:
    answer_with    = agent_answer(task, context=skill_text)
    answer_without = agent_answer(task, context="")
    win[task]      = llm_judge(task, answer_with, answer_without) == "with"

TSR = mean(win)
```

The LLM judge uses a pairwise preference prompt comparing the two answers
against the task's `ground_truth`. Each trial is repeated `trials_per_task`
times (default 3) with different seeds; wins are majority-voted.

**Range:** [0, 1]
**Interpretation:** 0.5 = skill provides no benefit; > 0.5 = skill helps;
< 0.5 = skill actively hurts (negative transfer).
**Acceptable value:** ≥ 0.6 for a skill to be considered useful.
**Stability:** Higher variance than other sub-metrics (CV ≤ 0.15 acceptable).

---

### Patch Stability (PS)

**What it measures:** Does the skill converge after repeated review cycles,
or does each review keep changing it substantially?

**Computation:**

```
def normalised_edit_distance(a, b):
    # Word-level Levenshtein distance, capped at 500 tokens each
    a_words = a.split()[:500]
    b_words = b.split()[:500]
    return levenshtein(a_words, b_words) / max(len(a_words), len(b_words), 1)

PS = 1 - mean( normalised_edit_distance(versions[i], versions[i+1])
               for i in range(len(versions) - 1) )
```

Requires at least 2 consecutive versions of the same skill.

**Range:** [0, 1]
**Interpretation:** 1 = skill is completely stable (no changes between reviews);
0 = completely rewritten on every review.
**Acceptable value:** ≥ 0.7 after the warm-up period (see Study 07).
**Stability:** CV < 0.08.

---

### Combined Quality Score (Q)

**Formula (default weights):**

```
Q = 0.30 * RR  +  0.20 * ID  +  0.40 * TSR  +  0.10 * PS
```

**Note:** Study 01 calibrates these weights empirically using ridge regression
on human-labelled skills. The calibrated weights from Study 01 should replace
the defaults above for all subsequent studies.

**Range:** [0, 1]
**Interpretation:** ≥ 0.6 = high-quality skill; 0.4–0.6 = acceptable;
< 0.4 = poor quality (candidate for Curator action).

---

## Part 2 — Library-level Metrics

Used in Study 08 to characterise the skill library as a whole.

---

### Library Coverage (LC)

**What it measures:** What fraction of the agent's task distribution is
"covered" by at least one skill?

**Computation:**

```
for task in evaluation_task_bank:
    covered[task] = any( RR(skill, task) >= 0.6 for skill in library )

LC = mean(covered)
```

**Range:** [0, 1]
**Interpretation:** 1 = every task has a relevant skill; 0 = no task covered.
**Target:** LC ≥ 0.75 for a mature library.

---

### Mean Pairwise Similarity (redundancy proxy)

**What it measures:** Average semantic overlap between all pairs of skills in
the library. High values indicate redundant skills.

**Computation:**

```
embeddings = [embed(skill.text) for skill in library]
pairs      = all combinations of (i, j) where i < j
mean_pairwise_sim = mean( cosine_sim(embeddings[i], embeddings[j])
                          for (i,j) in pairs )
```

**Range:** [0, 1]
**Interpretation:** > 0.5 signals problematic redundancy and is a trigger for
Phase 2 LLM consolidation.

---

### Gini Coefficient (usage inequality)

**What it measures:** How unequally are skills used? A high Gini means a few
skills are used constantly while most are never retrieved.

**Computation:**

```
use_counts = sorted([skill.use_count for skill in library])
n = len(use_counts)
gini = (2 * sum((i+1) * v for i,v in enumerate(use_counts)) /
        (n * sum(use_counts))) - (n+1)/n
```

Standard Gini coefficient of the use_count distribution.

**Range:** [0, 1]
**Interpretation:** 0 = perfectly equal usage; 1 = one skill is used for
everything. Gini > 0.7 may indicate stale skills dragging down coverage.

---

## Part 3 — Statistical Metrics

Used in Study 01 for metric validation and in Study 07 for effect size.

---

### Spearman's ρ (rank correlation)

**Used in:** Study 01 — correlating each sub-metric with human quality labels.

**Acceptable threshold:** ρ > 0.5 for a metric to be considered a valid proxy
for human judgment.

---

### Fleiss' κ (inter-rater agreement)

**Used in:** Study 01 — measuring agreement between 3 human raters labelling
30 skills as high/medium/low.

**Acceptable threshold:** κ > 0.65. Study 01 does not proceed to weight
calibration if κ ≤ 0.65.

---

### Coefficient of Variation (CV)

**Used in:** Study 01 — stability check for each sub-metric across 5 repeated
measurements.

**Formula:** `CV = std / mean`

**Acceptable thresholds:** CV < 0.08 for RR, ID, PS; CV < 0.15 for TSR.

---

### Cohen's d (effect size)

**Used in:** Study 07 — measuring the interference effect of within-session
skill patches.

**Formula:**

```
pooled_std = sqrt( ((n1-1)*std1^2 + (n2-1)*std2^2) / (n1+n2-2) )
d = (mean_treatment - mean_control) / pooled_std
```

**Interpretation:**

| |d| range | Magnitude |
|---|---|
| < 0.2 | Negligible |
| 0.2 – 0.5 | Small |
| 0.5 – 0.8 | Medium |
| > 0.8 | Large |

**Direction interpretation in Study 07:**
d > 0 = patch was helpful (treatment Q > control Q)
d < 0 = patch was harmful (negative interference)

---

## Part 4 — Trigger and Lifecycle Metrics

---

### Review Token Cost

**Used in:** Study 02.

```
cost_per_review = (avg_input_tokens * price_per_1M_in
                   + avg_output_tokens * price_per_1M_out) / 1_000_000
```

Prices default to OpenAI `gpt-4o-mini` rates. Override via `--cost-in` and
`--cost-out` CLI flags.

---

### F-score (lifecycle threshold sweep)

**Used in:** Study 05 — optimising `curator_stale_after_days` and
`curator_archive_after_days`.

```
FP_rate = |incorrectly_archived| / |still_useful|    (useful skills lost)
FN_rate = |incorrectly_kept|    / |genuinely_stale|  (stale skills kept)

F = FP_WEIGHT * FP_rate + FN_WEIGHT * FN_rate
```

Default weights: `FP_WEIGHT = 0.7`, `FN_WEIGHT = 0.3`. The asymmetry reflects
the business decision that losing a useful skill is worse than keeping a stale one.

Lower F is better. The optimal threshold pair minimises F.

---

### AUC-ROC (lifecycle predictor)

**Used in:** Study 05 Phase C — evaluating the `LifecyclePredictor` XGBoost
classifier.

Standard Area Under the Receiver Operating Characteristic curve, computed with
5-fold `StratifiedKFold` cross-validation.

**Acceptable threshold:** AUC > 0.75 to replace the fixed-threshold FSM in
the production Curator.

---

## Part 5 — Prompt and Model Metrics

---

### Q-drop (prompt ablation)

**Used in:** Study 06.

```
Q_drop(variant) = Q(full_prompt) - Q(ablated_variant)
```

**Classification:**
- Q_drop < 0.005 → element is dispensable (can be removed without loss)
- 0.005 ≤ Q_drop < 0.02 → element has minor value
- Q_drop ≥ 0.02 → element is critical (must be retained)

---

### Consolidation ΔLC

**Used in:** Study 08 Phase C.

```
ΔLC = LC_after_consolidation - LC_before_consolidation
```

Positive = consolidation improved coverage (merged redundant skills freed
semantic space for coverage gaps). Negative = consolidation destroyed useful
specificity.

**Benefit threshold:** ΔLC ≥ 0.005 for a consolidation pass to be classified
as beneficial.

---

## Summary table

| Metric | Range | Better | Primary study | Implementation |
|---|---|---|---|---|
| RR | [0,1] | ↑ | 01,02,03,04,05,07,08 | `shared/skill_quality_scorer.py` |
| ID | [0,1] | ↑ | 01 | `shared/skill_quality_scorer.py` |
| TSR | [0,1] | ↑ (≥0.5) | 01,03,04 | `shared/skill_quality_scorer.py` |
| PS | [0,1] | ↑ | 01,07 | `shared/skill_quality_scorer.py` |
| Q | [0,1] | ↑ | all | `shared/skill_quality_scorer.py` |
| LC | [0,1] | ↑ | 08 | `study_08_library_dynamics/coverage_metric.py` |
| Mean pairwise sim | [0,1] | ↓ | 08 | `study_08_library_dynamics/coverage_metric.py` |
| Gini | [0,1] | ↓ | 08 | `study_08_library_dynamics/coverage_metric.py` |
| Spearman ρ | [-1,1] | ↑ | 01 | `study_01_skill_quality_metrics/compute_validity.py` |
| Fleiss κ | [-1,1] | ↑ (>0.65) | 01 | `study_01_skill_quality_metrics/compute_validity.py` |
| CV | [0,∞) | ↓ | 01 | `study_01_skill_quality_metrics/compute_validity.py` |
| Cohen's d | (-∞,∞) | >0 = helpful | 07 | `study_07_skill_interference/interference_experiment.py` |
| F-score | [0,1] | ↓ | 05 | `study_05_lifecycle_optimization/threshold_sweep.py` |
| AUC-ROC | [0,1] | ↑ (>0.75) | 05 | `study_05_lifecycle_optimization/lifecycle_predictor.py` |
| Q-drop | [0,∞) | ↓ | 06 | `study_06_prompt_sensitivity/analyze.py` |
| ΔLC | (-1,1) | ↑ (>0.005) | 08 | `study_08_library_dynamics/consolidation_eval.py` |
