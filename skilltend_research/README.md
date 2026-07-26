# skilltend_research

Empirical research track for SkillTend. Every study produces a concrete,
actionable recommendation for a parameter or design decision in the
production `skilltend/` package.

See `docs/guide.md` for the full operational guide (what each study does, how to run it, how to read the results).

---

## Install

```bash
# Install the production system first
pip install -e ../

# Install the research package
pip install -e ".[dev]"
```

---

## Structure

```
shared/                          # Shared simulation and evaluation infrastructure
  task_bank.py                   # 60 curated evaluation tasks across 6 domains
  session_simulator.py           # Synthetic agent session generator
  skill_quality_scorer.py        # Canonical Q metric (RR, ID, TSR, PS)
  result_logger.py               # JSONL experiment output

study_01_skill_quality_metrics/  # Validate the Q metric itself
study_02_trigger_policy/         # Optimize skill_nudge_interval and memory_nudge_interval
study_03_review_model_calibration/ # Find minimum-cost review model
study_04_memory_abstraction/     # Compare inject-all vs. retrieval-based memory
study_05_lifecycle_optimization/ # Tune curator_stale_after_days / archive_after_days
study_06_prompt_sensitivity/     # Ablate review prompts; tune context window
study_07_skill_interference/     # Measure within-session patch interference
study_08_library_dynamics/       # Track coverage, redundancy, consolidation ROI
```

---

## Running a Study

Each study has a `runner.py` entry point:

```bash
# Study 02: trigger policy sweep
python -m study_02_trigger_policy.runner \
    --model gpt-4o-mini \
    --sessions-per-config 20 \
    --output results/study_02.jsonl

# Analyze results
python -m study_02_trigger_policy.analyze \
    --input results/study_02.jsonl \
    --plot results/study_02_pareto.png
```

All runners accept `--dry-run` to mock LLM calls (instant, free, for
CI or structural testing):

```bash
python -m study_02_trigger_policy.runner --dry-run
```

---

## Execution Order

Study 01 must complete before any other study, because it validates and
calibrates the Q metric all other studies depend on.

```
Study 01
  ├── Study 02
  ├── Study 03 ──► Study 06
  ├── Study 04
  ├── Study 05
  └── Study 07 ──► Study 08
```

---

## Results

Each runner writes JSONL to `results/study_NN.jsonl`. The `analyze.py`
scripts read that JSONL and produce:
- Summary statistics printed to stdout
- PNG plots saved to `results/`
- A `findings.md` with the actionable recommendation

Results directories are gitignored. Add your own `.gitignore` entry if
committing findings:

```
results/*.jsonl
results/*.png
!results/findings_*.md
```
