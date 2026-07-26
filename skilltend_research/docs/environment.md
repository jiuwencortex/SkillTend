# Environment Setup

This guide covers everything needed to run `skilltend_research` experiments from
a clean checkout.

---

## Python version

Python **3.11** or later is required. The lifecycle predictor uses
`StratifiedKFold` from scikit-learn ≥ 1.3 and XGBoost ≥ 2.0, both of which
drop support for Python 3.10 at those versions.

```bash
python --version      # must be ≥ 3.11
```

---

## Virtual environment

```bash
# Create and activate
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install production system first (editable)
pip install -e /path/to/SkillTend/

# Install the research package (editable, with dev extras)
pip install -e ".[dev]"
```

The `[dev]` extra pulls in `pytest`, `pytest-asyncio`, and `ruff` on top of the
core dependencies listed in `pyproject.toml`.

---

## Required environment variables

All LLM and embedding calls go through [litellm](https://github.com/BerriAI/litellm).
Set whichever keys correspond to the models you intend to use:

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | `openai/*` models (gpt-4o-mini, gpt-4.1, …) |
| `ANTHROPIC_API_KEY` | `anthropic/*` models (claude-sonnet-4, claude-opus-4, …) |
| `GROQ_API_KEY` | `groq/*` models (llama-3.3-70b, …) |
| `OPENROUTER_API_KEY` | `openrouter/*` models (qwen2.5-72b, …) |

A minimal `.env` for running with OpenAI only:

```
OPENAI_API_KEY=sk-...
```

Export before running (or use `direnv`):

```bash
export $(cat .env | xargs)
```

---

## Verifying the installation

Run every study in dry-run mode to confirm all imports resolve and the CLI
argument parsers work without making any LLM calls:

```bash
cd /path/to/SkillTend/skilltend_research

python -m study_01_skill_quality_metrics.runner  --dry-run --rounds 1
python -m study_02_trigger_policy.runner         --dry-run --sessions-per-config 1
python -m study_03_review_model_calibration.runner --dry-run
python -m study_04_memory_abstraction.runner     --dry-run
python -m study_05_lifecycle_optimization.runner --dry-run
python -m study_06_prompt_sensitivity.runner     --dry-run
python -m study_07_skill_interference.runner     --dry-run
python -m study_08_library_dynamics.runner       --dry-run
```

A successful dry-run prints `[dry-run]` prefixed lines and exits with code 0.
No files are written to `results/` in dry-run mode.

---

## Directory layout after first run

```
skilltend_research/
└── results/
    ├── study_01_metrics.jsonl
    ├── study_01_labels.jsonl      (written by label_tool.py)
    ├── study_02_trigger.jsonl
    ├── study_02_pareto.png
    ├── study_03_models.jsonl
    ├── study_04_memory.jsonl
    ├── study_05_lifecycle.jsonl
    ├── study_05_lifecycle_predictor.pkl
    ├── study_06_prompts.jsonl
    ├── study_07_interference.jsonl
    ├── study_08_dynamics.jsonl
    └── findings/
        ├── study_02_findings.json
        ├── study_03_findings.json
        ├── study_04_findings.json
        ├── study_05_findings.json
        ├── study_06_findings.json
        ├── study_07_findings.json
        └── study_08_findings.json
```

`*.jsonl`, `*.png`, and `*.pkl` are gitignored. Only `findings/*.json` and
any hand-written `findings_*.md` files are tracked by git.

---

## Running the full pipeline

After Study 01 finishes successfully (inter-rater κ > 0.65), run the
independent studies in parallel. Example with GNU parallel:

```bash
parallel python -m {}.runner --model openai/gpt-4o-mini --output results/{}.jsonl ::: \
    study_02_trigger_policy \
    study_03_review_model_calibration \
    study_04_memory_abstraction \
    study_05_lifecycle_optimization \
    study_07_skill_interference
```

Then Study 06 (depends on 03) and Study 08 (depends on 07):

```bash
python -m study_06_prompt_sensitivity.runner   --model openai/gpt-4o-mini
python -m study_08_library_dynamics.runner     --model openai/gpt-4o-mini
```

---

## Cost estimates

All cost numbers assume OpenAI API pricing as of mid-2025. Adjust for your
model selection.

| Study | Sessions / configs | Approximate tokens | Estimated cost |
|---|---|---|---|
| 01 | 30 skills × 5 rounds | ~450k | ~$0.20 |
| 02 (full sweep) | 7,680 sessions | ~30M | ~$15 |
| 02 (adaptive only) | 500 sessions | ~2M | ~$1 |
| 03 | 8 models × 40 sessions | ~1.3M | ~$1 per tier |
| 04 | 4 strategies × 50 sessions | ~800k | ~$0.40 |
| 05 | 180 sim days | ~500k | ~$0.25 |
| 06 | 7 variants × 30 sessions | ~630k | ~$0.30 |
| 07 | 5 Δ values × 40 sessions | ~800k | ~$0.40 |
| 08 | 200 sessions + consolidation | ~1.2M | ~$0.60 |

Study 02 dominates cost. Use `--sessions-per-config 5` for a cheap
exploratory run before committing to the full sweep.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'skilltend'`**
→ The production package was not installed. Run `pip install -e ../` from
inside the `skilltend_research/` directory.

**`litellm.exceptions.AuthenticationError`**
→ Check that the API key for your chosen model provider is exported.

**`xgboost.core.XGBoostError`**
→ Upgrade XGBoost: `pip install --upgrade xgboost`.

**Dry-run exits non-zero on Study 01**
→ Study 01's `runner.py` requires `--skills-dir` pointing to an existing
  directory even in dry-run mode. Create a temporary one:
  ```bash
  mkdir -p /tmp/test_skills
  python -m study_01_skill_quality_metrics.runner --dry-run --skills-dir /tmp/test_skills
  ```
