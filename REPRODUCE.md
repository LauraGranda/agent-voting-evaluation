# Reproducing the Experiments

This document is the step-by-step recipe for reproducing every result
in this repository starting from a clean checkout. It assumes no prior
knowledge of the project; if you are looking for an overview first,
read [`README.md`](./README.md).

All commands run from the repository root. Every Python command is
prefixed with `uv run` so it executes inside the locked virtual
environment created by `uv sync`.

## 1. Tested environment

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 (via WSL2 on Windows 11) |
| Kernel | Linux 6.6.x WSL2 |
| Python | 3.12 (pinned in `.python-version`) |
| `uv` | 0.5+ (any recent release works) |
| Disk | ≈ 3 GB free for `.venv`, dataset, outputs |
| RAM | 4 GB is enough; notebooks 06/07 use < 1 GB |

The pipeline does not need a GPU. macOS and native Linux should work
identically; native Windows has not been tested but should work if
`uv` is installed.

## 2. Bootstrap

```bash
git clone <repository-url>
cd agent-voting-evaluation
uv sync
```

`uv sync` reads `pyproject.toml` + `uv.lock`, creates `.venv/`, and
installs the exact pinned versions of every dependency. Re-runs are
no-ops if nothing changed.

Verify:

```bash
uv run python --version          # → Python 3.12.x
uv run python -c "import deepeval, openai, anthropic, google.genai; print('OK')"
```

## 3. API key configuration

The runners need three keys. Phase 5 (analysis notebooks) does **not**
need any of them, so if you only want to reproduce the analysis from
the committed JSON, you can skip this section — see §8.

### 3.1. Get the keys

| Provider | Console | Used by |
|---|---|---|
| OpenAI | <https://platform.openai.com/api-keys> | `gpt-4o` for G-Eval (Phase 3) and `judge_openai` (Phase 4) |
| Google | <https://aistudio.google.com/apikey> | `gemini-2.5-flash` for `judge_google` (Phase 4) |
| Anthropic | <https://console.anthropic.com/settings/keys> | `claude-haiku-4-5` for `judge_anthropic` (Phase 4) |

Each key only needs the default scope (chat completions / generate
content). No tool/file/admin scopes are required.

### 3.2. Write the `.env` file

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder values:

```dotenv
OPENAI_API_KEY=sk-...           # from https://platform.openai.com/api-keys
GOOGLE_API_KEY=AI...            # from https://aistudio.google.com/apikey
ANTHROPIC_API_KEY=sk-ant-...    # from https://console.anthropic.com/settings/keys
```

`.env` is git-ignored. Never commit it.

### 3.3. Verify connectivity

```bash
uv run python scripts/test_deepeval_setup.py
```

Expected: a single line per provider confirming that the API responds.
If any provider fails, fix it before continuing — `run_voting_system.py`
refuses to start if any of the three keys is missing or invalid.

## 4. Phase-by-phase reproduction

### Phase 1 — Dataset (≈ 30 s, no API)

```bash
uv run python scripts/download_dataset.py
uv run python scripts/transform_dataset.py
```

| File | Expected | Size |
|---|---|---|
| `data/raw/dailydialog_zhao/ACL2020_data.zip` | the original archive | ≈ 1.6 MB |
| `data/raw/dailydialog_zhao/dataset.json` | 900 entries with `turns[]`, `response`, `human_relevance_score`, `raw_relevance_scores` | ≈ 776 KB |
| `data/processed/deepeval_test_cases.json` | 900 entries in DeepEval `ConversationalTestCase` format | ≈ 1 MB |

Verify:

```bash
uv run python -c "import json; print(len(json.load(open('data/processed/deepeval_test_cases.json'))))"
# → 900
```

### Phase 2 — Exploratory data analysis (≈ 1 min, no API)

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb \
  --output 01_eda.ipynb --ExecutePreprocessor.timeout=180
```

| Output | What |
|---|---|
| `outputs/figures/01_histogram_relevance.png` | Histogram of human relevance scores |
| `outputs/figures/02_boxplot_by_model.png` | Boxplot per generator model |
| `outputs/figures/03_frequency_table.png` | Frequency table heatmap |
| `outputs/figures/04_interannotator_std.png` | Std among the 4 MTurk annotators |

### Phase 3 — G-Eval evaluation (≈ 37 min, ≈ USD 3.55)

```bash
uv run python scripts/run_geval.py
```

The runner checkpoints every 50 entries. Safe to interrupt and resume:
re-running picks up from the last checkpoint unless `--no-resume` is
passed.

| Output | What | Size |
|---|---|---|
| `outputs/geval_results.json` | 900 entries: `geval_score`, `reason`, `tokens_used`, `cost_usd`, `delta` | ≈ 696 KB |
| `outputs/geval_summary_stats.md` | Markdown summary (completion %, ρ, distribution, family breakdown) | ≈ 2 KB |
| `outputs/logs/geval_execution.log` | Structured log with `total wall time:` and `total cost:` on the last lines | ≈ 80 KB |

Verify the final summary line in the log:

```bash
grep -E "total wall time|total cost" outputs/logs/geval_execution.log | tail -2
# → total wall time:   2241.1s  (37m 21s)
# → total cost:        $3.5453
```

### Phase 4 — Voting system (≈ 157 min, ≈ USD 8.15)

```bash
uv run python scripts/run_voting_system.py
```

Three judges run in parallel per item. Tenacity retries on 5xx and
transient network errors; in the reference run, retries against
`gemini-2.5-flash` accounted for ~ 35 min of the total wall time
(157 min real vs ~ 120 min nominal). Re-running resumes from the last
saved item if interrupted.

| Output | What | Size |
|---|---|---|
| `outputs/voting_results.json` | 900 entries: `final_vote_score`, `individual_scores`, `cost_by_agent`, `metadata.std_deviation`, `agreement_level` | ≈ 728 KB |
| `outputs/voting_summary_stats.md` | Markdown summary (per-judge ρ, family breakdown, agreement levels, cost by agent) | ≈ 3 KB |
| `outputs/agent_scores/*.json` | One file per judge with its individual responses | ≈ 200 KB each |
| `outputs/logs/voting_execution.log` | Structured log; final lines have wall time + total cost + per-agent cost | ≈ 250 KB |

Verify:

```bash
grep -E "total wall time|total cost|judge_(openai|google|anthropic):" \
  outputs/logs/voting_execution.log | tail -5
# → total wall time:   9414.0s  (156m 54s)
# → total cost:        $8.1547
# →   judge_openai:      $4.3620
# →   judge_google:      $0.7426
# →   judge_anthropic:   $3.0501
```

### Phase 5 — Analysis notebooks (≈ 5 min total, no API)

Run them in numeric order — notebook 06 reads outputs from notebooks
04 and 05, and notebooks 07–08 reference 06.

```bash
for nb in 04_descriptive_analysis 05_correlation_analysis \
          06_significance_tests 07_error_analysis 08_cost_analysis; do
  uv run jupyter nbconvert --to notebook --execute \
    "notebooks/${nb}.ipynb" --output "${nb}.ipynb" \
    --ExecutePreprocessor.timeout=300
done
```

Each notebook produces a `outputs/<topic>_summary.md` and one or more
figures in `outputs/figures/10..21_*.png`. Approximate per-notebook
times on a modest laptop:

| Notebook | Time | Key outputs |
|---|---|---|
| 04 descriptive | ≈ 20 s | `descriptive_analysis_summary.md`, fig 10–13 |
| 05 correlation | ≈ 90 s (bootstrap n_iter = 10 000) | `correlation_analysis_summary.md`, fig 14 |
| 06 significance | ≈ 60 s | `significance_tests_summary.md`, fig 15–16 |
| 07 error | ≈ 30 s | `error_analysis_summary.md`, fig 17–19 |
| 08 cost | ≈ 10 s | `cost_analysis_summary.md`, fig 20–21 |

## 5. Cost & time budget summary

| Phase | Wall time | API cost | Notes |
|---|---|---|---|
| 1 Dataset | ~30 s | $0 | Single HTTP download from Zenodo |
| 2 EDA | ~1 min | $0 | Local only |
| 3 G-Eval | ~37 min | **$3.5453** | 900 calls to `gpt-4o`, 1 302 tokens/call |
| 4 Voting | ~157 min | **$8.1547** | 2 700 calls (3 judges × 900); includes Gemini 503 retries |
| 5 Analysis | ~5 min | $0 | Reads committed JSON only |
| **Total** | **~3.2 h** | **~$11.70** | Excluding human review of outputs |

Numbers come from `outputs/cost_analysis_summary.md` (notebook 08),
which derives them directly from `cost_usd` and `tokens_used` in the
result files plus the structured wall-time lines in the logs. They are
spot prices as of run dates (G-Eval: 2026-05-17; voting: 2026-06-07);
provider tariff changes invalidate these ratios.

## 6. Verification checklist

After a full reproduction, all of the following should hold:

```bash
# Lengths
uv run python -c "import json; [print(f, len(json.load(open(f)))) for f in ['outputs/geval_results.json', 'outputs/voting_results.json', 'data/processed/deepeval_test_cases.json']]"
# → outputs/geval_results.json 900
# → outputs/voting_results.json 900
# → data/processed/deepeval_test_cases.json 900

# Cost totals
uv run python -c "
import json
g = json.load(open('outputs/geval_results.json'))
v = json.load(open('outputs/voting_results.json'))
print(f'G-Eval total USD: {sum(r[\"cost_usd\"] for r in g):.4f}')
print(f'Voting total USD: {sum(r[\"cost_usd\"] for r in v):.4f}')
"
# → G-Eval total USD: 3.5453
# → Voting total USD: 8.1547

# Figures
ls outputs/figures/ | wc -l
# → 21
```

## 7. Known issues

- **Gemini 503 transient errors**. `gemini-2.5-flash` returns 503 under
  load. The voting runner uses `tenacity` with exponential backoff and
  10 retries; in the reference run, the inflated wall time of ~ 157 min
  vs the ~ 120 min nominal is entirely explained by these retries. If
  your run is much longer, the provider may be degraded — try again
  later or temporarily reduce parallelism in `configs/agents/agent_google.yaml`.
- **DeepEval cache**. The `.deepeval/` directory is not versioned and
  may grow large if you re-run G-Eval many times. Delete it freely.
- **Pre-commit reformatting after notebook edits**. The first
  `pre-commit run` after editing a notebook tends to reformat it
  (ruff-format on code cells). Re-run pre-commit a second time and it
  will pass.
- **Re-running deletes nothing**. The runners write atomically and
  resume from checkpoints, but the analysis notebooks **overwrite**
  the summary markdowns and figures on each execution. Commit them
  before re-running if you want a diff.
- **Pricing is spot at run time**. Costs in the logs and in
  `outputs/cost_analysis_summary.md` are the prices charged on the day
  of the run, not today. If you re-run, the totals will differ.

## 8. Analysis-only mode (skip Phases 3 and 4)

The committed JSON files in `outputs/` are the actual outputs of the
reference run, with the per-pair scores, rationales, costs and tokens
needed by every analysis notebook.

To reproduce the analysis without paying for any API call:

```bash
# 1. Make sure dataset.json + processed test cases exist
uv run python scripts/download_dataset.py
uv run python scripts/transform_dataset.py

# 2. Skip Phases 3 and 4 entirely. Just run the analysis:
for nb in 04_descriptive_analysis 05_correlation_analysis \
          06_significance_tests 07_error_analysis 08_cost_analysis; do
  uv run jupyter nbconvert --to notebook --execute \
    "notebooks/${nb}.ipynb" --output "${nb}.ipynb" \
    --ExecutePreprocessor.timeout=300
done
```

All five analysis notebooks read from `outputs/geval_results.json`,
`outputs/voting_results.json`, and `data/raw/dailydialog_zhao/dataset.json`.
None of them instantiates an LLM client.

Cost: $0. Wall time: ≈ 5 min.

## 9. Tests and quality gates

```bash
uv run pytest -q              # full test suite
uv run pre-commit run --all-files   # ruff + ruff-format + sanity hooks
```

These have no API dependencies and run in seconds.
