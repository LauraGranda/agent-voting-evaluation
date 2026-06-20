# Agentic Voting vs G-Eval for Relevance Evaluation in Conversational AI

> *Evaluación de Relevancia en Agentes Conversacionales de IA mediante un
> Sistema de Votación Agéntico en comparación con el Framework G-EVAL*
> — master's thesis, Universidad Pontificia Bolivariana.

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

## Overview

This repository contains the code, configuration and result artifacts for
a master's thesis that compares two automatic LLM-based evaluators of
response relevance in open-domain conversation: the published **G-Eval**
framework (single judge — `gpt-4o`) and an **agentic voting system** that
aggregates three judges from different providers (`gpt-4o` from OpenAI,
`gemini-2.5-flash` from Google and `claude-haiku-4-5` from Anthropic).
Both methods score 900 conversations from the DailyDialog–Zhao dataset
against four MTurk human relevance scores per item.

**Research question.** Given that both methods correlate strongly with
human judgment, does the additional cost of a three-judge panel buy a
measurable improvement on the dimensions that matter for downstream use
— ranking, scalar calibration, robustness — or is the single judge
statistically equivalent at a fraction of the cost?

**Headline finding** (HU-12 + HU-14). Both methods are statistically
equivalent in ranking (TOST on Δρ inside ±0.05) and in absolute error
(Wilcoxon p = 0.194, Cohen's d = 0.097 negligible). The voting panel
costs **2.30× more** (USD 8.15 vs USD 3.55 for n = 900) and takes
**4.20× longer** to run, but improves Cohen's weighted κ by +0.118
(0.643 vs 0.525, from *moderate* to *substantial*) and exact-agreement
rate by +8.4 pp. The dollar buys descriptive calibration, not
inferential improvement.

## Repository structure

```text
agent-voting-evaluation/
├── README.md                       # This file
├── REPRODUCE.md                    # Step-by-step reproduction guide
├── LICENSE                         # MIT license
├── CITATION.cff                    # Citation metadata (software + thesis)
├── CHANGELOG.md                    # Per-HU release notes
├── pyproject.toml                  # Project metadata + dependencies
├── uv.lock                         # Locked dependency versions
├── .env.example                    # Template for API keys
├── .python-version                 # Python 3.12
├── Makefile                        # Convenience targets (lint, docs)
├── configs/
│   ├── agents/                     # One YAML per judge (openai/google/anthropic)
│   ├── prompts/                    # Pilot prompts v1/v2/v3 + final G-Eval prompt
│   └── experiment_config.yaml      # Run-level defaults
├── data/
│   ├── raw/dailydialog_zhao/       # Original Zhao et al. (2020) dataset
│   └── processed/                  # DeepEval-format test cases
├── docs/                           # Methodological documentation (Spanish)
│   ├── dataset_selection.md
│   ├── agent_panel_design.md
│   ├── voting_scheme_analysis.md
│   ├── significance_tests_justification.md
│   └── limitations.md
├── scripts/                        # Executable scripts (CLI entry points)
│   ├── download_dataset.py
│   ├── transform_dataset.py
│   ├── run_geval.py
│   ├── run_judge.py                # Single-judge runner (used by voting)
│   ├── run_voting_system.py
│   ├── analyze_geval.py
│   ├── build_pilot_notebook.py
│   └── test_deepeval_setup.py
├── src/voting/                     # Library code (aggregator)
│   └── aggregator.py
├── notebooks/                      # Jupyter notebooks, executed end-to-end
│   ├── 01_eda.ipynb                # Exploratory data analysis (HU-03)
│   ├── 02_prompt_pilot.ipynb       # Prompt pilot v1/v2/v3 (HU-05)
│   ├── 03_voting_pilot.ipynb       # Voting pilot (HU-08)
│   ├── 04_descriptive_analysis.ipynb       # HU-10: ρ, κ, MAE per stratum
│   ├── 05_correlation_analysis.ipynb       # HU-11: Fisher Z + bootstrap CIs
│   ├── 06_significance_tests.ipynb         # HU-12: Wilcoxon, Steiger, TOST
│   ├── 07_error_analysis.ipynb             # HU-13: top-20 + taxonomy
│   └── 08_cost_analysis.ipynb              # HU-14: cost vs quality trade-off
├── outputs/                        # Generated artifacts (committed)
│   ├── geval_results.json          # Per-pair G-Eval scores + rationale
│   ├── voting_results.json         # Per-pair voting scores + per-judge breakdown
│   ├── *_summary.md                # Citable summaries per HU
│   ├── figures/                    # 21 PNG figures at 150 dpi
│   ├── logs/                       # Execution logs (with wall time + cost)
│   └── agent_scores/               # Per-judge JSON dumps
└── tests/                          # pytest test suite
```

## Requirements

- **Python ≥ 3.12** (pinned in `.python-version`).
- **[uv](https://github.com/astral-sh/uv)** as the package manager
  (handles virtualenv + lockfile automatically).
- **Three API keys** (only needed for Phases 3 and 4):
    - `OPENAI_API_KEY` — for G-Eval and the OpenAI judge.
    - `GOOGLE_API_KEY` — for the Gemini judge.
    - `ANTHROPIC_API_KEY` — for the Claude judge.
- ~ USD 12 of API budget and ~ 3.2 hours of wall time for a full
  reproduction (see [REPRODUCE.md](./REPRODUCE.md) for the breakdown).

## Installation

```bash
git clone <repository-url>
cd agent-voting-evaluation
cp .env.example .env       # then edit .env and fill in the three API keys
uv sync                    # creates .venv and installs locked dependencies
uv run python scripts/test_deepeval_setup.py   # sanity check
```

If you only want to **re-run the analysis** without paying for API calls,
the committed `outputs/geval_results.json` and `outputs/voting_results.json`
already contain all the raw scores. Skip Phases 3 and 4 — see the
"analysis-only mode" section of [REPRODUCE.md](./REPRODUCE.md).

## Reproduction in five phases

All commands are run from the repository root with the `uv run` prefix
so they execute inside the locked environment.

### Phase 1 — Dataset

```bash
uv run python scripts/download_dataset.py
uv run python scripts/transform_dataset.py
```

Downloads the DailyDialog–Zhao corpus (Zhao et al., 2020) and converts
it to the DeepEval `ConversationalTestCase` format
(`data/processed/deepeval_test_cases.json`, 900 entries).

### Phase 2 — Exploratory data analysis

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb \
  --output 01_eda.ipynb --ExecutePreprocessor.timeout=180
```

Produces `outputs/figures/01..04_*.png` plus the EDA section of the
descriptive summary.

### Phase 3 — G-Eval evaluation (≈ 37 min, ≈ USD 3.55)

```bash
uv run python scripts/run_geval.py
```

Single-judge G-Eval over the full corpus. Outputs:
`outputs/geval_results.json` (900 entries with `geval_score`, `reason`,
`cost_usd`, `tokens_used`) and `outputs/geval_summary_stats.md`.

### Phase 4 — Voting system (≈ 157 min, ≈ USD 8.15)

```bash
uv run python scripts/run_voting_system.py
```

Three-judge panel over the full corpus. Outputs:
`outputs/voting_results.json` (with `individual_scores`,
`final_vote_score`, `cost_by_agent`, `metadata.std_deviation`) and
`outputs/voting_summary_stats.md`. The runner has automatic retry on
transient 503 errors from Gemini.

### Phase 5 — Analysis notebooks (≈ 5 min total, no API calls)

```bash
for nb in 04_descriptive_analysis 05_correlation_analysis \
          06_significance_tests 07_error_analysis 08_cost_analysis; do
  uv run jupyter nbconvert --to notebook --execute \
    "notebooks/${nb}.ipynb" --output "${nb}.ipynb" \
    --ExecutePreprocessor.timeout=300
done
```

Each notebook reads `outputs/*.json` directly and produces
`outputs/<topic>_summary.md` plus figures `outputs/figures/10..21_*.png`.

See [REPRODUCE.md](./REPRODUCE.md) for expected outputs, verification
counts, troubleshooting, and the analysis-only mode that skips the
expensive Phases 3 and 4.

## Testing and code quality

```bash
uv run pytest -q              # run the test suite
uv run pre-commit run --all-files   # ruff + ruff-format + sanity hooks
make check                    # mypy + ruff + tests
```

## Documentation

The detailed methodological documents live in [`docs/`](./docs/) (in
Spanish, matching the thesis language):

- [`docs/dataset_selection.md`](./docs/dataset_selection.md) — corpus
  rationale (DailyDialog–Zhao).
- [`docs/agent_panel_design.md`](./docs/agent_panel_design.md) — judge
  selection and panel architecture.
- [`docs/voting_scheme_analysis.md`](./docs/voting_scheme_analysis.md) —
  aggregation scheme (arithmetic mean) and alternatives.
- [`docs/significance_tests_justification.md`](./docs/significance_tests_justification.md)
  — Wilcoxon vs t, Steiger overlapping, TOST equivalence.
- [`docs/limitations.md`](./docs/limitations.md) — dataset, G-Eval,
  voting, shared limitations, and directions for future work.

A user-facing release history is in [`CHANGELOG.md`](./CHANGELOG.md).

## How to cite

Citation metadata is provided in [`CITATION.cff`](./CITATION.cff) using
the [Citation File Format](https://citation-file-format.github.io/) v1.2.0.
GitHub renders a "Cite this repository" widget from that file. The
preferred citation points at the thesis; a software-style citation is
also available for code reuse.

## License

This project is released under the [MIT License](./LICENSE).

## Author

**Laura Granda** — Universidad Pontificia Bolivariana.
Master's thesis on automatic evaluation of conversational AI.
GitHub: [@LauraGranda](https://github.com/LauraGranda).
