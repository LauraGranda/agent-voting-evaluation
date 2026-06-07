# Voting System Run — Summary Statistics

- **Panel**: judge_openai (`gpt-4o`), judge_google (`gemini-2.5-flash`), judge_anthropic (`claude-haiku-4-5`)
- **Prompt**: `configs/prompts/geval_relevance_prompt.txt` (V3, shared)
- **Aggregator**: `arithmetic_mean` (see `docs/voting_scheme_analysis.md`)
- **Generated**: 2026-06-07T20:07:19.471292+00:00

## Run completion

| Metric | Value |
|---|---|
| Total pairs | 900 |
| Fully scored | 900 (100.00%) |
| Aggregate failed | 0 |
| Input tokens | 3,556,386 |
| Output tokens | 666,808 |
| Total cost | $8.1547 |
|  - judge_openai | $4.3620 |
|  - judge_google | $0.7426 |
|  - judge_anthropic | $3.0501 |

## final_vote_score distribution (1-5)

| Stat | Value |
|---|---|
| n | 900 |
| mean | 2.8093 |
| median | 2.67 |
| std | 1.4804 |
| min | 1.0 |
| max | 5.0 |

## Spearman correlation vs. human_score

| Source | rho | p-value | n |
|---|---|---|---|
| voting (panel mean) | 0.7443 | 1.24269e-159 | 900 |
| G-Eval baseline (`gpt-4o`) | 0.7565 | — | 900 |
| judge_openai (individual) | 0.7131 | 1.222e-140 | 900 |
| judge_google (individual) | 0.6777 | 4.83982e-122 | 900 |
| judge_anthropic (individual) | 0.6935 | 5.06211e-130 | 900 |

## Breakdown by model family

| Family | n | mean | median | std | min | max |
|---|---|---|---|---|---|---|
| GPT2 | 300 | 3.2433 | 3.165 | 1.37 | 1.0 | 5.0 |
| HRED | 95 | 2.5827 | 2.33 | 1.2952 | 1.0 | 5.0 |
| S2S | 200 | 2.44 | 2.0 | 1.3671 | 1.0 | 5.0 |
| VHRED | 105 | 2.2633 | 1.67 | 1.1675 | 1.0 | 5.0 |
| ground-truth | 100 | 4.5201 | 5.0 | 0.7592 | 1.67 | 5.0 |
| negative-sample | 100 | 1.3235 | 1.0 | 0.6476 | 1.0 | 4.67 |

## Panel agreement levels

| Level | Count | Percentage |
|---|---|---|
| high | 352 | 39.11% |
| medium | 398 | 44.22% |
| low | 150 | 16.67% |
| n/a | 0 | 0.00% |
