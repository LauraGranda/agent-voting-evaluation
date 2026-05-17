# G-Eval Run — Analysis Report

_Generated: 2026-05-17T23:49:40.887768+00:00_

## 1. Executive summary

G-Eval (gpt-4o) scored **900 of 900** conversation-response pairs (0 failed) for relevance against four human annotators.

- **Rank agreement with humans**: Spearman ρ = **0.756** (_strong_, p = 8.54e-168).
- **Human ceiling**: a single annotator predicts the 4-rater consensus at ρ = 0.580; G-Eval's ρ = 0.756 is +0.176 vs. that ceiling.
- **Linear agreement**: Pearson r = 0.711; Kendall τ = 0.571.
- **Error magnitude**: MAE = 0.90, RMSE = 1.14 points on the 1-5 scale.
- **Systematic bias**: mean Δ = -0.75 — G-Eval **under-rates** responses relative to humans (G-Eval mean 2.41 vs. human 3.16).
- **Tolerance hit-rate**: 36% of pairs within ±0.5, 63% within ±1.0 of the human score.

## 2. Agreement metrics

| Metric | Value | Reading |
|---|---|---|
| Pairs analysed (n) | 900 | successful evaluations |
| Spearman ρ | 0.7565 | rank correlation (strong) |
| Spearman p-value | 8.54e-168 | significance |
| Pearson r | 0.7106 | linear correlation |
| Kendall τ | 0.5705 | concordance |
| MAE | 0.9008 | mean absolute error (1-5 pts) |
| RMSE | 1.1434 | penalises large misses |
| Mean bias (Δ) | -0.7495 | + = G-Eval scores high |
| Within ±0.5 | 35.9% | tight agreement |
| Within ±1.0 | 62.6% | loose agreement |

## 3. Human ceiling — inter-annotator agreement

Spearman ρ can only be judged against how well the humans agree **with each other**. A metric cannot be expected to beat the noise floor of the labels it is graded on.

| Reference | ρ / coefficient | Meaning |
|---|---|---|
| Two single humans (pairwise ρ) | 0.4660 | one annotator vs. another |
| **Human ceiling (leave-one-out ρ)** | **0.5803** | one annotator vs. the mean of the other 3 |
| **G-Eval vs. consensus** | **0.7565** | one G-Eval pass vs. the 4-human mean |
| Krippendorff α (ordinal) | 0.4655 | standard reliability coefficient |
| ICC(2,1) single rater | 0.4661 | absolute-agreement reliability of one rating |

**Gap to ceiling: +0.1762.** G-Eval **exceeds the single-human ceiling** — it tracks the 4-rater consensus more closely than an individual annotator does. This is the expected outcome when the human labels are noisy: a *consistent* rater correlates with the average better than the noisy individuals correlate with each other. It does **not** mean G-Eval is 'better than humans' — the consensus is the target by construction — but it does mean a stronger evaluator model would buy essentially nothing: the gap left to ρ = 1.0 is mostly irreducible annotation noise, not model weakness.

## 4. Breakdown by model family

Families ordered by human-rated relevance (best first). `bias` is the mean G-Eval − human gap within the family.

| Family | n | Human mean | G-Eval mean | Bias | MAE |
|---|---|---|---|---|---|
| ground-truth | 100 | 4.38 | 3.74 | -0.64 | 0.84 |
| GPT2 | 300 | 3.60 | 2.69 | -0.91 | 1.06 |
| VHRED | 105 | 2.83 | 2.02 | -0.81 | 0.96 |
| HRED | 95 | 2.79 | 2.24 | -0.55 | 0.74 |
| S2S | 200 | 2.77 | 2.10 | -0.66 | 0.81 |
| negative-sample | 100 | 2.08 | 1.41 | -0.67 | 0.75 |

## 5. Breakdown by individual model

| Model | n | Human mean | G-Eval mean | Bias | MAE |
|---|---|---|---|---|---|
| ground-truth | 100 | 4.38 | 3.74 | -0.64 | 0.84 |
| GPT2_medium greedy_temp1.0_k0_p0.0 | 33 | 4.09 | 3.11 | -0.98 | 1.14 |
| GPT2_small top_temp1.0_k0_p0.5 | 37 | 3.91 | 3.01 | -0.91 | 1.07 |
| GPT2_medium top_temp1.0_k0_p0.5 | 46 | 3.79 | 3.00 | -0.78 | 0.97 |
| GPT2_small greedy_temp1.0_k0_p0.0 | 44 | 3.72 | 2.69 | -1.03 | 1.17 |
| GPT2_small top_temp1.0_k0_p0.9 | 52 | 3.38 | 2.49 | -0.89 | 1.07 |
| GPT2_medium sample_temp1.0_k0_p0.0 | 41 | 3.27 | 2.28 | -0.99 | 1.01 |
| HRED_attn top_temp1.0_k0_p0.5 | 32 | 3.26 | 2.65 | -0.61 | 0.82 |
| GPT2_small sample_temp1.0_k0_p0.0 | 47 | 3.26 | 2.41 | -0.84 | 1.02 |
| VHRED_attn greedy_temp1.0_k0_p0.0 | 37 | 3.07 | 2.27 | -0.79 | 1.02 |
| VHRED_attn top_temp1.0_k0_p0.5 | 26 | 2.96 | 2.15 | -0.82 | 0.95 |
| S2S_attn greedy_temp1.0_k0_p0.0 | 38 | 2.96 | 2.33 | -0.64 | 0.78 |
| S2S greedy_temp1.0_k0_p0.0 | 43 | 2.94 | 2.30 | -0.64 | 0.79 |
| S2S_attn top_temp1.0_k0_p0.5 | 35 | 2.89 | 2.06 | -0.83 | 0.90 |
| S2S top_temp1.0_k0_p0.5 | 32 | 2.87 | 2.26 | -0.60 | 0.82 |
| HRED_attn greedy_temp1.0_k0_p0.0 | 34 | 2.82 | 2.23 | -0.58 | 0.73 |
| S2S_attn sample_temp1.0_k0_p0.0 | 23 | 2.65 | 1.91 | -0.74 | 0.90 |
| VHRED_attn sample_temp1.0_k0_p0.0 | 42 | 2.53 | 1.71 | -0.82 | 0.92 |
| HRED_attn sample_temp1.0_k0_p0.0 | 29 | 2.23 | 1.79 | -0.44 | 0.66 |
| S2S sample_temp1.0_k0_p0.0 | 29 | 2.09 | 1.56 | -0.52 | 0.68 |
| negative-sample | 100 | 2.08 | 1.41 | -0.67 | 0.75 |

## 6. Score-band confusion matrix

Both scores rounded to the nearest integer (1-5). Exact-band agreement: **35.6%**. Rows = human, columns = G-Eval.

| Human ↓ / G-Eval → | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **5** | 0 | 23 | 20 | 51 | 22 |
| **4** | 5 | 140 | 67 | 75 | 18 |
| **3** | 17 | 104 | 31 | 11 | 0 |
| **2** | 90 | 142 | 8 | 5 | 0 |
| **1** | 50 | 19 | 2 | 0 | 0 |

## 7. Largest disagreements

### G-Eval over-rated (top 10 positive Δ)

| conversation_id | model | human | G-Eval | Δ |
|---|---|---|---|---|
| conv_43_VHRED_attn greedy_temp1.0_k0_p0.0 | VHRED | 1.00 | 3.23 | +2.23 |
| conv_18_GPT2_small top_temp1.0_k0_p0.9 | GPT2 | 1.00 | 2.85 | +1.85 |
| conv_5_S2S top_temp1.0_k0_p0.5 | S2S | 2.00 | 3.77 | +1.77 |
| conv_98_GPT2_medium greedy_temp1.0_k0_p0.0 | GPT2 | 2.75 | 4.28 | +1.53 |
| conv_66_GPT2_small top_temp1.0_k0_p0.5 | GPT2 | 2.50 | 4.01 | +1.51 |
| conv_45_ground-truth | ground-truth | 3.00 | 4.35 | +1.35 |
| conv_55_VHRED_attn greedy_temp1.0_k0_p0.0 | VHRED | 1.00 | 2.35 | +1.35 |
| conv_60_VHRED_attn top_temp1.0_k0_p0.5 | VHRED | 2.25 | 3.51 | +1.26 |
| conv_89_ground-truth | ground-truth | 2.50 | 3.72 | +1.22 |
| conv_41_ground-truth | ground-truth | 3.75 | 4.89 | +1.14 |

### G-Eval under-rated (top 10 negative Δ)

| conversation_id | model | human | G-Eval | Δ |
|---|---|---|---|---|
| conv_75_GPT2_small greedy_temp1.0_k0_p0.0 | GPT2 | 4.75 | 1.65 | -3.10 |
| conv_9_GPT2_small top_temp1.0_k0_p0.9 | GPT2 | 5.00 | 1.99 | -3.01 |
| conv_75_VHRED_attn greedy_temp1.0_k0_p0.0 | VHRED | 4.75 | 1.74 | -3.01 |
| conv_37_GPT2_medium greedy_temp1.0_k0_p0.0 | GPT2 | 4.75 | 1.78 | -2.97 |
| conv_62_GPT2_medium greedy_temp1.0_k0_p0.0 | GPT2 | 4.50 | 1.54 | -2.96 |
| conv_30_GPT2_small top_temp1.0_k0_p0.9 | GPT2 | 4.75 | 1.86 | -2.89 |
| conv_12_GPT2_small greedy_temp1.0_k0_p0.0 | GPT2 | 5.00 | 2.11 | -2.89 |
| conv_37_GPT2_small greedy_temp1.0_k0_p0.0 | GPT2 | 4.50 | 1.71 | -2.79 |
| conv_67_ground-truth | ground-truth | 4.75 | 1.96 | -2.79 |
| conv_36_HRED_attn top_temp1.0_k0_p0.5 | HRED | 4.75 | 1.97 | -2.78 |

## 8. Figures

![G-Eval vs human](outputs/figures/05_geval_vs_human_scatter.png)

![Residual histogram](outputs/figures/06_residual_histogram.png)

![Residuals by family](outputs/figures/07_delta_by_family_boxplot.png)

![Mean score by family](outputs/figures/08_mean_score_by_family.png)

![G-Eval vs human ceiling](outputs/figures/09_ceiling_comparison.png)

## 9. How to read this

- The ρ of 0.76 should be read against the 0.58 human ceiling, not against 1.0 — no metric can out-agree the label noise it is graded on.
- A Spearman ρ of 0.76 means G-Eval reproduces the human **ranking** of responses strongly — useful for picking the better of two responses even when absolute points differ.
- The -0.75 bias is a *calibration* offset: it can be subtracted out before comparing against the 1-5 human scale.
- Families where `bias` and `MAE` are largest are where the metric is least trustworthy and most worth a prompt revision.
