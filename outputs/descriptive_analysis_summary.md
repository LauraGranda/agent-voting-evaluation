## Summary: G-Eval vs Voting System vs Human Scores

### Descriptive Statistics

| Metric    | Human | G-Eval | Voting | OpenAI | Google | Anthropic |
|---|---|---|---|---|---|---|
| Mean | 3.158 | 2.408 | 2.809 | 2.534 | 3.091 | 2.802 |
| Median | 3.250 | 2.036 | 2.670 | 2.000 | 3.000 | 2.000 |
| Std | 1.186 | 1.066 | 1.481 | 1.658 | 1.700 | 1.411 |
| Min | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Max | 5.000 | 4.994 | 5.000 | 5.000 | 5.000 | 5.000 |
| Range | 4.000 | 3.994 | 4.000 | 4.000 | 4.000 | 4.000 |
| Q1 | 2.250 | 1.665 | 1.330 | 1.000 | 1.000 | 2.000 |
| Q3 | 4.250 | 3.067 | 4.330 | 4.250 | 5.000 | 4.000 |
| IQR | 2.000 | 1.402 | 3.000 | 3.250 | 4.000 | 2.000 |
| CV (%) | 37.566 | 44.275 | 52.725 | 65.400 | 55.001 | 50.366 |
| Skewness | -0.162 | 0.791 | 0.218 | 0.489 | -0.004 | 0.258 |
| Kurtosis | -1.156 | -0.499 | -1.466 | -1.410 | -1.690 | -1.283 |

### Correlation with Human Annotations (Spearman ρ)

| Method | ρ | p-value | Interpretation |
|---|---|---|---|
| G-Eval (gpt-4o) | 0.756 | 8.54e-168 | strong |
| Voting System | 0.744 | 1.24e-159 | strong |
| Agent OpenAI | 0.713 | 1.22e-140 | strong |
| Agent Google | 0.678 | 4.84e-122 | moderate |
| Agent Anthropic | 0.694 | 5.06e-130 | moderate |

### Inter-Rater Agreement

| Comparison | Weighted κ | Interpretation |
|---|---|---|
| Human vs G-Eval | 0.525 | moderate |
| Human vs Voting | 0.643 | substantial |
| G-Eval vs Voting | 0.733 | substantial |

Krippendorff α (3 raters, ordinal): **0.632** (substantial)

### Exact Agreement Rate (|Δ| ≤ 0.5)

| Comparison | Rate (%) |
|---|---|
| Human vs G-Eval | 35.89 |
| Human vs Voting | 44.33 |
| G-Eval vs Voting | 50.89 |
