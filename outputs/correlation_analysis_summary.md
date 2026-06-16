## Correlation Analysis: G-Eval vs Voting System (n=900)

### Spearman ρ vs Human Annotations

| Method | ρ | 95% CI (Fisher Z) | 95% CI (Bootstrap) | p-value |
|---|---|---|---|---|
| G-Eval (gpt-4o, V3) | 0.756 | [0.727, 0.783] | [0.725, 0.784] | 8.54e-168 |
| Voting System | 0.744 | [0.714, 0.772] | [0.713, 0.772] | 1.24e-159 |

### Pearson r vs Human Annotations

| Method | r | 95% CI (Fisher Z) | p-value |
|---|---|---|---|
| G-Eval (gpt-4o, V3) | 0.711 | [0.677, 0.742] | 2.87e-139 |
| Voting System | 0.733 | [0.701, 0.761] | 3.87e-152 |

### Observed Difference

| Metric | G-Eval | Voting | Δ (G-Eval − Voting) |
|---|---|---|---|
| Spearman ρ | 0.756 | 0.744 | +0.012 |
| Pearson r | 0.711 | 0.733 | -0.022 |

> **Nota**: La significancia estadística del Δρ se prueba en `notebooks/06_significance_tests.ipynb` (HU-12). Este notebook reporta el Δ observado pero no concluye sobre su significancia.
