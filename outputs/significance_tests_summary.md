## Resumen de Pruebas de Significancia (n=900)

### 1. Exactitud (|error| vs humano) — TEST PRINCIPAL

| Test | Estadístico | p-value | p Holm | α=0.05 (Holm) | Tamaño efecto |
|---|---|---|---|---|---|
| Wilcoxon signed-rank (primario) | W = 191287.0000 | 1.9364e-01 | 5.8091e-01 | No rechaza H0 | r = +0.0501 (negligible) |
| t pareado (complementario) | t = +2.8943 | 3.8914e-03 | 1.5565e-02 | Rechaza H0 | d = +0.0965 (negligible) |

### 2. Sesgo inter-método (geval − voting) — DIAGNÓSTICO

| Test | Estadístico | p-value | p Holm | α=0.05 (Holm) | Tamaño efecto |
|---|---|---|---|---|---|
| Wilcoxon signed-rank | W = 104126.0000 | 5.1540e-36 | 2.5770e-35 | Rechaza H0 | r = -0.4829 (medium) |

> Diagnóstico: voting puntúa +0.401 sobre G-Eval en promedio. **Esto es sesgo de escala, no exactitud** (el test de exactitud está arriba).

### 3. Ranking (Δρ vs humano) — STEIGER OVERLAPPING + BOOTSTRAP

| Método | Δρ observado | CI 95% | p-value | p Holm | α=0.05 (Holm) |
|---|---|---|---|---|---|
| Bootstrap pareado | +0.0122 | [-0.0096, +0.0334] | 0.2632 | 0.2632 | No rechaza H0 |
| Steiger overlapping | +0.0122 | — | 0.2094 | 0.4188 | No rechaza H0 |

> Correlación entre estimadores: **r_GV (Spearman) = 0.897** (alta, esperable por compartir prompt V3 y `gpt-4o` como juez). La fórmula correcta del Steiger overlapping incorpora esta correlación.

### 4. Equivalencia formal (TOST)

| Test | Región | 90% CI | Conclusión |
|---|---|---|---|
| TOST sobre Δρ | ±0.05 | [-0.0059, +0.0298] | **EQUIVALENCIA AFIRMADA** |

### 5. Diagnósticos

| Test | Estadístico | p-value | Interpretación |
|---|---|---|---|
| Shapiro-Wilk sobre `diff_abs` | W = 0.9688 | 5.9117e-13 | No normal (esperado con n grande) |
| Skewness de `diff_abs` | +0.380 | z-skew=+4.66 | Asimétrica |
