# Cost Analysis Summary -- G-Eval vs Voting (n=900)

Generado automáticamente por `notebooks/08_cost_analysis.ipynb` (HU-14). Cero llamadas a APIs.

## 1. Agregados globales

| Métrica | G-Eval | Voting | Ratio V/G |
|---|---|---|---|
| n evaluaciones | 900 | 900 | 1.00 |
| Costo total USD | 3.5453 | 8.1547 | 2.30 |
| Tokens input | 1,090,187 | 3,556,386 | 3.26 |
| Tokens output | 81,977 | 666,808 | 8.13 |
| Tokens total | 1,172,164 | 4,223,194 | 3.60 |
| Costo por pair USD | 0.00394 | 0.00906 | 2.30 |
| Tokens por pair | 1302.4 | 4692.4 | 3.60 |

## 2. Desglose por agente del voting

| Agente | Costo USD | % del total | USD por pair |
|---|---|---|---|
| judge_openai | 4.3620 | 53.5 | 0.00485 |
| judge_google | 0.7426 | 9.1 | 0.00083 |
| judge_anthropic | 3.0501 | 37.4 | 0.00339 |
| **Total voting** | **8.1547** | **100.0** | **0.00906** |

## 3. Tiempo de ejecución (parseado de logs)

| Métrica | G-Eval | Voting | Ratio V/G |
|---|---|---|---|
| Wall time total | 2241.1s (37.4 min) | 9414.0s (156.9 min) | 4.20 |
| Segundos por evaluación | 2.49s | 10.46s | 4.20 |
| Throughput (eval/min) | 24.10 | 5.74 | 0.24 |

Fuente: `outputs/logs/geval_execution.log` y `outputs/logs/voting_execution.log`. El wall time del voting incluye reintentos por errores 503 transitorios del proveedor Google.

## 4. Trade-off costo vs calidad

| Métrica | G-Eval | Voting | Δ (V − G) | Costo marginal |
|---|---|---|---|---|
| Costo total USD | 3.5453 | 8.1547 | +4.6095 | — |
| Spearman ρ | 0.756 | 0.744 | -0.012 | — (equivalencia TOST) |
| Cohen's κ ponderado | 0.525 | 0.643 | +0.118 | USD 39.06 por +0,001 κ |
| Exact-agreement (%) | 35.89 | 44.33 | +8.44 pp | USD 0.55 por +1 pp |

## 5. Conclusión

El sistema de votación cuesta **2.30 veces** lo que cuesta G-Eval (USD 8.15 vs USD 3.55) y tarda **4.2 veces** más en ejecutar. Dado que HU-12 estableció equivalencia estadística formal entre ambos métodos en ranking (TOST) y exactitud (Wilcoxon), el dólar adicional **no compra mejora inferencial** pero sí mejora descriptiva: κ ponderado +0.118, exact-agreement +8.44 pp, y MAE 40 % menor en estrato 3 (HU-10/13).

**Recomendación**: G-Eval para producción a escala donde el ranking es la métrica central; voting cuando importan calibración absoluta, robustez de proveedor y precisión en alta calidad. Para evaluación de investigación con n ≤ 1000, el costo absoluto de ambos métodos es bajo (< USD 10) y conviene priorizar voting por sus ventajas descriptivas.

## 6. Limitaciones

- Precios spot de los días de ejecución (G-Eval 17/05/2026; voting 07/06/2026). Cambios futuros en tarifas de proveedores invalidan los ratios.
- Wall time del voting incluye reintentos por errores 503 transitorios de gemini-2.5-flash.
- No se contabiliza overhead operativo (CI, monitoreo, almacenamiento).
- No se compara contra costo humano de anotación MTurk (no expuesto por el dataset).
