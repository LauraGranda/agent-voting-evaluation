# Prompt Iterations — G-Eval Relevance Pilot

Generated: 2026-04-22T04:49:09.851259+00:00
Evaluator model: `gpt-4o` — direct successor of GPT-4 (Liu et al., EMNLP 2023).

Each version was piloted on the same stratified 20-entry sample (configs/prompts/pilot_sample.json), seed=42.

## Version V1 — Generic baseline
- Model: `gpt-4o`
- Prompt file: `configs/prompts/v1_generic.txt`
- Changes from previous: Initial generic baseline.
- Pilot Spearman ρ: **0.937** (p = 0.0000)
- Mean |Δ|: 0.918   Max |Δ|: 2.841
- Key observations: Largest delta on stratum 3 (mean |Δ|=1.87); smallest delta on stratum 2 (mean |Δ|=0.16).
- Decision: **Rejected**

## Version V2 — Dialogue-aware CoT
- Model: `gpt-4o`
- Prompt file: `configs/prompts/v2_dialogue_cot.txt`
- Changes from previous: Added multi-turn task framing, 3-part relevance definition, and 5-step chain-of-thought.
- Pilot Spearman ρ: **0.904** (p = 0.0000)
- Mean |Δ|: 0.824   Max |Δ|: 2.887
- Key observations: Largest delta on stratum 3 (mean |Δ|=1.56); smallest delta on stratum 2 (mean |Δ|=0.20).
- Decision: **Rejected**

## Version V3 — Full CoT + anchored
- Model: `gpt-4o`
- Prompt file: `configs/prompts/v3_full_cot_anchored.txt`
- Changes from previous: Kept V2 structure; added anchored per-score rubric and 3 worked examples drawn from the pilot sample.
- Pilot Spearman ρ: **0.899** (p = 0.0000)
- Mean |Δ|: 0.848   Max |Δ|: 2.940
- Key observations: Largest delta on stratum 3 (mean |Δ|=1.67); smallest delta on stratum 2 (mean |Δ|=0.15).
- Decision: **Selected**

## Análisis Estadístico de Diferencias entre Versiones

Prueba de Steiger (1980) sobre muestra piloto (n=20):

| Comparación | ρ_A | ρ_B | Z | p-value | Conclusión |
|---|---|---|---|---|---|
| V1 vs V2 | 0.937 | 0.904 | 0.649 | 0.516 | NO significativa |
| V1 vs V3 | 0.937 | 0.899 | 0.720 | 0.471 | NO significativa |
| V2 vs V3 | 0.904 | 0.899 | 0.071 | 0.943 | NO significativa |

Interpretación: Con n=20 y Δρ_max < 0.05, ninguna versión demuestra superioridad estadística. La selección se basa en criterios metodológicos.

## Selección Final

**Versión seleccionada: V3 — Full CoT con rúbrica anclada**

**Modelo evaluador: gpt-4o**

**Fecha: 2026-04-22T04:49:09.851655+00:00**

Justificación: Ante ausencia de diferencias estadísticamente significativas entre versiones (prueba de Steiger, todos p > 0.05), se seleccionó V3 por criterios metodológicos: CoT auditable, rúbrica anclada que reduce varianza, y mayor fidelidad al diseño original de G-Eval (Liu et al., 2023).

Nota metodológica: gpt-4o seleccionado como sucesor directo de GPT-4 (Liu et al., 2023), manteniendo comparabilidad con baselines publicados (SummEval relevancia ρ=0.547).
