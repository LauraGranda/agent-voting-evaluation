# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Análisis de costo G-Eval vs Voting sobre 900 pares (HU-14)** - Quinto notebook del bloque de análisis. Cuantifica costo en USD, tokens y wall time, y los enmarca como trade-off frente a la equivalencia estadística formal de HU-12. **Cero llamadas a APIs**:
    - `notebooks/08_cost_analysis.ipynb` - 11 celdas ejecutadas end-to-end con 900/900 entradas. Lee costos desde `outputs/geval_results.json` y `outputs/voting_results.json`, parsea wall time desde `outputs/logs/{geval,voting}_execution.log` (usando el último match del summary final), y lee ρ/κ/exact-agreement desde los summaries existentes de HU-10 y HU-11 vía regex — ningún número hardcoded.
    - **Figura 20** `20_cost_breakdown.png` (150 dpi): barra apilada del costo total con stack por agente del voting + distribución del costo por evaluación (histograma comparativo).
    - **Figura 21** `21_cost_vs_quality_tradeoff.png` (150 dpi): scatter de los dos métodos en tres planos (ρ, κ, exact-agreement) vs costo total USD.
    - **Summary persistido** en `outputs/cost_analysis_summary.md` con 6 secciones: agregados globales, desglose por agente, tiempo de ejecución parseado, trade-off costo vs calidad con costo marginal por punto de métrica, conclusión y limitaciones.
    - **Resultados clave (n=900)**:
        - **Costo total**: G-Eval USD 3,5453 vs Voting USD 8,1547 → ratio voting/G-Eval = **2,30×**.
        - **Tokens**: G-Eval 1 172 164 vs Voting 4 223 194 → ratio 3,60×.
        - **Wall time**: G-Eval 2 241 s (37,4 min) vs Voting 9 414 s (156,9 min) → ratio 4,20×; throughput 24,1 vs 5,7 eval/min.
        - **Desglose voting por agente**: `judge_openai` USD 4,3620 (53,5 %); `judge_anthropic` USD 3,0501 (37,4 %); `judge_google` USD 0,7426 (9,1 %). Desbalance porque `gemini-2.5-flash` es sustancialmente más barato por token.
        - **Costo marginal**: USD 39,06 por +0,001 κ ponderado; USD 0,55 por +1 pp de exact-agreement; **indefinido para ρ** porque HU-12 estableció equivalencia formal (Δρ no significativo).
    - **Conclusión integrada (citable en la tesis)**: dado que HU-12 estableció equivalencia inferencial entre los dos métodos, el dólar adicional del voting **no compra mejora en ranking** pero sí mejora descriptiva sustantiva (κ +0,118, exact-agreement +8,4 pp, MAE 40 % menor en estrato 3). Recomendación condicionada: G-Eval para producción a escala donde ranking es la métrica central; voting cuando importan calibración absoluta, robustez de proveedor y precisión en alta calidad.

- **Análisis de errores G-Eval vs Voting sobre 900 pares + documento de limitaciones (HU-13)** - Cuarto y último notebook del bloque de análisis. Identifica patrones cualitativos de fallo que la equivalencia estadística global (HU-12) oculta. **Cero llamadas a APIs**:
    - `notebooks/07_error_analysis.ipynb` - 14 celdas ejecutadas end-to-end con 900/900 entradas alineadas. Distribución de errores con signo, top-20 divergencias por método, matriz de coocurrencia de errores severos, MAE por estrato, análisis de desacuerdo inter-agente, taxonomía emergente apoyada en datos y 5 case studies con texto real de `turns[]` y `response`. Narrativa en español, identificadores en inglés. `np.random.seed(42)`.
    - **`docs/limitations.md`** (NUEVO, 1 577 palabras, 5 secciones): limitaciones del dataset DailyDialog–Zhao, específicas de G-Eval, específicas del sistema de votación, compartidas, e implicaciones para investigación futura. Cada limitación cita evidencia numérica concreta de los notebooks 04, 06, 07.
    - **Figura 17** `17_error_distribution.png` (150 dpi): histogramas superpuestos del error con signo y del error absoluto, con MAE y umbral severo señalizados.
    - **Figura 18** `18_divergence_by_stratum.png` (150 dpi): boxplots de `|error|` por estrato/método + heatmap del sesgo medio por estrato.
    - **Figura 19** `19_inter_agent_disagreement.png` (150 dpi): scatter `std_judges` vs `|err_voting|` coloreado por estrato + distribución de `std_judges` por estrato.
    - **Summary persistido** en `outputs/error_analysis_summary.md` con 7 secciones: métricas globales, MAE/sesgo por estrato, matriz de errores severos, top-20 G-Eval, top-20 Voting, taxonomía y 5 case studies.
    - **Hallazgos centrales (n=900)** — la equivalencia global esconde fallos asimétricos:
        - **Sesgo de subestimación**: G-Eval sesgo medio = −0,750 (58,7 % de casos por debajo del humano); Voting sesgo medio = −0,348 (39,4 %). Ambos métodos son **más estrictos** que los humanos MTurk, contradiciendo la intuición de "LLM judges generosos".
        - **Estrato 3 (IA alta, n = 216)** confirmado como el régimen crítico: G-Eval MAE = 1,286 vs Voting MAE = 0,770; sesgo G-Eval = −1,228; **16 de los 20** casos del top-20 de G-Eval caen aquí.
        - **Estrato 5 (IA baja, n = 159)** único régimen donde G-Eval supera a voting: MAE 0,372 vs 0,595.
        - **Co-localización parcial**: ρ(|err_geval|, |err_voting|) = +0,420; 91 casos con fallo severo simultáneo, 99 solo G-Eval, 84 solo voting, 626 ambos correctos. 183 casos (20 %) son rescatados por exactamente uno de los dos métodos.
        - **`std_judges` no es señal útil de incertidumbre**: ρ(std_judges, |err_voting|) = +0,013, p = 0,705. La hipótesis "mayor desacuerdo inter-juez = mayor error vs humano" queda **rechazada por los datos**.
    - **Taxonomía de errores en 5 categorías derivadas de los datos**: (A) subestimación severa en estrato 3, (B) sesgo de subestimación global más fuerte en G-Eval, (C) estrato 5 como excepción a favor de G-Eval, (D) fallas correlacionadas en bloque del panel, (E) co-localización parcial de fallos cerrada con McNemar.
    - **Cuatro sesgos clásicos del criterio académico abordados explícitamente** (HU-13, Cell 9b): verbosidad (G-Eval ρ = −0,003 NS; voting ρ = −0,175 *** con efecto pequeño anti-verbosidad), posicional (N/A por diseño pointwise), ambigüedad de contexto (G-Eval ρ = +0,109, voting ρ = +0,377 sobre `std(raw_human_scores)` como proxy de ambigüedad — voting más sensible al ruido del gold standard), auto-preferencia de familia (ρ ∈ {0,499; 0,534; 0,568} entre errores absolutos de los tres jueces individuales).
    - **Adjudicación manual del top-20 con regla mecánica** (HU-13, Cell 13): 2/20 etiqueta dudosa en G-Eval (18/20 error genuino del método), 8/20 etiqueta dudosa en voting (12/20 error genuino). La asimetría refuerza la conclusión a favor de voting: una fracción importante de sus peores casos es ruido del gold standard MTurk, no fallo del método.
    - **McNemar sobre la matriz de errores severos** (HU-13, Cell 7): χ² = 1,07, p = 0,30 — ningún método comete significativamente más errores severos que el otro, cierre inferencial coherente con la equivalencia formal de HU-12.
    - **Reframing de la categoría D** (HU-13, Cell 10): se reemplaza "hipótesis RECHAZADA" por "no se encuentra evidencia de asociación con caveat de potencia" (std_judges toma sólo 4 valores discretos). El insight central pasa a ser: cuando el panel falla, falla en bloque (case studies 3 y 4 con std = 0). Las correlaciones de errores entre jueces individuales (ρ ≈ 0,5) cuantifican la limitación arquitectónica del enfoque de panel — tres jueces que comparten puntos ciegos del paradigma LLM no aportan tres muestras independientes.

## [0.7.0] - 2026-06-17

### Added

- **Pruebas de significancia estadística G-Eval vs Voting sobre 900 pares (HU-12)** - Tercer notebook del bloque de análisis, precedido del documento de justificación metodológica. **Cero llamadas a APIs**:
    - **`docs/significance_tests_justification.md`** (NUEVO) - Documento de investigación con 8 secciones que justifica la selección de tests sobre el diseño pareado y ordinal del estudio: naturaleza de los datos (§1), Steiger overlapping con `r_GV` (§2), Shapiro–Wilk con n grande (§3), Wilcoxon vs t pareado (§4), effect sizes incluyendo rank-biserial directo (§5), comparativa de pruebas alternativas (§6), comparaciones múltiples Holm (§7) y referencias verificadas con DOIs/URLs (§8). 12 fuentes citadas, todas verificadas contra su publicación de origen.
    - **`notebooks/06_significance_tests.ipynb`** - 14 celdas con 5 tests principales sobre la cantidad correcta para cada pregunta: Wilcoxon principal y t pareado complementario sobre `|err_g|` vs `|err_v|` (exactitud); Wilcoxon sobre `geval − voting` como diagnóstico de sesgo inter-método; bootstrap pareado y Steiger overlapping sobre Δρ (ranking); TOST con margen ±0,05 sobre Δρ (equivalencia formal). Corrección Holm-Bonferroni sobre los 5 p-valores. ρ exactamente reproducibles vs HU-11 (0,756 / 0,744). `np.random.seed(42)`.
    - **`steiger_overlapping(r₁, r₂, r_GV, n)`** implementación de la fórmula correcta del Steiger (1980) para correlaciones dependientes que comparten una variable. El SE depende explícitamente de `r_GV = corr(geval, voting) = 0,897`.
    - **`rank_biserial_signed(diff)`** computa el effect size del Wilcoxon directamente desde los rangos con signo (sin circular desde el p-valor) y reporta el N efectivo (pares no nulos).
    - **TOST sobre Δρ** con región de equivalencia predefinida ±0,05 (Schuirmann, 1987): el 90% CI bootstrap cabe íntegramente dentro, **afirma equivalencia formal** en ranking — declaración positiva, no solo ausencia de evidencia.
    - **Figura 15** `15_normality_check.png` (150 dpi): histograma + Q-Q plot de `diff_abs` con análisis de simetría (skewness, z-skew).
    - **Figura 16** `16_bootstrap_delta_rho.png` (150 dpi): distribución bootstrap de Δρ con CI 95% y región TOST sombreada.
    - **Summary persistido** en `outputs/significance_tests_summary.md` con 5 sub-tablas (exactitud, sesgo inter-método, ranking, equivalencia TOST, diagnósticos).
    - **Resultados clave (n=900)** — **prácticamente equivalentes** en lo que importa para aproximación al humano:
        - **Exactitud, Wilcoxon principal sobre `|error|`**: W=191.287, p=0,194, r=+0,050 (**negligible**). **No rechaza** H0.
        - **Exactitud, t pareado complementario**: t=+2,89, p=0,004 (p Holm=0,016), Cohen's d=+0,097 (**negligible**). Rechaza H0 pero con effect size irrelevante — discrepancia legítima por asimetría (skew=+0,38).
        - **Ranking, Bootstrap pareado de Δρ**: Δρ=+0,012, CI 95% [−0,010, +0,033], p=0,263. No rechaza.
        - **Ranking, Steiger overlapping**: Z=+1,26, p=0,209. La fórmula correcta depende de `r_GV=0,897` (alta correlación entre estimadores que comparten prompt V3 y `gpt-4o`).
        - **TOST sobre Δρ ±0,05**: 90% CI bootstrap [−0,006, +0,030] ⊂ [−0,05, +0,05]. **Equivalencia formal afirmada**.
        - **Sesgo inter-método, Wilcoxon sobre `geval − voting`**: W=104.126, p<10⁻³⁵, r=−0,48 (medium). Rechaza H0: voting puntúa +0,40 sobre G-Eval sistemáticamente. **No es exactitud** (`geval − voting` es algebraicamente independiente del humano).
        - **Diagnósticos**: Shapiro-Wilk W=0,969, p=5,9×10⁻¹³ (no normal, esperado con n grande); skewness +0,38, asimétrica.
    - **Lectura integrada del bloque (HU-10 + HU-11 + HU-12)**: G-Eval y voting son **estadísticamente equivalentes** en ranking (TOST formal con margen ±0,05) y en exactitud (effect size negligible). Difieren significativamente en sesgo de escala con effect size medium pero esto es descriptivo, no ventaja de exactitud. El voting conserva ventajas descriptivas sustantivas (Cohen's κ 0,643 vs 0,525, exact-agreement 44% vs 36%, MAE en Estrato 3 de IA alta 40% menor) más diversidad de proveedor. **Recomendación**: el sistema de votación, fundamentado en equivalencia formal en ranking y exactitud + ventajas descriptivas + robustez operativa, no en superioridad inferencial.

- **Análisis de correlación con CIs sobre 900 pares (HU-11)** - Segundo notebook del bloque de análisis. Cuantifica con qué fuerza cada método automático correlaciona con las anotaciones humanas, reportando intervalos de confianza al 95 % por dos vías independientes y contrastando contra los baselines publicados por Liu et al. (2023). **Cero llamadas a APIs**:
    - `notebooks/05_correlation_analysis.ipynb` - 12 celdas ejecutadas end-to-end con 900/900 entradas alineadas, ρ exactamente reproducible respecto a HU-10 (0.756 / 0.744). Narrativa y comentarios en español; identificadores técnicos en inglés.
    - **CIs de Fisher Z analíticos** (Fisher, 1915) implementados inline en `fisher_z_ci(rho, n)` — referencia de la literatura. Reusable para Spearman ρ y Pearson r.
    - **CIs bootstrap percentil** (`n_iter=10_000`, `seed=42`) en `bootstrap_spearman_ci()` como validación no paramétrica. **Match con Fisher Z confirmado** dentro de `|Δ| < 0.002` en ambos métodos.
    - **Pearson r** complementario con sus CIs: G-Eval r=0.711 [0.677, 0.742], Voting r=0.733 [0.701, 0.761] — **el orden se invierte respecto a Spearman**, hallazgo que documenta la prosa de la celda interpretativa.
    - **Figura 14** `outputs/figures/14_correlation_ci_plot.png` (150 dpi): dot-plot con error-bars del CI 95 % y líneas de referencia para los baselines publicados (SummEval relevance 0.547, Topical-Chat coherence 0.605).
    - **Summary table persistida** en `outputs/correlation_analysis_summary.md` con tres sub-tablas (Spearman + CIs por dos vías, Pearson + CI, Δ observado) y nota de cross-reference a HU-12 para significancia.
    - **Sección interpretativa académica** de ~750 palabras en prosa española con 5 subsecciones del spec: magnitud (Cohen 1988), interpretación de CIs, Pearson vs Spearman (incluyendo la inversión de orden), comparación contra Liu et al. (2023) y limitaciones.
    - **Resultados clave (n=900)**:
        - **Spearman ρ vs human**: G-Eval `0.756` 95% CI [0.727, 0.783] | Voting `0.744` 95% CI [0.714, 0.772]. **Δρ = +0.012** (a favor de G-Eval).
        - **Pearson r vs human**: G-Eval `0.711` 95% CI [0.677, 0.742] | Voting `0.733` 95% CI [0.701, 0.761]. **Δr = −0.022** (a favor de Voting).
        - **Validación cruzada**: bootstrap difiere de Fisher Z en `|Δ|≤0.002` (mucho menor que la tolerancia 0.01).
        - **Ambos métodos superan baselines de Liu et al. (2023)** por +0.15 a +0.20 ρ.
    - **Alcance acotado por diseño**: el test de significancia de Δρ (Steiger 1980, bootstrap pareado, Wilcoxon) **no** se ejecuta aquí — queda íntegro para HU-12 (`notebooks/06_significance_tests.ipynb`). Cada celda relevante referencia HU-12 explícitamente.

- **Análisis descriptivo G-Eval vs Voting vs Human sobre 900 pares (HU-10)** - Notebook que cierra el bloque comparativo descriptivo, sin llamadas a APIs (lee solo `outputs/voting_results.json`, `outputs/geval_results.json`, `data/raw/dailydialog_zhao/dataset.json`):
    - `notebooks/04_descriptive_analysis.ipynb` - 15 celdas ejecutadas end-to-end con 900/900 entradas alineadas por `conversation_id`, 0 excluidas por scores faltantes.
    - **Estrato derivado en el propio notebook** (1=ground-truth, 2=negative-sample, 3=AI h≥4, 4=2.5–3.5, 5=h≤2) porque `voting_results.json.stratum` quedó en `null` durante HU-09 (el runner leyó del nivel equivocado del JSON procesado; no afecta a este análisis, pero sí se debe corregir en una HU futura del runner).
    - **Cohen's κ ponderado implementado manualmente** con NumPy (no se añade `scikit-learn` solo para una función); `krippendorff>=0.8.2` añadido vía `uv add` para Krippendorff α ordinal.
    - **4 figuras 150 dpi** a `outputs/figures/` (renumeradas a 10–13 para no pisar las 06–09 del análisis G-Eval HU-04): `10_histograms_comparison.png`, `11_boxplots_comparison.png`, `12_concordance_heatmap.png`, `13_scatter_human_vs_methods.png`.
    - **Summary table persistida** en `outputs/descriptive_analysis_summary.md` con descriptivos (12 métricas × 6 fuentes), Spearman ρ vs humano, κ ponderado, Krippendorff α y tasa de acuerdo exacto (|Δ|≤0.5).
    - **Resultados clave (n=900, headline metric-dependent)**:
        - **Spearman ρ vs humano**: G-Eval `0.756` > Voting `0.744` > judge_openai `0.713` > judge_anthropic `0.694` > judge_google `0.678`. Voting supera a todos los jueces individuales por 3.1–6.6 puntos (efecto wisdom-of-crowds), queda 0.012 por debajo de G-Eval (rango plausible de ruido — pendiente bootstrap pareado).
        - **Weighted Cohen's κ vs humano**: **Voting `0.643` (substantial) > G-Eval `0.525` (moderate)** — el voting cruza el umbral Landis–Koch en categorical agreement.
        - **Krippendorff α (3 raters, ordinal)**: `0.632` (substantial).
        - **Exact agreement (|Δ|≤0.5)**: **Voting 44.3% > G-Eval 35.9%** — voting domina por +8.4 pp.
        - **Estrato 3 (IA alta relevancia, n=216)**: confirmado como el más difícil para ambos (ρ_G=0.27, ρ_V=0.22), pero voting reduce el MAE en **40%** vs G-Eval (0.770 vs 1.286).
    - **Sección interpretativa** de 1.043 palabras en prosa académica (Cell 14) cubriendo distribución, correlación, agreement, estrato, agentes individuales e implicaciones para la tesis. El hallazgo central documentado: la respuesta a "¿voting ≥ G-Eval?" **depende de la métrica** — Spearman favorece G-Eval marginalmente, κ/α/exact-agreement favorecen voting sustancialmente.
    - **Field-name reconciliation con la spec**: `human_relevance_score` (dataset crudo, no `human_score`), `final_vote_score` (no `vote_score`).

## [0.6.0] - 2026-06-07

### Added

- **Sistema de Votación — Runner completo sobre 900 pares (HU-09)** - Escala el pipeline del panel (validado en pilot HU-08) al dataset completo, produciendo los outputs que la HU posterior usará para el análisis comparativo formal contra G-Eval:
    - `scripts/run_voting_system.py` - Orquesta 2.700 llamadas reales (900 pares × 3 jueces) reusando `call_agent` de `scripts/run_judge.py` (no se reescribe) y `aggregate` de `src/voting/aggregator.py`. **Resumible por `conversation_id`**: re-ejecutar tras crash arranca del primer par pendiente, no desde cero (persistencia atómica vía `tmp` + `os.replace` cada 10 pares y en `finally`).
    - **Política de retries**: `tenacity` con `wait_exponential(min=2, max=60)`, `stop_after_attempt(5)`, retry sobre excepciones transitorias de los tres SDKs (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`, `google_errors.ServerError/APIError`). Errores fatales (`AuthenticationError`, `PermissionDeniedError`, `NotFoundError`) abortan el run en vez de quemar 90 min registrando 900 fallos idénticos.
    - **Tolerancia por-juez, no por-par**: si un juez devuelve `score=None`, `aggregate` lo reporta vía `metadata.missing_agents` y el par sigue. Si los 3 fallan simultáneamente se captura el `ValueError` del aggregador y se persiste con `aggregate_failed=true`.
    - **CLI mínima**: `--dataset`, `--output-dir`, `--agents-dir`, `--limit N` (smoke), `--sleep` (default 0.5 s entre pares).
    - **Logging dual** (consola + `outputs/logs/voting_execution.log`) con una línea por par mostrando scores individuales, voto final y costo acumulado.
    - **Persistencia**: `outputs/voting_results.json` (900 dicts con `final_vote_score`, `individual_scores`, `agreement_level`, `metadata`, `cost_by_agent`, `aggregate_failed`, etc.) más `outputs/agent_scores/full_agent_{openai,google,anthropic}.json` con los razonamientos completos por juez. Summary auto-generado en `outputs/voting_summary_stats.md` espejo estilístico de `geval_summary_stats.md`.
    - **Resultados del run completo (n=900, 100% éxito, 0 `aggregate_failed`)**:
        - **Spearman ρ vs `human_score`**: **voting (panel mean) 0.7443** | G-Eval baseline (`gpt-4o`) 0.7565 | `judge_openai` 0.7131 | `judge_anthropic` 0.6935 | `judge_google` 0.6777. El voting supera a cada juez individual pero queda 1.2 puntos por debajo de G-Eval (a confirmar significancia con bootstrap en la HU de análisis).
        - **Distribución de `agreement_level`**: 39.11% `high`, 44.22% `medium`, 16.67% `low`.
        - **Costo real**: **$8.1547** (OpenAI $4.36, Anthropic $3.05, Google $0.74), wall time **156 min 54 s** (~10.46 s/par secuencial), 4.22 M tokens.
        - **Confirmación de la advertencia metodológica del pilot**: las ρ del pilot eran optimistas por el muestreo estratificado (`judge_anthropic` cayó de 0.927 a 0.694; voting de 0.892 a 0.744), exactamente como se anticipó.
    - **Smoke + crash drill verificados antes del run caro**: `--limit 5` (5/5, $0.0436), re-ejecución del mismo comando salta los 5 procesados (0 nuevas llamadas), confirma el checkpoint por `conversation_id`.

- **Sistema de Votación — Pilot end-to-end sobre 20 conversaciones (HU-08)** - Ejecución del pipeline completo del panel sobre las 20 entradas de `configs/prompts/pilot_sample.json` con decisión **go/no-go** documentada para el run completo:
    - `scripts/run_judge.py` - Runner reutilizable de un solo juez (`call_agent`) con dispatch por proveedor (OpenAI, Google vía la nueva SDK `google-genai`, Anthropic). Constantes `Final[]` con tarifas list price por proveedor (siguiendo el patrón de `scripts/run_geval.py`). Estrategia de parsing del SCORE acordada con la usuaria: regex permisivo `(?:SCORE|Score|score)\s*[=:]\s*([1-5])` en el primer intento más una sola reejecución con sufijo de formato si falla, manteniendo el V3 íntegro en la primera llamada como exige el control científico de HU-06. Para Gemini 2.5 Flash se pasa `thinking_config.thinking_budget=0`, en cumplimiento de la decisión cerrada en `docs/agent_panel_design.md` §4.2.
    - `notebooks/03_voting_pilot.ipynb` - Notebook generado con `nbformat`, 14 celdas (título, setup, configs, una corrida por agente, agregación con `aggregate()` de HU-07, join con G-Eval, tabla comparativa, degenerate agent check, Spearman, costo, review manual, decisión).
    - **Dependencia nueva**: `google-genai>=2.8.0` añadida con `uv add`.
    - **Pre-flight checks 1-6**: imports, API keys, sample, prompt V3 compartido por los tres YAML, agregador, una llamada real a Gemini ($0.00068).
    - **Resultados del pilot (n=20)**:
        - **60/60 llamadas exitosas** (20 por agente, sin fallos de parsing; el regex permisivo capturó el formato natural `Score = N` del V3 sin disparar el retry de formato).
        - **Sin jueces degenerados**: std de `judge_openai` 1.76, `judge_google` 1.75, `judge_anthropic` 1.64 (todos > 0.3).
        - **Spearman ρ contra `human_score`**: `judge_anthropic` 0.927, G-Eval (gpt-4o) 0.902, **voting_final 0.892**, `judge_openai` 0.846, `judge_google` 0.802. El voting iguala el techo de G-Eval con n=20.
        - **Distribución del agreement_level**: 14 `high`, 5 `medium`, 1 `low`.
        - **Costo real del pilot**: $0.1742 (OpenAI $0.0947, Anthropic $0.0641, Google $0.0154). Proyectado para 900 pares: **$7.84**, cerca de la estimación de HU-06 ($7.61).
    - **Persistencia**: `outputs/agent_scores/pilot_agent_{openai,google,anthropic}.json` (20 dicts por archivo) más `outputs/voting_pilot_results.json` (20 dicts con `vote_final`, `agreement_level`, `agreement_continuous`, `median_score` y `individual_scores`).
    - **Decisión: PROCEED**. Pipeline validado end-to-end, sin degeneraciones, ρ del voting competitiva con G-Eval, costo proyectado dentro del rango. Próximo paso: `scripts/run_voting_system.py` reusando `call_agent` con paralelismo entre proveedores y reintentos con `tenacity`.
    - **Observación no bloqueante**: `judge_anthropic` individual supera al voting agregado en este pilot (ρ 0.927 vs 0.892); a contrastar con n=900 antes de declarar que el voting domina al mejor juez individual.

- **Sistema de Votación — Módulo agregador `src/voting/aggregator.py` (HU-07)** - Implementación pura de la lógica de agregación de puntuaciones de los jueces del panel, sin llamadas a APIs ni dependencias pesadas:
    - `src/voting/__init__.py` - Reexporta la API pública (`aggregate`, `SCHEME_NAME`).
    - `src/voting/aggregator.py` - Cuatro funciones con docstrings tipo NumPy/Google y constantes a nivel de módulo con `Final[]` siguiendo el patrón de `scripts/run_geval.py`:
        - `aggregate(scores)`: orquesta el flujo y devuelve el dict de salida.
        - `validate_scores(scores)`: filtra `None`, recoge los `missing_agents` para `metadata`, valida rango `[SCORE_MIN, SCORE_MAX]` y lanza `ValueError` con el nombre del agente cuando un score está fuera.
        - `compute_agreement(values)`: mapea la desviación estándar muestral a `"high"` (std ≤ 0.5), `"medium"` (std ≤ 1.0), `"low"` (std > 1.0) o `"n/a"` si n < 2.
        - `_compute_final_score(values)`: aplica el esquema seleccionado en HU-05 (media aritmética) con redondeo a 2 decimales.
    - **Esquema implementado** (HU-05): `SCHEME_NAME = "arithmetic_mean"`, media aritmética de los scores disponibles, resultado float continuo en [1, 5] **comparable directamente con G-Eval**.
    - **Cinco casos extremos cubiertos y documentados en código**: input vacío (`ValueError "No agent scores provided"`), score fuera de rango (`ValueError` con nombre del agente), score `None` o agente ausente (cálculo sobre los restantes, `metadata.missing_agents`), un solo agente (`agreement_level = "n/a"`), unanimidad (`final_score` igual al valor compartido, `agreement_level = "high"`).
    - **Estructura de retorno**: las tres claves exigidas por el AC del issue (`final_score`, `individual_scores`, `agreement_level`) más dos extras alineadas con el contrato del doc HU-05 (`scheme_used` y `metadata`). El `metadata` incluye `n_agents`, `std_deviation`, `min_score`, `max_score`, **`median_score`** (mediana reportada en paralelo como verificación de robustez del doc HU-05) y **`agreement_continuous`** (`1 - std/std_max` en [0, 1], `None` si n < 2, también del doc HU-05).
    - `tests/test_aggregator.py` - 13 tests pytest sin llamadas a API, con fixtures que usan los nombres canónicos de los jueces (`judge_openai`, `judge_google`, `judge_anthropic`). Cubre los cinco escenarios obligatorios del AC (unanimidad, mayoría, "empate" demostrado como caso resoluble bajo media, dispersión máxima, valor faltante) más output, precisión/tipo, mínimos, máximos, mediana en metadata, agreement continuo en rango y agente único.
    - **Verificación**: `13 passed` en `pytest tests/test_aggregator.py`; suite completa `119 passed` (sin regresiones); `pre-commit` (ruff + ruff-format + mypy) `Passed` en los tres archivos.

## [0.5.0] - 2026-05-31

### Added

- **Sistema de Votación — Diseño del panel de agentes juez (HU-06)** - Definición declarativa del cuerpo evaluador del sistema de votación. Sin llamadas a APIs ni ejecución, solo configs y diseño:
    - **Tres archivos YAML de agentes** en `configs/agents/`, uno por proveedor, con la estructura exacta del Definition of Done (`name`, `model`, `provider`, `api_key_env`, `temperature`, `max_tokens`, `prompt_file`, `evaluation_dimension`, `score_range`, `independence`, `methodology_note`) y un bloque de comentarios `INDEPENDENCE GUARANTEE` al inicio:
        - `agent_openai.yaml`: `judge_openai` con modelo `gpt-4o`
        - `agent_google.yaml`: `judge_google` con modelo `gemini-2.5-flash`
        - `agent_anthropic.yaml`: `judge_anthropic` con modelo `claude-haiku-4-5` (lanzamiento de octubre de 2025)
    - **`temperature: 0.0` en los tres jueces**, no negociable: garantiza determinismo del juicio dado el mismo input y reproducibilidad entre corridas.
    - **Prompt único V3 compartido por los tres agentes** (`configs/prompts/geval_relevance_prompt.txt`, sin modificar). Decisión metodológica explícita: se reemplaza el requisito de diversidad estilística de prompts por el de **control científico estricto**, de modo que las diferencias entre `judge_openai` y G-Eval queden atribuibles únicamente a la mecánica de scoring (form-filling con logprobs en DeepEval frente a prompting directo con parser), y las diferencias entre el panel agregado y G-Eval queden atribuibles a la mecánica de scoring más la agregación.
    - **`gpt-4o` aparece tanto en G-Eval como en el panel** de forma deliberada, para aislar el efecto del método de evaluación del efecto del modelo.
    - **Diversidad del panel** restringida a proveedor y modelo (tres proveedores independientes: OpenAI, Google, Anthropic).
    - `docs/agent_panel_design.md` — Documento académico en español con once secciones (contexto, fundamentación metodológica, arquitectura con diagrama ASCII, descripción por agente, garantía de diversidad, mecanismo de independencia, mitigación de sesgos, relación con G-Eval, limitaciones conocidas, estimación de costos y referencias en formato APA con URL consultable). Sin emojis ni caracteres tipo IA, en línea con el estilo de `docs/voting_scheme_analysis.md`.
    - **Estimación de costos del panel**: aproximadamente 7.61 USD para 900 pares (4.50 USD OpenAI, 0.89 USD Google, 2.22 USD Anthropic), llevando el total estimado de la tesis a entre 12 y 13 USD incluido el baseline G-Eval ya ejecutado.
    - **Limitaciones declaradas explícitamente**: sesgo de longitud no mitigado por el prompt V3 (a vigilar en el pilot del panel), sacrificio consciente de la diversidad de estilo de prompt como precio del control científico, asimetría potencial de calidad por la presencia de `claude-haiku-4-5` (modelo eficiente) junto a dos modelos de mayor escala.

- **Sistema de Votación — Análisis y selección del esquema de votación (HU-05)** - Documento de calidad académica que abre el bloque del sistema de votación agéntico y fija el operador de agregación del panel de jueces de IA:
    - `docs/voting_scheme_analysis.md` - Análisis en 6 secciones, en español, con el formato de `docs/dataset_selection.md`:
        - **Revisión de 5 esquemas** de agregación (mayoría/moda, media aritmética, mediana, votación ponderada y recuento de Borda), cada uno con descripción, ventajas, limitaciones y aplicabilidad a puntuaciones ordinales 1-5 emitidas por jueces de IA
        - **7 criterios de selección** justificados (robustez ante empates, simplicidad, fundamentación en literatura, compatibilidad con escala ordinal 1-5, comparabilidad directa con G-Eval, paralelismo con la etiqueta humana y producción de un `agreement_level` significativo) con tabla comparativa esquema × criterio
        - **Esquema seleccionado: media aritmética** (principal) + **mediana** reportada en paralelo como verificación de robustez; justificación de ~790 palabras (el AC exige ≥300). Argumento central: el `human_score` es la media de 4 anotadores, por lo que promediar el panel de IA lo vuelve el análogo metodológico exacto del panel humano
        - Discusión **ordinal vs. categórico** (por qué Borda, diseñado para rankings de candidatos, no encaja en scoring absoluto por ítem)
        - Definición del **`agreement_level`** en dos niveles: por ítem (`1 - std/std_max`) y a nivel dataset (Krippendorff α / ICC(2,1) / Spearman pairwise), reutilizando las funciones de `scripts/analyze_geval.py`
        - **9 referencias** en formato APA, todas reales y consultables (Galton 1907; Stevens 1946; Arrow 1951; Clemen 1989; Liu et al. 2023; Zheng et al. 2023; Verga et al. 2024; Pacuit 2019; Brandt et al. 2016)
    - Mapeo del esquema al contrato `{final_score, individual_scores, agreement_level}` del futuro módulo `src/voting/aggregator.py`, y nota para corregir en el issue del agregador la referencia errónea al issue #8 (debe apuntar a esta HU-05)

## [0.4.0] - 2026-05-17

### Added

- **G-Eval — Ejecución del run completo y análisis de resultados** - Ejecución de `scripts/run_geval.py` sobre el 100% del dataset y análisis interpretado de la línea base automática de relevancia:
    - **Run completo ejecutado**: 900/900 pares evaluados con éxito (0 fallos), evaluador `gpt-4o`, 37m 21s wall time, 1.172.164 tokens (1.090.187 input / 81.977 output), costo real **$3.5453**
    - **Resultado principal**: correlación de Spearman G-Eval vs. `human_score` = **0.7565** (`strong`, p ≈ 8.5e-168, n=900); Pearson r = 0.7106; Kendall τ = 0.5705
    - **Artefactos del run**: `outputs/geval_results.json` (900 resultados par-a-par), `outputs/geval_summary_stats.md`, `outputs/logs/geval_execution.log`
    - `scripts/analyze_geval.py` - Script de análisis reutilizable que lee `geval_results.json` + el dataset y produce un reporte interpretado y ordenado más figuras de soporte:
        - **Métricas de concordancia**: Spearman ρ, Pearson r, Kendall τ, MAE (0.90), RMSE (1.14), sesgo medio Δ (−0.75), hit-rate dentro de ±0.5 (36%) y ±1.0 (63%)
        - **Techo inter-anotador (human ceiling)**: implementación propia de la concordancia entre los 4 anotadores humanos como referencia para juzgar ρ — Spearman pairwise medio (0.466), Spearman leave-one-out (1 anotador vs. media de los otros 3 = 0.580), **Krippendorff α ordinal** (0.466) e **ICC(2,1)** (0.466). Hallazgo: G-Eval (0.756) **supera** el techo de un anotador individual por +0.176, porque las etiquetas humanas son ruidosas (α < 0.667) — un evaluador más potente no aportaría ganancia medible
        - **Desglose por familia y por modelo individual**: n, media humana, media G-Eval, sesgo y MAE por grupo
        - **Matriz de confusión por bandas** (scores redondeados 1-5) con acuerdo exacto de banda
        - **Top-10 mayores desacuerdos** en ambas direcciones (sobre- y sub-estimación)
        - Cuatro funciones de figura (`fig_scatter`, `fig_residuals`, `fig_delta_boxplot`, `fig_mean_by_family`, `fig_ceiling`) en modo headless (`matplotlib.use("Agg")`)
        - **CLI**: `--results` (default `outputs/geval_results.json`), `--output-dir` (default `outputs/`)
    - **Schema de `outputs/geval_analysis_report.md`**: reporte en 9 secciones — resumen ejecutivo, métricas de concordancia, techo inter-anotador, desglose por familia, desglose por modelo, matriz de confusión, mayores desacuerdos, figuras y guía de lectura
    - **Figuras nuevas** en `outputs/figures/`: `05_geval_vs_human_scatter.png`, `06_residual_histogram.png`, `07_delta_by_family_boxplot.png`, `08_mean_score_by_family.png`, `09_ceiling_comparison.png`
    - `.code_quality/ruff.toml` - `scripts/analyze_geval.py` añadido a `per-file-ignores` para `RUF001`/`RUF002`/`PLR2004` (letras griegas ρ/α en strings de salida y umbrales de correlación que son constantes de dominio, no magic numbers — mismo precedente que `build_pilot_notebook.py`)
    - `.gitignore` - Ignora `.claude/scheduled_tasks.lock` (lock de runtime de Claude Code, regenerado por sesión)

- **G-Eval Production Runner (HU-04)** - Script ejecutable sobre el 100% de los pares (900) para producir la línea base automática de relevancia, secuencial, con reintentos y checkpoint:
    - `scripts/run_geval.py` - Runner de producción (~570 LOC):
        - **Retries con backoff exponencial** vía `tenacity` (2s → 60s, 5 intentos) sobre errores transitorios de OpenAI: `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`. Errores fatales (`AuthenticationError`, `PermissionDeniedError`, `NotFoundError`) abortan el run en la primera entrada en vez de gastar 75 minutos en 900 fallos idénticos
        - **Persistencia incremental**: `outputs/geval_results.json`, `outputs/geval_summary_stats.md` y `outputs/.geval_checkpoint.json` se escriben juntos cada 25 entradas y en el bloque `finally` del loop. Garantía: pase lo que pase (Ctrl+C, fallo fatal, crash), los artefactos AC quedan en disco con lo evaluado al momento
        - **Resume automático** desde checkpoint en re-corrida del mismo comando; `--no-resume` lo ignora; checkpoint corrupto se rota a `.bak` y se arranca limpio en vez de crashear
        - **Atomic write** (`.tmp` → rename) para los tres artefactos
        - **Token counting determinista** con `tiktoken` (cl100k_base para gpt-4o), permite cómputo de costo offline reproducible en tests sin llamar a la API
        - **Logging dual** (stdlib `logging`) a stdout + `outputs/logs/geval_execution.log` con per-entry INFO line + bloque RUN SUMMARY al final (tiempo total, mean per entry, total tokens, total cost, n_ok/n_fail)
    - **Schema de `outputs/geval_results.json`** (campos del AC + auxiliares):
        - AC: `conversation_id`, `geval_score` (escala 1-5), `human_score`, `model_used`, `timestamp` (ISO UTC), `tokens_used` (input/output/total), `cost_usd`
        - Auxiliares: `geval_score_raw` (raw [0,1] de DeepEval), `delta`, `reason`, `attempts`
        - Entradas fallidas: los 7 campos AC con `null` en `geval_score`/`tokens_used`/`cost_usd`, más `error: "<ExcType>: <msg>"`
    - **Schema de `outputs/geval_summary_stats.md`**:
        - Tabla "Run completion": total/successful/failed/tokens/cost
        - Tabla "G-Eval score distribution (1-5)": n/mean/median/std/min/max
        - Tabla "Spearman correlation vs. human_score": rho/p-value/n
        - Tabla "Breakdown by model family" (ground-truth / negative-sample / GPT2 / S2S / HRED / VHRED)
        - Tabla "Failed entries" con conversation_id/attempts/error
    - **CLI**: `--limit N` (smoke test), `--no-resume` (ignora checkpoint), `--model` (default `gpt-4o`), `--output-dir` (default `outputs/`)
    - **Códigos de salida**: `0` éxito | `1` error inesperado | `2` falta `OPENAI_API_KEY` | `3` error fatal de API (auth/modelo) | `130` Ctrl+C
    - **Refactor de calidad** según code review:
        - `compute_summary()` (lógica numérica pura) y `render_summary_markdown()` (formateo) separadas, ambas testeable sin parsear markdown
        - 5 helpers privados de render por sección (`_render_completion_table`, `_render_distribution_table`, `_render_spearman_table`, `_render_family_table`, `_render_failed_table`)
    - `tests/test_run_geval.py` - 41 tests sin llamadas a API (mocks de GEval):
        - `_rescale_0_1_to_1_5` con casos límite (0/1/0.5/0.8)
        - `build_test_case` (drop del trailing assistant turn, formato `[Turn N] Role: content`)
        - `estimate_tokens_and_cost` (positividad y monotonicidad sobre la longitud del reason)
        - `_model_family` parametrizado para 7 nombres
        - `_basic_stats` (vacío y poblado)
        - `compute_summary` (counts, agregación tokens/cost, NaN cuando n<2 para Spearman, agrupación por familia)
        - `render_summary_markdown` (todas las secciones, omite tablas opcionales vacías, escapa `|` en mensajes de error)
        - `generate_summary_stats` integración (idempotente, sobrevive a 100% fallos)
        - `_evaluate_one` con métrica mockeada: éxito, fallo no-retryable, agotamiento de retries, propagación fatal
        - `evaluate_dataset` (escritura incremental cada checkpoint, persistencia post-fatal vía finally)
        - `load_checkpoint` (missing, corrupto rotado a `.bak`)
        - CLI parser (defaults + `--limit`/`--no-resume`)
    - **Smoke test E2E** verificado contra API real: 3/3 entries exitosos, ~$0.012, 13s; resume desde checkpoint sin duplicación
    - **Costo estimado del run completo**: ~$3-5 en gpt-4o list pricing; ~30-60 minutos wall time (~3-4s/par)

- **Speaker-labeled dataset schema** (fix al pipeline HU-02 surgido en el piloto HU-03) - Los roles `user`/`assistant` ahora se derivan del speaker label en vez de la paridad del índice del turno. La paridad fallaba cuando la longitud del contexto y el `response_speaker` no estaban alineados, produciendo turnos consecutivos del mismo rol que rompían la extracción del `input` en `serialize_test_case`:
    - `scripts/download_dataset.py` - Preserva `{speaker, text}` en `turns[]` (antes se descartaba el speaker) y agrega `response_speaker` derivado de `reference[0]`. Parsing defensivo del `reference` faltante o malformado
    - `scripts/transform_dataset.py` - `build_turns()` mapea `speaker == response_speaker → assistant`, todo lo demás → `user`. `serialize_test_case()` busca el último turno *user* (no `turns[-1]`) para el campo `input`
    - `data/README.md` + template del generador - Documentan que `"A"`/`"B"` son los identificadores anónimos crudos del corpus DailyDialog-Zhao, no marcadores de género ni persona
    - `configs/prompts/pilot_sample.json` - Regenerado con el campo `response_speaker` en metadata
    - 30 tests parametrizados/regresión añadidos en `tests/test_download_dataset.py` y `tests/test_transform.py`

- **G-Eval Relevance Prompt Pilot (HU-03)** - Diseño, ejecución y selección del prompt final de evaluación de relevancia para G-Eval sobre DailyDialog-Zhao, con evaluador `gpt-4o` (sucesor directo de GPT-4 en Liu et al., EMNLP 2023):
    - `configs/prompts/select_pilot_sample.py` - Sampler estratificado determinista (seed=42) que produce 20 entradas distribuidas en 5 estratos × 4 entradas:
        - Estrato 1: top-4 `ground-truth` por `human_score`
        - Estrato 2: bottom-4 `negative-sample` por `human_score`
        - Estrato 3: IA con `human_score >= 4.0`, balanceado entre familias GPT2/S2S/HRED/VHRED
        - Estrato 4: IA con `2.5 <= human_score <= 3.5`, balanceado entre familias
        - Estrato 5: IA con `human_score <= 2.0`, balanceado entre familias
    - Tres versiones iteradas del prompt G-Eval en `configs/prompts/`:
        - `v1_generic.txt` (526 B) - baseline genérico, rúbrica 1-5 sin CoT
        - `v2_dialogue_cot.txt` (1861 B) - tarea multi-turno + definición 3-partes (a/b/c) + 5 pasos CoT
        - `v3_full_cot_anchored.txt` (5125 B) - V2 + rúbrica anclada por score + 3 ejemplos trabajados del sample piloto
    - `configs/prompts/pilot_sample.json` - Las 20 entradas sampleadas, cada una taggeada con `stratum`
    - `configs/prompts/pilot_results_v{1,2,3}.json` - Scores por entrada (raw `[0,1]` y reescalado a `[1,5]`), `human_score`, delta y `reason` del evaluador (20/20 exitosos en las tres versiones)
    - `configs/prompts/geval_relevance_prompt.txt` - Prompt final seleccionado (copia de V3) para el run completo de 900 pares
    - `scripts/build_pilot_notebook.py` - Generador programático del notebook (29 celdas como strings en Python) para mantener el source diff-reviewable
    - `notebooks/02_prompt_pilot.ipynb` (145 KB) - Notebook ejecutado con outputs embebidos, organizado en 11 secciones numeradas en español:
        1. Setup y carga del sample piloto
        2. Distribución por estrato
        3. Catálogo de prompts y estimación de costo
        4. Helpers reutilizables
        5. Piloto V1 - baseline genérico
        6. Piloto V2 - CoT consciente del diálogo
        7. Piloto V3 - CoT completo + rúbrica anclada
        8. Comparación cuantitativa y visual (scatter 1x3 coloreado por estrato)
        9. Prueba de Steiger - comparación de correlaciones
        10. Selección final: V3
        11. Conclusiones
    - Patrón de carga cacheada en las celdas de piloto: lee `pilot_results_v*.json` si existe, evita re-llamar la API en re-ejecuciones
    - Figura `outputs/figures/05_prompt_pilot_comparison.png` (150 dpi, embebida en el output de la celda 23)
    - `docs/prompt_iterations.md` - Documentación metodológica auto-generada desde la celda 28:
        - Tabla por versión con `Changes from previous`, Spearman rho, p-value, mean/max |delta| y observaciones por estrato
        - Tabla de Steiger (Fisher Z) con las tres comparaciones pairwise
        - Sección "Selección Final" con los 4 argumentos metodológicos
    - **Resultados del piloto** (n=20, 20/20 exitosos en cada versión):
        - V1 Spearman rho = 0.937 (p < 0.0001)
        - V2 Spearman rho = 0.904 (p < 0.0001)
        - V3 Spearman rho = 0.899 (p < 0.0001)
        - Prueba de Steiger: las tres comparaciones pairwise son NO significativas (todos p > 0.45)
    - **Versión seleccionada: V3** por criterios metodológicos (ausencia de diferencia estadística entre versiones):
        - Transparencia: CoT auditable por par evaluado
        - Robustez: rúbrica anclada reduce varianza a n=900
        - Comparabilidad: razonamiento explícito alineado con el baseline agéntico
        - Contribución: fidelidad al diseño original de G-Eval (Liu et al., EMNLP 2023)
    - Costo del piloto: ~\$1.80 (una vez). Estimación del experimento completo con V3: ~\$4-6

- **Dataset Transformation Pipeline (HU-02)** - Conversión de dataset raw a formato DeepEval `ConversationalTestCase`:
    - `scripts/transform_dataset.py` - Pipeline que convierte 900 pares contexto-respuesta:
        - Carga dataset raw desde `data/raw/dailydialog_zhao/dataset.json`
        - Construye objetos `ConversationalTestCase` preservando historias de conversación
        - Mapea turnos (alternancia user/assistant) preservando contexto completo
        - Almacena métricas humanas en `additional_metadata` para análisis posterior
        - Serializa a `data/processed/deepeval_test_cases.json` con campos: input, actual_output, turns, metadata
        - Valida integridad: count match, ids preservados, scores exactos, rango [1.0, 5.0]
    - `tests/test_transform.py` - Suite completa con 26 tests:
        - **TestBuildTurns** (5 tests): count, alternating roles, content preservation, edge cases
        - **TestEntryToTestCase** (9 tests): transformación de 3 tipos de entries, metadata preservation
        - **TestSerializeTestCase** (4 tests): JSON serializable, keys requeridas, structure
        - **TestValidateTransform** (5 tests): passing, count mismatch, score mismatch, null response, wrong ID
        - **TestLoadDataset** (2 tests): file reading, error handling
        - **Integration test** (1 test): full pipeline end-to-end
    - Código completamente tipado (PEP 585) y con docstrings Google-style en todas las funciones
    - Constantes con Final[] para evitar magic numbers (PLR2004 compliance)

### Changed

- **Ruff configuration** (`.code_quality/ruff.toml`) - `[lint.per-file-ignores]` acotado a los archivos nuevos del piloto para permitir notación científica y thresholds estadísticos legítimos; el resto del codebase mantiene los `select`/`ignore` originales:
    - `notebooks/**`: `RUF001`, `RUF002` (letras griegas y signos `×`/`–`/`—` en strings y docstrings), `PLR2004` (thresholds como 0.05, 0.4, 0.7, 1.5), `B018`
    - `scripts/build_pilot_notebook.py`: `RUF001`, `RUF002`, `PLR2004` (el builder embebe el source del notebook como strings literales, por lo que hereda los mismos casos)
    - `configs/prompts/select_pilot_sample.py`: `PLR2004`, `S311` (`random.Random` con seed=42 para reproducibilidad, no criptografía)
- **`.gitignore`** - Dos cambios:
    - Añadido `.deepeval/` para excluir el directorio de telemetría que DeepEval crea bajo el cwd al instanciar la métrica
    - Reemplazado el patrón broad `outputs` (heredado de Hydra) por patrones específicos: `outputs/geval_results.json`, `outputs/geval_summary_stats.md`, `outputs/.geval_checkpoint.json`, `outputs/logs/*.log`, `outputs/figures/*.{png,pdf,svg}`. Los `.gitkeep` en `outputs/`, `outputs/logs/` y `outputs/figures/` versionan la estructura para que un clone fresco pueda ejecutar `run_geval.py` sin `mkdir` manual
- `scripts/transform_dataset.py` - `zip(..., strict=True)` tras `ruff --fix` (B905); seguro porque las longitudes ya se asertan antes del zip
- `scripts/build_pilot_notebook.py` - Vuelto al patrón convencional `import nbformat  # type: ignore[import-untyped]` (con comentario explicativo) en vez de `importlib.import_module(...)` con typing como `Any`. El nuevo patrón mantiene el type-checking donde mypy puede inferir y queda alineado con el resto del repo (e.g. `import requests  # type: ignore[import-untyped]` en `download_dataset.py`)
- **Dependencies** - Nuevas librerías:
    - `tenacity` (>= 9.0.0) - Reintentos con backoff exponencial sobre errores transitorios de la API OpenAI (usado por `run_geval.py`)
    - `tiktoken` (>= 0.8.0) - Token counting determinista (cl100k_base) para gpt-4o; permite cómputo de costo offline reproducible en tests sin gastar API tokens
    - `jupyter` (>= 1.1.0) - Entorno Jupyter para ejecutar .ipynb
    - `nbconvert` (>= 7.16.0) - Conversión y ejecución de notebooks (para CI/CD)

## [0.3.0] - 2026-04-12

### Added

- **Exploratory Data Analysis Notebook (Dataset_exploration branch)** - Análisis completo de las puntuaciones de relevancia humana:
    - `notebooks/01_eda.ipynb` - Notebook Jupyter con 16 celdas (243 KB):
        - Análisis descriptivo: mean, median, std, min, max, Q1, Q3, IQR, skewness
        - Estadísticas por familia de modelos (8 familias colapsadas de 21 variantes)
        - Histograma de distribución de puntuaciones (con líneas media/mediana)
        - Boxplot horizontal por familia de modelos (colores distintivos: verde ground-truth, rojo negative-sample)
        - Gráfico de frecuencia de buckets con porcentajes
        - Análisis de outliers (detección por IQR fence + desacuerdo inter-anotador alto, std > 1.5)
        - Histograma de desacuerdo entre anotadores
        - Verificación de balance del dataset (ground-truth siempre más alto, negative-sample siempre más bajo)
        - Conclusiones con 4 subsecciones: resumen de distribución, patrones de rendimiento, acuerdo anotadores, implicaciones para experimentos G-Eval
    - Cuatro figuras de 150 dpi listos para publicación en `outputs/figures/`:
        - `01_histogram_relevance.png` (53 KB) - Distribución de puntuaciones
        - `02_boxplot_by_model.png` (60 KB) - Comparación por familia de modelos
        - `03_frequency_table.png` (53 KB) - Distribución de buckets
        - `04_interannotator_std.png` (63 KB) - Distribución de desacuerdo inter-anotador
    - Código completamente tipado (PEP 585: list[float], dict[str, Any], etc.)
    - Docstrings Google-style en todas las funciones helper
    - Constantes como Final[] para evitar magic numbers

## [0.2.0] - 2026-04-11

### Added

- **Dataset Selection Investigation (HU-00 / #8)** - Investigación y selección del dataset para evaluación de relevancia:
    - `docs/dataset_selection.md` con 6 datasets candidatos documentados y análisis comparativo
    - Matriz de evaluación con 6 criterios (anotaciones humanas, respuestas IA, disponibilidad, resultados previos, tamaño, compatibilidad)
    - Justificación del dataset ganador **DailyDialog-Zhao** (Zhao et al., ACL 2020) con 650+ palabras

- **Dataset Acquisition & Processing Pipeline (HU-01)** - Descarga y procesamiento automático del dataset DailyDialog-Zhao desde Zenodo:
    - `scripts/download_dataset.py` - Script de descarga idempotente que:
        - Descarga `ACL2020_data.zip` (118.2 KB) desde Zenodo (DOI: 10.5281/zenodo.3828180)
        - Extrae y parsea 900 pares contexto-respuesta con anotaciones humanas
        - Valida integridad contra valores del paper (100 diálogos, 9 respuestas/diálogo, 4 anotadores)
        - Mapea dimensiones: `relevance → relevance`, `content → appropriateness`
        - Genera `data/raw/dailydialog_zhao/dataset.json` con 8 campos por entrada
        - Auto-genera `data/README.md` con estadísticas y metadatos
    - `data/raw/dailydialog_zhao/` - Directorio con zip descargado y dataset.json procesado (586 KB)
    - `data/README.md` - Documentación auto-generada con:
        - Información de fuente (paper, Zenodo, fecha descarga)
        - Licencia y esquema de campos
        - Distribución de puntuaciones y verificación de integridad
        - Lista de 21 modelos generativos incluidos

- **Test Suite for Data Pipeline** - Cobertura comprehensiva de funciones de procesamiento de datos:
    - `tests/test_download_dataset.py` - 27 tests divididos en 4 clases:
        - **TestParseAnnotations** (9 tests): parsing JSON → dataset limpio
            - Single entry parsing, múltiples modelos, cálculo de medias, extracción de turns, formato conversation_id
            - Ground-truth y negative-sample incluidos, input vacío, scores faltantes
        - **TestRunIntegrityChecks** (6 tests): validación de integridad del dataset
            - 900 pares pasan, total incorrecto falla, scores fuera de rango, anotadores insuficientes, reportes completos
        - **TestPrintSummary** (4 tests): generación de resumen de estadísticas
            - Dataset mínimo y completo sin crash, reporta conteo de pares y modelos
        - **TestGenerateReadme** (8 tests): generación de documentación README
            - Archivo creado, URL Zenodo, licencia, estadísticas, modelos, esquema, sección de integridad, fecha

### Changed

- **Dependencies** - Nuevas librerías para descarga y type-checking:
    - `requests` (>= 2.28.0) - HTTP client para descargar desde Zenodo
    - `types-requests` (>= 2.28.0) - Type stubs para mypy (dev dependency)

## [0.1.0] - 2026-04-05

### Added

- **Project Structure** - Reorganized repository to align with thesis plan for "Evaluación de Relevancia en Agentes Conversacionales de IA mediante un Sistema de Votación Agéntico en comparación con el Framework G-EVAL"
- **Configuration System** - Created thesis-specific configuration structure:
    - `configs/prompts/` - Directory for prompt templates
    - `configs/agents/` - Directory for agent configurations
    - `configs/experiment_config.yaml` - Experiment configuration with model parameters (model_name, temperature, n_executions, seed)
- **Data Structure** - Simplified data layers for thesis research:
    - `data/raw/` - Raw unprocessed data
    - `data/processed/` - Processed and cleaned data
- **Output Management** - Created organized output directories:
    - `outputs/figures/` - Visualization outputs
    - `outputs/logs/` - Execution logs and metrics
- **Scripts** - Added utility scripts:
    - `scripts/test_deepeval_setup.py` - Setup verification script for DeepEval integration with ConversationalTestCase validation
- **Environment Configuration** - Added `.env.example` with API key placeholders for OpenAI and Anthropic
- **Production Dependencies** - Integrated 9 core libraries:
    - `deepeval` (3.9.5) - LLM evaluation framework
    - `openai` (2.30.0) - OpenAI API client
    - `anthropic` (0.89.0) - Anthropic API client
    - `pandas` (3.0.2) - Data manipulation
    - `scipy` (1.17.1) - Scientific computing
    - `matplotlib` (3.10.8) - Data visualization
    - `seaborn` (0.13.2) - Statistical visualization
    - `python-dotenv` (1.2.2) - Environment variable management
    - `pyyaml` (6.0.3) - YAML file processing

### Changed

- **README.md** - Complete rewrite in Spanish with:
    - Project title and thesis context
    - New repository structure documentation
    - Setup instructions using `uv sync`
    - Configuration and API key setup guide
- **Configuration Directory** - Renamed `conf/` to `configs/` for clarity
    - Removed generic ML pipeline configuration subdirectories
- **.gitignore** - Updated to reflect new project structure:
    - Removed stale `data/README.md` exception
    - Added explicit exceptions for `data/raw/` and `data/processed/` directories

### Removed

- **Generic ML Boilerplate** - Removed cookiecutter template artifacts:
    - 8 generic notebook directories (`1-data` through `8-reports`)
    - `notebooks/notebook_template.ipynb`
    - `notebooks/README.md`
- **Kedro Data Layers** - Removed unused data layer structure:
    - `data/01_raw/` through `data/08_reporting/`
    - `data/README.md` (Kedro-specific documentation)
- **ML Pipeline Scaffolding** - Removed placeholder modules:
    - `src/data/`, `src/inference/`, `src/model/`, `src/pipelines/` (empty modules)
    - `src/tmp_mock.py` (mock test function)
    - `src/README.md` (cookiecutter documentation)
    - `tests/data/`, `tests/inference/`, `tests/model/`, `tests/pipelines/` (empty test modules)
    - `tests/test_mock.py` (mock tests)
- **Generic Configuration Modules** - Removed unused conf/ subdirectories:
    - `conf/data_extraction/`, `conf/data_preparation/`, `conf/data_validation/`
    - `conf/model_evaluation/`, `conf/model_serving/`, `conf/model_train/`, `conf/model_validation/`

### Fixed

- **Type Hints** - Ensured all new code includes proper static typing with `Any` type annotations
- **Code Quality** - Maintained compliance with pre-commit hooks (Ruff, MyPy, YAML validation)

### Notes

- Initial setup focused on establishing the thesis research infrastructure
- Project version: 0.1.0 (development stage)
- Python requirement: >= 3.12
- Dependency management: UV package manager
- All external dependencies are production-level and verified for compatibility
