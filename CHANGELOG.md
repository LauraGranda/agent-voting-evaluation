# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
