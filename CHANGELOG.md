# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- **Dependencies** - Nuevas librerías para ejecutar notebooks:
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
