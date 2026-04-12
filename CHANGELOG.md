# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Exploratory Data Analysis Notebook** - Complete analysis of human relevance scores:
    - `notebooks/01_eda.ipynb` - Jupyter notebook with 16 cells (243 KB):
        - Descriptive statistics: mean, median, std, min, max, Q1, Q3, IQR, skewness
        - Statistics by model family (8 families collapsed from 21 variants)
        - Histogram of score distribution (with mean/median lines)
        - Horizontal boxplot by model family (distinctive colors: green ground-truth, red negative-sample)
        - Frequency graph of buckets with percentages
        - Outlier analysis (IQR fence detection + inter-annotator disagreement, std > 1.5)
        - Inter-annotator disagreement histogram
        - Dataset balance verification (ground-truth always highest, negative-sample always lowest)
        - Conclusions with 4 subsections: distribution summary, performance patterns, annotator agreement, implications for G-Eval experiments
    - Four publication-ready figures at 150 dpi in `outputs/figures/`:
        - `01_histogram_relevance.png` (53 KB) - Score distribution
        - `02_boxplot_by_model.png` (60 KB) - Comparison by model family
        - `03_frequency_table.png` (53 KB) - Bucket distribution
        - `04_interannotator_std.png` (63 KB) - Inter-annotator disagreement distribution
    - Fully typed code (PEP 585: list[float], dict[str, Any], etc.)
    - Google-style docstrings on all helper functions
    - Constants as Final[] to avoid magic numbers

### Changed

- **Dependencies** - New libraries for notebook execution:
    - `jupyter` (>= 1.1.0) - Jupyter notebook environment
    - `nbconvert` (>= 7.16.0) - Notebook conversion and execution (for CI/CD)

## [0.2.0] - 2026-04-11

### Added

- **Dataset Selection Investigation (HU-00 / #8)** - Dataset investigation and selection for relevance evaluation:
    - `docs/dataset_selection.md` with 6 candidate datasets and comparative analysis
    - Evaluation matrix with 6 criteria (human annotations, AI responses, availability, prior results, size, compatibility)
    - Justification of winning dataset **DailyDialog-Zhao** (Zhao et al., ACL 2020) with 650+ words

- **Dataset Acquisition & Processing Pipeline (HU-01)** - Automatic download and processing of DailyDialog-Zhao dataset from Zenodo:
    - `scripts/download_dataset.py` - Idempotent download script that:
        - Downloads `ACL2020_data.zip` (118.2 KB) from Zenodo (DOI: 10.5281/zenodo.3828180)
        - Extracts and parses 900 context-response pairs with human annotations
        - Validates integrity against paper values (100 dialogues, 9 responses/dialogue, 4 annotators)
        - Maps dimensions: `relevance -> relevance`, `content -> appropriateness`
        - Generates `data/raw/dailydialog_zhao/dataset.json` with 8 fields per entry
        - Auto-generates `data/README.md` with statistics and metadata
    - `data/raw/dailydialog_zhao/` - Directory with downloaded zip and processed dataset.json (586 KB)
    - `data/README.md` - Auto-generated documentation with:
        - Source information (paper, Zenodo, download date)
        - License and field schema
        - Score distribution and integrity verification
        - List of 21 generative models included

- **Test Suite for Data Pipeline** - Comprehensive coverage of data processing functions:
    - `tests/test_download_dataset.py` - 27 tests divided into 4 classes:
        - **TestParseAnnotations** (9 tests): JSON parsing -> clean dataset
            - Single entry parsing, multiple models, mean calculation, turn extraction, conversation_id format
            - Ground-truth and negative-sample included, empty input, missing scores
        - **TestRunIntegrityChecks** (6 tests): dataset integrity validation
            - 900 pairs pass, incorrect total fails, scores out of range, insufficient annotators, complete reports
        - **TestPrintSummary** (4 tests): statistics summary generation
            - Minimal and complete dataset without crash, reports pair and model counts
        - **TestGenerateReadme** (8 tests): README documentation generation
            - File created, Zenodo URL, license, statistics, models, schema, integrity section, date

### Changed

- **Dependencies** - New libraries for download and type-checking:
    - `requests` (>= 2.28.0) - HTTP client for downloading from Zenodo
    - `types-requests` (>= 2.28.0) - Type stubs for mypy (dev dependency)

## [0.1.0] - 2026-04-05

### Added

- **Project Structure** - Reorganized repository to align with thesis plan for "Relevance Evaluation in Conversational AI Agents through an Agentic Voting System compared with G-EVAL Framework"
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
