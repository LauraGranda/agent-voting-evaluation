# Evaluación de Relevancia en Agentes Conversacionales de IA mediante un Sistema de Votación Agéntico en comparación con el Framework G-EVAL

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

## Descripción del Proyecto

Este proyecto de investigación para tesis de maestría implementa un sistema de evaluación de relevancia para agentes conversacionales de IA. El enfoque propuesto compara un sistema innovador de votación agéntica con el marco de evaluación G-EVAL.

**Autor**: Laura Granda

## 🏗️ Estructura del Proyecto

```text
agent-voting-evaluation/
├── .env.example                    # Template de variables de entorno
├── .github/                        # Configuración de GitHub Actions y templates
├── .code_quality/                  # Configuración de herramientas de calidad
│   ├── mypy.ini                    # Configuración de type checking
│   └── ruff.toml                   # Configuración de linter y formatter
├── configs/                        # Archivos de configuración
│   ├── prompts/                    # Templates de prompts para LLMs
│   ├── agents/                     # Configuración de agentes
│   └── experiment_config.yaml      # Configuración de experimentos
├── data/                           # Datos del proyecto
│   ├── raw/                        # Datos crudos sin procesar
│   └── processed/                  # Datos procesados y listos para usar
├── docs/                           # Documentación del proyecto
├── models/                         # Modelos entrenados y serializados
├── notebooks/                      # Notebooks Jupyter para experimentación
├── outputs/                        # Outputs generados
│   ├── figures/                    # Figuras y visualizaciones
│   └── logs/                       # Logs de ejecución
├── scripts/                        # Scripts ejecutables
│   └── test_deepeval_setup.py      # Script para verificar setup de DeepEval
├── src/                            # Código fuente
│   ├── data/                       # Módulos para procesamiento de datos
│   ├── inference/                  # Módulos para inferencia
│   ├── model/                      # Módulos para modelos
│   └── pipelines/                  # Pipelines de procesamiento
├── tests/                          # Tests del proyecto
├── .gitignore                      # Archivo de exclusión de git
├── .pre-commit-config.yaml         # Configuración de pre-commit hooks
├── Makefile                        # Comandos útiles
├── codecov.yml                     # Configuración de cobertura
├── mkdocs.yml                      # Configuración de documentación
├── pyproject.toml                  # Configuración de dependencias
└── README.md                       # Este archivo
```

## 🚀 Instalación

### Requisitos Previos

- Python >= 3.12
- [UV](https://github.com/astral-sh/uv) (gestor de dependencias moderno)

### Pasos de Instalación

1. **Clonar el repositorio**:

   ```bash
   git clone <repository-url>
   cd agent-voting-evaluation
   ```

2. **Configurar variables de entorno**:

   ```bash
   cp .env.example .env
   # Editar .env y añadir tus claves de API
   export OPENAI_API_KEY=tu-openai-api-key
   export ANTHROPIC_API_KEY=tu-anthropic-api-key
   ```

3. **Instalar dependencias**:

   ```bash
   uv sync
   ```

4. **Verificar la instalación**:

   ```bash
   uv run python scripts/test_deepeval_setup.py
   ```

## 📋 Dependencias Principales

El proyecto incluye las siguientes dependencias de producción:

- **deepeval** - Framework de evaluación para LLMs
- **openai** - Cliente de API de OpenAI
- **anthropic** - Cliente de API de Anthropic
- **pandas** - Procesamiento de datos
- **scipy** - Computación científica
- **matplotlib** - Visualización de datos
- **seaborn** - Visualización estadística
- **python-dotenv** - Gestión de variables de entorno
- **pyyaml** - Procesamiento de archivos YAML

### Dependencias de Desarrollo

- **pytest** - Framework de testing
- **pre-commit** - Control de calidad automático
- **mypy** - Type checking
- **ruff** - Linting y formatting
- **mkdocs** - Generación de documentación

## 🔧 Configuración de Experimentos

Edita `configs/experiment_config.yaml` para configurar tus experimentos:

```yaml
model_name: "gpt-4o"           # Modelo LLM a usar
temperature: 0.0               # Creatividad del modelo (0-1)
n_executions: 10               # Número de ejecuciones
seed: 42                       # Seed para reproducibilidad
```

## 🧪 Desarrollo y Testing

### Ejecutar tests

```bash
uv run pytest
```

### Ejecutar tests con cobertura

```bash
uv run pytest --cov=src
```

### Verificar calidad de código

```bash
make check
```

### Formatear código

```bash
make lint
```

## 📝 Notas de Desarrollo

- Todos los scripts deben incluir type hints (static typing)
- Docstrings en formato Google
- Pre-commit hooks se ejecutan automáticamente antes de cada commit
- Covertura mínima requerida: 90%

## 📚 Documentación

Para más información sobre el proyecto, consulta la documentación en `docs/`.

Para servir la documentación localmente:

```bash
make docs
```

## ✨ Características Principales

- **Sistema de Votación Agéntica**: Implementación de un novedoso sistema donde múltiples agentes LLM votan sobre la relevancia
- **Comparación con G-EVAL**: Benchmarking contra el framework G-EVAL existente
- **DeepEval Integration**: Uso de DeepEval para métricas de evaluación
- **Reproducibilidad**: Configuración completa para experimentos reproducibles
- **Código de Calidad**: Type checking, linting y testing automáticos

## 📄 Licencia

Este proyecto está bajo licencia MIT License.

## 👤 Autor

**Laura Granda** - [@LauraGranda](https://github.com/LauraGranda)
