# Dataset Selection — HU-00
## Evaluación de Relevancia en Agentes de IA Conversacionales: G-Eval vs. Sistema de Votación Agéntico

> **Tesis de Maestría** | Framework: DeepEval | Métrica objetivo: Relevance (respuesta vs. pregunta/contexto)

---

## 1. Listado de Datasets Candidatos

### Dataset 1 — DailyDialog-Zhao

| Campo | Detalle |
|---|---|
| **Nombre** | DailyDialog-Zhao (Zhao et al., 2020) |
| **Fuente** | ACL 2020 — *"Designing Precise and Robust Dialogue Response Evaluators"* |
| **Tamaño** | 900 pares contexto-respuesta (100 diálogos × 9 respuestas por contexto) |
| **Dominio** | Diálogo abierto en vida cotidiana (open-domain, daily conversation) |
| **Tipo de anotaciones** | Humanas explícitas (MTurk, 4 anotadores por par). Dimensiones: **Relevance** y Appropriateness, escala Likert. Krippendorff α > 0.8 tras eliminación de outliers por MAD |
| **Escala de puntuación** | 1 a 5 (Likert continua, compatible nativamente con G-Eval) |
| **Idioma** | Inglés |
| **URL** | https://github.com/ZHAOTING/dialog-processing |
| **Respuestas generadas por IA** | Sí: LSTM Seq2Seq, HRED, CVAE, GPT-2 small, GPT-2 medium + decodificación variada; 1 respuesta humana de referencia por contexto |

---

### Dataset 2 — FED-Turn (Fine-grained Evaluation of Dialogue)

| Campo | Detalle |
|---|---|
| **Nombre** | FED-Turn (Mehri & Eskenazi, 2020b) |
| **Fuente** | SIGDIAL 2020 — *"Unsupervised Evaluation of Interactive Dialog with DialoGPT"* |
| **Tamaño** | 375 turnos conversacionales (40 Human-Meena, 44 Human-Mitsuku, 40 Human-Human) |
| **Dominio** | Diálogo abierto con asistentes conversacionales (open-domain chatbot) |
| **Tipo de anotaciones** | Humanas explícitas (AMT, 5 trabajadores por turno). Dimensión: **"Relevant"** — *"Is the response relevant to the conversation?"*. IAA Spearman = 0.753 para relevancia (el más alto del dataset). 18 dimensiones de calidad anotadas en total |
| **Escala de puntuación** | No / Somewhat / Yes (equivalente a 1–2–3) |
| **Idioma** | Inglés |
| **URL** | https://github.com/Shikib/fed — datos directos: http://shikib.com/fed_data.json |
| **Respuestas generadas por IA** | Sí: Meena (Google, 2.6B params) y Mitsuku (basado en reglas AIML, 5 veces ganador del Turing Test Loebner Prize) |

---

### Dataset 3 — Topical-Chat USR

| Campo | Detalle |
|---|---|
| **Nombre** | Topical-Chat USR (Mehri & Eskenazi, 2020a) |
| **Fuente** | ACL 2020 — *"USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation"* |
| **Tamaño** | 360 pares contexto-respuesta (60 contextos × 6 respuestas: 4 estrategias de decodificación + 1 humana + 1 ground-truth) |
| **Dominio** | Diálogo informativo grounded en conocimiento factual (Topical-Chat dataset de Amazon) |
| **Tipo de anotaciones** | Humanas explícitas (CMU, 3 anotadores expertos por par). Dimensiones: Maintains Context, Interesting, Uses Knowledge, Fluent, Overall. IAA Spearman = 0.56 para Maintains Context |
| **Escala de puntuación** | 1 a 3 (limitada; requiere normalización para comparar con G-Eval 1-5) |
| **Idioma** | Inglés |
| **URL** | http://shikib.com/tc_usr_data.json — paper: https://aclanthology.org/2020.acl-main.64/ |
| **Respuestas generadas por IA** | Sí: Transformer 6 capas, 4 estrategias de decodificación distintas; 1 respuesta humana |

---

### Dataset 4 — HUMOD

| Campo | Detalle |
|---|---|
| **Nombre** | HUMOD — Human Moderated Dialogue Dataset (Merdivan et al., 2020) |
| **Fuente** | IEEE Access 2020 — *"Human Annotations for Conversational AI"* |
| **Tamaño** | 9,500 pares diálogo-respuesta (8,500 train + 1,000 test) |
| **Dominio** | Diálogo de películas (Cornell Movie Dialogue Corpus) — dominio abierto |
| **Tipo de anotaciones** | Humanas explícitas (AMT, 3 anotadores por par). **Relevancia es la ÚNICA dimensión anotada**, definida como grado en que la respuesta es pertinente al contexto conversacional |
| **Escala de puntuación** | 1 a 5 (1 = Irrelevant, 5 = Highly relevant) |
| **Idioma** | Inglés |
| **URL** | https://github.com/erincmer/HUMOD |
| **Respuestas generadas por IA** | No — respuestas son humanas originales del corpus cinematográfico; limitación para este contexto |

---

### Dataset 5 — DSTC10 Metric Track (Topical-DSTC10 + Persona-DSTC10)

| Campo | Detalle |
|---|---|
| **Nombre** | DSTC10 Track 5 — Automatic Evaluation of Open-Domain Dialogue (Zhang et al., 2021) |
| **Fuente** | DSTC10 Challenge 2021 — *"Automatic Evaluation and Moderation of Open-domain Dialogue Systems"* |
| **Tamaño** | ~2,000 pares turn-level (combinación de Topical-DSTC10 y Persona-DSTC10 como hidden test sets) |
| **Dominio** | Dual: diálogo basado en conocimiento (Topical-Chat) y diálogo basado en persona (PersonaChat) |
| **Tipo de anotaciones** | Humanas explícitas. Dimensiones: Appropriateness, Content, Grammar y **Relevance** (1-5). Utilizado como benchmark oficial de la comunidad DSTC. LLM-Eval (2023) reportó Spearman para Relevance: Claude v1.3 ρ = 0.428 (Topical) / 0.521 (Persona); ChatGPT ρ = 0.373 / 0.488 |
| **Escala de puntuación** | 1 a 5 |
| **Idioma** | Inglés |
| **URL** | https://github.com/e0397123/dstc10_metric_track — plataforma: https://chateval.org/dstc10 |
| **Respuestas generadas por IA** | Sí: múltiples sistemas de diálogo participantes en el challenge DSTC10 |

---

### Dataset 6 — Zhang et al. 12 Datasets (AAAI 2024 Comprehensive Analysis)

| Campo | Detalle |
|---|---|
| **Nombre** | Comp-Analysis (Zhang et al., AAAI 2024) — compilación de 12 sub-datasets |
| **Fuente** | AAAI 2024 — *"A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators"* |
| **Tamaño** | ~3,000–5,000 pares aprox. (12 sub-datasets: 6 turn-level incluyendo DailyDialog-Zhao, FED-Turn, Topical-USR, Persona-Zhao, Persona-USR, ConTurE-Turn; 6 dialogue-level) |
| **Dominio** | Múltiple: diálogo abierto, basado en persona, grounded en conocimiento |
| **Tipo de anotaciones** | Mixtas — hereda anotaciones de cada sub-dataset. Evaluación con 30 LLMs (GPT-4, PaLM-2, ChatGPT, LLaMA-2, Falcon, etc.). GPT-4 logró Pearson r = 0.704 promediado para relevancia sobre los 6 datasets turn-level |
| **Escala de puntuación** | Heterogénea según sub-dataset (1-3, 1-5); requiere normalización |
| **Idioma** | Inglés |
| **URL** | https://github.com/e0397123/comp-analysis — paper: https://ojs.aaai.org/index.php/AAAI/article/view/29923 |
| **Respuestas generadas por IA** | Sí: todos los sub-datasets incluyen respuestas de modelos IA; mezcla de modelos 2019-2023 |

---

## 2. Matriz de Evaluación

> **Leyenda:** ✅ Cumple plenamente | ⚠️ Cumple parcialmente o con limitaciones | ❌ No cumple

| Criterio | DailyDialog-Zhao | FED-Turn | Topical-Chat USR | HUMOD | DSTC10 | Zhang AAAI 2024 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1. Anotaciones humanas explícitas de RELEVANCIA** | ✅ Dimensión "Relevance" explícita, escala 1-5, α > 0.8 | ✅ Dimensión "Relevant" con el nombre exacto, IAA Spearman 0.753 | ⚠️ "Maintains Context" como proxy de relevancia, sin dimensión nominal | ✅ Relevancia como única dimensión, 3 anotadores | ✅ Dimensión "Relevance" explícita (1-5), resultados LLM publicados | ⚠️ Hereda anotaciones de cada sub-dataset; escalas heterogéneas |
| **C2. Respuestas generadas por IA** | ✅ 6 modelos: LSTM, HRED, CVAE, GPT-2 sm/md + variantes de decodificación | ✅ Meena (2.6B) y Mitsuku; modelos IA reales de producción | ✅ Transformer 6 capas, 4 estrategias de decodificación | ❌ Respuestas humanas del corpus cinematográfico | ✅ Múltiples sistemas del DSTC10 challenge | ✅ Mezcla de modelos 2019-2023 incluyendo GPT-2 y modelos modernos |
| **C3. Disponibilidad pública sin restricciones** | ✅ GitHub público, descarga directa, sin registro | ✅ GitHub público, JSON descargable directamente | ✅ URL directa JSON sin registro | ✅ GitHub público, licencia abierta | ⚠️ Requiere solicitud a organizadores DSTC; parte disponible en ChatEval | ✅ GitHub público, licencia MIT, datos incluidos |
| **C4. Resultados previos publicados (G-Eval o LLM-as-Judge)** | ⚠️ Zhang et al. AAAI 2024 con 30 LLMs (GPT-4 Pearson r ≈ 0.704 promediado); NO con G-Eval canónico | ⚠️ Zhang et al. AAAI 2024 incluido; métricas existentes máx. Spearman 0.152 para relevancia | ✅ ÚNICO con G-Eval canónico publicado: Coherence ρ = 0.605 GPT-4, Engagingness ρ = 0.631 (Liu et al., EMNLP 2023) | ❌ Sin resultados LLM-as-Judge ni G-Eval publicados | ✅ LLM-Eval 2023: Relevance Spearman Claude 0.428-0.521, ChatGPT 0.373-0.488 | ✅ Resultados más completos: 30 LLMs, GPT-4 mejor en relevancia (Pearson 0.704 promediado) |
| **C5. Tamaño manejable (< 5,000 pares)** | ✅ 900 pares — óptimo. Costo GPT-4o ≈ USD $9 total | ✅ 375 turnos — compacto. Costo GPT-4o ≈ USD $3-4 | ✅ 360 pares — mínimo estadístico. Costo GPT-4o ≈ USD $3.60 | ⚠️ 9,500 pares (subsampling recomendado). Costo full ≈ USD $85 | ⚠️ ~2,000 pares, acceso parcial; tamaño exacto no siempre público | ⚠️ 3,000-5,000 pares aprox.; normalización de escalas requerida |
| **C6. Compatibilidad con DeepEval** | ✅ JSON estándar → mapeo directo a `LLMTestCase(input=context, actual_output=response)`. Escala 1-5 nativa con G-Eval | ✅ JSON con estructura turn/response/scores. Escala 1-3 requiere mapeo simple | ⚠️ JSON directo, pero escala 1-3 requiere normalización; "Maintains Context" no es relevancia nominal | ✅ JSON estándar, escala 1-5 nativa, preprocesamiento mínimo | ⚠️ Requiere acceso completo a datos; estructura DSTC puede necesitar adaptación | ⚠️ Heterogeneidad de escalas y formatos; normalización entre sub-datasets necesaria |

### Resumen de puntuación (✅ = 2 pts, ⚠️ = 1 pt, ❌ = 0 pts)

| Dataset | C1 | C2 | C3 | C4 | C5 | C6 | **Total / 12** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DailyDialog-Zhao | 2 | 2 | 2 | 1 | 2 | 2 | **11** |
| FED-Turn | 2 | 2 | 2 | 1 | 2 | 2 | **11** |
| Topical-Chat USR | 1 | 2 | 2 | 2 | 2 | 1 | **10** |
| HUMOD | 2 | 0 | 2 | 0 | 1 | 2 | **7** |
| DSTC10 | 2 | 2 | 1 | 2 | 1 | 1 | **9** |
| Zhang AAAI 2024 | 1 | 2 | 2 | 2 | 1 | 1 | **9** |

---

## 3. Dataset Ganador: DailyDialog-Zhao

### Justificación

**DailyDialog-Zhao** (Zhao, Lala & Kawahara, ACL 2020) es el dataset más adecuado para los experimentos de esta tesis de maestría centrada en la evaluación de relevancia en agentes de IA conversacionales usando G-Eval versus un sistema de votación agéntico. Esta conclusión surge de la evaluación sistemática de seis criterios técnicos y metodológicos, y se sustenta en argumentos tanto cuantitativos como estratégicos para la investigación.

**En primer lugar**, DailyDialog-Zhao es uno de los pocos datasets de diálogo que posee la dimensión **Relevance anotada de forma explícita** por humanos calificados, con una escala Likert de 1 a 5. Esta no es una dimensión aproximada ni derivada de otra: es la anotación directa que se define como el grado en que la respuesta se ajusta al contexto de la conversación, que es precisamente la variable dependiente que esta tesis busca medir. La alta correlación Pearson (r = 0.91) entre Relevance y Appropriateness, lejos de ser una debilidad, confirma la validez de constructo: ambas dimensiones capturan aspectos conceptualmente cercanos de la calidad conversacional, y la comunidad académica las trata como dimensiones complementarias. El acuerdo inter-anotador robusto (Krippendorff α > 0.8 tras eliminación de outliers con la Median Absolute Deviation) garantiza que el ground truth es confiable y reproducible como referencia comparativa.

**En segundo lugar**, las **respuestas fueron generadas por seis modelos de IA distintos** con múltiples estrategias de decodificación: LSTM Seq2Seq, HRED, CVAE, GPT-2 small y GPT-2 medium, más sus variantes de decodificación (greedy, beam search, top-k). Esto crea una distribución de calidad variada y realista — con puntuaciones que van de respuestas claramente irrelevantes a respuestas apropiadas — que es esencial para que tanto G-Eval como el sistema de votación agéntico tengan rango suficiente para demostrar capacidad discriminativa. Un dataset donde todas las respuestas son de alta calidad comprimiría artificialmente las correlaciones de Spearman.

**En tercer lugar**, el **tamaño de 900 pares** es óptimo para el alcance de una tesis de maestría por tres razones complementarias. Primero, es estadísticamente suficiente para obtener intervalos de confianza estrechos en las correlaciones de Spearman (con n = 900 y α = 0.05, el error estándar de ρ ≈ 0.033, significativamente mejor que los 360 pares de Topical-Chat USR con error estándar ≈ 0.053). Segundo, las 900 evaluaciones son ejecutables en pocas horas de tiempo de cómputo, permitiendo iteraciones rápidas durante el desarrollo de los prompts de evaluación.

**En cuarto lugar**, la **compatibilidad con DeepEval es nativa y directa**. La escala 1-5 coincide exactamente con la escala de puntuación que G-Eval utiliza internamente en DeepEval (`GEval` con `threshold` entre 0 y 1 normalizado desde la escala 1-5), eliminando cualquier transformación que pudiera introducir sesgos metodológicos. El formato JSON de descarga directa mapea sin ambigüedad a la clase `LLMTestCase(input=context, actual_output=response, expected_output=ground_truth)`, y los scores de relevancia humana pueden cargarse como referencia para calcular correlaciones de Spearman con los puntajes de G-Eval y del sistema de votación agéntico.

**En quinto lugar**, desde una perspectiva estratégica para la investigación, la investigación exhaustiva realizada revela que **no existe ningún resultado de G-Eval canónico publicado para la dimensión relevancia sobre datasets de diálogo**. G-Eval (Liu et al., EMNLP 2023) evaluó relevancia únicamente sobre SummEval (sumarización, ρ = 0.547), y su benchmark en diálogo (Topical-Chat USR) no incluyó una dimensión de relevancia nominal. Esto significa que **aplicar G-Eval canónico sobre DailyDialog-Zhao generaría los primeros resultados publicados de G-Eval para relevancia en diálogo**, constituyendo por sí misma una contribución original al campo. Los números de referencia de Zhang et al. (AAAI 2024) con 30 LLMs —incluyendo GPT-4 con Pearson r = 0.704 promediado sobre datasets turn-level— proporcionan un benchmark razonable para contextualizar los resultados, aun cuando no sean G-Eval canónico.

**Finalmente**, DailyDialog-Zhao ha sido utilizado como benchmark en al menos cinco estudios académicos recientes y relevantes —Zhang et al. (AAAI 2024), Vasselli et al. (COLING 2025), Mendonca et al. (NLP4ConvAI 2024), Robichaud et al. (2023) y el DSTC10 Track 5— lo que le otorga validez comunitaria y permite que los resultados de esta tesis sean contextualizados e interpretados por el jurado evaluador y la comunidad académica de NLP en diálogo.

Es de tener en cuenta que: Las respuestas evaluadas fueron generadas por modelos de lenguaje generativos de la generación anterior y no por agentes conversacionales modernos con arquitecturas RAG, memoria o herramientas. La validación de los métodos sobre conversaciones de agentes modernos se propone como línea de trabajo futuro.
---

## 4. Referencias

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *Proceedings of EMNLP 2023*, 2511–2522. https://aclanthology.org/2023.emnlp-main.153/

Mehri, S., & Eskenazi, M. (2020a). USR: An unsupervised and reference free evaluation metric for dialog generation. *Proceedings of ACL 2020*, 681–707. https://aclanthology.org/2020.acl-main.64/

Mehri, S., & Eskenazi, M. (2020b). Unsupervised evaluation of interactive dialog with DialoGPT. *Proceedings of the 2nd Workshop on NLP for ConvAI*. https://arxiv.org/abs/2006.12719

Merdivan, E., Singh, D., Hanke, S., & Holzinger, A. (2020). Human annotated dialogues dataset for natural conversational agents. *IEEE Access*. https://github.com/erincmer/HUMOD

Zhang, C., D'Haro, L. F., Chen, Y., Zhang, M., & Li, H. (2024). A comprehensive analysis of the effectiveness of large language models as automatic dialogue evaluators. *Proceedings of AAAI 2024*, 38(17), 19515–19524. https://ojs.aaai.org/index.php/AAAI/article/view/29923

Zhang, C., D'Haro, L. F., Banchs, R. E., Friedrichs, T., & Li, H. (2021). Automatic evaluation and moderation of open-domain dialogue systems. *DSTC10 Track 5*. https://github.com/e0397123/dstc10_metric_track

Zhao, T., Lala, D., & Kawahara, T. (2020). Designing precise and robust dialogue response evaluators. *Proceedings of ACL 2020*, 26–33. https://aclanthology.org/2020.acl-main.4/

Lin, Z., & Chen, Y. (2023). LLM-Eval: Unified multi-dimensional automatic evaluation for open-domain conversations with large language models. *Proceedings of NLP4ConvAI at ACL 2023*. https://aclanthology.org/2023.nlp4convai-1.5/

Vasselli, J., Bhatt, A., Kang, J., & Solberg, L. (2025). Measuring the robustness of reference-free dialogue evaluation systems. *Proceedings of COLING 2025*. https://arxiv.org/abs/2501.06728

---

