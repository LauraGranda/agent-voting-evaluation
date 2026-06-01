# Agent Panel Design - HU-06
## Diseño del Panel de Agentes Juez del Sistema de Votación Agéntico

> Tesis de Maestría | Componente: Sistema de Votación Agéntico | Métrica objetivo: Relevance | Escala: 1 a 5 ordinal

---

## 1. Contexto de la tesis

Este panel constituye el cuerpo evaluador del sistema de votación agéntico que se contrasta con G-Eval (Liu et al., 2023) sobre el dataset DailyDialog-Zhao. Se eligen tres agentes juez para satisfacer simultáneamente dos requisitos del Definition of Done: diversidad de proveedores (al menos dos proveedores distintos de modelos de lenguaje) y un número impar de votos que facilite la agregación por media y por mediana, esta última siempre definida sin reglas de desempate.

---

## 2. Fundamentación metodológica

La decisión metodológica central es deliberada: el modelo `gpt-4o` aparece simultáneamente como evaluador único en el baseline G-Eval y como uno de los tres jueces del panel. La elección no es accidental, sino el mecanismo principal de control experimental.

En esta tesis se sostiene constante el prompt de evaluación de relevancia, V3, en los tres jueces del panel. V3 es exactamente el mismo prompt utilizado por el baseline G-Eval. Con el prompt constante y con `gpt-4o` presente en ambos sistemas, el juez `judge_openai` del panel difiere del baseline G-Eval únicamente en dos factores: (i) la mecánica de extracción del score (G-Eval emplea `GEval` de DeepEval, que rellena un formulario sobre las probabilidades de los tokens; el juez del panel usa prompting directo y un parser explícito del score en la salida) y (ii) la participación en un panel agregado en lugar de un juicio individual. Por tanto, la comparación `judge_openai` frente a G-Eval queda como una prueba controlada del cambio de mecánica de scoring, y la comparación del panel agregado frente a G-Eval queda como una prueba del cambio combinado de mecánica de scoring y agregación. Cualquier diferencia en la correlación de Spearman contra el `human_score` se interpreta a la luz de estas dos contribuciones, sin que la variación de prompt sea una hipótesis alternativa que el director o el jurado puedan oponer.

Esta decisión sustituye, de manera explícita, el requisito de diversidad estilística de prompts por el requisito de control científico de la comparación. La consecuencia en la mitigación de sesgos se documenta en la Sección 7 y la limitación derivada en la Sección 9.

---

## 3. Arquitectura del panel

El siguiente diagrama representa la topología de evaluación. Cada juez recibe el mismo input y produce su puntuación de manera aislada; las tres puntuaciones se entregan al agregador definido en la historia de usuario anterior.

```text
  Conversation Input (turns + response)
           |
           v
  +-----------------------------------------+
  |              AGENT PANEL                |
  |  +----------+ +----------+ +----------+ |
  |  | Agent 1  | | Agent 2  | | Agent 3  | |
  |  |  gpt-4o  | |  gemini  | |  claude  | |
  |  | (OpenAI) | | 2.5-flash| | haiku-4-5| |
  |  +----------+ +----------+ +----------+ |
  |   Score 1      Score 2      Score 3     |
  |        [evaluated independently]        |
  +-----------------------------------------+
           |
           v
     Aggregator (HU anterior: media + mediana de robustez)
           |
           v
     Final Vote Score (1 a 5)
```

---

## 4. Descripción de los agentes

### 4.1 Agent 1: `judge_openai` (`gpt-4o`)

Modelo `gpt-4o` del proveedor OpenAI, configurado en `configs/agents/agent_openai.yaml`. La selección obedece al control metodológico descrito en la Sección 2: al coincidir con el modelo del baseline G-Eval y al usar el mismo prompt V3, este juez aísla el efecto del cambio de mecánica de scoring respecto al baseline. El estilo de razonamiento del prompt es el del V3: cinco pasos de chain-of-thought, rúbrica anclada por nivel y tres ejemplos contextualizados. La temperatura se fija en 0.0 para garantizar reproducibilidad del juicio dado el mismo input; los logs y resultados podrán por tanto compararse de manera determinista entre corridas.

### 4.2 Agent 2: `judge_google` (`gemini-2.5-flash`)

Modelo `gemini-2.5-flash` del proveedor Google, configurado en `configs/agents/agent_google.yaml`. Aporta la segunda familia de proveedor al panel, requisito explícito del Definition of Done. La elección de la variante `flash` privilegia un coste sustancialmente menor que `gpt-4o` con calidad comparable para tareas de evaluación, lo cual hace posible mantener el presupuesto del experimento dentro del orden de magnitud del baseline. El prompt es V3, idéntico al usado por `judge_openai`. La temperatura se fija en 0.0 por la misma razón de determinismo. El modo de razonamiento extendido (thinking) se desactivará en el runner posterior para preservar la comparabilidad con los otros dos jueces.

### 4.3 Agent 3: `judge_anthropic` (`claude-haiku-4-5`)

Modelo `claude-haiku-4-5` del proveedor Anthropic, lanzado en octubre de 2025, configurado en `configs/agents/agent_anthropic.yaml`. Aporta la tercera familia de proveedor y completa la diversidad requerida. El prompt es V3. La temperatura se fija en 0.0. El razonamiento extendido (extended thinking) no se utiliza, por la misma razón de comparabilidad de condiciones de evaluación entre los tres jueces.

---

## 5. Garantía de diversidad

La diversidad del panel se restringe a los ejes de proveedor y modelo, como consecuencia directa de la decisión metodológica de prompt único descrita en la Sección 2.

| Dimensión | Agent 1 | Agent 2 | Agent 3 |
|---|---|---|---|
| Proveedor | OpenAI | Google | Anthropic |
| Modelo | gpt-4o | gemini-2.5-flash | claude-haiku-4-5 |

Nota explícita: el diseño no busca diversidad de estilo de prompt. Ese eje, que en otras propuestas de panel de jueces se utiliza como mecanismo adicional de mitigación de sesgo (Verga et al., 2024), se sacrifica de manera consciente para preservar el control científico de la comparación con G-Eval. El precio metodológico de esta decisión se asume y se documenta como limitación en la Sección 9.

---

## 6. Mecanismo de independencia

Cada juez del panel cumple las siguientes condiciones de independencia, ya declaradas en el bloque de comentarios `INDEPENDENCE GUARANTEE` al inicio de cada YAML:

- Recibe el mismo input de manera aislada, sin información lateral.
- No tiene acceso a las puntuaciones de los otros jueces.
- No tiene acceso a las puntuaciones de G-Eval.
- No tiene acceso a las anotaciones humanas ni a derivados de estas.
- Usa `temperature: 0.0`, lo que garantiza una salida determinista para el mismo input.

El runner que se implementará en una HU posterior debe respetar este contrato mediante tres llamadas independientes a tres APIs distintas, sin compartir contexto entre llamadas.

---

## 7. Estrategia de mitigación de sesgos

Con la decisión de prompt único, tres de los cuatro riesgos del diseño original quedan mitigados; el cuarto, el sesgo de longitud, se transfiere a la Sección 9 como limitación conocida.

| Riesgo | Mecanismo de mitigación |
|---|---|
| Autopreferencia del modelo (un modelo puntúa mejor las respuestas que se parecen a su propia distribución) | Tres proveedores distintos en el panel. La autopreferencia, si existe, no se reforzaría por consenso. |
| Compresión de la distribución hacia el centro (tendencia a puntuar 3) | Rúbrica anclada del V3, con un descriptor conductual por nivel (1 a 5) y tres ejemplos. Esto fuerza al juez a usar todo el rango y no refugiarse en el valor medio. |
| No determinismo entre corridas | `temperature: 0.0` en los tres YAML. Cualquier diferencia entre re-ejecuciones será atribuible a cambios externos al panel. |

El sesgo de longitud (la posibilidad de que los jueces sub-puntúen respuestas breves pero relevantes) no queda mitigado por el prompt V3, pues este no incluye una instrucción explícita en ese sentido. Se documenta y se vigila en la Sección 9 y en el pilot posterior.

---

## 8. Relación con G-Eval

| Aspecto | G-Eval (HU-04) | Sistema de votación (HU-06 y siguientes) |
|---|---|---|
| Modelo o modelos | gpt-4o | gpt-4o, gemini-2.5-flash, claude-haiku-4-5 |
| Número de jueces | 1 | 3 |
| Agregación | Form-filling sobre probabilidades en DeepEval (`GEval`) | Esquema de votación de la HU-05 (media + mediana de robustez) |
| Prompt | V3 con anclas y CoT | V3 con anclas y CoT, idéntico al de G-Eval |
| Costo aproximado por 900 pares | aproximadamente 5 USD | aproximadamente 7.61 USD |

La fila clave es la del prompt: al ser el mismo en ambos sistemas, las únicas variables que cambian entre G-Eval y el sistema de votación son la mecánica de scoring (form-filling con logprobs frente a prompting directo con parser) y el número de jueces con su agregación.

---

## 9. Limitaciones conocidas

- El modelo `gpt-4o` aparece tanto en el baseline G-Eval como en el panel. Si el modelo tiene un sesgo sistemático en la evaluación de relevancia, este sesgo se replica en parte en el panel. La mitigación parcial proviene de la diferencia de mecánica de scoring entre ambos sistemas.
- El modelo `claude-haiku-4-5` pertenece a la familia eficiente de Anthropic y es, por diseño, un modelo más pequeño que `gpt-4o` y `gemini-2.5-flash`. Existe la posibilidad de una asimetría de calidad de evaluación dentro del panel, que se vigilará en el pilot.
- Los tres modelos son propietarios. Una réplica del experimento con alternativas open-source mejoraría la reproducibilidad y se propone como trabajo futuro.
- Los tres modelos han sido entrenados con esquemas de aprendizaje por refuerzo a partir de retroalimentación humana (RLHF). Pueden compartir sesgos de evaluación heredados del entrenamiento, no presentes en los anotadores humanos del dataset.
- Sesgo de longitud no mitigado por el prompt: el V3 no contiene una instrucción explícita que indique al juez no penalizar respuestas cortas si estas son relevantes. En el pilot del panel se vigilará si los tres jueces sub-puntúan respuestas breves; si se detecta el patrón, se propondrá una corrección puntual del prompt o una extensión del análisis para reportar la sensibilidad de los resultados a la longitud de la respuesta.
- Diversidad de estilo de prompt sacrificada por control científico: en otras configuraciones (Verga et al., 2024) los jueces de un panel usan prompts deliberadamente distintos para introducir diversidad de razonamiento. Esta tesis descarta esa diversidad para asegurar el aislamiento experimental con G-Eval. Es una decisión consciente y se asume como limitación.

---

## 10. Estimación de costos

Los costos se estiman a partir de las tarifas públicas de cada proveedor a la fecha de redacción y el conteo de tokens del baseline G-Eval (entrada: 1.090.187 tokens; salida: 81.977 tokens para 900 pares). Los valores son orientativos.

| Agente | Modelo | Input USD por millón | Output USD por millón | Costo estimado 900 pares |
|---|---|---|---|---|
| judge_openai | gpt-4o | 2.50 | 10.00 | aproximadamente 4.50 USD |
| judge_google | gemini-2.5-flash | 0.30 | 2.50 | aproximadamente 0.89 USD |
| judge_anthropic | claude-haiku-4-5 | 1.00 | 5.00 | aproximadamente 2.22 USD |
| Total panel | | | | aproximadamente 7.61 USD |
| G-Eval (ya ejecutado) | gpt-4o | | | aproximadamente 5.00 USD |
| Total estimado de la tesis | | | | entre 12 y 13 USD |

El runner posterior reportará los costos reales por agente al final de la ejecución, tal como hace `scripts/run_geval.py` para el baseline.

---

## 11. Referencias

Anthropic. (2025, octubre). *Claude Haiku 4.5.* Anthropic News. https://www.anthropic.com/news/claude-haiku-4-5

Google DeepMind. (2025). *Gemini 2.5 Flash.* Google DeepMind Models. https://deepmind.google/models/gemini/flash/

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *Proceedings of EMNLP 2023*, 2511–2522. https://aclanthology.org/2023.emnlp-main.153/

OpenAI. (2024). *GPT-4o system card.* OpenAI. https://openai.com/index/gpt-4o-system-card/

Verga, P., Hofstatter, S., Althammer, S., Su, Y., Piktus, A., Arkhangorodsky, A., Xu, M., White, N., & Lewis, P. (2024). Replacing judges with juries: Evaluating LLM generations with a panel of diverse models. *arXiv preprint*. https://arxiv.org/abs/2404.18796

---
