# Voting Scheme Analysis - HU-05

## Evaluación de Relevancia en Agentes de IA Conversacionales: G-Eval frente a Sistema de Votación Agéntico

> Tesis de Maestría | Componente: Sistema de Votación Agéntico | Métrica objetivo: Relevance (respuesta frente a contexto) | Escala: 1 a 5 ordinal

---

## 1. Contexto y objetivo

Esta tesis contrasta dos enfoques para la evaluación automática de relevancia conversacional sobre el dataset DailyDialog-Zhao (900 pares contexto-respuesta, anotados por cuatro evaluadores humanos en una escala Likert de 1 a 5):

1. G-Eval, un juez único (`gpt-4o` mediante DeepEval), ya implementado y ejecutado sobre el 100% del dataset. El resultado de referencia es una correlación de Spearman ρ = 0.7565 frente al `human_score` (n = 900, p ≈ 8.5e-168).
2. Sistema de votación agéntico, un panel de varios agentes-juez de IA cuyas puntuaciones individuales se agregan en un único score final. Este documento define el esquema de agregación de ese panel.

El problema central que aquí se resuelve es de agregación de juicios: dadas las puntuaciones ordinales de 1 a 5 que emiten *k* jueces independientes sobre un mismo par, ¿cómo se combinan en un único score final, en escala de 1 a 5, directamente comparable con G-Eval mediante correlación de Spearman contra el `human_score`?

Tres restricciones delimitan la elección:

| Restricción | Implicación para el esquema |
|---|---|
| Escala ordinal de 1 a 5 | Los valores tienen orden (1 menor que 2, y así sucesivamente) pero la equidistancia entre niveles no está garantizada. El esquema debe respetar el orden y no tratar las puntuaciones como categorías nominales sin relación. |
| Comparabilidad con G-Eval | G-Eval produce una puntuación continua en el intervalo de 1 a 5 (reescalada desde el intervalo de 0 a 1). El score final del panel debe ocupar el mismo rango y, en lo posible, tener granularidad comparable para que las correlaciones de Spearman sean equiparables. |
| Contrato del módulo agregador | La historia posterior (`src/voting/aggregator.py`) recibirá `{agent_name: score}` y devolverá `{final_score, individual_scores, agreement_level}`. El esquema debe, por tanto, producir tanto `final_score` como un nivel de acuerdo interpretable. |

Un hecho del diseño experimental orienta toda la decisión: el `human_score` contra el cual se mide cada correlación es la media aritmética de los cuatro anotadores humanos (`raw_relevance_scores`). Es decir, la etiqueta de oro ya es, en sí misma, el resultado de un esquema de agregación, el promedio, aplicado a un panel humano. Este paralelismo es central en la Sección 4.

---

## 2. Esquemas de votación candidatos

Para cada esquema se describe su funcionamiento, sus ventajas, sus limitaciones y su aplicabilidad concreta a la agregación de puntuaciones de 1 a 5 emitidas por jueces de IA. La taxonomía y las descripciones de los métodos clásicos de agregación se apoyan en los tratamientos panorámicos de la teoría de la elección social de Pacuit (2019), una revisión abierta y consultable de los principales sistemas de votación, y del Handbook of Computational Social Choice de Brandt et al. (2016).

### 2.1 Mayoría simple o moda

Descripción. Cada puntuación de 1 a 5 se trata como un voto categórico; el score final es el valor más frecuente (la moda) entre los *k* jueces.

Ventajas. Máxima simplicidad e interpretabilidad. No asume nada sobre la distancia entre niveles.

Limitaciones. Primero, con tres jueces y cinco categorías los empates son la norma y no la excepción: configuraciones como {2, 3, 5} no tienen moda única. Segundo, ignora la ordinalidad, pues trata el 1 y el 5 como categorías sin relación y descarta que ocupan extremos opuestos de la escala. Tercero, descarta la magnitud: {4, 4, 1} y {4, 4, 4} producen la misma moda (4) pese a reflejar acuerdos muy distintos.

Aplicabilidad. Baja para una escala de 1 a 5 con pocos jueces. La frecuencia de empates lo vuelve poco operativo sin reglas de desempate arbitrarias.

### 2.2 Media aritmética

Descripción. El score final es el promedio de las *k* puntuaciones. Produce un valor continuo en el intervalo de 1 a 5.

Ventajas. Utiliza toda la información de magnitud de cada voto. El resultado es continuo, idéntico en naturaleza al de G-Eval, lo que maximiza la comparabilidad y la granularidad para la correlación de Spearman. Los empates son virtualmente imposibles sobre un dominio continuo, lo que aporta robustez ante empates por construcción. Mantiene un paralelismo metodológico exacto con la etiqueta humana, que es a su vez la media de cuatro anotadores. Cuenta con fundamentación clásica en la sabiduría de las multitudes y en la combinación de estimaciones.

Limitaciones. Es sensible a valores atípicos: un único juez degenerado, que siempre puntúe alto o bajo, desplaza la media. Asume de forma implícita un tratamiento de intervalo de la escala, discutible para datos ordinales (véase la Sección 3, criterio d).

Aplicabilidad. Alta. Es el esquema con mayor compatibilidad simultánea con las tres restricciones de la Sección 1.

### 2.3 Mediana

Descripción. El score final es el valor central de las *k* puntuaciones ordenadas.

Ventajas. Es robusta a valores atípicos, pues un juez degenerado no la arrastra. Es la medida de tendencia central teóricamente correcta para datos puramente ordinales, ya que solo utiliza el orden y no la distancia.

Limitaciones. Con tres jueces, la mediana es simplemente el valor de en medio, por lo que queda confinada a la grilla entera {1, 2, 3, 4, 5} y pierde la granularidad continua que sí aporta la media. Esa discretización comprime la correlación de Spearman frente a una puntuación continua. Para un número par de jueces requiere una regla adicional (el promedio de los dos valores centrales).

Aplicabilidad. Alta como complemento de robustez; moderada como agregador único, por la pérdida de granularidad con pocos jueces.

### 2.4 Votación ponderada

Descripción. Es una media ponderada en la que el peso de cada juez refleja su fiabilidad o competencia, por ejemplo su correlación histórica con el juicio humano.

Ventajas. Puede atenuar el efecto de jueces débiles y amplificar el de los fiables, e incorpora conocimiento previo sobre la calidad de cada modelo.

Limitaciones. Requiere datos de calibración para estimar los pesos de forma principista. Fijarlos a mano es arbitrario, y estimarlos sobre el mismo dataset de evaluación introduce riesgo de sobreajuste y circularidad metodológica. Añade complejidad y reduce la transparencia.

Aplicabilidad. Moderada. Es potente, pero prematura para una línea base; conviene documentarla como trabajo futuro, una vez se disponga de un conjunto de calibración independiente.

### 2.5 Recuento de Borda (Borda count)

Descripción. Es un método clásico de elección social para agregar ordenamientos: cada votante ordena un conjunto de candidatos y a cada candidato se le asignan puntos según su posición; gana el de mayor suma.

Ventajas. Utiliza información ordinal de forma natural y es robusto. Está ampliamente estudiado en la teoría de la elección social.

Limitaciones. Presenta una desalineación conceptual con el problema. Borda agrega ordenamientos de varios candidatos por votante, mientras que aquí cada juez asigna una puntuación absoluta a una única respuesta, no un ordenamiento de respuestas alternativas. Aplicarlo exigiría reformular la tarea como un problema de ordenamiento (ordenar respuestas dentro de una conversación), lo que no produce una puntuación absoluta de 1 a 5 comparable con G-Eval, que es el objetivo. Además, hereda las paradojas de la elección social, como la sensibilidad a alternativas irrelevantes (Arrow, 1951).

Aplicabilidad. Baja para puntuación absoluta. Solo sería pertinente si la tarea se redefiniera como selección u ordenamiento de la mejor respuesta, lo cual queda fuera del alcance de esta tesis.

---

## 3. Criterios de selección

La elección se evalúa contra siete criterios. Los cuatro primeros son los exigidos por la definición de terminado de la historia; los tres últimos son específicos del diseño de esta tesis y refuerzan la decisión. En la tabla, "Sí" indica que el esquema satisface el criterio, "Parcial" que lo satisface con reservas y "No" que no lo satisface.

| Criterio | Mayoría/moda | Media | Mediana | Ponderada | Borda |
|---|---|---|---|---|---|
| (a) Robustez ante empates | No | Sí | Parcial | Sí | Parcial |
| (b) Simplicidad e interpretabilidad | Sí | Sí | Sí | Parcial | No |
| (c) Fundamentación en literatura | Sí | Sí | Sí | Sí | Sí |
| (d) Compatibilidad con escala ordinal de 1 a 5 | No | Sí | Sí | Sí | Parcial |
| (e) Comparabilidad directa con G-Eval (continuo de 1 a 5) | No | Sí | Parcial | Sí | No |
| (f) Paralelismo con la etiqueta humana (media de 4 anotadores) | No | Sí | Parcial | Parcial | No |
| (g) Produce un `agreement_level` significativo | Parcial | Sí | Parcial | Sí | Parcial |

Definición y justificación de cada criterio:

- (a) Robustez ante empates. Con tres jueces y cinco niveles, un esquema categórico produce empates en una fracción alta de los casos y obliga a reglas de desempate arbitrarias que introducen sesgo. Un esquema definido sobre un dominio continuo elimina el problema de raíz.
- (b) Simplicidad e interpretabilidad. Una línea base de tesis debe ser transparente y reproducible; el director y el jurado deben poder entender y auditar la regla de agregación sin parámetros ocultos.
- (c) Fundamentación en literatura. La regla debe apoyarse en cuerpos teóricos establecidos (elección social, sabiduría de las multitudes, combinación de pronósticos, paneles de jueces basados en modelos de lenguaje) y no en una heurística improvisada.
- (d) Compatibilidad con la escala ordinal de 1 a 5. La escala tiene orden; el esquema no debe destruirlo, como hace la moda al tratar los niveles como categorías nominales, ni asumir más estructura de la defendible.
- (e) Comparabilidad directa con G-Eval. Para que el contraste central de la tesis sea limpio, el score del panel debe ocupar el mismo rango continuo de 1 a 5 y permitir las mismas correlaciones de Spearman que G-Eval.
- (f) Paralelismo con la etiqueta humana. El `human_score` es la media de un panel humano; usar el mismo operador en el panel de IA hace que la comparación sea entre objetos metodológicamente equivalentes.
- (g) Nivel de acuerdo significativo. El contrato del agregador exige un nivel de acuerdo; el esquema debe admitir una medida de dispersión interpretable y continua.

---

## 4. Esquema seleccionado y justificación

Esquema seleccionado: la media aritmética como agregador principal, acompañada de la mediana reportada en paralelo como verificación de robustez.

La media aritmética es el esquema que satisface de forma simultánea los siete criterios de la Sección 3, y su elección se sostiene en cinco argumentos que, en conjunto, la hacen la opción más rigurosa y defendible para esta tesis.

En primer lugar, y con el mayor peso, está el paralelismo metodológico con la etiqueta de oro. El `human_score` contra el cual se mide toda correlación no es un valor único, sino la media aritmética de los cuatro anotadores humanos. La etiqueta de referencia es, por construcción, el resultado de promediar un panel. Si el sistema de votación agéntico promedia su propio panel de jueces, entonces el score del panel de IA se convierte en el análogo exacto del score humano, es decir, un panel de evaluadores agregado con idéntico operador. Esta simetría convierte el experimento en una comparación limpia entre un panel humano (cuatro anotadores, media) y un panel de IA (k jueces, media), y elimina una variable de confusión que cualquier otro operador introduciría.

En segundo lugar, está la comparabilidad directa con G-Eval. G-Eval emite una puntuación continua en el intervalo de 1 a 5. La media produce igualmente un valor continuo en ese mismo intervalo, de modo que ambos sistemas son comparables sin transformaciones adicionales y sus correlaciones de Spearman contra el juicio humano se interpretan en pie de igualdad. La mediana, en cambio, con solo tres jueces quedaría confinada a la grilla entera de uno a cinco; esa discretización comprime de forma artificial la correlación de Spearman y restaría granularidad al contraste, precisamente el riesgo que el propio documento de selección del dataset advierte sobre las escalas que comprimen las correlaciones.

En tercer lugar, está la robustez ante empates por construcción. Sobre un dominio continuo, la probabilidad de que dos pares distintos reciban exactamente el mismo `final_score` es prácticamente nula, y no existe el concepto de empate sin resolver que aqueja a la mayoría simple. El criterio (a) de la definición de terminado queda así satisfecho sin necesidad de reglas de desempate.

En cuarto lugar, está la fundamentación en literatura. El promedio de juicios independientes es el mecanismo canónico de la sabiduría de las multitudes, cuyo origen empírico se remonta a Galton (1907), y de la combinación de estimaciones, ámbito en el que Clemen (1989) documenta que promediar pronósticos supera de forma consistente a los pronosticadores individuales. En el dominio específico de la evaluación con modelos de lenguaje, Verga et al. (2024) muestran que un panel de jueces diversos reduce el sesgo intramodelo y mejora la alineación con el juicio humano respecto a un juez único, y Zheng et al. (2023) documentan los sesgos sistemáticos (de posición, de verbosidad y de autopreferencia) que motivan agregar varios jueces en lugar de confiar en uno solo.

En quinto lugar, está la simplicidad e interpretabilidad. La media no tiene parámetros ocultos ni requiere datos de calibración, a diferencia de la votación ponderada, por lo que es plenamente reproducible y auditable por el director y el jurado.

Manejo de empates y casos extremos. Ante el valor faltante de un juez (un fallo de la interfaz de programación tras los reintentos), la media se calcula sobre los jueces disponibles y el caso se marca con una bandera; el `agreement_level` se reporta con el número efectivo de jueces. Ante un juez degenerado (varianza casi nula a lo largo del dataset, detectable en el pilotaje de la historia correspondiente), se conmuta a la mediana o se corrige el prompt del agente antes del run completo. Ante una discrepancia máxima, por ejemplo una pareja de votos {1, 5}, la media la refleja como un valor intermedio y, lo más importante, el `agreement_level` la señala de forma explícita como un caso de bajo consenso.

Definición del nivel de acuerdo. Se define en dos niveles complementarios. A nivel de ítem, como campo de salida del agregador, se usa una medida de dispersión normalizada de las puntuaciones, calculada como uno menos el cociente entre la desviación estándar observada y la desviación máxima posible en la escala de 1 a 5 para ese número de jueces; un valor de uno indica acuerdo perfecto y un valor de cero, desacuerdo máximo. A nivel de dataset, para el análisis comparativo, se usa la fiabilidad entre jueces mediante el alfa de Krippendorff ordinal, el coeficiente de correlación intraclase ICC(2,1) y la correlación de Spearman por pares, reutilizando las funciones ya implementadas en `scripts/analyze_geval.py` (`krippendorff_alpha_ordinal`, `icc_2_1` y `human_ceiling`). Esto produce, de forma simétrica al análisis del techo humano de G-Eval, una referencia para juzgar si el panel de IA concuerda internamente tanto como lo hace el panel humano.

---

## 5. Implicaciones para el módulo agregador

El esquema seleccionado se traduce de forma directa en el contrato del módulo `src/voting/aggregator.py`:

| Campo de salida | Cómo se produce con el esquema seleccionado |
|---|---|
| `final_score` | Media aritmética de las puntuaciones disponibles, en el intervalo de 1 a 5. |
| `individual_scores` | El diccionario `{agent_name: score}` de entrada, sin modificar, para trazabilidad. |
| `agreement_level` | Uno menos el cociente entre la desviación estándar y la desviación máxima sobre las puntuaciones del ítem (continuo entre 0 y 1). |

---

## 6. Referencias

Arrow, K. J. (2012). *Social choice and individual values* (3.ª ed.). Yale University Press. (Obra original publicada en 1951). ISBN 978-0-300-17931-6. <https://yalebooks.yale.edu/book/9780300179316/social-choice-and-individual-values/>

Brandt, F., Conitzer, V., Endriss, U., Lang, J., & Procaccia, A. D. (Eds.). (2016). *Handbook of computational social choice.* Cambridge University Press. <https://doi.org/10.1017/CBO9781107446984>

Clemen, R. T. (1989). Combining forecasts: A review and annotated bibliography. *International Journal of Forecasting*, 5(4), 559–583. <https://doi.org/10.1016/0169-2070(89)90012-5>

Galton, F. (1907). Vox populi. *Nature*, 75(1949), 450–451. <https://doi.org/10.1038/075450a0>

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *Proceedings of EMNLP 2023*, 2511–2522. <https://aclanthology.org/2023.emnlp-main.153/>

Pacuit, E. (2019). Voting methods. En E. N. Zalta (Ed.), *The Stanford Encyclopedia of Philosophy* (edición de otoño de 2019). Metaphysics Research Lab, Stanford University. <https://plato.stanford.edu/entries/voting-methods/>

Stevens, S. S. (1946). On the theory of scales of measurement. *Science*, 103(2684), 677–680. <https://doi.org/10.1126/science.103.2684.677>

Verga, P., Hofstatter, S., Althammer, S., Su, Y., Piktus, A., Arkhangorodsky, A., Xu, M., White, N., & Lewis, P. (2024). Replacing judges with juries: Evaluating LLM generations with a panel of diverse models. *arXiv preprint*. <https://arxiv.org/abs/2404.18796>

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. <https://arxiv.org/abs/2306.05685>

---
