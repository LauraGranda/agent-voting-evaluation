# Limitaciones del estudio: evaluación automática de relevancia en conversación

Este documento recoge las limitaciones empíricas y metodológicas del
análisis comparativo entre G-Eval (juez único con `gpt-4o`) y el
sistema agéntico de votación (panel de tres jueces: `gpt-4o` de
OpenAI, `gemini-2.5-flash` de Google, `claude-haiku-4-5` de
Anthropic) sobre las 900 conversaciones de DailyDialog–Zhao. Cada
sección cita evidencia numérica concreta producida por los notebooks
04 (descriptivo), 05 (correlación), 06 (significancia) y 07 (análisis
de errores). El objetivo no es minimizar los hallazgos sino acotar su
generalización y orientar la investigación futura.

## 1. Limitaciones del dataset DailyDialog–Zhao

DailyDialog–Zhao es un dataset estratificado de tamaño moderado y con
varias restricciones que conviene declarar antes de extrapolar los
resultados a otros contextos.

- **Tamaño y cobertura**. n = 900 conversaciones distribuidas en cinco
  estratos: 100 ground-truth, 100 negative-sample, 216 respuestas de
  IA con relevancia humana alta (h ≥ 4), 325 respuestas de IA con
  relevancia intermedia (2 < h < 4) y 159 respuestas de IA con
  relevancia baja (h ≤ 2). La cobertura es **desbalanceada**: el
  estrato 4 (IA media) duplica al estrato 5 (IA baja) y triplica a los
  estratos 1 y 2. Las conclusiones por estrato heredan errores
  estándar proporcionales a √n por subgrupo, y los estratos 1 y 2
  alcanzan apenas n = 100, suficientes para estimar MAE pero al límite
  para inferencia fina sobre ρ intra-estrato (HU-10: ρ estrato 1 =
  0,260 y 0,181 para G-Eval y voting respectivamente, indicando alta
  varianza muestral).
- **Solo cuatro anotadores MTurk por ítem**. El campo
  `raw_relevance_scores` muestra cuatro juicios humanos por
  conversación, con una promediación aritmética como puntaje gold.
  Krippendorff α inter-anotador sobre los humanos no está calculado en
  este trabajo (queda como direction futura). El α inter-juez
  reportado en HU-10 (0,632 entre los tres LLMs del panel) **no es
  comparable** al techo humano hasta que se mida formalmente.
- **Diálogos cortos y únicamente en inglés**. La mediana de turnos por
  conversación es de tres turnos antes de la respuesta evaluada, y
  todo el corpus es monolingüe en inglés. Cualquier afirmación sobre
  diálogos largos (cinco o más intercambios), multi-idioma o con
  cambio de código requiere replicación independiente. Los nueve
  modelos generadores cubren arquitecturas pre-2020 (GPT-2 small/
  medium, S2S, HRED_attn, VHRED_attn) que producen respuestas con
  fallos cualitativamente distintos a los de LLMs modernos.

## 2. Limitaciones específicas de G-Eval

G-Eval con `gpt-4o` y prompt rúbrico de tres dimensiones (intención,
tema, alineamiento) muestra un perfil de fallo concentrado y
diagnosticable.

- **Sesgo sistemático de subestimación**. Sesgo medio global =
  −0,750 (HU-13, notebook 07, Cell 3). En 528 de 900 casos (58,7 %)
  G-Eval queda al menos 0,5 puntos por debajo del humano; solo en 49
  casos (5,4 %) lo supera por más de 0,5. Esto invierte la intuición
  habitual sobre "LLM judges como over-raters generosos": en este
  dataset, `gpt-4o` bajo la rúbrica G-Eval es marcadamente más
  estricto que los anotadores MTurk.
- **Colapso de discriminación en el estrato 3 (HU-10 + HU-13)**. En
  respuestas de IA que humanos calificaron como muy relevantes (h ≥ 4,
  n = 216), G-Eval obtiene MAE = 1,286, casi 70 % más alto que voting
  (MAE = 0,770). El sesgo medio del estrato es −1,228: G-Eval
  subestima por más de un punto en escala 1–5. **Dieciséis de los
  veinte** casos del top-20 de divergencia de G-Eval caen en este
  estrato (HU-13, notebook 07, Cell 5).
- **Dependencia de un único modelo sin redundancia**. G-Eval consulta
  únicamente a `gpt-4o`. Una sola anomalía del modelo (drift,
  actualización mayor, deprecación) afecta el 100 % de los puntajes.
  No hay señal interna de incertidumbre que el usuario pueda consumir:
  el `score` numérico no viene acompañado de un intervalo ni de una
  probabilidad calibrada.
- **Costo unitario menor pero rationale no calibrado**. Costo total
  para n = 900 = USD 3,55 (1 302 tokens promedio por conversación).
  El campo `reason` provee chain-of-thought rico pero **no está
  alineado numéricamente con el puntaje**: en los casos del top-20
  (HU-13, Cell 11) la prosa identifica correctamente el problema
  cualitativo pero el puntaje numérico es desproporcionadamente bajo
  respecto al humano.

## 3. Limitaciones específicas del sistema de votación

El panel de tres jueces aporta robustez y mejor calibración promedio,
pero introduce costos y supuestos propios.

- **Costo y latencia 2,3× respecto a G-Eval**. Costo total para
  n = 900 = USD 8,15 (1 564 tokens promedio por conversación,
  agregando los tres jueces). El operativo "tres llamadas paralelas a
  proveedores distintos" amplifica también los fallos de
  disponibilidad: cualquier juez caído degrada el sistema.
- **Sesgo de subestimación también presente, pero atenuado**. Sesgo
  medio global = −0,348 (HU-13, Cell 3); el 39,4 % de los casos
  subestiman por más de 0,5 puntos. Es la mitad del problema de
  G-Eval, no su ausencia. En el estrato 2 (negative-sample) el sesgo
  llega a −0,759: voting puntúa respuestas claramente irrelevantes
  como aún más irrelevantes que los humanos MTurk — un over-rejection
  que conviene documentar.
- **Heterogeneidad de familias y vintages**. Los tres jueces provienen
  de proveedores distintos y de generaciones temporales no idénticas
  (`gpt-4o` 2024, `gemini-2.5-flash` 2025, `claude-haiku-4-5` 2025).
  Krippendorff α inter-juez = 0,632 (HU-10), substantial pero lejos
  del consenso. La agregación por media oculta esta heterogeneidad.
- **`std_judges` no resulta útil como señal de incertidumbre respecto
  al humano (con caveat de potencia)**. La correlación de Spearman
  entre `std_judges` y `|err_voting|` es ρ = +0,013 con p = 0,705
  (HU-13, Cell 9): no se encuentra evidencia de asociación. **La
  lectura correcta no es "hipótesis rechazada"** — el test tiene baja
  potencia por construcción, porque `std_judges` toma sólo cuatro
  valores discretos (0, 0,577, 1,0, 1,155) sobre la escala 1–5. Aun
  así, la implicación práctica es la misma: en este corpus, la
  varianza inter-juez no funciona como prior de incertidumbre frente
  al estándar humano, y el insight relevante de los case studies 3 y
  4 es complementario — cuando voting falla, los tres jueces convergen
  en error con `std = 0`, es decir, en consenso erróneo. La señal de
  incertidumbre útil debe buscarse en otra parte (entropía del
  razonamiento textual, log-probabilidades del token, re-prompting con
  paráfrasis), no en la varianza del puntaje numérico.
- **Estrato 5: voting es peor que G-Eval**. En IA con baja relevancia
  (h ≤ 2, n = 159), G-Eval MAE = 0,372 vs voting MAE = 0,595 (HU-10,
  HU-13). Es el único estrato donde G-Eval domina. El panel mixto
  introduce ruido de promediación cuando el caso es fácil y unánime.

## 3b. Los cuatro sesgos del criterio: lectura puntual

El criterio académico de la tesis enumera cuatro sesgos clásicos de
los evaluadores automáticos. Cada uno se testea o se declara N/A con
justificación (HU-13, Cell 9b):

- **Verbosidad** — correlación entre longitud de la respuesta y error
  con signo. G-Eval: ρ = −0,003 (p = 0,920), indistinguible de cero;
  Voting: ρ = −0,175 (p ≈ 10⁻⁷), señal débil en la dirección
  *anti-verbosidad* (respuestas largas reciben puntaje algo más bajo
  por el panel). El efecto es estadísticamente significativo en voting
  por n grande, pero el tamaño es pequeño y va contra la dirección
  tradicional reportada en la literatura (LLM-as-judge típicamente
  premia verbosidad). Cabe en el corpus DailyDialog, donde la longitud
  media de respuesta es de 50–70 caracteres con baja varianza.
- **Posicional** — **N/A por diseño**. El sesgo posicional aplica a
  evaluación *pairwise* (cuál de dos respuestas A/B es mejor según el
  orden de presentación). G-Eval y voting hacen scoring *pointwise* de
  una sola respuesta sobre la escala 1–5; no hay posición que sesgar.
- **Ambigüedad de contexto** — correlación entre la desviación estándar
  de los cuatro anotadores MTurk (proxy de ambigüedad del caso) y el
  error absoluto del método. G-Eval: ρ = +0,109 (p = 0,001); Voting:
  ρ = +0,377 (p ≈ 10⁻³¹). Ambos métodos fallan más donde los humanos
  discrepan, pero **voting es mucho más sensible** a la ambigüedad.
  El MAE en alto desacuerdo humano (std raw ≥ 1,25) sube notoriamente
  en voting. Implicación: voting es menos robusto en casos donde el
  gold standard mismo no es confiable.
- **Auto-preferencia de familia** — la auto-preferencia clásica
  ("el juez premia el texto de su propia familia") **no es directamente
  testeable** porque los generadores son GPT-2/VHRED/HRED/S2S y los
  jueces son `gpt-4o`/`gemini-2.5-flash`/`claude-haiku-4-5`. La versión
  relevante para este estudio es la **correlación de errores entre
  jueces individuales**: ρ ∈ {0,499; 0,534; 0,568} entre los tres
  jueces del voting, y ρ = 0,499 entre el juez `gpt-4o` del voting y
  G-Eval (que también usa `gpt-4o`). Las correlaciones son todas
  moderadas-altas y similares — los tres jueces fallan en bloque
  independientemente del proveedor, no por compartir específicamente
  `gpt-4o`. Esto es el insight arquitectónico central: **el panel tiene
  menos información efectiva que la nominal** porque sus tres miembros
  comparten puntos ciegos vinculados al paradigma LLM moderno, no a la
  familia específica de modelo.

## 3c. Adjudicación manual del Top-20: cuánto del "error del método" es ruido del gold standard

Reportar como "error del método" todo lo que diverge del humano
sobreestima la tasa real de fallo si el gold standard mismo contiene
ruido. HU-13 (Cell 13) adjudica cada caso del top-20 con una regla
mecánica: **etiqueta dudosa** si `std(raw_human_scores) ≥ 1,25` o si
hay incoherencia evidente (negative-sample con humano > 3, ground-truth
con humano < 2,5); **error del método** si el otro método acertó
(`|err_otro| < 0,8`); **ambiguo genuino** si ambos métodos fallaron.

- **Top-20 G-Eval**: 2/20 etiqueta dudosa, **18/20 error genuino del
  método**. G-Eval falla por mérito propio — el panel humano es
  mayoritariamente confiable y G-Eval realmente subestima estos
  casos.
- **Top-20 Voting**: 8/20 etiqueta dudosa, **12/20 error genuino del
  método**. Casi la mitad de los "peores" casos del voting son ruido
  del gold standard (anotadores discrepantes, negative-samples mal
  etiquetados, ground-truth subvalorado). La tasa real de fallo
  efectivo de voting es **sustancialmente menor** que la sugerida por
  los `abs_err` crudos.

Esta asimetría refuerza la conclusión de HU-12 a favor de voting: no
sólo es equivalente a G-Eval en exactitud y ranking globales, sino que
cuando aparenta fallar en colas, en una fracción importante el problema
está en el humano, no en el método.

## 4. Limitaciones compartidas por ambos métodos

- **Colapso de ρ en estrato 3**. Tanto G-Eval (ρ = 0,271) como voting
  (ρ = 0,217) tienen correlación intra-estrato baja en el régimen de
  IA de alta calidad (HU-10). Ambos métodos pierden capacidad para
  rankear conversaciones cuando todas son razonablemente buenas. Es el
  techo de discriminación que define el caso de uso más exigente y que
  ningún método actual resuelve.
- **Equivalencia estadística global (HU-12)**. La prueba TOST sobre
  Δρ con margen ±0,05 afirma equivalencia formal (90 % CI ⊂
  [−0,05; +0,05]); Wilcoxon sobre `diff_abs` no rechaza H0
  (p = 0,194); t pareado da Cohen's d = 0,097 (negligible) aún siendo
  significativo (p = 0,004). **No existe método dominante** a nivel
  global; cualquier elección debe condicionarse al estrato de uso.
- **Sin calibración contra techo humano**. Ningún análisis aquí
  calcula el inter-annotator agreement humano (los cuatro anotadores
  MTurk) como techo de referencia. Sin ese baseline, no se puede
  saber qué fracción del error residual de los métodos es ruido humano
  irreducible vs. fallo del modelo.
- **Co-localización parcial de fallos**. ρ(|err_geval|, |err_voting|)
  = +0,420 (HU-13, Cell 7): los métodos comparten un patrón pero
  también fallan en regiones independientes — 91 casos donde ambos
  fallan severamente, 99 donde solo G-Eval, 84 donde solo voting, 626
  donde ambos aciertan. El **test de McNemar** sobre los pares
  discordantes (84 vs 99, con corrección de continuidad) da
  χ² = 1,07 con p = 0,30: ningún método comete significativamente más
  errores severos que el otro, cierre inferencial coherente con la
  equivalencia formal de HU-12. Ningún método aisladamente cubre los
  274 casos con al menos un fallo severo.
- **Fallas correlacionadas del panel (limitación arquitectónica de
  voting)**. Las correlaciones de Spearman entre los errores absolutos
  de los tres jueces individuales del panel son ρ ∈ {0,499; 0,534;
  0,568} (HU-13, Cell 9b). Esto significa que **los tres jueces fallan
  en bloque**: cuando uno se equivoca, los otros también se equivocan
  con alta probabilidad. Los case studies 3 y 4 de HU-13 lo ilustran:
  los tres jueces convergen en error con `std_judges = 0`. La
  consecuencia es que el panel tiene **menos información efectiva que
  la nominal**: tres jueces con errores correlacionados no son tres
  muestras independientes. Esto explica de raíz por qué HU-12 encontró
  equivalencia estadística entre G-Eval (un juez) y voting (tres
  jueces) — más jueces no implican proporcionalmente más información
  cuando comparten puntos ciegos vinculados al paradigma LLM.

## 5. Implicaciones para investigación futura

- **Calcular el techo humano**. Krippendorff α entre los cuatro
  anotadores MTurk define el techo discriminativo del problema. Sin
  ese número, los ρ = 0,76 / 0,74 reportados no se pueden colocar en
  perspectiva de qué tan cerca están del límite teórico.
- **Combinar G-Eval + voting en producción**. Dado que la matriz de
  errores severos muestra 183 casos rescatados por exactamente uno de
  los dos (HU-13), un esquema híbrido — usar voting por defecto y
  G-Eval como árbitro en casos con baja confianza, o viceversa —
  podría reducir el error a costo intermedio. Cuantificar este
  trade-off requiere una HU dedicada.
- **Buscar mejor señal de incertidumbre que `std_judges`**. La
  varianza inter-juez no predice error vs. humano (ρ = +0,013).
  Alternativas a explorar: entropía del razonamiento textual,
  log-probabilidades del token de puntaje (cuando el proveedor las
  expone), o re-prompting con paráfrasis del prompt rúbrico.
- **Profundizar en el estrato 3**. Es donde más se pierde valor
  discriminativo. Estudios cualitativos por categoría de error
  (verbosidad, ambigüedad, antropomorfismo, código mixto) sobre los
  216 casos podrían informar prompts rúbricos mejor calibrados.
- **Ampliar a multi-idioma y diálogos largos**. La generalización del
  hallazgo de equivalencia estadística requiere replicación en corpora
  con propiedades distintas: español, mandarín, código, diálogos de
  cinco o más turnos, dominios técnicos. Sin esa replicación, las
  conclusiones se restringen a diálogos cortos en inglés tipo
  DailyDialog.
- **Panel heterogéneo con árbitro o reasoning chain**. El voting
  actual es una media simple; experimentar con esquemas tipo
  *judge-of-judges* o *debate* puede aprovechar la heterogeneidad sin
  el ruido de la promediación. El estrato 5 (donde voting pierde
  frente a G-Eval) es un buen banco de prueba para ese tipo de
  refinamiento.

---

**Cierre.** La tesis concluye con dos métodos *estadísticamente
equivalentes* a nivel global pero con perfiles de fallo *cualitativamente
distintos*. La recomendación operativa de HU-12 (favorecer voting cuando
la precisión absoluta y la calibración escalar son críticas; favorecer
G-Eval cuando la prioridad es ranking) emerge precisamente porque las
limitaciones documentadas arriba se reparten asimétricamente entre los
dos métodos. La utilidad práctica del estudio depende de respetar ese
condicionamiento y no extrapolar la equivalencia global a contextos
fuera del corpus, los estratos y los modelos aquí evaluados.
