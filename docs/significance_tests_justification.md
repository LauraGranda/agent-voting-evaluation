# Justificación de Pruebas de Significancia Estadística

**Propósito.** Este documento justifica metodológicamente la selección
de las pruebas estadísticas usadas en
`notebooks/06_significance_tests.ipynb` (HU-12) para contrastar si la
diferencia observada entre G-Eval y el sistema de votación agéntico es
estadísticamente significativa o atribuible a ruido de muestreo. La
selección de tests del notebook debe coincidir exactamente con lo que se
justifica aquí.

Todas las fuentes citadas en este documento han sido verificadas
contra su publicación de origen (DOIs y URLs en §8).

## Contexto del estudio

Sobre las 900 conversaciones del conjunto DailyDialog–Zhao
(`data/raw/dailydialog_zhao/dataset.json`) se comparan dos métodos
automáticos de evaluación de relevancia conversacional contra el juicio
humano (`human_relevance_score`, promedio de 4 anotadores MTurk en
escala 1–5):

- **G-Eval** (`gpt-4o` con el prompt V3, single judge, score ponderado
  por logprobs de DeepEval — Liu et al., 2023).
- **Sistema de votación agéntico** (panel de 3 jueces:
  `gpt-4o`, `gemini-2.5-flash`, `claude-haiku-4-5`, todos con el mismo
  prompt V3, agregado por media aritmética).

Resultados clave heredados (HU-11):

| Magnitud | G-Eval | Voting |
|---|---|---|
| Spearman ρ vs human | 0,756 (95 % CI [0,727; 0,783]) | 0,744 (95 % CI [0,714; 0,772]) |
| Pearson r vs human | 0,711 (95 % CI [0,677; 0,742]) | 0,733 (95 % CI [0,701; 0,761]) |

La diferencia observada es **Δρ = +0,012** (a favor de G-Eval) y
**Δr = −0,022** (a favor de votación). La pregunta abierta tras HU-11 es
si esas diferencias son estadísticamente distintas de cero. Este
documento sustenta la respuesta metodológica para `Δρ` y para la
diferencia entre errores por par.

Características de los datos que condicionan la elección de tests:

- **Muestras dependientes (pareadas):** ambos métodos evalúan
  exactamente las mismas 900 conversaciones. No son muestras
  independientes; tests para comparar dos muestras independientes
  (Mann–Whitney, t no pareado) son inadecuados.
- **Escalas ordinales 1–5:** los puntajes humanos y los del panel se
  emiten en una escala ordinal de 5 niveles; G-Eval emite valores casi
  continuos en [1, 5] por la ponderación por logprobs. Tratar todos
  como escala de intervalo introduce un supuesto fuerte.
- **n grande (900):** el tamaño de muestra hace que tests de
  significancia tradicionales tengan potencia muy alta y que el
  Teorema Central del Límite asegure aproximación a normalidad en
  los promedios muestrales — pero no necesariamente en cada
  observación individual.

## 1. Naturaleza de los datos

Los datos sobre los que se ejecutan los tests de significancia son tres
arrays paralelos de longitud 900:

- `human` (`human_relevance_score`, escala 1–5).
- `geval` (`geval_score` de `outputs/geval_results.json`).
- `voting` (`final_vote_score` de `outputs/voting_results.json`).

Para el contraste de **exactitud** se computan los errores con signo y
sus magnitudes:

- `error_geval = geval − human`.
- `error_voting = voting − human`.
- `abs_err_geval = |error_geval|`.
- `abs_err_voting = |error_voting|`.

La cantidad analizada por el Wilcoxon principal y el t pareado
complementario es **`diff_abs = abs_err_geval − abs_err_voting`** (un
valor por conversación). Esta es la diferencia de magnitudes de error
contra el humano, que es exactamente lo que mide "qué método se acerca
más al gold standard". Es un diseño *paired-samples*: dos mediciones
por unidad (cada conversación) y la pregunta es si una tiende a
producir mayores errores absolutos que la otra. El test estándar para
esta estructura es el Wilcoxon de rangos signados (Wilcoxon, 1945).

**Nota crítica sobre la cancelación algebraica.** Una elección natural
pero **incorrecta** para esta misma pregunta sería testear directamente
`diff_signed = error_geval − error_voting`. Algebraicamente,
`diff_signed = (geval − human) − (voting − human) = geval − voting`:
el humano se cancela completamente, así que ese contraste mide
**sesgo inter-método** (¿los dos métodos producen scores distintos
entre sí?), no exactitud contra el humano. Por construcción, ese test
es matemáticamente independiente del gold standard, así que no puede
informar sobre cuál método se acerca más al juicio humano. El notebook
mantiene el contraste sobre `diff_signed` etiquetado explícitamente
como diagnóstico de sesgo, y deja el contraste de exactitud para
`diff_abs = |error_geval| − |error_voting|`.

Para el contraste de **correlaciones**, se utilizan los dos coeficientes
de Spearman `ρ_geval` y `ρ_voting`, ambos calculados sobre la misma
muestra de 900 pares contra el mismo gold standard humano. Son
correlaciones **dependientes** que comparten una variable (`human`),
así que el test correcto para esta estructura es el caso *overlapping*
del Steiger (1980), que **depende explícitamente** de
`r_GV = corr(geval, voting)`. En nuestros datos `r_GV (Spearman) = 0,897`,
una correlación muy alta entre los dos estimadores que tiene sentido:
ambos métodos comparten el prompt V3, evalúan exactamente las mismas
900 conversaciones, y G-Eval usa `gpt-4o` como juez único mientras
que voting lo usa como uno de tres jueces. Cuando dos estimadores
están tan correlacionados, la varianza de su diferencia se reduce
muchísimo y el test gana mucha potencia. Una versión del test que
asuma independencia (`SE = √(2/(n−3))`) infla el SE en un factor ≈ 3
y vuelve el contraste demasiado conservador, así que la fórmula
overlapping con `r_GV` es la única correcta para este diseño.

## 2. Test de Steiger para comparar correlaciones dependientes

El test de Steiger (1980) contrasta `H0: ρ₁ = ρ₂`. Su familia incluye
varios casos según el patrón de dependencia entre las dos
correlaciones; el que aplica aquí es el caso **overlapping** (dos
correlaciones que comparten exactamente una variable).

**Fórmula correcta para el caso overlapping.** Sea `r₁ = corr(X, Y)`,
`r₂ = corr(X, Z)` y `r₃ = corr(Y, Z)` la correlación entre los dos
estimadores. Con `zᵢ = arctanh(rᵢ)`:

```
r̄² = (r₁² + r₂²) / 2
f   = (1 − r₃) / (2 (1 − r̄²))
h   = (1 − f r̄²) / (1 − r̄²)
Z   = (z₁ − z₂) · √( (n − 3) / (2 (1 − r₃) h) )
```

con p-valor de dos colas desde la normal estándar. En nuestro caso
`X = human`, `Y = geval`, `Z = voting`, así que `r₃ = r_GV`. La
implementación está en `notebooks/06_significance_tests.ipynb`,
celda 8, como `steiger_overlapping`.

**Por qué `r_GV` es crucial.** Cuando dos estimadores están altamente
correlacionados (`r_GV → 1`), la varianza de su diferencia se reduce
mucho. Intuitivamente: si dos termómetros producen lecturas casi
idénticas, una pequeña diferencia entre ellos contra una temperatura
real desconocida es informativa; si producen lecturas
independientes, esa misma pequeña diferencia es ruido. El SE del
Steiger overlapping baja con `r_GV` para reflejar esto, y un test
que ignora `r_GV` (asumiendo independencia) infla el SE y se vuelve
demasiado conservador.

**Por qué Spearman vs. Pearson.** La derivación clásica de Steiger
es para correlaciones de Pearson bajo normalidad bivariada. Para
Spearman se puede aplicar la corrección de Bonett & Wright (2000),
`SE(z) = √((1 + ρ²/2) / (n − 3))`. En la práctica, la diferencia es
pequeña para `ρ ≈ 0,75` (el factor de corrección es `1 + 0,28 ≈ 1,14`,
así que el SE crece un 7 %). Para evitar la complicación del
método híbrido se usa además un **bootstrap pareado** del Δρ como
validación no paramétrica que no asume ninguna distribución
transformada. Si Steiger y bootstrap dan conclusiones cualitativas
distintas, el bootstrap es la referencia.

**Predicción y resultado.** Con `r_GV = 0,897` (Spearman), el SE
correcto es ≈ 0,016 (≈ 3× menor que el `√(2/(n−3)) = 0,047` del
caso independiente). Para `Δρ = 0,012`, eso da `Z ≈ 1,26, p ≈ 0,21`.
El bootstrap pareado da `p ≈ 0,26` con CI `[−0,010, +0,033]`. Ninguno
rechaza H0; coinciden cualitativamente y dan p-valores en el mismo
orden de magnitud, lo cual robustece la conclusión.

**Solapamiento de CIs ≠ test de equivalencia.** Comparar visualmente
los dos CIs de Fisher Z calculados independientemente para ρ_geval y
ρ_voting (lo que HU-11 hizo como diagnóstico) **no** es un test de
hipótesis sobre la diferencia: dos CIs solapados pueden corresponder
a una diferencia significativa entre dos correlaciones pareadas, y
dos CIs disjuntos pueden corresponder a una diferencia no
significativa una vez que la covariación entre las dos correlaciones
se contabiliza. El Steiger overlapping incorpora esa covariación
directamente vía `r_GV`.

**TOST como prueba positiva de equivalencia.** "No rechazar H0" en el
Steiger o el bootstrap no equivale a "los dos métodos son equivalentes":
puede deberse simplemente a falta de potencia. Para afirmar
equivalencia positivamente se usa TOST (Two One-Sided Tests;
Schuirmann, 1987), que requiere fijar una **región de equivalencia**
`±δ` a priori y rechaza la no-equivalencia si el 90 % CI bootstrap de
`Δρ` cae completamente dentro de `[−δ, +δ]`. El notebook usa
`δ = 0,05` sobre Δρ, defendible como "menos de 5 puntos de Spearman
se considera práctica clínicamente irrelevante para evaluación de
relevancia conversacional". Otras elecciones son defendibles; el
resultado debería repetirse para varios `δ` como análisis de
sensibilidad si la región elegida es polémica.

## 3. Prueba de normalidad: Shapiro-Wilk

El test de Shapiro–Wilk (Shapiro & Wilk, 1965) contrasta
`H0: la muestra proviene de una distribución normal`. Se aplica aquí
sobre el array `diff` (diferencia entre errores), no sobre los scores
brutos, porque la suposición de normalidad del t-test pareado se
refiere a la distribución de las diferencias intra-sujeto, no a las
distribuciones marginales.

**Comportamiento esperado con n = 900.** La estadística inferencial
sobre el Shapiro–Wilk con muestras grandes está bien documentada:
el test gana potencia con el tamaño muestral, y con n del orden de
varios cientos detecta desviaciones cuantitativamente pequeñas de la
normalidad que no tienen consecuencias prácticas. Razali y Wah (2011)
mostraron mediante simulación Monte Carlo que Shapiro–Wilk es el más
potente entre los cuatro tests evaluados (frente a
Kolmogorov–Smirnov, Lilliefors y Anderson–Darling) en escenarios con
muestras crecientes. Ghasemi y Zahediasl (2012) explicitan la
recomendación práctica: con muestras grandes el p-valor de los tests
de normalidad pierde valor inferencial y debe acompañarse de
inspección gráfica (histograma + Q-Q plot).

**Implicación operativa.** El p-valor de Shapiro–Wilk se reporta como
dato diagnóstico, pero la decisión de seleccionar el test principal
**no** depende únicamente de ese p-valor. Se combina con
**inspección visual** del array de diferencias vía un histograma con
curva normal superpuesta y un Q-Q plot (figura 15
`outputs/figures/15_normality_check.png`). Si el histograma es
aproximadamente simétrico y campaniforme y el Q-Q sigue la línea de
referencia salvo en colas extremas, la desviación de normalidad es
trivial en la práctica.

## 4. Selección del test principal: Wilcoxon vs. t-test

Con los datos pareados ordinales descritos en §1, hay dos candidatos
naturales para contrastar `H0: la mediana (Wilcoxon) o media (t) de
la diferencia entre errores es cero`:

- **Test de Wilcoxon de rangos signados** (Wilcoxon, 1945):
  no paramétrico, contrasta la mediana, requiere solo que las
  diferencias sean simétricas alrededor de su mediana bajo H0. No
  asume distribución específica.
- **Test t pareado**: paramétrico, contrasta la media, requiere
  normalidad aproximada de las diferencias (relajada por el Teorema
  Central del Límite cuando n es grande).

**Regla de decisión.** El test principal seleccionado para esta tesis
es el **Wilcoxon de rangos signados**, por tres razones:

1. Los puntajes humanos son ordinales en escala 1–5; el voting
   también emite scores que después de la agregación viven en una
   escala continua acotada, pero el voting subyacente es ordinal por
   juez. Esta naturaleza ordinal favorece tests no paramétricos
   incluso cuando la suposición de normalidad del paramétrico no
   está violada.
2. Independientemente del resultado de Shapiro–Wilk, la sensibilidad
   del test a desviaciones triviales con n = 900 implica que el
   p-valor del Shapiro–Wilk no es un criterio de decisión sólido por
   sí solo; el Wilcoxon es robusto a la forma exacta de la
   distribución de diferencias.
3. La literatura clásica de estadística no paramétrica
   (Wilcoxon, 1945; Conover, 1999, cap. 5) recomienda el Wilcoxon
   como contraste por defecto para datos pareados ordinales.

**Por qué se reporta también el t pareado.** Con n = 900 el Teorema
Central del Límite asegura que la distribución muestral de la media
de diferencias es aproximadamente normal, por lo que el t-test es
válido aunque las diferencias individuales no sean estrictamente
normales. En esa frontera n-grande / ordinalidad-discutible, los dos
tests usualmente dan p-valores y conclusiones casi idénticos. Que
coincidan robustece la conclusión. Si difirieran, el Wilcoxon (más
conservador para ordinales) sería el de referencia.

## 5. Tamaño del efecto

Reportar p-valores sin tamaño de efecto es prácticamente
indefendible con n = 900: cualquier diferencia sustantivamente
trivial puede ser estadísticamente significativa, y cualquier
diferencia sustantivamente importante puede no serlo si el ruido es
suficiente. Los tres tests del notebook reportan effect size
acompañando al p-valor.

**Wilcoxon — r como rank-biserial.** Para muestras pareadas, el
estimador directo del rank-biserial es
`r = (W⁺ − W⁻) / (W⁺ + W⁻)`, donde `W⁺` es la suma de rangos de las
diferencias positivas y `W⁻` la de las negativas, ambas calculadas
sobre los pares **con diferencia distinta de cero** (Wilcoxon descarta
los empates exactos en la diferencia). Esta forma directa desde los
rangos es la preferida (Tomczak & Tomczak, 2014); el notebook la
implementa en la función `rank_biserial_signed`. Una forma alternativa
sustituye Z por `norm.ppf(p/2)` y calcula `r = |Z|/√N`, pero es
circular (deriva el effect size del p-valor que se quiere
contextualizar) y pierde precisión. El N para `√N` debe ser además el
N efectivo (pares con diferencia no nula), no el N total — en este
dataset son 897 de 900.

La interpretación sigue los umbrales convencionales (Cohen, 1988):

- `r < 0,10`: efecto negligible.
- `0,10 ≤ r < 0,30`: efecto pequeño.
- `0,30 ≤ r < 0,50`: efecto mediano.
- `r ≥ 0,50`: efecto grande.

**Test t pareado — Cohen's d.** Se computa
`d = mean(diff) / std(diff, ddof=1)` donde `diff` es el array de
diferencias entre errores. Los umbrales de Cohen (1988) son los
canónicos para Cohen's d:

- `|d| < 0,20`: efecto negligible.
- `0,20 ≤ |d| < 0,50`: efecto pequeño.
- `0,50 ≤ |d| < 0,80`: efecto mediano.
- `|d| ≥ 0,80`: efecto grande.

Lakens (2013) ofrece una discusión contemporánea sobre cuándo
reportar Cohen's d vs. sus parientes (Hedges' g, Glass's Δ) en
contextos de psicología experimental.

**Steiger — Δρ como effect size natural.** Para la comparación de
correlaciones, el effect size es el propio `Δρ` con su intervalo de
confianza, que ya viene calculado en HU-11. No hay un effect size
estandarizado distinto al usado en HU-11.

## 6. Comparativa de pruebas estadísticas alternativas

Para que la elección de tests sea defendible es importante explicar
también qué se descartó y por qué. La tabla y la prosa a continuación
listan las alternativas más comunes en este escenario (comparación
pareada de dos evaluadores automáticos contra un gold standard humano).

### 6.1 Pruebas para la diferencia entre errores

| Test | Tipo | Asume | Ventajas | Desventajas | ¿Por qué (no) aquí? |
|---|---|---|---|---|---|
| **Wilcoxon signed-rank** (Wilcoxon, 1945) | No paramétrico, pareado | Diferencias simétricas alrededor de la mediana | Robusto a no-normalidad; apropiado para datos ordinales; usa rangos en vez de valores absolutos | Menor potencia que t cuando los datos son verdaderamente normales; descarta información cuantitativa de las magnitudes | **Test principal**: cumple los tres criterios (pareado, ordinal, robusto). |
| **t pareado** (Student) | Paramétrico, pareado | Diferencias aproximadamente normales | Máxima potencia bajo normalidad; effect size familiar (Cohen's d) | Sensible a outliers; supuesto de normalidad puede ser cuestionable con escalas ordinales | **Reportado como complemento**: con n=900 el TCL garantiza su validez; sirve como sanity check del Wilcoxon. |
| **Sign test** | No paramétrico, pareado | Solo simetría de los signos bajo H0 | Mínimos supuestos; muy simple | Descarta toda la magnitud de las diferencias; potencia más baja que Wilcoxon | **Descartado**: el Wilcoxon domina por usar el rango de las diferencias. |
| **Mann–Whitney U** | No paramétrico, **no pareado** | Muestras independientes | Robusto; estándar para dos grupos independientes | **No aplica**: los datos son pareados por `conversation_id` | **Descartado**: usarlo ignoraría la dependencia y subestimaría la potencia. |
| **Permutation test** (paired) | No paramétrico, pareado | Intercambiabilidad bajo H0 | Distribución exacta bajo H0; sin supuestos paramétricos | Costo computacional; effect size no tan estándar | **No usado**: el Wilcoxon es equivalente en este escenario y mucho más rápido; ya hay reproducibilidad fija con seed=42. |
| **TOST (Two One-Sided Tests)** | Paramétrico, equivalencia | Región de equivalencia predefinida | Permite **probar equivalencia** (no solo "no rechazar superioridad") | Requiere fijar una región de equivalencia antes del análisis | **No usado en HU-12** pero **señalado como extensión futura**: HU-12 prueba superioridad y deja la equivalencia para HUs siguientes (limitación reconocida en la conclusión). |

### 6.2 Pruebas para la diferencia entre correlaciones

| Test | Aplicación | Ventajas | Desventajas | ¿Por qué (no) aquí? |
|---|---|---|---|---|
| **Steiger Z overlapping** (Steiger, 1980) | Comparar dos correlaciones que comparten una variable | Cierre analítico; usa Fisher Z; explícitamente contabiliza `r_GV` | Derivado para Pearson bajo normalidad bivariada; SE inflado si se aplica Spearman sin corrección Bonett–Wright | **Test usado** con la fórmula correcta del caso overlapping (depende de `r_GV = 0,897`). |
| **Fisher Z para correlaciones independientes** | Comparar dos correlaciones medidas en muestras distintas | Forma cerrada; estándar de la literatura | **No aplica**: nuestras correlaciones no son independientes (mismo target) | **Descartado**: aplicarlo ignora la dependencia y produce SE incorrecto. |
| **Williams test** | Caso especial con ajuste de Hotelling-Williams para muestras pequeñas | Más conservador con n pequeño | Marginalmente distinto del Steiger overlapping con n grande | **No necesario** con n=900; Steiger overlapping es la forma estándar. |
| **Bootstrap pareado de Δρ** | Remuestrear con reemplazo y calcular CI percentil del Δρ | No paramétrico; respeta automáticamente la dependencia entre las dos correlaciones; sin supuestos sobre la transformación Fisher Z | Sin p-valor de forma cerrada (se obtiene del CI) | **Test usado** como método principal robusto. Con n=900 son milisegundos, no costoso. Es la referencia cuando Steiger overlapping y bootstrap discrepan. |
| **Bonett & Wright (2000)** | Corrección del SE de Fisher Z para Spearman | Apropiado cuando se aplica Steiger a Spearman | Marginal con `ρ ≈ 0,75` (factor ≈ 1,07) | **Mencionado** pero no implementado: el bootstrap pareado resuelve el problema sin necesitar la corrección analítica. |
| **TOST** (Schuirmann, 1987) | Probar equivalencia con región predefinida | **Único** método para afirmar equivalencia positivamente, no solo "no rechazo" | Requiere fijar `δ` a priori; resultado depende de esa elección | **Test usado** con `δ = 0,05` sobre Δρ. Es la pieza que permite cerrar la pregunta con afirmación, no con ausencia de evidencia. |

### 6.3 Pruebas de normalidad consideradas

| Test | Característica | ¿Por qué (no) aquí? |
|---|---|---|
| **Shapiro–Wilk** (Shapiro & Wilk, 1965) | Más potente entre los tests de normalidad clásicos (Razali & Wah, 2011) | **Usado como diagnóstico**: complementado con Q-Q plot por la limitación con n grande. |
| **Kolmogorov–Smirnov** (Lilliefors) | Menos potente que Shapiro–Wilk en simulaciones | **Descartado**: el Shapiro es estricto dominante en este rango. |
| **Anderson–Darling** | Más sensible a colas | **Descartado** por simplicidad; conclusión cualitativa no cambiaría. |
| **Q-Q plot + histograma** | Diagnóstico visual sin p-valor | **Usado en paralelo al Shapiro–Wilk**: con n grande es el indicador decisivo de la desviación práctica. |

### 6.4 Síntesis

La combinación seleccionada — **Shapiro–Wilk** + inspección visual +
análisis de simetría como diagnósticos; **Wilcoxon signed-rank**
primario y **t pareado** complementario sobre `|error|` para
exactitud; **Wilcoxon** sobre `geval − voting` para diagnosticar
sesgo inter-método; **Steiger overlapping** y **bootstrap pareado**
para Δρ; **TOST** ±0,05 para afirmar equivalencia formal;
**Holm-Bonferroni** sobre los p-valores principales — cubre los
planos del análisis con el test apropiado en cada uno. Los
alternativos descartados son o bien incompatibles con la dependencia
entre muestras (Mann–Whitney, Fisher Z independiente) o bien
estrictamente dominados por la opción elegida (sign test, KS,
Anderson–Darling) en el régimen de los datos de la tesis.

## 7. Comparaciones múltiples

El notebook reporta cinco p-valores sobre hipótesis de interés
(Wilcoxon sobre `|error|`, t sobre `|error|`, Wilcoxon sobre
`geval − voting`, bootstrap Δρ, Steiger overlapping). Para
controlar la tasa de error familywise se aplica la corrección de
**Holm-Bonferroni** (Holm, 1979): los p-valores se ordenan
ascendentemente y cada uno se compara contra
`α / (k − rank + 1)`, donde `k = 5`. Es menos conservadora que
Bonferroni puro y tiene mejor potencia. El notebook reporta el
p-valor crudo y el ajustado lado a lado para cada test.

## 8. Referencias bibliográficas

Todas las fuentes han sido verificadas contra su publicación de
origen. Los URLs y DOIs apuntan a la versión canónica.

**Fundacionales (estadística):**

- Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
  estimating Pearson, Kendall and Spearman correlations.
  *Psychometrika*, 65(1), 23–28.
  <https://doi.org/10.1007/BF02294183>
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
  Sciences* (2nd ed.). Hillsdale, NJ: Lawrence Erlbaum Associates.
  ISBN 978-0805802832.
- Conover, W. J. (1999). *Practical Nonparametric Statistics*
  (3rd ed.). New York: John Wiley & Sons. ISBN 978-0471160687.
  <https://www.wiley.com/en-us/Practical+Nonparametric+Statistics%2C+3rd+Edition-p-9780471160687>
- Fisher, R. A. (1915). Frequency distribution of the values of the
  correlation coefficient in samples from an indefinitely large
  population. *Biometrika*, 10(4), 507–521.
  <https://doi.org/10.1093/biomet/10.4.507>
- Holm, S. (1979). A simple sequentially rejective multiple test
  procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
  <https://www.jstor.org/stable/4615733>
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests
  procedure and the power approach for assessing the equivalence of
  average bioavailability. *Journal of Pharmacokinetics and
  Biopharmaceutics*, 15(6), 657–680.
  <https://doi.org/10.1007/BF01068419>
- Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
  for normality (complete samples). *Biometrika*, 52(3–4), 591–611.
  <https://doi.org/10.1093/biomet/52.3-4.591>
- Steiger, J. H. (1980). Tests for comparing elements of a
  correlation matrix. *Psychological Bulletin*, 87(2), 245–251.
  <https://doi.org/10.1037/0033-2909.87.2.245>
- Wilcoxon, F. (1945). Individual comparisons by ranking methods.
  *Biometrics Bulletin*, 1(6), 80–83.
  <https://doi.org/10.2307/3001968>

**Aplicadas (effect size y normalidad):**

- Ghasemi, A., & Zahediasl, S. (2012). Normality tests for
  statistical analysis: A guide for non-statisticians.
  *International Journal of Endocrinology and Metabolism*, 10(2),
  486–489. <https://doi.org/10.5812/ijem.3505> ·
  PubMed: <https://pubmed.ncbi.nlm.nih.gov/23843808/>
- Lakens, D. (2013). Calculating and reporting effect sizes to
  facilitate cumulative science: A practical primer for t-tests and
  ANOVAs. *Frontiers in Psychology*, 4, 863.
  <https://doi.org/10.3389/fpsyg.2013.00863>
- Razali, N. M., & Wah, Y. B. (2011). Power comparisons of
  Shapiro–Wilk, Kolmogorov–Smirnov, Lilliefors and Anderson–Darling
  tests. *Journal of Statistical Modeling and Analytics*, 2(1),
  21–33.
  <https://www.nrc.gov/docs/ML1714/ML17143A100.pdf>
- Tomczak, M., & Tomczak, E. (2014). The need to report effect size
  estimates revisited: An overview of some recommended measures of
  effect size. *Trends in Sport Sciences*, 1(21), 19–25.
  <https://tss.awf.poznan.pl/pdf-188960-110189?filename=The+need+to+report+effect.pdf>

**NLP (evaluadores y dataset):**

- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023).
  G-Eval: NLG evaluation using GPT-4 with better human alignment.
  En *Proceedings of EMNLP 2023*, 2511–2522.
  ACL Anthology: <https://aclanthology.org/2023.emnlp-main.153/> ·
  arXiv: <https://arxiv.org/abs/2303.16634>
- Zhao, T., Lala, D., & Kawahara, T. (2020). Designing precise and
  robust dialogue response evaluators. En *Proceedings of ACL 2020*,
  26–33. ACL Anthology:
  <https://aclanthology.org/2020.acl-main.4/> ·
  arXiv: <https://arxiv.org/abs/2004.04908>
