# Error Analysis Summary — G-Eval vs Voting (n=900)

Generado automáticamente por `notebooks/07_error_analysis.ipynb` (HU-13). Cero llamadas a APIs.

## 1. Metricas globales

| Metrica | G-Eval | Voting |
|---|---|---|
| MAE global | 0.901 | 0.832 |
| Sesgo medio (estim - human) | -0.750 | -0.348 |
| Casos sobreestimados (>0.5) | 49 (5.4%) | 146 (16.2%) |
| Casos subestimados (<-0.5) | 528 (58.7%) | 355 (39.4%) |

## 2. MAE y sesgo por estrato

| Estrato | n | MAE G-Eval | MAE Voting | Sesgo G-Eval | Sesgo Voting |
|---|---|---|---|---|---|
| 1: ground-truth | 100 | 0.838 | 0.593 | -0.640 | +0.140 |
| 2: negative-sample | 100 | 0.752 | 0.829 | -0.674 | -0.759 |
| 3: IA alta (h>=4) | 216 | 1.286 | 0.770 | -1.228 | -0.392 |
| 4: IA media (2<h<4) | 325 | 0.968 | 1.063 | -0.863 | -0.531 |
| 5: IA baja (h<=2) | 159 | 0.372 | 0.595 | +0.018 | +0.036 |

## 3. Matriz de errores severos (|err| >= 1.5) + McNemar

| | Voting ok | Voting severo |
|---|---|---|
| G-Eval ok | 626 | 84 |
| G-Eval severo | 99 | 91 |

- McNemar (con corrección de continuidad): chi2 = 1.071, p = 0.3007 -> ningún método comete significativamente más errores severos que el otro.
- Spearman rho(|err_geval|, |err_voting|) = 0.420 (p = 1.11e-39)
- Spearman rho(std_judges, |err_voting|) = +0.013 (p = 7.05e-01); std_judges toma solo 4 valores discretos -> test de baja potencia, no concluyente.

## 4. Cuatro sesgos del criterio

| Sesgo | Test | G-Eval | Voting | Lectura |
|---|---|---|---|---|
| Verbosidad | rho(len(response), err_signed) | -0.003 (p=0.920) | -0.175 (p=0.000) | señal débil/nula en este corpus |
| Posicional | --- | N/A | N/A | scoring pointwise, no pairwise: no aplica |
| Ambigüedad de contexto | rho(std(raw_human), |err|) | +0.109 (p=1.09e-03) | +0.377 (p=1.02e-31) | mayor error donde anotadores discrepan |
| Auto-preferencia de familia | rho(|err_juez_i|, |err_juez_j|) | ver §4b | ver §4b | jueces comparten puntos ciegos vía gpt-4o |

Correlación de errores absolutos entre jueces individuales (auto-preferencia de familia):

- rho(|err_openai|,    |err_google|)    = +0.568
- rho(|err_openai|,    |err_anthropic|) = +0.534
- rho(|err_google|,    |err_anthropic|) = +0.499
- rho(|err_openai|,    |err_geval|)     = +0.499  (comparten gpt-4o)
- rho(|err_google|,    |err_geval|)     = +0.279
- rho(|err_anthropic|, |err_geval|)     = +0.421

## 5. Top 20 divergencias G-Eval (mayor |err_geval|) con adjudicación

| conversation_id | stratum | human | raw | geval | voting | |err_g| | |err_v| | adjudicación |
|---|---|---|---|---|---|---|---|---|
| `conv_75_GPT2_small greedy_temp1.0_k0_p0.0` | 3 | 4.75 | [5, 4, 5, 5] | 1.65 | 2.33 | 3.104 | 2.420 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_9_GPT2_small top_temp1.0_k0_p0.9` | 3 | 5.00 | [5, 5, 5, 5] | 1.99 | 3.00 | 3.006 | 2.000 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_75_VHRED_attn greedy_temp1.0_k0_p0.0` | 3 | 4.75 | [5, 4, 5, 5] | 1.74 | 3.00 | 3.006 | 1.750 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_37_GPT2_medium greedy_temp1.0_k0_p0.0` | 3 | 4.75 | [5, 4, 5, 5] | 1.78 | 5.00 | 2.972 | 0.250 | error del metodo (el otro acerto) |
| `conv_62_GPT2_medium greedy_temp1.0_k0_p0.0` | 3 | 4.50 | [5, 3, 5, 5] | 1.54 | 3.00 | 2.962 | 1.500 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_30_GPT2_small top_temp1.0_k0_p0.9` | 3 | 4.75 | [5, 5, 4, 5] | 1.86 | 1.33 | 2.894 | 3.420 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_12_GPT2_small greedy_temp1.0_k0_p0.0` | 3 | 5.00 | [5, 5, 5, 5] | 2.11 | 3.33 | 2.891 | 1.670 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_37_GPT2_small greedy_temp1.0_k0_p0.0` | 3 | 4.50 | [5, 3, 5, 5] | 1.71 | 2.33 | 2.789 | 2.170 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_67_ground-truth` | 1 | 4.75 | [5, 5, 4, 5] | 1.96 | 5.00 | 2.788 | 0.250 | error del metodo (el otro acerto) |
| `conv_36_HRED_attn top_temp1.0_k0_p0.5` | 3 | 4.75 | [5, 5, 4, 5] | 1.97 | 3.00 | 2.783 | 1.750 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_74_GPT2_small top_temp1.0_k0_p0.5` | 3 | 4.75 | [5, 5, 5, 4] | 2.00 | 4.67 | 2.747 | 0.080 | error del metodo (el otro acerto) |
| `conv_92_negative-sample` | 2 | 3.75 | [5, 3, 2, 5] | 1.01 | 1.00 | 2.736 | 2.750 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_58_negative-sample` | 2 | 3.75 | [1, 5, 4, 5] | 1.03 | 1.00 | 2.724 | 2.750 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_67_S2S_attn top_temp1.0_k0_p0.5` | 3 | 4.75 | [4, 5, 5, 5] | 2.03 | 3.33 | 2.722 | 1.420 | no clasificado |
| `conv_62_VHRED_attn sample_temp1.0_k0_p0.0` | 3 | 4.25 | [5, 4, 4, 4] | 1.54 | 2.67 | 2.715 | 1.580 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_51_ground-truth` | 1 | 5.00 | [5, 5, 5, 5] | 2.33 | 4.33 | 2.669 | 0.670 | error del metodo (el otro acerto) |
| `conv_43_VHRED_attn sample_temp1.0_k0_p0.0` | 3 | 4.75 | [5, 5, 5, 4] | 2.10 | 2.67 | 2.647 | 2.080 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_69_S2S greedy_temp1.0_k0_p0.0` | 3 | 4.75 | [4, 5, 5, 5] | 2.18 | 4.00 | 2.567 | 0.750 | error del metodo (el otro acerto) |
| `conv_77_HRED_attn greedy_temp1.0_k0_p0.0` | 3 | 4.25 | [3, 4, 5, 5] | 1.70 | 1.67 | 2.554 | 2.580 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_54_GPT2_small top_temp1.0_k0_p0.5` | 3 | 4.50 | [4, 4, 5, 5] | 1.97 | 5.00 | 2.527 | 0.500 | error del metodo (el otro acerto) |

## 6. Top 20 divergencias Voting (mayor |err_voting|) con adjudicación

| conversation_id | stratum | human | raw | geval | voting | |err_v| | |err_g| | adjudicación |
|---|---|---|---|---|---|---|---|---|
| `conv_30_GPT2_small top_temp1.0_k0_p0.9` | 3 | 4.75 | [5, 5, 4, 5] | 1.86 | 1.33 | 3.420 | 2.894 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_43_VHRED_attn greedy_temp1.0_k0_p0.0` | 5 | 1.00 | [1, 1, 1, 1] | 3.23 | 4.00 | 3.000 | 2.232 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_48_ground-truth` | 1 | 2.00 | [1, 4, 1, 2] | 2.29 | 5.00 | 3.000 | 0.287 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_21_VHRED_attn greedy_temp1.0_k0_p0.0` | 3 | 4.25 | [5, 5, 4, 3] | 1.95 | 1.33 | 2.920 | 2.304 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_3_GPT2_medium top_temp1.0_k0_p0.5` | 3 | 4.50 | [4, 4, 5, 5] | 2.14 | 1.67 | 2.830 | 2.363 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_31_negative-sample` | 2 | 3.75 | [2, 3, 5, 5] | 1.62 | 1.00 | 2.750 | 2.127 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_58_negative-sample` | 2 | 3.75 | [1, 5, 4, 5] | 1.03 | 1.00 | 2.750 | 2.724 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_65_VHRED_attn greedy_temp1.0_k0_p0.0` | 4 | 2.25 | [2, 3, 3, 1] | 2.62 | 5.00 | 2.750 | 0.365 | error del metodo (el otro acerto) |
| `conv_92_negative-sample` | 2 | 3.75 | [5, 3, 2, 5] | 1.01 | 1.00 | 2.750 | 2.736 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_18_GPT2_small top_temp1.0_k0_p0.9` | 5 | 1.00 | [1, 1, 1, 1] | 2.85 | 3.67 | 2.670 | 1.848 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_26_S2S greedy_temp1.0_k0_p0.0` | 5 | 2.00 | [1, 2, 4, 1] | 2.48 | 4.67 | 2.670 | 0.483 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_59_S2S_attn sample_temp1.0_k0_p0.0` | 5 | 2.00 | [1, 1, 2, 4] | 1.96 | 4.67 | 2.670 | 0.041 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_9_GPT2_medium sample_temp1.0_k0_p0.0` | 3 | 4.25 | [5, 5, 5, 2] | 1.94 | 1.67 | 2.580 | 2.309 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_15_GPT2_small sample_temp1.0_k0_p0.0` | 3 | 4.25 | [4, 5, 3, 5] | 1.93 | 1.67 | 2.580 | 2.319 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_77_HRED_attn greedy_temp1.0_k0_p0.0` | 3 | 4.25 | [3, 4, 5, 5] | 1.70 | 1.67 | 2.580 | 2.554 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_30_GPT2_small top_temp1.0_k0_p0.5` | 4 | 3.50 | [3, 5, 3, 3] | 1.84 | 1.00 | 2.500 | 1.661 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_31_S2S_attn greedy_temp1.0_k0_p0.0` | 3 | 4.50 | [5, 5, 4, 4] | 2.35 | 2.00 | 2.500 | 2.147 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_15_VHRED_attn top_temp1.0_k0_p0.5` | 4 | 3.75 | [4, 5, 3, 3] | 1.92 | 1.33 | 2.420 | 1.828 | caso ambiguo (ambos fallan, anotadores consistentes) |
| `conv_56_GPT2_small sample_temp1.0_k0_p0.0` | 4 | 3.75 | [5, 4, 5, 1] | 1.79 | 1.33 | 2.420 | 1.964 | etiqueta dudosa (alto desacuerdo MTurk) |
| `conv_75_GPT2_small greedy_temp1.0_k0_p0.0` | 3 | 4.75 | [5, 4, 5, 5] | 1.65 | 2.33 | 2.420 | 3.104 | caso ambiguo (ambos fallan, anotadores consistentes) |

Resumen adjudicación: 2/20 casos del top-20 de G-Eval y 8/20 del de Voting son etiqueta dudosa. La tasa real de fallo del método es menor que la sugerida por los |err| crudos.

## 7. Taxonomia emergente (5 categorias)

- **A. Subestimacion severa en estrato 3 (IA alta, n=216)**: G-Eval MAE = 1.286 vs Voting MAE = 0.770 (delta = +0.516); sesgo G-Eval = -1.228; 16/20 casos del top-20 de G-Eval caen aqui.
- **B. Sesgo de subestimacion global**: G-Eval sesgo medio = -0.750, Voting sesgo medio = -0.348; G-Eval subestima por mas de 0,5 puntos en 58.7% de los casos vs 39.4% de voting. En estrato 1 (ground-truth): sesgo G-Eval = -0.640, sesgo Voting = +0.140.
- **C. Estrato 5 como unico regimen donde G-Eval supera a voting**: G-Eval MAE = 0.372 vs Voting MAE = 0.595; el resto de estratos favorece a voting.
- **D. Fallas correlacionadas en bloque**: cuando el panel falla, falla con `std_judges = 0` (case studies 3 y 4). rho(std_judges, |err_voting|) = +0.013 (p = 7.05e-01) no es evidencia de ausencia de asociacion -- el test tiene baja potencia porque std_judges toma solo 4 valores discretos. La auto-preferencia de familia (jueces compartiendo `gpt-4o`) produce errores correlacionados que reducen el numero efectivo de muestras del panel.
- **E. Co-localizacion parcial**: rho(|err_geval|, |err_voting|) = +0.420; ambos severos = 91 (10.1%), solo uno severo = 183 (20.3%). McNemar p = 0.301 -> equivalencia de severos. Valor diagnostico real en consultar ambos metodos.

## 8. Case studies (5 ejemplos detallados)

### Case study 1 — Solo G-Eval falla

- **conversation_id**: `conv_37_GPT2_medium greedy_temp1.0_k0_p0.0`
- **modelo generador**: `GPT2_medium greedy_temp1.0_k0_p0.0`
- **estrato**: 3 (3: IA alta (h>=4))
- **human_relevance_score**: 4.750 (raw MTurk: [5, 4, 5, 5])
- **geval_score**: 1.778 (|err| = 2.972)
- **final_vote_score**: 5.000 (|err| = 0.250)
- **individual_scores**: openai=5, google=5, anthropic=5 (std=0.000)

**Diálogo:**
```
  B: is there someone i can talk to about a payment question ?
  A: yes , we can handle that here . how can i help you ?
  --> RESPONSE (B): i 'd like to make a payment of 100 yuan to my bank account .
```

**Razonamiento G-Eval (extracto):**
> The Actual Output does not address the user's intent from the final turn, which is to offer help with a payment question. Instead, it introduces a new intent of making a payment, which is not aligned with the user's offer to assist. The output is not topically consistent with the conversation history, as it shifts focus from discussing a payment question to executing a payment transaction.


### Case study 2 — Solo G-Eval falla

- **conversation_id**: `conv_67_ground-truth`
- **modelo generador**: `ground-truth`
- **estrato**: 1 (1: ground-truth)
- **human_relevance_score**: 4.750 (raw MTurk: [5, 5, 4, 5])
- **geval_score**: 1.962 (|err| = 2.788)
- **final_vote_score**: 5.000 (|err| = 0.250)
- **individual_scores**: openai=5, google=5, anthropic=5 (std=0.000)

**Diálogo:**
```
  B: wake up , it 's time for school .
  --> RESPONSE (A): i 'm so tired . let me sleep for five more minutes .
```

**Razonamiento G-Eval (extracto):**
> The communicative intent of the user is a request to wake up for school. The key topic is waking up for school. The Actual Output does not address the intent directly, as it expresses a desire to continue sleeping, which is contrary to the request. It is topically related as it discusses sleep, but it does not align with the intent to wake up.


### Case study 3 — Solo Voting falla

- **conversation_id**: `conv_48_ground-truth`
- **modelo generador**: `ground-truth`
- **estrato**: 1 (1: ground-truth)
- **human_relevance_score**: 2.000 (raw MTurk: [1, 4, 1, 2])
- **geval_score**: 2.287 (|err| = 0.287)
- **final_vote_score**: 5.000 (|err| = 3.000)
- **individual_scores**: openai=5, google=5, anthropic=5 (std=0.000)

**Diálogo:**
```
  B: i can drink ten cups in fifteen minutes . i ' m sure .
  --> RESPONSE (A): you ? you are full of hot air .
```

**Razonamiento G-Eval (extracto):**
> The communicative intent of the user is a statement expressing confidence in their ability to drink ten cups in fifteen minutes. The key topic is the user's claim about drinking capacity. The Actual Output does not directly address the user's statement but instead responds with a dismissive comment, suggesting disbelief. While the response is topically related to the user's claim, it does not engage with the intent of expressing confidence, resulting in only partial alignment.


### Case study 4 — Solo Voting falla

- **conversation_id**: `conv_65_VHRED_attn greedy_temp1.0_k0_p0.0`
- **modelo generador**: `VHRED_attn greedy_temp1.0_k0_p0.0`
- **estrato**: 4 (4: IA media (2<h<4))
- **human_relevance_score**: 2.250 (raw MTurk: [2, 3, 3, 1])
- **geval_score**: 2.615 (|err| = 0.365)
- **final_vote_score**: 5.000 (|err| = 2.750)
- **individual_scores**: openai=5, google=5, anthropic=5 (std=0.000)

**Diálogo:**
```
  B: from the looks of this house from the outside , i don ' t really think that i want to go inside .
  A: from here , it does look like it could use a little work . let ' s see what the interior looks like .
  --> RESPONSE (B): i ' m sure you ' re right . i ' m looking forward to it .
```

**Razonamiento G-Eval (extracto):**
> The communicative intent of the user's final turn is a suggestion to explore the interior of the house. The key topics are the house's appearance and the intention to see the interior. The Actual Output partially addresses the intent by expressing agreement and anticipation but lacks a direct response to the suggestion to see the interior. It is topically consistent with the conversation but does not fully engage with the user's proposal.


### Case study 5 — Ambos fallan

- **conversation_id**: `conv_30_GPT2_small top_temp1.0_k0_p0.9`
- **modelo generador**: `GPT2_small top_temp1.0_k0_p0.9`
- **estrato**: 3 (3: IA alta (h>=4))
- **human_relevance_score**: 4.750 (raw MTurk: [5, 5, 4, 5])
- **geval_score**: 1.855 (|err| = 2.894)
- **final_vote_score**: 1.330 (|err| = 3.420)
- **individual_scores**: openai=1, google=1, anthropic=2 (std=0.577)

**Diálogo:**
```
  B: hello , i am calling about my cable service .
  A: what 's the problem ?
  B: for the past week my cable has n't been working .
  --> RESPONSE (A): what is the problem ?
```

**Razonamiento G-Eval (extracto):**
> The Actual Output fails to address the user's intent, which is to report a problem with their cable service. It repeats a question already asked by the assistant in Turn 2, showing no progression in addressing the user's issue. The response is topically consistent with the conversation about cable service problems but does not advance the conversation or provide any resolution.

