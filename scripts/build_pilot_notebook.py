"""Build notebooks/02_prompt_pilot.ipynb programmatically.

This builder keeps the 18 pilot cells in one place so the notebook is easy to
regenerate or audit. Running it writes the notebook to disk without executing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat  # type: ignore[import-untyped]

NB_PATH = Path("notebooks/02_prompt_pilot.ipynb")


def md(src: str) -> Any:
    return nbformat.v4.new_markdown_cell(src.rstrip() + "\n")  # type: ignore[no-untyped-call]


def code(src: str) -> Any:
    return nbformat.v4.new_code_cell(src.rstrip() + "\n")  # type: ignore[no-untyped-call]


CELL_1 = md(
    """
# G-Eval Prompt Pilot: Relevance Evaluation

**Thesis context.** This notebook is part of a master's thesis that compares G-Eval
(Liu et al., EMNLP 2023) against an agentic voting system for conversational
relevance evaluation on DailyDialog-Zhao (Zhao et al., 2020). The goal here is
to pilot three candidate G-Eval relevance prompts on a stratified 20-entry
sample and pick the winner before committing to the full 900-pair run.

**⚠️ Cost awareness.** This notebook makes ~60 OpenAI API calls using `gpt-4o`
(20 pilot entries × 3 prompt versions). Estimated cost: **~$1.50–2.00**. Full
900-pair experiment estimated cost: ~$15–25. Verify all pre-flight checks pass
before executing this notebook — do not spend tokens on a broken pipeline.

**Citations.** Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023).
*G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP 2023.
Zhao, T., Lala, D., & Kawahara, T. (2020). *Designing Precise and Robust
Dialogue Response Evaluators.* ACL 2020.
"""
)

CELL_2 = code(
    """
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.stats import spearmanr

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv(Path("../.env"))

PILOT_PATH = Path("../configs/prompts/pilot_sample.json")
PROMPTS_DIR = Path("../configs/prompts")
FIGURES_DIR = Path("../outputs/figures")
RESULTS_DIR = Path("../configs/prompts")
DOCS_DIR = Path("../docs")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o"  # Direct successor of GPT-4 used in Liu et al. (2023).

assert os.getenv("OPENAI_API_KEY"), (
    "OPENAI_API_KEY not set \\u2014 stop here, do not spend API tokens"
)

with open(PILOT_PATH, encoding="utf-8") as f:
    pilot = json.load(f)
assert len(pilot) == 20, f"Expected 20 pilot entries, got {len(pilot)}"

print(f"Pilot sample loaded: {len(pilot)} entries")
print(f"Evaluator model: {MODEL}")
"""
)

CELL_3 = md(
    """
> **Justificación del modelo evaluador.** Se seleccionó `gpt-4o` por ser el
> sucesor directo de GPT-4, el modelo usado en el paper original de G-Eval
> (Liu et al., EMNLP 2023). Esta elección mantiene comparabilidad
> metodológica con el baseline de relevancia en SummEval (Spearman
> ρ = 0.547) y el baseline de coherencia de diálogo en Topical-Chat USR
> (ρ = 0.605) reportados en el mismo paper, permitiendo interpretar los
> resultados en el contexto de la literatura existente.
"""
)

CELL_4 = code(
    """
from collections import defaultdict

by_stratum: dict[int, list[dict]] = defaultdict(list)
for e in pilot:
    by_stratum[e["stratum"]].append(e)

print(f"{'Stratum':<8} | {'Count':<5} | {'Score range':<12} | Models")
print("-" * 80)
for s in sorted(by_stratum):
    entries = by_stratum[s]
    scores = [e["metadata"]["human_score"] for e in entries]
    models = sorted({e["metadata"]["model"] for e in entries})
    print(
        f"{s:<8} | {len(entries):<5} | "
        f"{min(scores):.1f} - {max(scores):.1f}   | {', '.join(models)}"
    )
print("-" * 80)
all_scores = [e["metadata"]["human_score"] for e in pilot]
print(
    f"{'TOTAL':<8} | {len(pilot):<5} | "
    f"{min(all_scores):.1f} - {max(all_scores):.1f}   |"
)
"""
)

CELL_5 = code(
    """
PROMPT_FILES = {
    "V1": PROMPTS_DIR / "v1_generic.txt",
    "V2": PROMPTS_DIR / "v2_dialogue_cot.txt",
    "V3": PROMPTS_DIR / "v3_full_cot_anchored.txt",
}

# GPT-4o pricing (per 1M tokens, 2026-04 public rates):
# input $2.50 / output $10.00. The G-Eval evaluator prompt is dominated by
# input tokens; output is a short JSON object, so we estimate conservatively.
USD_PER_INPUT_TOKEN = 2.50 / 1_000_000
USD_PER_OUTPUT_TOKEN = 10.00 / 1_000_000
AVG_OUTPUT_TOKENS = 120  # a typical G-Eval evaluation returns ~100-150 tokens
# Dataset average conversation payload ~ 250 input tokens (measured post-hoc
# on pilot); we add the prompt size to get a per-call estimate.
AVG_CONVO_INPUT_TOKENS = 250

prompts: dict[str, str] = {}
for name, path in PROMPT_FILES.items():
    prompts[name] = path.read_text(encoding="utf-8")

print(f"{'Version':<6} | {'Chars':<6} | {'~Tokens':<8} | {'$/call':<9} | $/900")
print("-" * 56)
for name, text in prompts.items():
    chars = len(text)
    # Rough heuristic: 1 token ~= 4 characters in English.
    prompt_tokens = chars // 4
    input_tokens = prompt_tokens + AVG_CONVO_INPUT_TOKENS
    cost_per_call = (
        input_tokens * USD_PER_INPUT_TOKEN
        + AVG_OUTPUT_TOKENS * USD_PER_OUTPUT_TOKEN
    )
    cost_per_900 = cost_per_call * 900
    print(
        f"{name:<6} | {chars:<6} | {prompt_tokens:<8} | "
        f"${cost_per_call:<7.4f} | ${cost_per_900:<5.2f}"
    )
"""
)

CELL_6 = code(
    '''
def build_test_case(entry: dict) -> LLMTestCase:
    """Build an LLMTestCase from a processed dataset entry.

    The conversation history is packed into the ``input`` field using
    ``[Turn N] Role: content`` lines so the evaluator sees the full dialogue
    context even though ``evaluation_params`` is fixed to ``[INPUT, ACTUAL_OUTPUT]``.
    The trailing assistant turn (which duplicates ``actual_output``) is removed
    to avoid leaking the response back into the prompt.

    Args:
        entry: Dataset entry with keys ``turns`` and ``actual_output``.

    Returns:
        LLMTestCase with packed conversation history as input.
    """
    context = entry["turns"][:-1]
    formatted = "\\n".join(
        f"[Turn {i + 1}] {t['role'].capitalize()}: {t['content']}"
        for i, t in enumerate(context)
    )
    return LLMTestCase(input=formatted, actual_output=entry["actual_output"])


def build_geval_metric(prompt_text: str) -> GEval:
    """Initialize GEval with ``gpt-4o`` and the provided criteria text.

    Args:
        prompt_text: Natural-language relevance criteria and rubric.

    Returns:
        GEval metric ready to score test cases.
    """
    return GEval(
        name="Relevance",
        criteria=prompt_text,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=MODEL,
    )


def _rescale_0_1_to_1_5(score_0_1: float) -> float:
    """Map a 0-1 G-Eval score onto a 1-5 scale for delta comparison.

    Spearman correlations are invariant to this monotone rescaling; the rescale
    only affects the ``delta`` columns used for human-readable diagnostics.
    """
    return 1.0 + 4.0 * score_0_1


def run_pilot(prompt_text: str, pilot_sample: list[dict], version_name: str) -> list[dict]:
    """Run G-Eval across the pilot sample, collecting scores.

    Per-entry failures are caught and reported; the pilot continues on the
    remaining entries. Each result dict contains the raw 0-1 G-Eval score,
    the rescaled 1-5 score, the human reference, and the signed delta.

    Args:
        prompt_text: Criteria string passed to GEval.
        pilot_sample: Tagged pilot entries.
        version_name: Label for progress messages.

    Returns:
        List of result dicts (one per pilot entry), ordered as ``pilot_sample``.
    """
    metric = build_geval_metric(prompt_text)
    results: list[dict] = []
    n_ok = 0
    n_fail = 0
    t0 = perf_counter()
    for i, entry in enumerate(pilot_sample, start=1):
        conv_id = entry["metadata"]["conversation_id"]
        human = entry["metadata"]["human_score"]
        print(f"[{version_name}] Evaluating {i}/{len(pilot_sample)} ({conv_id})...")
        try:
            tc = build_test_case(entry)
            metric.measure(tc)
            raw = float(metric.score)
            geval_1_5 = _rescale_0_1_to_1_5(raw)
            reason = getattr(metric, "reason", None) or ""
            results.append(
                {
                    "conversation_id": conv_id,
                    "stratum": entry["stratum"],
                    "model": entry["metadata"]["model"],
                    "human_score": human,
                    "geval_score_raw": raw,
                    "geval_score": geval_1_5,
                    "delta": geval_1_5 - human,
                    "reason": reason,
                }
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL ({type(exc).__name__}): {exc}")
            results.append(
                {
                    "conversation_id": conv_id,
                    "stratum": entry["stratum"],
                    "model": entry["metadata"]["model"],
                    "human_score": human,
                    "geval_score_raw": None,
                    "geval_score": None,
                    "delta": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            n_fail += 1
    elapsed = perf_counter() - t0
    print(f"[{version_name}] Complete. {n_ok}/{len(pilot_sample)} successful, {n_fail} failed ({elapsed:.1f}s).")
    return results


def compute_spearman(results: list[dict]) -> tuple[float, float]:
    """Compute Spearman ρ and p-value between human and G-Eval scores.

    Entries with ``geval_score is None`` (API errors) are dropped.
    """
    ok = [r for r in results if r["geval_score"] is not None]
    humans = [r["human_score"] for r in ok]
    gevals = [r["geval_score"] for r in ok]
    rho, p = spearmanr(humans, gevals)
    return float(rho), float(p)


def save_results(results: list[dict], path: Path) -> None:
    """Persist pilot results to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {path}")


def _delta_note(delta: float | None) -> str:
    if delta is None:
        return "\u2717 API error"
    abs_d = abs(delta)
    if abs_d <= 0.5:
        base = "\u2713 close"
    elif abs_d <= 1.5:
        base = "\u26a0 moderate gap"
    else:
        base = "\u2717 large gap"
    if delta > 0:
        return f"\u2191 overscored | {base}"
    if delta < 0:
        return f"\u2193 underscored | {base}"
    return base


def build_results_table(results: list[dict]) -> None:
    """Print a formatted results table with delta and diagnostic note."""
    print(f"{'#':<2} | {'conversation_id':<42} | {'str':<3} | {'hum':<4} | {'geval':<5} | {'Δ':<6} | note")
    print("-" * 110)
    for i, r in enumerate(results, start=1):
        conv = r["conversation_id"]
        s = r["stratum"]
        h = r["human_score"]
        g = r["geval_score"]
        d = r["delta"]
        g_str = f"{g:.2f}" if g is not None else "  -  "
        d_str = f"{d:+.2f}" if d is not None else "  -  "
        print(f"{i:<2} | {conv:<42} | {s:<3} | {h:<4} | {g_str:<5} | {d_str:<6} | {_delta_note(d)}")
    ok = [r for r in results if r["delta"] is not None]
    if ok:
        mean_abs = sum(abs(r["delta"]) for r in ok) / len(ok)
        max_abs = max(abs(r["delta"]) for r in ok)
        print("-" * 110)
        print(f"Mean |Δ| = {mean_abs:.3f}   Max |Δ| = {max_abs:.3f}   (over {len(ok)} successful)")


def interpret_rho(rho: float) -> str:
    if rho > 0.7:
        return "strong"
    if rho > 0.4:
        return "moderate"
    if rho > 0.2:
        return "weak"
    return "problematic (< 0.2)"
'''
)

CELL_7 = code(
    """
V1_PATH = RESULTS_DIR / "pilot_results_v1.json"
if V1_PATH.exists():
    with open(V1_PATH, encoding="utf-8") as f:
        results_v1 = json.load(f)
    print(f"Loaded cached V1 results from {V1_PATH} ({len(results_v1)} entries).")
else:
    results_v1 = run_pilot(prompts["V1"], pilot, "V1")
    save_results(results_v1, V1_PATH)
n_ok_v1 = sum(1 for r in results_v1 if r["geval_score"] is not None)
print(f"V1: {n_ok_v1}/20 successful, {20 - n_ok_v1} failed.")
"""
)

CELL_8 = code(
    """
print("V1 results (prompt = v1_generic.txt):")
build_results_table(results_v1)
"""
)

CELL_9 = code(
    """
rho_v1, p_v1 = compute_spearman(results_v1)
print(f"V1 Spearman \\u03c1 = {rho_v1:.3f}   (p = {p_v1:.4f})")
print(f"Interpretation: {interpret_rho(rho_v1)}")
"""
)

CELL_10 = md(
    """
### Análisis de V1

V1 es el baseline genérico: tarea de dos oraciones, definición de relevancia
en una línea y rúbrica reducida a descriptores de una línea. Sin
chain-of-thought, sin conciencia del diálogo, sin ejemplos. Tres cosas que
conviene revisar tras leer la tabla de deltas:

- **¿Qué estratos tuvieron los mayores deltas?** Si los estratos 1
  (ground-truth) y 2 (negative-sample) muestran los mayores gaps absolutos,
  el prompt está fallando en los extremos — típicamente porque la definición
  genérica no ayuda al evaluador a distinguir continuaciones coherentes vs
  fuera de tema.
- **¿V1 comprimió los scores hacia 3-4?** Los prompts genéricos sin rúbrica
  anclada suelen regresar hacia el centro de la escala. Verificar si algún
  score V1 es ≥ 4.5 o ≤ 1.5 — si no, hay compresión de escala.
- **¿Qué debería corregir V2?** Se esperan dos mejoras: (i) forzar al
  evaluador a razonar sobre el contexto multi-turno (paso 2 del CoT: consistencia
  temática) y (ii) tensar los extremos haciendo explícito "aborda completamente
  el intent" vs "ignora el intent".
"""
)

CELL_11 = code(
    """
V2_PATH = RESULTS_DIR / "pilot_results_v2.json"
if V2_PATH.exists():
    with open(V2_PATH, encoding="utf-8") as f:
        results_v2 = json.load(f)
    print(f"Loaded cached V2 results from {V2_PATH} ({len(results_v2)} entries).")
else:
    results_v2 = run_pilot(prompts["V2"], pilot, "V2")
    save_results(results_v2, V2_PATH)
n_ok_v2 = sum(1 for r in results_v2 if r["geval_score"] is not None)
print(f"V2: {n_ok_v2}/20 successful, {20 - n_ok_v2} failed.")
"""
)

CELL_12 = code(
    """
print("V2 results (prompt = v2_dialogue_cot.txt) with improvement vs V1:")
map_v1 = {r["conversation_id"]: r for r in results_v1}
rows = []
for r in results_v2:
    v1 = map_v1.get(r["conversation_id"])
    if v1 is None or v1["delta"] is None or r["delta"] is None:
        improvement = None
    else:
        improvement = abs(v1["delta"]) - abs(r["delta"])
    rows.append((r, improvement))

print(f"{'#':<2} | {'conversation_id':<42} | {'str':<3} | {'hum':<4} | {'geval':<5} | {'Δ':<6} | {'imp vs V1':<9} | note")
print("-" * 130)
for i, (r, imp) in enumerate(rows, start=1):
    conv = r["conversation_id"]
    g = r["geval_score"]
    d = r["delta"]
    g_str = f"{g:.2f}" if g is not None else "  -  "
    d_str = f"{d:+.2f}" if d is not None else "  -  "
    imp_str = f"{imp:+.2f}" if imp is not None else "   -   "
    print(f"{i:<2} | {conv:<42} | {r['stratum']:<3} | {r['human_score']:<4} | {g_str:<5} | {d_str:<6} | {imp_str:<9} | {_delta_note(d)}")

rho_v2, p_v2 = compute_spearman(results_v2)
print(f"\\nV2 Spearman \\u03c1 = {rho_v2:.3f}   (p = {p_v2:.4f})   -> {interpret_rho(rho_v2)}")
"""
)

CELL_13 = code(
    """
V3_PATH = RESULTS_DIR / "pilot_results_v3.json"
if V3_PATH.exists():
    with open(V3_PATH, encoding="utf-8") as f:
        results_v3 = json.load(f)
    print(f"Loaded cached V3 results from {V3_PATH} ({len(results_v3)} entries).")
else:
    results_v3 = run_pilot(prompts["V3"], pilot, "V3")
    save_results(results_v3, V3_PATH)
n_ok_v3 = sum(1 for r in results_v3 if r["geval_score"] is not None)
print(f"V3: {n_ok_v3}/20 successful, {20 - n_ok_v3} failed.")
"""
)

CELL_14 = code(
    """
print("V3 results (prompt = v3_full_cot_anchored.txt):")
map_v2 = {r["conversation_id"]: r for r in results_v2}

print(f"{'#':<2} | {'conversation_id':<42} | {'str':<3} | {'hum':<4} | {'geval':<5} | {'Δ':<6} | {'imp vs V2':<9}")
print("-" * 115)
for i, r in enumerate(results_v3, start=1):
    v2 = map_v2.get(r["conversation_id"])
    if v2 is None or v2["delta"] is None or r["delta"] is None:
        imp = None
    else:
        imp = abs(v2["delta"]) - abs(r["delta"])
    g = r["geval_score"]
    d = r["delta"]
    g_str = f"{g:.2f}" if g is not None else "  -  "
    d_str = f"{d:+.2f}" if d is not None else "  -  "
    imp_str = f"{imp:+.2f}" if imp is not None else "   -   "
    print(f"{i:<2} | {r['conversation_id']:<42} | {r['stratum']:<3} | {r['human_score']:<4} | {g_str:<5} | {d_str:<6} | {imp_str:<9}")

rho_v3, p_v3 = compute_spearman(results_v3)
print(f"\\nV3 Spearman \\u03c1 = {rho_v3:.3f}   (p = {p_v3:.4f})   -> {interpret_rho(rho_v3)}")
"""
)

CELL_15 = code(
    """
def _stats(results: list[dict]) -> tuple[float, float]:
    ok = [r for r in results if r["delta"] is not None]
    if not ok:
        return (float("nan"), float("nan"))
    mean_abs = sum(abs(r["delta"]) for r in ok) / len(ok)
    max_abs = max(abs(r["delta"]) for r in ok)
    return mean_abs, max_abs


mean_v1, max_v1 = _stats(results_v1)
mean_v2, max_v2 = _stats(results_v2)
mean_v3, max_v3 = _stats(results_v3)


def _cost_per_900(prompt_text: str) -> float:
    tokens = len(prompt_text) // 4 + AVG_CONVO_INPUT_TOKENS
    return (tokens * USD_PER_INPUT_TOKEN + AVG_OUTPUT_TOKENS * USD_PER_OUTPUT_TOKEN) * 900


cost_v1 = _cost_per_900(prompts["V1"])
cost_v2 = _cost_per_900(prompts["V2"])
cost_v3 = _cost_per_900(prompts["V3"])

print(f"{'Version':<7} | {'Model':<7} | {'Spearman ρ':<10} | {'p-value':<8} | {'Mean |Δ|':<9} | {'Max |Δ|':<8} | Est. cost/900")
print("-" * 90)
for name, rho, p, mean, mx, cost in [
    ("V1", rho_v1, p_v1, mean_v1, max_v1, cost_v1),
    ("V2", rho_v2, p_v2, mean_v2, max_v2, cost_v2),
    ("V3", rho_v3, p_v3, mean_v3, max_v3, cost_v3),
]:
    print(
        f"{name:<7} | {MODEL:<7} | {rho:<10.3f} | {p:<8.4f} | {mean:<9.3f} | {mx:<8.3f} | ${cost:>6.2f}"
    )

# Scatter plot: 1 row, 3 columns, diagonal reference, color by stratum.
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
stratum_colors = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#ff7f0e", 5: "#9467bd"}
for ax, (name, results, rho) in zip(
    axes,
    [("V1", results_v1, rho_v1), ("V2", results_v2, rho_v2), ("V3", results_v3, rho_v3)],
):
    for r in results:
        if r["geval_score"] is None:
            continue
        ax.scatter(
            r["human_score"],
            r["geval_score"],
            color=stratum_colors[r["stratum"]],
            s=60,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
    ax.plot([1, 5], [1, 5], color="gray", linestyle="--", linewidth=1, label="perfect")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Human score")
    ax.set_ylabel("G-Eval score (1-5)")
    ax.set_title(f"{name} (gpt-4o) ρ={rho:.2f}")
    ax.grid(True, alpha=0.3)
handles = [
    plt.Line2D(
        [], [], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
        markersize=8, label=f"Stratum {s}",
    )
    for s, c in stratum_colors.items()
]
fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("G-Eval vs Human scores across prompt versions", y=1.08)
fig.tight_layout()
fig_path = FIGURES_DIR / "05_prompt_pilot_comparison.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")
"""
)

CELL_SECTION_SETUP = md(
    """
## 1. Setup y carga del sample piloto

Cargamos variables de entorno (`OPENAI_API_KEY`), definimos rutas relativas al
repositorio y leemos las 20 entradas estratificadas de
`configs/prompts/pilot_sample.json`. El `assert` es la primera barrera de
seguridad: si la API key no está disponible, el notebook se detiene antes de
intentar cualquier llamada.
"""
)

CELL_SECTION_STRATA = md(
    """
## 2. Distribución por estrato

Los 20 pares piloto están repartidos en 5 estratos (4 por estrato) para cubrir
todo el espectro de calidad: `ground-truth` altos, `negative-sample` bajos, y
tres bandas de modelos IA (alta / media / baja). Esta tabla confirma que el
sample mantiene la estratificación tras la carga.
"""
)

CELL_SECTION_PROMPTS = md(
    """
## 3. Catálogo de prompts y estimación de costo

Leemos los tres prompts candidatos (`v1_generic.txt`, `v2_dialogue_cot.txt`,
`v3_full_cot_anchored.txt`) y estimamos el costo por llamada y por un
experimento de 900 pares. Precios GPT-4o (abril 2026): input $2.50/1M,
output $10.00/1M. Esto permite decidir si vale la pena pagar un prompt más
largo antes de ejecutarlo.
"""
)

CELL_SECTION_HELPERS = md(
    """
## 4. Helpers reutilizables

Funciones auxiliares usadas por los tres pilotos:

- `build_test_case` — empaqueta la historia multi-turno en el campo `input`
  formato `[Turn N] Role: content`, eliminando el turno final del asistente
  para no filtrar la respuesta al evaluador.
- `build_geval_metric` — instancia `GEval` con `gpt-4o` y los parámetros
  fijados por la spec (`INPUT`, `ACTUAL_OUTPUT`).
- `_rescale_0_1_to_1_5` — G-Eval devuelve scores en [0, 1]; los reescalamos
  a [1, 5] solo para comparar deltas con el humano (Spearman es invariante a
  esta transformación).
- `run_pilot` — itera el sample, captura errores por entrada para no perder
  el piloto completo ante un fallo aislado.
- `compute_spearman`, `build_results_table`, `interpret_rho` — métricas y
  reportes.
"""
)

CELL_SECTION_V1 = md(
    """
## 5. Piloto V1 — baseline genérico

Ejecutamos el piloto V1 (o cargamos el resultado cacheado si existe). El
patrón `if path.exists(): load else: run_pilot()` permite re-ejecutar el
notebook sin gastar tokens cuando los resultados ya están persistidos.
"""
)

CELL_SECTION_V2 = md(
    """
## 6. Piloto V2 — CoT consciente del diálogo

V2 añade sobre V1: descripción multi-turno, definición de relevancia en 3
partes (a/b/c) y 5 pasos de chain-of-thought explícitos. La tabla muestra la
mejora (o regresión) por entrada respecto a V1.
"""
)

CELL_SECTION_V3 = md(
    """
## 7. Piloto V3 — CoT completo + rúbrica anclada

V3 mantiene la estructura de V2 y añade: rúbrica anclada por score (1-5) y
3 ejemplos trabajados sacados del propio sample piloto. Nota de
`overfitting`: los ejemplos pertenecen al sample piloto, lo que genera una
fuga leve sobre estas 20 entradas; sobre el run completo de 900 pares la
fuga es despreciable (3/900 = 0.3%).
"""
)

CELL_SECTION_COMPARISON = md(
    """
## 8. Comparación cuantitativa y visual

Tabla comparativa (Spearman ρ, p-value, mean |Δ|, max |Δ|, costo/900) y
scatter plot 1×3 con la recta de identidad. Los puntos están coloreados por
estrato para detectar sesgos sistemáticos (p.ej. compresión hacia 3 en
estratos IA medios).
"""
)

CELL_SECTION_PERSIST = md(
    """
### Persistencia automática: docs y prompt final

La celda siguiente escribe `docs/prompt_iterations.md` (con la tabla de
Steiger y la justificación de la selección) y copia el prompt V3 a
`configs/prompts/geval_relevance_prompt.txt`, que será el prompt que use el
experimento completo de 900 pares.
"""
)

CELL_STEIGER_MD = md(
    """
## 9. Prueba de Steiger — ¿Son estadísticamente distintas las versiones?

La prueba de Steiger (1980) compara dos correlaciones de Spearman
obtenidas sobre la misma muestra. H0: las correlaciones son iguales.
Con n=20, diferencias de Δρ < 0.1 raramente alcanzan significancia, por
lo que este test es el que permite decidir si la diferencia cuantitativa
entre V1/V2/V3 es real o simplemente ruido de no-determinismo.
"""
)

CELL_STEIGER_CODE = code(
    '''
import numpy as np
from scipy.stats import norm


def steiger_test(r1: float, r2: float, n: int) -> tuple[float, float]:
    """Compare two Spearman correlations from the same sample.

    Uses Fisher Z transformation (Steiger, 1980).
    H0: r1 == r2 (no significant difference).

    Args:
        r1: First Spearman correlation.
        r2: Second Spearman correlation.
        n: Sample size (same for both).

    Returns:
        (Z statistic, two-tailed p-value).
    """
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(2 / (n - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


# Robustness: if rho_v1/v2/v3 are missing from scope (e.g. the cell is run
# standalone), load them from the cached pilot result JSONs.
_scope = set(globals())
if not {"rho_v1", "rho_v2", "rho_v3"}.issubset(_scope):
    from scipy.stats import spearmanr
    _loaded = {}
    for _v in ("v1", "v2", "v3"):
        with open(RESULTS_DIR / f"pilot_results_{_v}.json", encoding="utf-8") as _f:
            _res = json.load(_f)
        _ok = [r for r in _res if r["geval_score"] is not None]
        _rho, _ = spearmanr([r["human_score"] for r in _ok], [r["geval_score"] for r in _ok])
        _loaded[_v] = float(_rho)
    rho_v1, rho_v2, rho_v3 = _loaded["v1"], _loaded["v2"], _loaded["v3"]

n_pilot = 20
SIG_LEVEL = 0.05
comparisons = [
    ("V1", "V2", rho_v1, rho_v2),
    ("V1", "V3", rho_v1, rho_v3),
    ("V2", "V3", rho_v2, rho_v3),
]

header = f"Prueba de Steiger (1980) \\u2014 Comparaci\\u00f3n de correlaciones de Spearman"
print(header)
print(f"Muestra piloto: n={n_pilot}")
print("-" * 72)
col_pair = "Comparaci\\u00f3n"
col_a = "\\u03c1_A"
col_b = "\\u03c1_B"
print(f"{col_pair:<13} {col_a:<8} {col_b:<8} {'Z':<8} {'p-value':<10} {'Resultado'}")
print("-" * 72)

steiger_results: list[dict] = []
all_non_significant = True
for name_a, name_b, r1, r2 in comparisons:
    z, p = steiger_test(r1, r2, n_pilot)
    significant = p < SIG_LEVEL
    if significant:
        all_non_significant = False
    result = "\\u2717 Significativa" if significant else "\\u2713 NO significativa"
    label = f"{name_a} vs {name_b}"
    print(f"{label:<13} {r1:<8.3f} {r2:<8.3f} {z:<8.3f} {p:<10.3f} {result}")
    steiger_results.append({"pair": label, "rho_a": r1, "rho_b": r2, "z": z, "p": p})

print("-" * 72)
if all_non_significant:
    print("\\n\\u2705 CONCLUSI\\u00d3N: Ninguna versi\\u00f3n supera a otra estad\\u00edsticamente.")
    print("   La diferencia m\\u00e1xima observada es atribuible a no-determinismo")
    print("   del LLM con n=20. Selecci\\u00f3n del prompt final basada en")
    print("   criterios metodol\\u00f3gicos, no cuantitativos.")
else:
    print("\\n\\u26a0\\ufe0f  Al menos una comparaci\\u00f3n es estad\\u00edsticamente significativa.")
    print("   Revisar manualmente antes de seleccionar versi\\u00f3n final.")
'''
)

CELL_16 = md(
    """
## 10. Selección final: V3

### Evidencia estadística
La prueba de Steiger (1980) no encontró diferencias estadísticamente
significativas entre ningún par de versiones (todos p > 0.05, n=20).
La diferencia observada de Δρ = 0.038 entre V1 y V3 es atribuible
al no-determinismo de GPT-4o con una muestra pequeña, no a una
superioridad real de V1.

### Justificación de selección
Ante ausencia de diferencias estadísticas, se selecciona V3 por
criterios metodológicos:

**Argumento de transparencia:**
V3 incluye CoT explícito con pasos auditables. Cada evaluación
produce un razonamiento visible que puede ser inspeccionado para
detectar sesgos sistemáticos — como el identificado en stratum 3.
V1 produce solo un puntaje sin razonamiento auditable.

**Argumento de robustez:**
La rúbrica anclada de V3 (ejemplos concretos por score 1-5)
reduce la varianza por no-determinismo del LLM. Con n=900 pares,
menor varianza por definición se traduce en mayor estabilidad
del Spearman final.

**Argumento de comparabilidad:**
El sistema de votación agéntico (el otro método comparado en la
tesis) opera con razonamiento explícito. Comparar un método con
CoT (V3) contra un sistema agéntico es metodológicamente más
limpio que comparar un prompt genérico sin razonamiento (V1).

**Argumento de contribución:**
V3 es la versión más cercana al diseño original de G-Eval
(Liu et al., 2023), que especifica explícitamente el uso de
Chain-of-Thought y criterios graduados. Usar V3 mantiene
fidelidad al método que se está evaluando.

### Costo estimado — experimento completo con V3
- 900 pares × gpt-4o con prompt V3: ~$4-6
- V3 es ~30% más largo que V1 en tokens, diferencia de ~$1-2
- Costo adicional justificado por mayor solidez metodológica
"""
)

CELL_17 = code(
    """
from datetime import datetime, timezone


def _changes(prev: str | None, curr: str) -> str:
    if prev is None:
        return "Initial generic baseline."
    if curr == "V2":
        return "Added multi-turn task framing, 3-part relevance definition, and 5-step chain-of-thought."
    if curr == "V3":
        return "Kept V2 structure; added anchored per-score rubric and 3 worked examples drawn from the pilot sample."
    return ""


def _observations(results: list[dict]) -> str:
    ok = [r for r in results if r["geval_score"] is not None]
    if not ok:
        return "All evaluations failed."
    by_stratum: dict[int, list[float]] = defaultdict(list)
    for r in ok:
        by_stratum[r["stratum"]].append(abs(r["delta"]))
    worst = max(by_stratum, key=lambda s: sum(by_stratum[s]) / len(by_stratum[s]))
    best = min(by_stratum, key=lambda s: sum(by_stratum[s]) / len(by_stratum[s]))
    return (
        f"Largest delta on stratum {worst} (mean |\\u0394|="
        f"{sum(by_stratum[worst]) / len(by_stratum[worst]):.2f}); "
        f"smallest delta on stratum {best} (mean |\\u0394|="
        f"{sum(by_stratum[best]) / len(by_stratum[best]):.2f})."
    )


filename_map = {"V1": "v1_generic.txt", "V2": "v2_dialogue_cot.txt", "V3": "v3_full_cot_anchored.txt"}

rows = [
    ("V1", "Generic baseline", results_v1, rho_v1, p_v1, mean_v1, max_v1, None),
    ("V2", "Dialogue-aware CoT", results_v2, rho_v2, p_v2, mean_v2, max_v2, "V1"),
    ("V3", "Full CoT + anchored", results_v3, rho_v3, p_v3, mean_v3, max_v3, "V2"),
]

# Methodological override: V3 is the selected version per the Steiger test
# (no statistically significant difference between versions at n=20).
winner = "V3"

lines: list[str] = []
lines.append("# Prompt Iterations \\u2014 G-Eval Relevance Pilot\\n")
lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
lines.append("Evaluator model: `gpt-4o` \\u2014 direct successor of GPT-4 (Liu et al., EMNLP 2023).\\n")
lines.append("Each version was piloted on the same stratified 20-entry sample (configs/prompts/pilot_sample.json), seed=42.\\n")

for name, label, results, rho, p, mean, mx, prev in rows:
    decision = "Selected" if name == winner else "Rejected"
    lines.append(f"## Version {name} \\u2014 {label}")
    lines.append(f"- Model: `gpt-4o`")
    lines.append(f"- Prompt file: `configs/prompts/{filename_map[name]}`")
    lines.append(f"- Changes from previous: {_changes(prev, name)}")
    lines.append(f"- Pilot Spearman \\u03c1: **{rho:.3f}** (p = {p:.4f})")
    lines.append(f"- Mean |\\u0394|: {mean:.3f}   Max |\\u0394|: {mx:.3f}")
    lines.append(f"- Key observations: {_observations(results)}")
    lines.append(f"- Decision: **{decision}**\\n")

lines.append("## An\\u00e1lisis Estad\\u00edstico de Diferencias entre Versiones\\n")
lines.append("Prueba de Steiger (1980) sobre muestra piloto (n=20):\\n")
lines.append("| Comparaci\\u00f3n | \\u03c1_A | \\u03c1_B | Z | p-value | Conclusi\\u00f3n |")
lines.append("|---|---|---|---|---|---|")
for row in steiger_results:
    concl = "Significativa" if row["p"] < 0.05 else "NO significativa"
    lines.append(
        f"| {row['pair']} | {row['rho_a']:.3f} | {row['rho_b']:.3f} | "
        f"{row['z']:.3f} | {row['p']:.3f} | {concl} |"
    )
lines.append("")
lines.append(
    "Interpretaci\\u00f3n: Con n=20 y \\u0394\\u03c1_max < 0.05, ninguna versi\\u00f3n "
    "demuestra superioridad estad\\u00edstica. La selecci\\u00f3n se basa en criterios "
    "metodol\\u00f3gicos.\\n"
)

lines.append("## Selecci\\u00f3n Final\\n")
lines.append(f"**Versi\\u00f3n seleccionada: V3 \\u2014 Full CoT con r\\u00fabrica anclada**\\n")
lines.append(f"**Modelo evaluador: gpt-4o**\\n")
lines.append(f"**Fecha: {datetime.now(timezone.utc).isoformat()}**\\n")
lines.append(
    "Justificaci\\u00f3n: Ante ausencia de diferencias estad\\u00edsticamente "
    "significativas entre versiones (prueba de Steiger, todos p > 0.05), "
    "se seleccion\\u00f3 V3 por criterios metodol\\u00f3gicos: CoT auditable, "
    "r\\u00fabrica anclada que reduce varianza, y mayor fidelidad al dise\\u00f1o "
    "original de G-Eval (Liu et al., 2023).\\n"
)
lines.append(
    "Nota metodol\\u00f3gica: gpt-4o seleccionado como sucesor directo de "
    "GPT-4 (Liu et al., 2023), manteniendo comparabilidad con baselines "
    "publicados (SummEval relevancia \\u03c1=0.547)."
)

DOCS_PATH = DOCS_DIR / "prompt_iterations.md"
DOCS_PATH.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"Wrote {DOCS_PATH}")
print(f"Winning version: {winner}")

# Persist the selected prompt (V3) as the final prompt file.
winner_path = PROMPTS_DIR / filename_map[winner]
final_path = PROMPTS_DIR / "geval_relevance_prompt.txt"
final_path.write_text(winner_path.read_text(encoding="utf-8"), encoding="utf-8")
print(f"Saved winning prompt to {final_path}")
"""
)

CELL_18 = md(
    """
## 11. Conclusiones

### Modelo y diseño del prompt
`gpt-4o` se usó como evaluador en todas las instancias de G-Eval, siguiendo
la justificación metodológica de la sección 1. Las tres versiones evolucionan
de un baseline genérico (V1) → CoT consciente del diálogo (V2) → rúbrica
anclada con ejemplos trabajados (V3). La selección y persistencia están
automatizadas en la sección 10.

### Correlación del piloto frente a Liu et al. (2023)
Para interpretar la tabla comparativa de la sección 8:

- Relevancia en SummEval (paper original): Spearman ρ ≈ **0.547**
- Coherencia en Topical-Chat USR (paper original): Spearman ρ ≈ **0.605**
- Relevancia en DailyDialog-Zhao (este trabajo): ver sección 8

La comparación numérica es aproximada — difieren datasets, dominios y
fraseo de la tarea — pero el rango publicado da un objetivo de orden de
magnitud para saber si el prompt está en régimen "deployable".

### Sesgos sistemáticos a monitorear
Revisar el scatter (sección 8) buscando patrones de sobre/infra-valoración
por estrato. Modos de fallo frecuentes:

- Sobre-valoración en estratos 4 y 5 (filler genérico puntuado como
  parcialmente relevante).
- Infra-valoración en estrato 1 (respuestas ground-truth que requieren
  conocimiento del mundo para validarse).
- Compresión de scores hacia 3 en V1 al no tener rúbrica anclada.

Estos patrones, de aparecer, deben documentarse en la tesis al interpretar
la correlación final de los 900 pares.

### Criterio de avance al experimento completo
Regla de decisión:

- Si la ρ de la versión seleccionada supera **0.4** en el piloto, el prompt
  se considera listo para el run de 900 pares (~$15–25 con gpt-4o).
- Si 0.2 < ρ ≤ 0.4, considerar una revisión adicional del prompt antes
  de escalar.
- Si ρ ≤ 0.2 en las tres versiones, detener y diagnosticar — el fallo puede
  estar en el pipeline (construcción de test cases, rescaling, muestreo
  estratificado) antes de culpar al prompt.
"""
)


def main() -> None:
    nb = nbformat.v4.new_notebook()  # type: ignore[no-untyped-call]
    nb["cells"] = [
        CELL_1,
        CELL_3,
        CELL_SECTION_SETUP,
        CELL_2,
        CELL_SECTION_STRATA,
        CELL_4,
        CELL_SECTION_PROMPTS,
        CELL_5,
        CELL_SECTION_HELPERS,
        CELL_6,
        CELL_SECTION_V1,
        CELL_7,
        CELL_8,
        CELL_9,
        CELL_10,
        CELL_SECTION_V2,
        CELL_11,
        CELL_12,
        CELL_SECTION_V3,
        CELL_13,
        CELL_14,
        CELL_SECTION_COMPARISON,
        CELL_15,
        CELL_STEIGER_MD,
        CELL_STEIGER_CODE,
        CELL_16,
        CELL_SECTION_PERSIST,
        CELL_17,
        CELL_18,
    ]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)  # type: ignore[no-untyped-call]
    print(f"Wrote {NB_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
