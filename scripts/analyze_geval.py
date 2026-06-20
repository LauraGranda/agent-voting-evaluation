"""Analyse a completed G-Eval run and produce an interpreted report.

Reads the per-pair results written by ``run_geval.py`` plus the source
dataset, then computes agreement metrics against the human relevance
scores, breaks them down by model family and individual model, and writes
an ordered markdown report with four supporting figures.

Usage:
    uv run python scripts/analyze_geval.py
    uv run python scripts/analyze_geval.py --results outputs/geval_results.json

Outputs (under ``outputs/``):
    - geval_analysis_report.md          interpreted, ordered analysis
    - figures/05_geval_vs_human_scatter.png
    - figures/06_residual_histogram.png
    - figures/07_delta_by_family_boxplot.png
    - figures/08_mean_score_by_family.png
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display server
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "deepeval_test_cases.json"
DEFAULT_RESULTS: Final[Path] = PROJECT_ROOT / "outputs" / "geval_results.json"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"

# Agreement tolerances on the shared 1-5 scale.
TOL_TIGHT: Final[float] = 0.5
TOL_LOOSE: Final[float] = 1.0

# Number of largest-error pairs to surface in the report.
TOP_N_ERRORS: Final[int] = 10


# ─── Correlation interpretation ──────────────────────────────────────────
def interpret_rho(rho: float) -> str:
    """Plain-language band for an absolute correlation coefficient."""
    a = abs(rho)
    if a < 0.20:
        return "negligible"
    if a < 0.40:
        return "weak"
    if a < 0.60:
        return "moderate"
    if a < 0.80:
        return "strong"
    return "very strong"


def model_family(model_name: str) -> str:
    """Collapse a dataset model name onto a coarse family label."""
    if model_name in ("ground-truth", "negative-sample"):
        return model_name
    for family in ("GPT2", "VHRED", "HRED", "S2S"):
        if model_name.startswith(family):
            return family
    return "other"


# ─── Loading ─────────────────────────────────────────────────────────────
def load_json(path: Path) -> Any:
    """Read and parse a JSON file as UTF-8."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rel(path: Path) -> Path:
    """Path relative to the project root, for portable markdown links."""
    return path.relative_to(PROJECT_ROOT)


def join_results(results: list[dict], dataset: list[dict]) -> list[dict]:
    """Attach the dataset ``model`` and family to each successful result."""
    id_to_model = {e["metadata"]["conversation_id"]: e["metadata"]["model"] for e in dataset}
    joined = []
    for r in results:
        if r.get("geval_score") is None:
            continue
        model = id_to_model.get(r["conversation_id"], "unknown")
        joined.append(
            {
                "conversation_id": r["conversation_id"],
                "geval": float(r["geval_score"]),
                "human": float(r["human_score"]),
                "delta": float(r["geval_score"]) - float(r["human_score"]),
                "model": model,
                "family": model_family(model),
                "reason": r.get("reason", ""),
            }
        )
    return joined


# ─── Metrics ─────────────────────────────────────────────────────────────
def agreement_metrics(rows: list[dict]) -> dict[str, float]:
    """Correlation + error metrics for G-Eval vs human scores."""
    geval = np.array([r["geval"] for r in rows])
    human = np.array([r["human"] for r in rows])
    delta = geval - human

    rho, rho_p = spearmanr(human, geval)
    pear, pear_p = pearsonr(human, geval)
    tau, tau_p = kendalltau(human, geval)

    return {
        "n": len(rows),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(pear),
        "pearson_p": float(pear_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "bias": float(np.mean(delta)),
        "within_tight": float(np.mean(np.abs(delta) <= TOL_TIGHT)),
        "within_loose": float(np.mean(np.abs(delta) <= TOL_LOOSE)),
        "geval_mean": float(np.mean(geval)),
        "human_mean": float(np.mean(human)),
    }


def per_group(rows: list[dict], key: str) -> list[dict]:
    """Per-group (family or model) mean scores and bias, sorted by human mean."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out = []
    for name, grp in groups.items():
        g = np.array([x["geval"] for x in grp])
        h = np.array([x["human"] for x in grp])
        out.append(
            {
                "name": name,
                "n": len(grp),
                "human_mean": float(np.mean(h)),
                "geval_mean": float(np.mean(g)),
                "bias": float(np.mean(g - h)),
                "mae": float(np.mean(np.abs(g - h))),
            }
        )
    return sorted(out, key=lambda d: d["human_mean"], reverse=True)


def score_band_crosstab(rows: list[dict]) -> tuple[np.ndarray, float]:
    """5x5 count matrix of rounded human vs G-Eval scores, plus exact-band agreement."""
    matrix = np.zeros((5, 5), dtype=int)
    hits = 0
    for r in rows:
        h = min(5, max(1, round(r["human"])))
        g = min(5, max(1, round(r["geval"])))
        matrix[5 - h, g - 1] += 1  # row 0 = human 5 (top), col 0 = geval 1
        if h == g:
            hits += 1
    return matrix, hits / len(rows)


# ─── Inter-annotator agreement (human ceiling) ───────────────────────────
def _rating_matrix(dataset: list[dict]) -> np.ndarray:
    """(n_units, 4) integer matrix of raw human relevance votes."""
    rows = [
        e["metadata"]["raw_relevance_scores"]
        for e in dataset
        if e["metadata"].get("raw_relevance_scores")
    ]
    return np.array(rows, dtype=int)


def krippendorff_alpha_ordinal(ratings: np.ndarray) -> float:
    """Krippendorff's alpha with the ordinal difference metric.

    ``ratings`` is a (units x coders) matrix with no missing values. The
    ordinal metric weights disagreements by the marginal mass between the
    two categories, so a 1-vs-5 split costs far more than a 4-vs-5 split.
    """
    values = np.arange(1, 6)  # rating scale 1..5
    idx = {int(v): i for i, v in enumerate(values)}
    v_n = len(values)
    n_units, m = ratings.shape

    coincidence = np.zeros((v_n, v_n))
    for unit in ratings:
        counts = np.zeros(v_n)
        for r in unit:
            counts[idx[int(r)]] += 1
        for c in range(v_n):
            for k in range(v_n):
                pairs = counts[c] * (counts[c] - 1) if c == k else counts[c] * counts[k]
                coincidence[c, k] += pairs / (m - 1)

    n_c = coincidence.sum(axis=1)
    n = n_c.sum()

    def delta2(c: int, k: int) -> float:
        """Squared interval distance between ordinal categories ``c`` and ``k``."""
        lo, hi = (c, k) if c <= k else (k, c)
        gap = n_c[lo : hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2.0
        return float(gap * gap)

    d_obs = sum(coincidence[c, k] * delta2(c, k) for c in range(v_n) for k in range(v_n)) / n
    d_exp = sum(n_c[c] * n_c[k] * delta2(c, k) for c in range(v_n) for k in range(v_n)) / (
        n * (n - 1)
    )
    return float(1 - d_obs / d_exp) if d_exp else 1.0


def icc_2_1(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way random-effects, single-rater, absolute agreement.

    Answers "how reliable is one annotator's raw score" on the 1-5 scale —
    a stricter, value-level companion to the rank-based Spearman ceiling.
    """
    n, k = ratings.shape
    grand = ratings.mean()
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((ratings - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom else 0.0


def human_ceiling(dataset: list[dict]) -> dict[str, float]:
    """Inter-annotator agreement metrics — the ceiling G-Eval is judged against.

    - pairwise: mean Spearman over the 6 annotator-column pairs (two single
      humans agreeing with each other).
    - leave_one_out: mean Spearman of each annotator column vs. the mean of
      the other 3 — directly comparable to G-Eval, which is also one rater
      predicting the human consensus.
    """
    ratings = _rating_matrix(dataset)
    n_units, k = ratings.shape

    pairwise = [
        spearmanr(ratings[:, i], ratings[:, j]).statistic for i in range(k) for j in range(i + 1, k)
    ]
    loo = []
    for j in range(k):
        others = np.delete(ratings, j, axis=1).mean(axis=1)
        loo.append(spearmanr(ratings[:, j], others).statistic)

    return {
        "n_units": n_units,
        "n_annotators": k,
        "pairwise_spearman": float(np.mean(pairwise)),
        "loo_spearman": float(np.mean(loo)),
        "krippendorff_alpha": krippendorff_alpha_ordinal(ratings),
        "icc_2_1": icc_2_1(ratings.astype(float)),
    }


def ceiling_verdict(geval_rho: float, loo: float) -> str:
    """Plain-language verdict comparing G-Eval to the human ceiling."""
    if geval_rho > loo + 0.03:
        return (
            "G-Eval **exceeds the single-human ceiling** — it tracks the 4-rater "
            "consensus more closely than an individual annotator does. This is the "
            "expected outcome when the human labels are noisy: a *consistent* rater "
            "correlates with the average better than the noisy individuals correlate "
            "with each other. It does **not** mean G-Eval is 'better than humans' — "
            "the consensus is the target by construction — but it does mean a "
            "stronger evaluator model would buy essentially nothing: the gap left to "
            "ρ = 1.0 is mostly irreducible annotation noise, not model weakness."
        )
    if geval_rho >= loo - 0.02:
        return (
            "G-Eval **matches the human ceiling** — a single G-Eval pass agrees "
            "with the consensus about as well as a single human annotator does. "
            "A stronger evaluator model would yield little to no measurable gain."
        )
    if geval_rho >= loo - 0.10:
        return (
            "G-Eval sits **just below the human ceiling** — close enough that the "
            "remaining gap is within annotation noise; switching evaluator models "
            "is unlikely to be worth the cost."
        )
    return (
        "G-Eval sits **measurably below the human ceiling** — there is genuine "
        "headroom, so a stronger evaluator model or a revised prompt could help."
    )


# ─── Figures ─────────────────────────────────────────────────────────────
def fig_scatter(rows: list[dict], path: Path) -> None:
    """Save the human-vs-G-Eval scatter with best-fit line at ``path``."""
    geval = np.array([r["geval"] for r in rows])
    human = np.array([r["human"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(human, geval, alpha=0.35, s=20, edgecolor="none")
    ax.plot([1, 5], [1, 5], "r--", lw=1, label="perfect agreement")
    coef = np.polyfit(human, geval, 1)
    xs = np.array([1, 5])
    ax.plot(xs, coef[0] * xs + coef[1], "b-", lw=1.5, label="best fit")
    ax.set_xlabel("Human relevance score (1-5)")
    ax.set_ylabel("G-Eval relevance score (1-5)")
    ax.set_title("G-Eval vs. human relevance")
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_residuals(rows: list[dict], path: Path) -> None:
    """Save the histogram of residuals ``Δ = G-Eval − human`` at ``path``."""
    delta = np.array([r["delta"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(delta, bins=40, color="#4477aa", edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.axvline(
        float(np.mean(delta)), color="black", lw=1.5, label=f"mean bias {np.mean(delta):+.2f}"
    )
    ax.set_xlabel("Δ = G-Eval − human")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_delta_boxplot(rows: list[dict], path: Path) -> None:
    """Save a boxplot of residuals broken down by model family at ``path``."""
    fams = sorted({r["family"] for r in rows})
    data = [[r["delta"] for r in rows if r["family"] == f] for f in fams]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=fams, showmeans=True)
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_ylabel("Δ = G-Eval − human")
    ax.set_title("Residuals by model family")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_mean_by_family(family_rows: list[dict], path: Path) -> None:
    """Save a paired bar plot of human vs G-Eval mean scores per family."""
    names = [g["name"] for g in family_rows]
    human = [g["human_mean"] for g in family_rows]
    geval = [g["geval_mean"] for g in family_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, human, 0.4, label="human", color="#ee6677")
    ax.bar(x + 0.2, geval, 0.4, label="G-Eval", color="#4477aa")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Mean relevance score (1-5)")
    ax.set_title("Mean score by model family: human vs. G-Eval")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_ceiling(geval_rho: float, ceiling: dict[str, float], path: Path) -> None:
    """Save a horizontal bar plot comparing G-Eval against the human agreement ceiling."""
    labels = [
        "Two single humans\n(pairwise)",
        "Human ceiling\n(1 rater vs consensus)",
        "G-Eval\n(vs consensus)",
    ]
    vals = [ceiling["pairwise_spearman"], ceiling["loo_spearman"], geval_rho]
    colors = ["#bbbbbb", "#ee6677", "#4477aa"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, vals, color=colors)
    ax.axvline(ceiling["loo_spearman"], color="#ee6677", ls="--", lw=1)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spearman ρ vs. consensus")
    ax.set_title("G-Eval vs. the human agreement ceiling")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─── Report ──────────────────────────────────────────────────────────────
def build_report(  # noqa: PLR0913
    rows: list[dict],
    metrics: dict[str, float],
    ceiling: dict[str, float],
    by_family: list[dict],
    by_model: list[dict],
    crosstab: np.ndarray,
    band_agreement: float,
    n_total: int,
    n_fail: int,
    fig_dir: Path,
) -> str:
    """Render the full markdown report string from the precomputed pieces."""
    rho = metrics["spearman_rho"]
    bias = metrics["bias"]
    direction = "over-rates" if bias > 0 else "under-rates"
    loo = ceiling["loo_spearman"]
    gap = rho - loo
    over = sorted(rows, key=lambda r: r["delta"], reverse=True)[:TOP_N_ERRORS]
    under = sorted(rows, key=lambda r: r["delta"])[:TOP_N_ERRORS]

    L: list[str] = [
        "# G-Eval Run — Analysis Report\n",
        f"_Generated: {datetime.now(UTC).isoformat()}_\n",
        "## 1. Executive summary\n",
        f"G-Eval (gpt-4o) scored **{metrics['n']} of {n_total}** conversation-response "
        f"pairs ({n_fail} failed) for relevance against four human annotators.\n",
        f"- **Rank agreement with humans**: Spearman ρ = **{rho:.3f}** "
        f"(_{interpret_rho(rho)}_, p = {metrics['spearman_p']:.2e}).",
        f"- **Human ceiling**: a single annotator predicts the 4-rater consensus at "
        f"ρ = {loo:.3f}; G-Eval's ρ = {rho:.3f} is {gap:+.3f} vs. that ceiling.",
        f"- **Linear agreement**: Pearson r = {metrics['pearson_r']:.3f}; "
        f"Kendall τ = {metrics['kendall_tau']:.3f}.",
        f"- **Error magnitude**: MAE = {metrics['mae']:.2f}, RMSE = {metrics['rmse']:.2f} "
        "points on the 1-5 scale.",
        f"- **Systematic bias**: mean Δ = {bias:+.2f} — G-Eval **{direction}** responses "
        f"relative to humans (G-Eval mean {metrics['geval_mean']:.2f} vs. human "
        f"{metrics['human_mean']:.2f}).",
        f"- **Tolerance hit-rate**: {metrics['within_tight']:.0%} of pairs within ±{TOL_TIGHT}, "
        f"{metrics['within_loose']:.0%} within ±{TOL_LOOSE} of the human score.\n",
        "## 2. Agreement metrics\n",
        "| Metric | Value | Reading |",
        "|---|---|---|",
        f"| Pairs analysed (n) | {metrics['n']} | successful evaluations |",
        f"| Spearman ρ | {rho:.4f} | rank correlation ({interpret_rho(rho)}) |",
        f"| Spearman p-value | {metrics['spearman_p']:.2e} | significance |",
        f"| Pearson r | {metrics['pearson_r']:.4f} | linear correlation |",
        f"| Kendall τ | {metrics['kendall_tau']:.4f} | concordance |",
        f"| MAE | {metrics['mae']:.4f} | mean absolute error (1-5 pts) |",
        f"| RMSE | {metrics['rmse']:.4f} | penalises large misses |",
        f"| Mean bias (Δ) | {bias:+.4f} | + = G-Eval scores high |",
        f"| Within ±{TOL_TIGHT} | {metrics['within_tight']:.1%} | tight agreement |",
        f"| Within ±{TOL_LOOSE} | {metrics['within_loose']:.1%} | loose agreement |\n",
        "## 3. Human ceiling — inter-annotator agreement\n",
        "Spearman ρ can only be judged against how well the humans agree **with each "
        "other**. A metric cannot be expected to beat the noise floor of the labels "
        "it is graded on.\n",
        "| Reference | ρ / coefficient | Meaning |",
        "|---|---|---|",
        f"| Two single humans (pairwise ρ) | {ceiling['pairwise_spearman']:.4f} | "
        "one annotator vs. another |",
        f"| **Human ceiling (leave-one-out ρ)** | **{loo:.4f}** | one annotator vs. "
        "the mean of the other 3 |",
        f"| **G-Eval vs. consensus** | **{rho:.4f}** | one G-Eval pass vs. the 4-human mean |",
        f"| Krippendorff α (ordinal) | {ceiling['krippendorff_alpha']:.4f} | standard "
        "reliability coefficient |",
        f"| ICC(2,1) single rater | {ceiling['icc_2_1']:.4f} | absolute-agreement "
        "reliability of one rating |\n",
        f"**Gap to ceiling: {gap:+.4f}.** {ceiling_verdict(rho, loo)}\n",
        "## 4. Breakdown by model family\n",
        "Families ordered by human-rated relevance (best first). `bias` is the mean "
        "G-Eval − human gap within the family.\n",
        "| Family | n | Human mean | G-Eval mean | Bias | MAE |",
        "|---|---|---|---|---|---|",
    ]
    for g in by_family:
        L.append(
            f"| {g['name']} | {g['n']} | {g['human_mean']:.2f} | {g['geval_mean']:.2f} | "
            f"{g['bias']:+.2f} | {g['mae']:.2f} |"
        )

    L += [
        "\n## 5. Breakdown by individual model\n",
        "| Model | n | Human mean | G-Eval mean | Bias | MAE |",
        "|---|---|---|---|---|---|",
    ]
    for g in by_model:
        L.append(
            f"| {g['name']} | {g['n']} | {g['human_mean']:.2f} | {g['geval_mean']:.2f} | "
            f"{g['bias']:+.2f} | {g['mae']:.2f} |"
        )

    L += [
        "\n## 6. Score-band confusion matrix\n",
        f"Both scores rounded to the nearest integer (1-5). Exact-band agreement: "
        f"**{band_agreement:.1%}**. Rows = human, columns = G-Eval.\n",
        "| Human ↓ / G-Eval → | 1 | 2 | 3 | 4 | 5 |",
        "|---|---|---|---|---|---|",
    ]
    for i, h in enumerate(range(5, 0, -1)):
        cells = " | ".join(str(int(crosstab[i, j])) for j in range(5))
        L.append(f"| **{h}** | {cells} |")

    L += [
        "\n## 7. Largest disagreements\n",
        f"### G-Eval over-rated (top {TOP_N_ERRORS} positive Δ)\n",
        "| conversation_id | model | human | G-Eval | Δ |",
        "|---|---|---|---|---|",
    ]
    for r in over:
        L.append(
            f"| {r['conversation_id']} | {r['family']} | {r['human']:.2f} | "
            f"{r['geval']:.2f} | {r['delta']:+.2f} |"
        )
    L += [
        f"\n### G-Eval under-rated (top {TOP_N_ERRORS} negative Δ)\n",
        "| conversation_id | model | human | G-Eval | Δ |",
        "|---|---|---|---|---|",
    ]
    for r in under:
        L.append(
            f"| {r['conversation_id']} | {r['family']} | {r['human']:.2f} | "
            f"{r['geval']:.2f} | {r['delta']:+.2f} |"
        )

    L += [
        "\n## 8. Figures\n",
        f"![G-Eval vs human]({_rel(fig_dir)}/05_geval_vs_human_scatter.png)\n",
        f"![Residual histogram]({_rel(fig_dir)}/06_residual_histogram.png)\n",
        f"![Residuals by family]({_rel(fig_dir)}/07_delta_by_family_boxplot.png)\n",
        f"![Mean score by family]({_rel(fig_dir)}/08_mean_score_by_family.png)\n",
        f"![G-Eval vs human ceiling]({_rel(fig_dir)}/09_ceiling_comparison.png)\n",
        "## 9. How to read this\n",
        f"- The ρ of {rho:.2f} should be read against the {loo:.2f} human ceiling, "
        "not against 1.0 — no metric can out-agree the label noise it is graded on.",
        f"- A Spearman ρ of {rho:.2f} means G-Eval reproduces the human **ranking** "
        f"of responses {interpret_rho(rho)}ly — useful for picking the better of two "
        "responses even when absolute points differ.",
        f"- The {bias:+.2f} bias is a *calibration* offset: it can be subtracted out "
        "before comparing against the 1-5 human scale.",
        "- Families where `bias` and `MAE` are largest are where the metric is least "
        "trustworthy and most worth a prompt revision.",
        "",
    ]
    return "\n".join(L)


# ─── Orchestration ───────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the analyse-G-Eval entry point."""
    p = argparse.ArgumentParser(description="Analyse a completed G-Eval run.")
    p.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS, help="Path to geval_results.json."
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return p.parse_args(argv)


def main() -> None:
    """Entry point: load results, compute metrics, render figures and report."""
    args = parse_args()
    results = load_json(args.results)
    dataset = load_json(DATA_PATH)

    n_total = len(results)
    n_fail = sum(1 for r in results if r.get("geval_score") is None)
    rows = join_results(results, dataset)
    if not rows:
        raise SystemExit("No successful results to analyse.")

    metrics = agreement_metrics(rows)
    ceiling = human_ceiling(dataset)
    by_family = per_group(rows, "family")
    by_model = per_group(rows, "model")
    crosstab, band_agreement = score_band_crosstab(rows)

    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_scatter(rows, fig_dir / "05_geval_vs_human_scatter.png")
    fig_residuals(rows, fig_dir / "06_residual_histogram.png")
    fig_delta_boxplot(rows, fig_dir / "07_delta_by_family_boxplot.png")
    fig_mean_by_family(by_family, fig_dir / "08_mean_score_by_family.png")
    fig_ceiling(metrics["spearman_rho"], ceiling, fig_dir / "09_ceiling_comparison.png")

    report = build_report(
        rows,
        metrics,
        ceiling,
        by_family,
        by_model,
        crosstab,
        band_agreement,
        n_total,
        n_fail,
        fig_dir,
    )
    report_path = args.output_dir / "geval_analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Analysis written to {report_path}")
    print(f"Figures written to {fig_dir}")
    print(
        f"n={metrics['n']}  Spearman={metrics['spearman_rho']:.3f}  "
        f"MAE={metrics['mae']:.2f}  bias={metrics['bias']:+.2f}"
    )
    print(
        f"Human ceiling: leave-one-out ρ={ceiling['loo_spearman']:.3f}  "
        f"pairwise ρ={ceiling['pairwise_spearman']:.3f}  "
        f"Krippendorff α={ceiling['krippendorff_alpha']:.3f}  "
        f"gap={metrics['spearman_rho'] - ceiling['loo_spearman']:+.3f}"
    )


if __name__ == "__main__":
    main()
