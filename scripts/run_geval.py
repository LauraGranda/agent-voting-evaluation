"""Run G-Eval relevance scoring over the full DailyDialog-Zhao dataset.

Sequential, retried, checkpointed. Writes per-entry results, an execution
log, and a summary-statistics markdown. The 900-pair run consumes ~$5 USD
of OpenAI gpt-4o credits and ~30-60 minutes wall time.

Usage:
    uv run python scripts/run_geval.py                 # full 900-pair run
    uv run python scripts/run_geval.py --limit 2       # smoke test (~$0.01)
    uv run python scripts/run_geval.py --no-resume     # ignore checkpoint

Outputs (under ``outputs/``):
    - geval_results.json         per-pair scores, tokens, cost, timestamps
    - geval_summary_stats.md     descriptive stats + Spearman vs human
    - logs/geval_execution.log   per-entry + final summary log
    - .geval_checkpoint.json     written every ``CHECKPOINT_EVERY`` entries
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Any, Final

import openai
import tiktoken
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
from scipy.stats import spearmanr
from tenacity import (
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ─── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "deepeval_test_cases.json"
PROMPT_PATH: Final[Path] = PROJECT_ROOT / "configs" / "prompts" / "geval_relevance_prompt.txt"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"

# ─── Run constants ───────────────────────────────────────────────────────
DEFAULT_MODEL: Final[str] = "gpt-4o"
CHECKPOINT_EVERY: Final[int] = 25

# gpt-4o pricing (2026-04, per 1M tokens). Same source as the pilot.
PRICE_INPUT_PER_M: Final[float] = 2.50
PRICE_OUTPUT_PER_M: Final[float] = 10.00

# Score rescale anchors (0-1 raw G-Eval → 1-5 human scale).
SCALE_MIN: Final[float] = 1.0
SCALE_RANGE: Final[float] = 4.0

# Retry policy.
RETRY_MAX_ATTEMPTS: Final[int] = 5
RETRY_WAIT_MIN_S: Final[int] = 2
RETRY_WAIT_MAX_S: Final[int] = 60

# Minimum sample size for a meaningful Spearman correlation.
MIN_FOR_SPEARMAN: Final[int] = 2

# Transient OpenAI errors worth retrying. Auth/validation errors fail fast.
RETRYABLE_EXCEPTIONS: Final[tuple[type[Exception], ...]] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

# Errors that mean the run cannot succeed at all (bad credentials, wrong
# model name, account suspended). Re-raising aborts the loop instead of
# burning ~75 minutes recording 900 identical failures.
FATAL_EXCEPTIONS: Final[tuple[type[Exception], ...]] = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
)

logger = logging.getLogger("run_geval")


# ─── Logging setup ───────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> None:
    """Configure root logger to emit INFO to both stdout and ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(sh)

    logger.propagate = False


# ─── Pure helpers (unit-tested) ──────────────────────────────────────────
def _rescale_0_1_to_1_5(score_0_1: float) -> float:
    """Map a 0-1 G-Eval score onto a 1-5 scale for delta comparison."""
    return SCALE_MIN + SCALE_RANGE * score_0_1


def build_test_case(entry: dict[str, Any]) -> tuple[LLMTestCase, str]:
    """Pack the conversation into ``input`` and return the LLMTestCase plus
    the packed input string (needed downstream for token counting)."""
    context = entry["turns"][:-1]
    formatted = "\n".join(
        f"[Turn {i + 1}] {t['role'].capitalize()}: {t['content']}" for i, t in enumerate(context)
    )
    return LLMTestCase(input=formatted, actual_output=entry["actual_output"]), formatted


def build_geval_metric(prompt_text: str, model: str) -> GEval:
    """Initialize GEval with the given criteria and evaluator model."""
    return GEval(
        name="Relevance",
        criteria=prompt_text,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
    )


def estimate_tokens_and_cost(
    prompt_text: str,
    packed_input: str,
    actual_output: str,
    reason: str,
) -> tuple[int, int, float]:
    """Estimate (input_tokens, output_tokens, cost_usd) using tiktoken.

    Input tokens cover the criteria + packed conversation + response.
    Output tokens cover the reason text returned by GEval (plus a small
    overhead for the score JSON wrapper that DeepEval requests).
    Costs are computed at gpt-4o list pricing.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    input_tokens = (
        len(enc.encode(prompt_text))
        + len(enc.encode(packed_input))
        + len(enc.encode(actual_output))
    )
    # +20 tokens of overhead for DeepEval's JSON envelope (score + reason wrapper).
    output_tokens = len(enc.encode(reason)) + 20
    cost_usd = (
        input_tokens * PRICE_INPUT_PER_M / 1_000_000
        + output_tokens * PRICE_OUTPUT_PER_M / 1_000_000
    )
    return input_tokens, output_tokens, round(cost_usd, 6)


# ─── Retry-wrapped evaluator call ────────────────────────────────────────
def _measure_with_retry(metric: GEval, tc: LLMTestCase) -> tuple[int, str]:
    """Run ``metric.measure(tc)`` with exponential backoff. Returns the
    number of attempts taken and the final reason. Re-raises if all attempts
    are exhausted (caller catches and records the failure)."""
    attempts = 0
    last_reason = ""
    for attempt in Retrying(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN_S, max=RETRY_WAIT_MAX_S),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            attempts = attempt.retry_state.attempt_number
            metric.measure(tc)
            last_reason = getattr(metric, "reason", None) or ""
    return attempts, last_reason


# ─── Checkpointing ───────────────────────────────────────────────────────
def load_checkpoint(checkpoint_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Return (results_so_far, completed_ids). Empty if no checkpoint exists.

    A corrupt checkpoint (truncated JSON, partial write) is rotated to a
    ``.bak`` sibling and treated as "no checkpoint" so the run can restart
    rather than crash on startup. The .bak file lets the user inspect what
    was lost (atomic write + rename normally prevents this, but a kill -9
    or disk-full mid-rename can leave a malformed file).
    """
    if not checkpoint_path.exists():
        return [], set()
    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            results: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        backup = checkpoint_path.with_suffix(checkpoint_path.suffix + ".bak")
        try:
            checkpoint_path.replace(backup)
            logger.warning(
                "Corrupt checkpoint at %s (%s); moved to %s and starting fresh",
                checkpoint_path,
                exc,
                backup,
            )
        except OSError:
            logger.warning("Corrupt checkpoint at %s (%s); starting fresh", checkpoint_path, exc)
        return [], set()
    ids = {r["conversation_id"] for r in results}
    logger.info("Resuming from checkpoint: %d entries already evaluated", len(results))
    return results, ids


def save_checkpoint(results: list[dict[str, Any]], checkpoint_path: Path) -> None:
    """Atomic write: dump to .tmp, then rename."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.replace(checkpoint_path)


def write_results(results: list[dict[str, Any]], path: Path) -> None:
    """Atomic final write of the results JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ─── Main loop ───────────────────────────────────────────────────────────
def evaluate_dataset(  # noqa: PLR0913
    dataset: list[dict[str, Any]],
    prompt_text: str,
    model: str,
    output_dir: Path,
    limit: int | None,
    resume: bool,
) -> list[dict[str, Any]]:
    """Iterate over ``dataset`` and run G-Eval per entry. Returns results.

    Every ``CHECKPOINT_EVERY`` entries the script writes ALL three artifacts
    (checkpoint, results.json, summary.md) so a Ctrl+C / power loss leaves
    usable AC artifacts on disk — never just a checkpoint that needs recovery.
    """
    checkpoint_path = output_dir / ".geval_checkpoint.json"
    results_path = output_dir / "geval_results.json"
    summary_path = output_dir / "geval_summary_stats.md"

    if resume:
        results, done_ids = load_checkpoint(checkpoint_path)
    else:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        results, done_ids = [], set()

    pending = [e for e in dataset if e["metadata"]["conversation_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
    total = len(pending) + len(results)

    metric = build_geval_metric(prompt_text, model)

    n_ok = sum(1 for r in results if r.get("geval_score") is not None)
    n_fail = len(results) - n_ok
    t0 = perf_counter()

    try:
        for idx, entry in enumerate(pending, start=len(results) + 1):
            conv_id = entry["metadata"]["conversation_id"]
            human_score = entry["metadata"]["human_score"]
            result = _evaluate_one(entry, metric, prompt_text, model)

            if result.get("geval_score") is not None:
                n_ok += 1
                elapsed = result.pop("_elapsed_s", 0.0)
                logger.info(
                    "[%03d/%03d] %s  geval=%.2f  human=%.2f  Δ=%+.2f  tok=%d  $%.4f  %.2fs",
                    idx,
                    total,
                    conv_id,
                    result["geval_score"],
                    human_score,
                    result["delta"],
                    result["tokens_used"]["total"],
                    result["cost_usd"],
                    elapsed,
                )
            else:
                n_fail += 1
                logger.error(
                    "[%03d/%03d] %s  FAILED after %d attempts: %s",
                    idx,
                    total,
                    conv_id,
                    result.get("attempts", 0),
                    result.get("error", "unknown"),
                )

            results.append(result)
            if idx % CHECKPOINT_EVERY == 0:
                _persist_partial(
                    results, checkpoint_path, results_path, summary_path, dataset, model
                )
                logger.info("Checkpoint + outputs saved at %d/%d", idx, total)
    finally:
        # Persist whatever we have on success, on Ctrl+C, on fatal exception.
        _persist_partial(results, checkpoint_path, results_path, summary_path, dataset, model)

    elapsed_total = perf_counter() - t0
    _log_run_summary(results, n_ok, n_fail, total, elapsed_total)
    return results


def _persist_partial(  # noqa: PLR0913
    results: list[dict[str, Any]],
    checkpoint_path: Path,
    results_path: Path,
    summary_path: Path,
    dataset: list[dict[str, Any]],
    model: str,
) -> None:
    """Write checkpoint + results.json + summary.md atomically. Never raises."""
    try:
        save_checkpoint(results, checkpoint_path)
        write_results(results, results_path)
        generate_summary_stats(results, summary_path, dataset=dataset, model=model)
    except OSError:
        # Disk full, permission denied, etc. Log but don't kill the run.
        logger.exception("Failed to persist partial results")


def _evaluate_one(
    entry: dict[str, Any],
    metric: GEval,
    prompt_text: str,
    model: str,
) -> dict[str, Any]:
    """Run GEval on a single entry and return a result dict (success or failure)."""
    conv_id = entry["metadata"]["conversation_id"]
    human_score = entry["metadata"]["human_score"]
    timestamp = datetime.now(UTC).isoformat()
    t0 = perf_counter()

    try:
        tc, packed_input = build_test_case(entry)
        attempts, reason = _measure_with_retry(metric, tc)
        raw_score = float(metric.score)
        geval_score = _rescale_0_1_to_1_5(raw_score)
        in_tok, out_tok, cost = estimate_tokens_and_cost(
            prompt_text, packed_input, entry["actual_output"], reason
        )
        elapsed = perf_counter() - t0
        return {
            "conversation_id": conv_id,
            "geval_score": round(geval_score, 4),
            "human_score": human_score,
            "model_used": model,
            "timestamp": timestamp,
            "tokens_used": {"input": in_tok, "output": out_tok, "total": in_tok + out_tok},
            "cost_usd": cost,
            "geval_score_raw": round(raw_score, 6),
            "delta": round(geval_score - human_score, 4),
            "reason": reason,
            "attempts": attempts,
            "_elapsed_s": elapsed,
        }
    except FATAL_EXCEPTIONS:
        # Bad credentials / wrong model / no access. The next 899 entries
        # will fail identically, so abort the run instead of burning 75
        # minutes on guaranteed failures. The finally-block in the caller
        # still persists what we have.
        raise
    except (RetryError, *RETRYABLE_EXCEPTIONS) as exc:
        return _failure_result(conv_id, human_score, model, timestamp, exc, retried=True)
    except Exception as exc:
        # Broad catch is intentional: never let one entry's quirk kill the whole run.
        return _failure_result(conv_id, human_score, model, timestamp, exc, retried=False)


def _failure_result(  # noqa: PLR0913
    conv_id: str,
    human_score: float,
    model: str,
    timestamp: str,
    exc: BaseException,
    *,
    retried: bool,
) -> dict[str, Any]:
    """Build a failure-result dict matching the success schema (with nulls)."""
    return {
        "conversation_id": conv_id,
        "geval_score": None,
        "human_score": human_score,
        "model_used": model,
        "timestamp": timestamp,
        "tokens_used": None,
        "cost_usd": None,
        "geval_score_raw": None,
        "delta": None,
        "reason": "",
        "attempts": RETRY_MAX_ATTEMPTS if retried else 1,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _log_run_summary(
    results: list[dict[str, Any]],
    n_ok: int,
    n_fail: int,
    total: int,
    elapsed_s: float,
) -> None:
    """Emit the end-of-run summary block to the logger."""
    ok = [r for r in results if r.get("geval_score") is not None]
    in_tok = sum(r["tokens_used"]["input"] for r in ok)
    out_tok = sum(r["tokens_used"]["output"] for r in ok)
    cost = sum(r["cost_usd"] for r in ok)
    avg = elapsed_s / max(1, len(results))
    logger.info("──────────────────── RUN SUMMARY ────────────────────")
    logger.info("total entries:     %5d", total)
    logger.info("successful:        %5d (%.2f%%)", n_ok, 100 * n_ok / max(1, total))
    logger.info("failed:            %5d (documented in results.json)", n_fail)
    logger.info(
        "total wall time:  %7.1fs  (%dm %02ds)",
        elapsed_s,
        int(elapsed_s // 60),
        int(elapsed_s % 60),
    )
    logger.info("mean per entry:    %5.2fs", avg)
    logger.info("total tokens:   %10d  (input %d, output %d)", in_tok + out_tok, in_tok, out_tok)
    logger.info("total cost:        $%.4f", cost)


# ─── Stats markdown ──────────────────────────────────────────────────────
def _model_family(model_name: str) -> str:
    """Collapse a dataset model name onto a coarse family label."""
    if model_name in ("ground-truth", "negative-sample"):
        return model_name
    for family in ("GPT2", "S2S", "HRED", "VHRED"):
        if model_name.startswith(family):
            return family
    return "other"


def _basic_stats(values: Iterable[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(vals),
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def compute_summary(
    results: list[dict[str, Any]],
    dataset: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute descriptive stats from a results list — pure, no I/O.

    Separated from rendering so the numbers can be unit-tested directly
    without parsing markdown.
    """
    ok = [r for r in results if r.get("geval_score") is not None]
    fail = [r for r in results if r.get("geval_score") is None]

    overall = _basic_stats(r["geval_score"] for r in ok)

    if len(ok) >= MIN_FOR_SPEARMAN:
        rho_val, p_val = spearmanr([r["human_score"] for r in ok], [r["geval_score"] for r in ok])
        rho, p = float(rho_val), float(p_val)
    else:
        rho, p = float("nan"), float("nan")

    by_family: dict[str, dict[str, float]] = {}
    if dataset is not None:
        id_to_model = {e["metadata"]["conversation_id"]: e["metadata"]["model"] for e in dataset}
        family_scores: dict[str, list[float]] = {}
        for r in ok:
            m = id_to_model.get(r["conversation_id"], "")
            family = _model_family(m) if m else "unknown"
            family_scores.setdefault(family, []).append(r["geval_score"])
        for family, scores in family_scores.items():
            by_family[family] = _basic_stats(scores)

    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "n_fail": len(fail),
        "input_tokens": sum(r["tokens_used"]["input"] for r in ok),
        "output_tokens": sum(r["tokens_used"]["output"] for r in ok),
        "total_cost_usd": sum(r["cost_usd"] for r in ok),
        "overall": overall,
        "spearman_rho": rho,
        "spearman_p": p,
        "n_paired": len(ok),
        "by_family": by_family,
        "failed_entries": [
            {
                "conversation_id": r["conversation_id"],
                "attempts": r.get("attempts", "n/a"),
                "error": r.get("error", ""),
            }
            for r in fail
        ],
    }


def render_summary_markdown(
    summary: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    prompt_path: Path = PROMPT_PATH,
) -> str:
    """Render a precomputed summary dict as a markdown string — pure, no I/O."""
    rel_prompt = prompt_path.relative_to(PROJECT_ROOT) if prompt_path.is_absolute() else prompt_path
    lines: list[str] = [
        "# G-Eval Run — Summary Statistics\n",
        f"- **Model**: `{model}`",
        f"- **Prompt**: `{rel_prompt}`",
        f"- **Generated**: {datetime.now(UTC).isoformat()}\n",
        *_render_completion_table(summary),
        *_render_distribution_table(summary["overall"]),
        *_render_spearman_table(summary),
    ]
    if summary["by_family"]:
        lines.extend(_render_family_table(summary["by_family"]))
    if summary["failed_entries"]:
        lines.extend(_render_failed_table(summary["failed_entries"]))
    return "\n".join(lines)


def _render_completion_table(s: dict[str, Any]) -> list[str]:
    pct_ok = 100 * s["n_ok"] / max(1, s["n_total"])
    return [
        "## Run completion\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Total entries | {s['n_total']} |",
        f"| Successful | {s['n_ok']} ({pct_ok:.2f}%) |",
        f"| Failed | {s['n_fail']} |",
        f"| Input tokens | {s['input_tokens']:,} |",
        f"| Output tokens | {s['output_tokens']:,} |",
        f"| Total cost | ${s['total_cost_usd']:.4f} |\n",
    ]


def _render_distribution_table(overall: dict[str, float]) -> list[str]:
    lines = [
        "## G-Eval score distribution (1-5)\n",
        "| Stat | Value |",
        "|---|---|",
    ]
    for k in ("n", "mean", "median", "std", "min", "max"):
        lines.append(f"| {k} | {overall[k]} |")
    lines.append("")
    return lines


def _render_spearman_table(s: dict[str, Any]) -> list[str]:
    return [
        "## Spearman correlation vs. human_score\n",
        "| Metric | Value |",
        "|---|---|",
        f"| rho | {s['spearman_rho']:.4f} |",
        f"| p-value | {s['spearman_p']:.6g} |",
        f"| n (paired) | {s['n_paired']} |\n",
    ]


def _render_family_table(by_family: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "## Breakdown by model family\n",
        "| Family | n | mean | median | std | min | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for family in sorted(by_family):
        st = by_family[family]
        lines.append(
            f"| {family} | {st['n']} | {st['mean']} | {st['median']} | "
            f"{st['std']} | {st['min']} | {st['max']} |"
        )
    lines.append("")
    return lines


def _render_failed_table(failed: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Failed entries\n",
        "| conversation_id | attempts | error |",
        "|---|---|---|",
    ]
    for f in failed:
        err = str(f.get("error") or "").replace("|", "\\|")
        lines.append(f"| {f['conversation_id']} | {f.get('attempts', 'n/a')} | {err} |")
    lines.append("")
    return lines


def generate_summary_stats(
    results: list[dict[str, Any]],
    out_path: Path,
    *,
    dataset: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    prompt_path: Path = PROMPT_PATH,
) -> None:
    """Compute summary stats, render markdown, and write to ``out_path``.

    Thin orchestrator. Splitting compute_summary / render_summary_markdown
    keeps the numerical logic testable without parsing markdown.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(results, dataset)
    md = render_summary_markdown(summary, model=model, prompt_path=prompt_path)
    out_path.write_text(md, encoding="utf-8")


# ─── CLI ─────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run G-Eval over the full DailyDialog-Zhao dataset.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N entries.")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and start fresh.",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Evaluator model (default: {DEFAULT_MODEL})."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results, logs, and checkpoint.",
    )
    return p.parse_args(argv)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    log_path = output_dir / "logs" / "geval_execution.log"

    setup_logging(log_path)
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set — refusing to start (no API tokens spent)")
        sys.exit(2)

    logger.info("Loading dataset from %s", DATA_PATH)
    dataset = load_dataset(DATA_PATH)
    logger.info("Loaded %d entries", len(dataset))

    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    logger.info("Loaded prompt from %s (%d chars)", PROMPT_PATH, len(prompt_text))
    logger.info(
        "Evaluator model: %s | limit=%s | resume=%s", args.model, args.limit, not args.no_resume
    )

    # evaluate_dataset writes results.json + summary.md + checkpoint every
    # CHECKPOINT_EVERY entries AND in its finally-block, so the AC artifacts
    # exist on disk regardless of how the run ends (success / Ctrl+C / fatal).
    evaluate_dataset(
        dataset=dataset,
        prompt_text=prompt_text,
        model=args.model,
        output_dir=output_dir,
        limit=args.limit,
        resume=not args.no_resume,
    )
    logger.info("Run complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user — partial results saved to outputs/geval_results.json. "
            "Re-run the same command to resume from checkpoint."
        )
        sys.exit(130)
    except FATAL_EXCEPTIONS as exc:
        logger.exception(
            "Fatal API error (%s). "
            "This means bad credentials, wrong model name, or no access — fix and re-run. "
            "Partial results saved to outputs/geval_results.json.",
            type(exc).__name__,
        )
        sys.exit(3)
    except Exception:
        logger.exception("Unexpected error — see traceback above")
        sys.exit(1)
