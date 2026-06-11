"""Run the agentic voting system on the full DailyDialog-Zhao dataset (HU-09).

Orchestrates 900 conversation-response pairs through the three-judge panel
(judge_openai, judge_google, judge_anthropic) defined in
``configs/agents/agent_*.yaml`` and aggregates each triple via
:func:`src.voting.aggregator.aggregate`. The per-call workhorse is
:func:`scripts.run_judge.call_agent`; this module only adds checkpointing,
retries on transient SDK errors, dual logging, and summary stats.

The full run is on the order of a few USD of API spend across the three
providers and roughly an hour or two of sequential wall time; the exact
cost, token counts and wall time of the latest run live in
``CHANGELOG.md`` and ``outputs/voting_summary_stats.md`` so this docstring
does not drift as prices, providers or dataset size change. The script is
**resumable** by ``conversation_id``: re-running after a crash starts from
the first pending pair, not from zero.

Usage:
    uv run python scripts/run_voting_system.py                 # full 900-pair run
    uv run python scripts/run_voting_system.py --limit 5       # smoke test

Outputs (under ``outputs/``):
    - voting_results.json                      per-pair aggregated scores
    - voting_summary_stats.md                  descriptive stats + Spearman vs human
    - agent_scores/full_agent_{openai,google,anthropic}.json   per-judge raw rows
    - logs/voting_execution.log                per-entry + final summary log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Any, Final

import anthropic
import openai
import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv
from google.genai import errors as google_errors
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
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_judge import call_agent  # noqa: E402
from src.voting.aggregator import aggregate  # noqa: E402

DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "deepeval_test_cases.json"
AGENTS_DIR: Final[Path] = PROJECT_ROOT / "configs" / "agents"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
GEVAL_RESULTS_PATH: Final[Path] = DEFAULT_OUTPUT_DIR / "geval_results.json"

# ─── Agent ordering and file map ─────────────────────────────────────────
# Fixed ordering keeps log columns and JSON keys stable across runs and
# matches the order used in notebooks/03_voting_pilot.ipynb.
AGENT_FILES: Final[tuple[tuple[str, str], ...]] = (
    ("judge_openai", "agent_openai.yaml"),
    ("judge_google", "agent_google.yaml"),
    ("judge_anthropic", "agent_anthropic.yaml"),
)

# ─── Run constants ───────────────────────────────────────────────────────
CHECKPOINT_EVERY: Final[int] = 10
DEFAULT_SLEEP_S: Final[float] = 0.5

# Retry policy for transient SDK exceptions raised by call_agent.
RETRY_MAX_ATTEMPTS: Final[int] = 5
RETRY_WAIT_MIN_S: Final[int] = 2
RETRY_WAIT_MAX_S: Final[int] = 60

# Minimum sample size for a meaningful Spearman correlation.
MIN_FOR_SPEARMAN: Final[int] = 2

# Transient SDK errors worth retrying. Auth and validation errors fail fast.
RETRYABLE_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    google_errors.ServerError,
    google_errors.APIError,
)

# Errors that mean the run cannot succeed at all. Re-raising aborts the
# loop instead of burning ~90 minutes recording 900 identical failures.
FATAL_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
)

logger = logging.getLogger("run_voting")


# ─── Logging setup ───────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> None:
    """Configure ``logger`` to emit INFO to both stdout and ``log_path``."""
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


# ─── Loaders ─────────────────────────────────────────────────────────────
def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load all entries from the processed dataset."""
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def load_agent_configs(agents_dir: Path) -> list[dict[str, Any]]:
    """Load the three judge YAMLs in the fixed AGENT_FILES order.

    Returned dicts are ready to pass to :func:`call_agent`: paths in
    ``prompt_file`` are resolved against the project root so the runner
    works regardless of the caller's CWD.
    """
    configs: list[dict[str, Any]] = []
    for expected_name, filename in AGENT_FILES:
        with open(agents_dir / filename, encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        if cfg["name"] != expected_name:
            raise ValueError(
                f"YAML {filename} declares name={cfg['name']!r}, expected {expected_name!r}"
            )
        # call_agent reads the prompt file with Path(...).read_text; the YAML
        # uses a repo-relative path. Resolve once so call_agent's cache key is
        # stable regardless of the runner's working directory.
        cfg["prompt_file"] = str((PROJECT_ROOT / cfg["prompt_file"]).resolve())
        configs.append(cfg)
    return configs


def load_existing_results(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Return (results_so_far, done_ids). Empty if no file exists.

    A corrupt JSON (truncated, partial write) is rotated to ``.bak`` and
    treated as no checkpoint, so the run can restart instead of crashing.
    """
    if not path.exists():
        return [], set()
    try:
        with open(path, encoding="utf-8") as f:
            results: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            path.replace(backup)
            logger.warning(
                "Corrupt results at %s (%s); moved to %s and starting fresh",
                path,
                exc,
                backup,
            )
        except OSError:
            logger.warning("Corrupt results at %s (%s); starting fresh", path, exc)
        return [], set()
    ids = {r["conversation_id"] for r in results}
    return results, ids


def load_existing_agent_rows(path: Path) -> list[dict[str, Any]]:
    """Return per-agent JSON rows already on disk, or an empty list."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return data


# ─── Pure helpers ────────────────────────────────────────────────────────
def format_conversation(turns: list[dict[str, str]]) -> str:
    """Format ``turns[:-1]`` as ``[Turn N] User|Assistant: ...`` joined by newlines.

    Replicates the formatting used in ``notebooks/03_voting_pilot.ipynb``:
    the final assistant turn (which equals ``actual_output``) is excluded;
    every preceding turn becomes one line with its 1-based index and
    capitalised role.
    """
    context = turns[:-1]
    return "\n".join(
        f"[Turn {i + 1}] {t['role'].capitalize()}: {t['content']}" for i, t in enumerate(context)
    )


def _write_atomic(payload: Any, path: Path) -> None:
    """Atomic JSON dump: write to ``.tmp`` and rename. Mirrors run_geval.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ─── Per-judge call with tenacity retry on transient SDK errors ──────────
def call_agent_with_retry(
    agent_config: dict[str, Any],
    conversation_input: str,
    actual_output: str,
) -> dict[str, Any]:
    """Call :func:`call_agent` with exponential backoff on transient errors.

    ``call_agent`` already handles parse-failure retries inside the prompt
    flow (annotated as ``retry_used`` in its return dict); this wrapper
    handles **SDK exceptions** (rate limits, timeouts, transient 5xx) and
    annotates the result with ``sdk_attempts`` — the number of times the
    SDK call was actually issued (1 on first-try success, up to
    ``RETRY_MAX_ATTEMPTS`` when tenacity retried). If all retries are
    exhausted the exception is caught and a failure-shaped dict is returned
    so the par-loop can keep going.
    """
    sdk_attempts = 0
    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN_S, max=RETRY_WAIT_MAX_S),
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                sdk_attempts = attempt.retry_state.attempt_number
                result = call_agent(agent_config, conversation_input, actual_output)
                result["sdk_attempts"] = sdk_attempts
                return result
    except FATAL_EXCEPTIONS:
        raise
    except (RetryError, *RETRYABLE_EXCEPTIONS) as exc:
        return {
            "agent": agent_config["name"],
            "model": agent_config["model"],
            "score": None,
            "reasoning": "",
            "tokens_used": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "timestamp": datetime.now(UTC).isoformat(),
            "sdk_attempts": sdk_attempts or RETRY_MAX_ATTEMPTS,
            "error": f"api_failure: {type(exc).__name__}: {exc}",
        }
    # The Retrying loop always either returns from inside the with-block or
    # raises; this line is unreachable but keeps mypy happy.
    raise RuntimeError("unreachable")


# ─── Per-pair processing ─────────────────────────────────────────────────
def process_pair(
    entry: dict[str, Any],
    agent_configs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Score one dataset entry through all three judges and aggregate.

    Returns a ``(aggregated_row, per_agent_rows)`` pair so the caller can
    persist both the panel-level result and the raw judge rows.
    """
    conv_id = entry["metadata"]["conversation_id"]
    human_score = entry["metadata"]["human_score"]
    model_name = entry["metadata"]["model"]
    stratum = entry.get("stratum")

    conv_input = format_conversation(entry["turns"])
    actual_output = entry["actual_output"]

    per_agent_rows: dict[str, dict[str, Any]] = {}
    scores: dict[str, float | None] = {}
    cost_by_agent: dict[str, float] = {}
    tokens_in = 0
    tokens_out = 0
    cost_total = 0.0

    for cfg in agent_configs:
        row = call_agent_with_retry(cfg, conv_input, actual_output)
        # Tag the per-agent row with conversation_id so the dumped JSONs are
        # self-joinable against voting_results.json without external keys.
        row_tagged = {"conversation_id": conv_id, **row}
        per_agent_rows[cfg["name"]] = row_tagged
        scores[cfg["name"]] = row["score"]
        cost_by_agent[cfg["name"]] = row["cost_usd"]
        tokens_in += row["tokens_in"]
        tokens_out += row["tokens_out"]
        cost_total += row["cost_usd"]

    aggregate_failed = False
    final_vote_score: float | None = None
    individual_scores: dict[str, float | None] = scores
    agreement_level = "n/a"
    metadata: dict[str, Any] = {}

    try:
        agg = aggregate(scores)
        final_vote_score = agg["final_score"]
        individual_scores = agg["individual_scores"]
        agreement_level = agg["agreement_level"]
        metadata = agg["metadata"]
    except ValueError as exc:
        # All three judges returned None; aggregate refuses to combine.
        aggregate_failed = True
        metadata = {"error": str(exc)}

    aggregated_row = {
        "conversation_id": conv_id,
        "human_score": human_score,
        "model": model_name,
        "stratum": stratum,
        "final_vote_score": final_vote_score,
        "individual_scores": individual_scores,
        "agreement_level": agreement_level,
        "metadata": metadata,
        "timestamp": datetime.now(UTC).isoformat(),
        "tokens_used": tokens_in + tokens_out,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost_total, 6),
        "cost_by_agent": {k: round(v, 6) for k, v in cost_by_agent.items()},
        "aggregate_failed": aggregate_failed,
    }
    return aggregated_row, per_agent_rows


# ─── Main loop ───────────────────────────────────────────────────────────
def evaluate_dataset(
    dataset: list[dict[str, Any]],
    agent_configs: list[dict[str, Any]],
    output_dir: Path,
    sleep_s: float,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Iterate over pending pairs, call the panel, aggregate, persist.

    Writes ``voting_results.json``, the three ``full_agent_*.json`` files,
    and the summary markdown every ``CHECKPOINT_EVERY`` pairs and again in
    a finally block, so the AC artifacts always exist on disk regardless
    of how the run ends.
    """
    results_path = output_dir / "voting_results.json"
    summary_path = output_dir / "voting_summary_stats.md"
    geval_results_path = output_dir / "geval_results.json"
    agent_scores_dir = output_dir / "agent_scores"
    per_agent_paths = {
        name: agent_scores_dir / f"full_agent_{name.removeprefix('judge_')}.json"
        for name, _ in AGENT_FILES
    }

    results, done_ids = load_existing_results(results_path)
    per_agent_rows = {
        name: load_existing_agent_rows(path) for name, path in per_agent_paths.items()
    }

    pending = [e for e in dataset if e["metadata"]["conversation_id"] not in done_ids]
    if limit is not None:
        pending = pending[: max(0, limit - len(results))]
    total = len(pending) + len(results)

    logger.info(
        "Checkpoint: %d pairs already processed, %d pending (of %d total in scope)",
        len(results),
        len(pending),
        total,
    )

    n_ok = sum(1 for r in results if not r.get("aggregate_failed", False))
    n_fail = len(results) - n_ok
    cost_acc = sum(r.get("cost_usd") or 0.0 for r in results)
    t0 = perf_counter()

    try:
        for idx, entry in enumerate(pending, start=len(results) + 1):
            conv_id = entry["metadata"]["conversation_id"]
            agg_row, agent_rows = process_pair(entry, agent_configs)
            results.append(agg_row)
            for name, row in agent_rows.items():
                per_agent_rows[name].append(row)

            if agg_row["aggregate_failed"]:
                n_fail += 1
                logger.error(
                    "[%03d/%03d] %s  AGGREGATE_FAILED (all judges returned None)",
                    idx,
                    total,
                    conv_id,
                )
            else:
                n_ok += 1
            cost_acc += agg_row["cost_usd"]
            sc = agg_row["individual_scores"]
            logger.info(
                "[%03d/%03d] %s | openai=%s google=%s anthropic=%s | vote=%s | $%.4f (acc $%.4f)",
                idx,
                total,
                conv_id,
                sc.get("judge_openai"),
                sc.get("judge_google"),
                sc.get("judge_anthropic"),
                agg_row["final_vote_score"] if agg_row["final_vote_score"] is not None else "None",
                agg_row["cost_usd"],
                cost_acc,
            )

            if idx % CHECKPOINT_EVERY == 0:
                _persist_partial(
                    results,
                    per_agent_rows,
                    results_path,
                    per_agent_paths,
                    summary_path,
                    dataset,
                    geval_results_path,
                )
                logger.info("Checkpoint + outputs saved at %d/%d", idx, total)

            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        _persist_partial(
            results,
            per_agent_rows,
            results_path,
            per_agent_paths,
            summary_path,
            dataset,
            geval_results_path,
        )

    elapsed = perf_counter() - t0
    _log_run_summary(results, n_ok, n_fail, total, elapsed)
    return results


def _persist_partial(  # noqa: PLR0913
    results: list[dict[str, Any]],
    per_agent_rows: dict[str, list[dict[str, Any]]],
    results_path: Path,
    per_agent_paths: dict[str, Path],
    summary_path: Path,
    dataset: list[dict[str, Any]],
    geval_results_path: Path,
) -> None:
    """Atomically write voting_results.json + full_agent_*.json + summary.md."""
    try:
        _write_atomic(results, results_path)
        for name, rows in per_agent_rows.items():
            _write_atomic(rows, per_agent_paths[name])
        generate_summary_stats(
            results,
            summary_path,
            dataset=dataset,
            geval_results_path=geval_results_path,
        )
    except OSError:
        logger.exception("Failed to persist partial results")


def _log_run_summary(
    results: list[dict[str, Any]],
    n_ok: int,
    n_fail: int,
    total: int,
    elapsed_s: float,
) -> None:
    """Emit the end-of-run summary block."""
    cost = sum(r.get("cost_usd") or 0.0 for r in results)
    cost_per_agent = {name: 0.0 for name, _ in AGENT_FILES}
    for r in results:
        for name, v in (r.get("cost_by_agent") or {}).items():
            cost_per_agent[name] = cost_per_agent.get(name, 0.0) + v
    in_tok = sum(r.get("tokens_in") or 0 for r in results)
    out_tok = sum(r.get("tokens_out") or 0 for r in results)
    avg = elapsed_s / max(1, len(results))
    logger.info("──────────────────── RUN SUMMARY ────────────────────")
    logger.info("total pairs:       %5d", total)
    logger.info("fully scored:      %5d (%.2f%%)", n_ok, 100 * n_ok / max(1, total))
    logger.info("aggregate failed:  %5d", n_fail)
    logger.info(
        "total wall time:  %7.1fs  (%dm %02ds)",
        elapsed_s,
        int(elapsed_s // 60),
        int(elapsed_s % 60),
    )
    logger.info("mean per pair:     %5.2fs", avg)
    logger.info("total tokens:   %10d  (input %d, output %d)", in_tok + out_tok, in_tok, out_tok)
    logger.info("total cost:        $%.4f", cost)
    for name, c in cost_per_agent.items():
        logger.info("  %-18s $%.4f", name + ":", c)


# ─── Stats ───────────────────────────────────────────────────────────────
def _model_family(model_name: str) -> str:
    if model_name in ("ground-truth", "negative-sample"):
        return model_name
    for family in ("GPT2", "S2S", "HRED", "VHRED"):
        if model_name.startswith(family):
            return family
    return "other"


def _basic_stats(values: Iterable[float]) -> dict[str, int | float]:
    """Return n + mean/median/std/min/max for ``values``.

    ``n`` is an ``int`` (count); the other fields are floats. The union
    return type reflects that mix so downstream consumers and mypy don't
    have to coerce when they want an integer count.
    """
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


def _spearman_or_nan(xs: list[float], ys: list[float]) -> tuple[float, float, int]:
    if len(xs) < MIN_FOR_SPEARMAN or len(ys) < MIN_FOR_SPEARMAN:
        return float("nan"), float("nan"), len(xs)
    rho, p = spearmanr(xs, ys)
    return float(rho), float(p), len(xs)


def _load_geval_rho(geval_results_path: Path) -> tuple[float, int] | None:
    """Return (rho, n) of G-Eval vs human from ``geval_results_path``, or None.

    The path is parameterised so a run with a custom ``--output-dir`` compares
    against the G-Eval baseline that lives in the same output tree, not the
    default one in the project root.
    """
    if not geval_results_path.exists():
        return None
    try:
        with open(geval_results_path, encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    pairs = [
        (r["human_score"], r["geval_score"])
        for r in data
        if r.get("geval_score") is not None and r.get("human_score") is not None
    ]
    if len(pairs) < MIN_FOR_SPEARMAN:
        return None
    rho, _, _ = _spearman_or_nan([p[0] for p in pairs], [p[1] for p in pairs])
    return rho, len(pairs)


def compute_summary(  # noqa: C901
    results: list[dict[str, Any]],
    dataset: list[dict[str, Any]] | None = None,
    geval_results_path: Path | None = None,
) -> dict[str, Any]:
    """Compute descriptive stats — pure, no I/O."""
    ok = [r for r in results if not r.get("aggregate_failed", False)]
    fail = [r for r in results if r.get("aggregate_failed", False)]

    vote_overall = _basic_stats(
        r["final_vote_score"] for r in ok if r.get("final_vote_score") is not None
    )

    paired = [
        (r["human_score"], r["final_vote_score"])
        for r in ok
        if r.get("final_vote_score") is not None and r.get("human_score") is not None
    ]
    rho_vote, p_vote, n_paired = _spearman_or_nan([p[0] for p in paired], [p[1] for p in paired])

    per_judge_rho: dict[str, dict[str, float]] = {}
    for name, _ in AGENT_FILES:
        xs: list[float] = []
        ys: list[float] = []
        for r in ok:
            s = (r.get("individual_scores") or {}).get(name)
            if s is not None and r.get("human_score") is not None:
                xs.append(r["human_score"])
                ys.append(s)
        rho_j, p_j, n_j = _spearman_or_nan(xs, ys)
        per_judge_rho[name] = {"rho": rho_j, "p": p_j, "n": n_j}

    by_family: dict[str, dict[str, int | float]] = {}
    if dataset is not None:
        id_to_model = {e["metadata"]["conversation_id"]: e["metadata"]["model"] for e in dataset}
        family_scores: dict[str, list[float]] = {}
        for r in ok:
            if r.get("final_vote_score") is None:
                continue
            m = id_to_model.get(r["conversation_id"], "")
            family = _model_family(m) if m else "unknown"
            family_scores.setdefault(family, []).append(r["final_vote_score"])
        for family, scores in family_scores.items():
            by_family[family] = _basic_stats(scores)

    agreement_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "n/a": 0}
    for r in ok:
        lvl = r.get("agreement_level", "n/a")
        agreement_counts[lvl] = agreement_counts.get(lvl, 0) + 1

    cost_by_agent: dict[str, float] = {name: 0.0 for name, _ in AGENT_FILES}
    for r in results:
        for name, v in (r.get("cost_by_agent") or {}).items():
            cost_by_agent[name] = cost_by_agent.get(name, 0.0) + v

    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "n_fail": len(fail),
        "input_tokens": sum(r.get("tokens_in") or 0 for r in results),
        "output_tokens": sum(r.get("tokens_out") or 0 for r in results),
        "total_cost_usd": sum(r.get("cost_usd") or 0.0 for r in results),
        "cost_by_agent": cost_by_agent,
        "overall": vote_overall,
        "spearman_rho": rho_vote,
        "spearman_p": p_vote,
        "n_paired": n_paired,
        "per_judge": per_judge_rho,
        "by_family": by_family,
        "agreement_counts": agreement_counts,
        "geval_reference": (
            _load_geval_rho(geval_results_path) if geval_results_path is not None else None
        ),
        "failed_entries": [
            {
                "conversation_id": r["conversation_id"],
                "error": (r.get("metadata") or {}).get("error", ""),
            }
            for r in fail
        ],
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    """Render the summary dict as a markdown string."""
    lines: list[str] = [
        "# Voting System Run — Summary Statistics\n",
        "- **Panel**: judge_openai (`gpt-4o`), judge_google (`gemini-2.5-flash`),"
        " judge_anthropic (`claude-haiku-4-5`)",
        "- **Prompt**: `configs/prompts/geval_relevance_prompt.txt` (V3, shared)",
        "- **Aggregator**: `arithmetic_mean` (see `docs/voting_scheme_analysis.md`)",
        f"- **Generated**: {datetime.now(UTC).isoformat()}\n",
        *_render_completion_table(summary),
        *_render_distribution_table(summary["overall"]),
        *_render_spearman_table(summary),
    ]
    if summary["by_family"]:
        lines.extend(_render_family_table(summary["by_family"]))
    lines.extend(_render_agreement_table(summary["agreement_counts"]))
    if summary["failed_entries"]:
        lines.extend(_render_failed_table(summary["failed_entries"]))
    return "\n".join(lines)


def _render_completion_table(s: dict[str, Any]) -> list[str]:
    pct_ok = 100 * s["n_ok"] / max(1, s["n_total"])
    lines = [
        "## Run completion\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Total pairs | {s['n_total']} |",
        f"| Fully scored | {s['n_ok']} ({pct_ok:.2f}%) |",
        f"| Aggregate failed | {s['n_fail']} |",
        f"| Input tokens | {s['input_tokens']:,} |",
        f"| Output tokens | {s['output_tokens']:,} |",
        f"| Total cost | ${s['total_cost_usd']:.4f} |",
    ]
    for name, c in s["cost_by_agent"].items():
        lines.append(f"|  - {name} | ${c:.4f} |")
    lines.append("")
    return lines


def _render_distribution_table(overall: dict[str, int | float]) -> list[str]:
    lines = [
        "## final_vote_score distribution (1-5)\n",
        "| Stat | Value |",
        "|---|---|",
    ]
    for k in ("n", "mean", "median", "std", "min", "max"):
        lines.append(f"| {k} | {overall[k]} |")
    lines.append("")
    return lines


def _render_spearman_table(s: dict[str, Any]) -> list[str]:
    lines = [
        "## Spearman correlation vs. human_score\n",
        "| Source | rho | p-value | n |",
        "|---|---|---|---|",
        f"| voting (panel mean) | {s['spearman_rho']:.4f} | {s['spearman_p']:.6g}"
        f" | {s['n_paired']} |",
    ]
    geval = s.get("geval_reference")
    if geval is not None:
        rho_g, n_g = geval
        lines.append(f"| G-Eval baseline (`gpt-4o`) | {rho_g:.4f} | — | {n_g} |")
    for name, st in s["per_judge"].items():
        lines.append(f"| {name} (individual) | {st['rho']:.4f} | {st['p']:.6g} | {st['n']} |")
    lines.append("")
    return lines


def _render_family_table(by_family: dict[str, dict[str, int | float]]) -> list[str]:
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


def _render_agreement_table(counts: dict[str, int]) -> list[str]:
    total = sum(counts.values())
    lines = [
        "## Panel agreement levels\n",
        "| Level | Count | Percentage |",
        "|---|---|---|",
    ]
    for level in ("high", "medium", "low", "n/a"):
        n = counts.get(level, 0)
        pct = 100 * n / max(1, total)
        lines.append(f"| {level} | {n} | {pct:.2f}% |")
    lines.append("")
    return lines


def _render_failed_table(failed: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Failed pairs (aggregate refused)\n",
        "| conversation_id | error |",
        "|---|---|",
    ]
    for f in failed:
        err = str(f.get("error") or "").replace("|", "\\|")
        lines.append(f"| {f['conversation_id']} | {err} |")
    lines.append("")
    return lines


def generate_summary_stats(
    results: list[dict[str, Any]],
    out_path: Path,
    *,
    dataset: list[dict[str, Any]] | None = None,
    geval_results_path: Path | None = None,
) -> None:
    """Compute and render summary stats to ``out_path``.

    ``geval_results_path`` should point at the G-Eval baseline JSON inside
    the same output tree as ``out_path``; callers pass it explicitly so a
    custom ``--output-dir`` never silently compares against a stale baseline
    from a different run.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(results, dataset, geval_results_path)
    out_path.write_text(render_summary_markdown(summary), encoding="utf-8")


# ─── CLI ─────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the agentic voting system over the full DailyDialog-Zhao dataset.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process only the first N pending pairs."
    )
    p.add_argument(
        "--dataset", type=Path, default=DATA_PATH, help="Path to deepeval_test_cases.json."
    )
    p.add_argument(
        "--agents-dir", type=Path, default=AGENTS_DIR, help="Directory with agent_*.yaml files."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results, logs, and per-agent JSONs.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_S,
        help=f"Seconds to sleep between pairs (default: {DEFAULT_SLEEP_S}).",
    )
    return p.parse_args(argv)


def _require_keys(agent_configs: list[dict[str, Any]]) -> None:
    """Abort before any API call if a key is missing."""
    missing = [c["api_key_env"] for c in agent_configs if not os.getenv(c["api_key_env"])]
    if missing:
        logger.error("Missing API keys in env: %s — refusing to start", ", ".join(missing))
        sys.exit(2)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    log_path = output_dir / "logs" / "voting_execution.log"

    setup_logging(log_path)
    load_dotenv(PROJECT_ROOT / ".env")

    logger.info("Loading dataset from %s", args.dataset)
    dataset = load_dataset(args.dataset)
    logger.info("Loaded %d entries", len(dataset))

    agent_configs = load_agent_configs(args.agents_dir)
    logger.info(
        "Loaded %d agents: %s",
        len(agent_configs),
        ", ".join(f"{c['name']} ({c['model']})" for c in agent_configs),
    )
    _require_keys(agent_configs)

    logger.info("Starting voting system run | limit=%s | sleep=%.2fs", args.limit, args.sleep)
    evaluate_dataset(
        dataset=dataset,
        agent_configs=agent_configs,
        output_dir=output_dir,
        sleep_s=args.sleep,
        limit=args.limit,
    )
    logger.info("Run complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user — partial results saved to outputs/voting_results.json. "
            "Re-run the same command to resume from checkpoint."
        )
        sys.exit(130)
    except FATAL_EXCEPTIONS as exc:
        logger.exception(
            "Fatal API error (%s). Bad credentials, wrong model, or no access — fix and re-run."
            " Partial results saved to outputs/voting_results.json.",
            type(exc).__name__,
        )
        sys.exit(3)
    except Exception:
        logger.exception("Unexpected error — see traceback above")
        sys.exit(1)
