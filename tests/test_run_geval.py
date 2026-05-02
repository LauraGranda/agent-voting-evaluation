"""Unit tests for scripts/run_geval.py — no live API calls."""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import openai
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_geval import (
    CHECKPOINT_EVERY,
    SCALE_MIN,
    SCALE_RANGE,
    _basic_stats,
    _evaluate_one,
    _model_family,
    _rescale_0_1_to_1_5,
    build_geval_metric,
    build_test_case,
    compute_summary,
    estimate_tokens_and_cost,
    evaluate_dataset,
    generate_summary_stats,
    load_checkpoint,
    parse_args,
    render_summary_markdown,
    save_checkpoint,
    write_results,
)

# Constants for test assertions (avoids ruff PLR2004 magic-value warnings)
EXPECTED_HUMAN_SCORE = 4.5
EXPECTED_RESCALE_HALF = 3.0
EXPECTED_RESCALE_FACTOR_AT_08 = round(SCALE_MIN + SCALE_RANGE * 0.8, 4)
EXPECTED_BASIC_N = 5
EXPECTED_BASIC_MEAN = 3.0
EXPECTED_BASIC_MIN = 1.0
EXPECTED_BASIC_MAX = 5.0
EXPECTED_CLI_LIMIT = 5


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def entry() -> dict[str, Any]:
    """Mirror of an entry in data/processed/deepeval_test_cases.json."""
    return {
        "input": "user line",
        "actual_output": "model response",
        "turns": [
            {"role": "user", "content": "user line"},
            {"role": "assistant", "content": "model response"},
        ],
        "metadata": {
            "human_score": 4.5,
            "raw_relevance_scores": [5, 5, 3, 5],
            "human_appropriateness_score": 4.25,
            "raw_appropriateness_scores": [4, 5, 4, 4],
            "conversation_id": "conv_0_ground-truth",
            "model": "ground-truth",
            "response_speaker": "A",
        },
    }


@pytest.fixture
def fake_results() -> list[dict[str, Any]]:
    """Five-entry results list with one failure for stats tests."""
    return [
        _ok("conv_0_ground-truth", 4.5, 4.6, 0.9, 1300, 90, 0.0042),
        _ok("conv_1_negative-sample", 1.5, 1.7, 0.175, 1200, 80, 0.0038),
        _ok("conv_2_GPT2_small", 3.0, 3.4, 0.6, 1250, 85, 0.004),
        _ok("conv_3_HRED_attn", 2.5, 2.8, 0.45, 1230, 82, 0.0039),
        _fail("conv_4_S2S", 4.0, "RateLimitError: 429"),
    ]


def _ok(  # noqa: PLR0913
    conv_id: str,
    human: float,
    geval: float,
    raw: float,
    in_tok: int,
    out_tok: int,
    cost: float,
) -> dict[str, Any]:
    return {
        "conversation_id": conv_id,
        "geval_score": geval,
        "human_score": human,
        "model_used": "gpt-4o",
        "timestamp": "2026-05-01T18:00:00+00:00",
        "tokens_used": {"input": in_tok, "output": out_tok, "total": in_tok + out_tok},
        "cost_usd": cost,
        "geval_score_raw": raw,
        "delta": round(geval - human, 4),
        "reason": "stub",
        "attempts": 1,
    }


def _fail(conv_id: str, human: float, err: str) -> dict[str, Any]:
    return {
        "conversation_id": conv_id,
        "geval_score": None,
        "human_score": human,
        "model_used": "gpt-4o",
        "timestamp": "2026-05-01T18:00:00+00:00",
        "tokens_used": None,
        "cost_usd": None,
        "geval_score_raw": None,
        "delta": None,
        "reason": "",
        "attempts": 5,
        "error": err,
    }


# ─── _rescale_0_1_to_1_5 ──────────────────────────────────────────────────


def test_rescale_zero() -> None:
    assert _rescale_0_1_to_1_5(0.0) == SCALE_MIN


def test_rescale_one() -> None:
    assert _rescale_0_1_to_1_5(1.0) == SCALE_MIN + SCALE_RANGE


def test_rescale_half() -> None:
    assert _rescale_0_1_to_1_5(0.5) == EXPECTED_RESCALE_HALF


# ─── build_test_case ─────────────────────────────────────────────────────


def test_build_test_case_drops_trailing_assistant(entry: dict[str, Any]) -> None:
    """Trailing assistant turn must not leak into the packed input."""
    _tc, packed = build_test_case(entry)
    assert "model response" not in packed
    assert "user line" in packed


def test_build_test_case_format(entry: dict[str, Any]) -> None:
    """Packed input uses ``[Turn N] Role: content`` lines."""
    _tc, packed = build_test_case(entry)
    assert packed == "[Turn 1] User: user line"


def test_build_test_case_actual_output(entry: dict[str, Any]) -> None:
    """actual_output passes through unchanged."""
    tc, _packed = build_test_case(entry)
    assert tc.actual_output == "model response"


# ─── build_geval_metric ──────────────────────────────────────────────────


def test_build_geval_metric_uses_given_model() -> None:
    metric = build_geval_metric("criteria text", "gpt-4o-mini")
    # GEval stores the model on either ``evaluation_model`` or ``model`` depending on version.
    model_attr = getattr(metric, "evaluation_model", None) or getattr(metric, "model", None)
    assert model_attr == "gpt-4o-mini" or "gpt-4o-mini" in str(model_attr)


# ─── estimate_tokens_and_cost ────────────────────────────────────────────


def test_estimate_tokens_returns_positive() -> None:
    in_tok, out_tok, cost = estimate_tokens_and_cost(
        prompt_text="some prompt",
        packed_input="[Turn 1] User: hi",
        actual_output="hello",
        reason="The response is on topic.",
    )
    assert in_tok > 0
    assert out_tok > 0
    assert cost > 0


def test_estimate_tokens_cost_monotonic() -> None:
    """A longer reason produces strictly more output tokens (and ≥ cost)."""
    short = estimate_tokens_and_cost("p", "i", "o", "ok")
    long = estimate_tokens_and_cost("p", "i", "o", "ok " * 100)
    assert long[1] > short[1]
    assert long[2] >= short[2]


# ─── _model_family / _basic_stats ────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ground-truth", "ground-truth"),
        ("negative-sample", "negative-sample"),
        ("GPT2_small top_temp1.0_k0_p0.9", "GPT2"),
        ("HRED_attn greedy_temp1.0_k0_p0.0", "HRED"),
        ("S2S sample_temp1.0_k0_p0.0", "S2S"),
        ("VHRED_attn greedy_temp1.0_k0_p0.0", "VHRED"),
        ("foo_bar_unknown", "other"),
    ],
)
def test_model_family(name: str, expected: str) -> None:
    assert _model_family(name) == expected


def test_basic_stats_empty() -> None:
    s = _basic_stats([])
    assert s["n"] == 0
    assert s["mean"] == 0.0


def test_basic_stats_nonempty() -> None:
    s = _basic_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == EXPECTED_BASIC_N
    assert s["mean"] == EXPECTED_BASIC_MEAN
    assert s["min"] == EXPECTED_BASIC_MIN
    assert s["max"] == EXPECTED_BASIC_MAX


# ─── Checkpoint round-trip ───────────────────────────────────────────────


def test_checkpoint_round_trip(tmp_path: Path, fake_results: list[dict[str, Any]]) -> None:
    cp = tmp_path / ".geval_checkpoint.json"
    save_checkpoint(fake_results, cp)
    assert cp.exists()
    loaded, ids = load_checkpoint(cp)
    assert loaded == fake_results
    assert ids == {r["conversation_id"] for r in fake_results}


def test_load_checkpoint_missing(tmp_path: Path) -> None:
    """Missing checkpoint returns empty results, no crash."""
    cp = tmp_path / "nope.json"
    loaded, ids = load_checkpoint(cp)
    assert loaded == []
    assert ids == set()


def test_load_checkpoint_corrupt_rotates_to_bak(tmp_path: Path) -> None:
    """A malformed-JSON checkpoint must not crash the run; it's rotated to .bak.

    Regression guard: a kill -9 mid-write or a disk-full event during the
    atomic rename can leave a truncated checkpoint. The next run must
    degrade gracefully rather than blow up on startup.
    """
    cp = tmp_path / "checkpoint.json"
    cp.write_text("[", encoding="utf-8")  # malformed JSON
    loaded, ids = load_checkpoint(cp)
    assert loaded == []
    assert ids == set()
    # Original is gone (renamed); .bak preserves the corrupt content for inspection.
    assert not cp.exists()
    assert (tmp_path / "checkpoint.json.bak").exists()


def test_save_checkpoint_atomic_no_tmp_left(
    tmp_path: Path, fake_results: list[dict[str, Any]]
) -> None:
    """The .tmp file is renamed away on success."""
    cp = tmp_path / "cp.json"
    save_checkpoint(fake_results, cp)
    assert not (tmp_path / "cp.json.tmp").exists()


# ─── write_results / results JSON shape ──────────────────────────────────


def test_write_results_atomic(tmp_path: Path, fake_results: list[dict[str, Any]]) -> None:
    out = tmp_path / "geval_results.json"
    write_results(fake_results, out)
    assert json.loads(out.read_text()) == fake_results


def test_results_contain_required_ac_fields(fake_results: list[dict[str, Any]]) -> None:
    """All AC-required fields present on successful entries."""
    required = {
        "conversation_id",
        "geval_score",
        "human_score",
        "model_used",
        "timestamp",
        "tokens_used",
        "cost_usd",
    }
    ok = [r for r in fake_results if r.get("geval_score") is not None]
    for entry in ok:
        assert required.issubset(entry.keys())


# ─── compute_summary (pure, no I/O) ──────────────────────────────────────


def test_compute_summary_counts_ok_and_fail(fake_results: list[dict[str, Any]]) -> None:
    s = compute_summary(fake_results)
    assert s["n_total"] == len(fake_results)
    assert s["n_ok"] == len([r for r in fake_results if r["geval_score"] is not None])
    assert s["n_fail"] == len([r for r in fake_results if r["geval_score"] is None])


def test_compute_summary_aggregates_tokens_and_cost(
    fake_results: list[dict[str, Any]],
) -> None:
    s = compute_summary(fake_results)
    expected_in = sum(r["tokens_used"]["input"] for r in fake_results if r["tokens_used"])
    expected_out = sum(r["tokens_used"]["output"] for r in fake_results if r["tokens_used"])
    expected_cost = sum(r["cost_usd"] for r in fake_results if r["cost_usd"] is not None)
    assert s["input_tokens"] == expected_in
    assert s["output_tokens"] == expected_out
    assert s["total_cost_usd"] == pytest.approx(expected_cost)


def test_compute_summary_spearman_when_n_lt_min() -> None:
    """With fewer than MIN_FOR_SPEARMAN paired entries, rho/p are NaN."""
    one_result = [_ok("c0", 4.0, 4.5, 0.875, 1000, 80, 0.003)]
    s = compute_summary(one_result)
    assert s["spearman_rho"] != s["spearman_rho"]  # NaN check
    assert s["spearman_p"] != s["spearman_p"]


def test_compute_summary_groups_by_family(
    fake_results: list[dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    """Family bucketing collapses GPT2_*, S2S_*, etc. to coarse families."""
    fake_dataset = []
    for r in fake_results:
        e = json.loads(json.dumps(entry))
        e["metadata"]["conversation_id"] = r["conversation_id"]
        e["metadata"]["model"] = r["conversation_id"].split("_", 2)[-1]
        fake_dataset.append(e)
    s = compute_summary(fake_results, fake_dataset)
    # Failed entry is excluded; remaining 4 OK ones span ground-truth/negative-sample/GPT2/HRED
    assert "ground-truth" in s["by_family"]
    assert "GPT2" in s["by_family"]
    # Each family bucket carries a full _basic_stats dict
    assert {"n", "mean", "median", "std", "min", "max"} <= set(s["by_family"]["GPT2"].keys())


# ─── render_summary_markdown (pure, no I/O) ──────────────────────────────


def test_render_markdown_contains_all_sections(fake_results: list[dict[str, Any]]) -> None:
    """All five expected H2 sections render when there are failures and families."""
    summary = compute_summary(fake_results)
    md = render_summary_markdown(summary)
    assert "## Run completion" in md
    assert "## G-Eval score distribution (1-5)" in md
    assert "## Spearman correlation vs. human_score" in md
    assert "## Failed entries" in md  # one failure in fake_results


def test_render_markdown_skips_optional_sections_when_empty() -> None:
    """No families and no failures → those tables are omitted."""
    summary = compute_summary([_ok("c0", 4.0, 4.5, 0.875, 1000, 80, 0.003)])
    md = render_summary_markdown(summary)
    assert "## Failed entries" not in md
    assert "## Breakdown by model family" not in md


def test_render_markdown_escapes_pipe_in_error_message() -> None:
    """Pipe in error string must be escaped so the markdown table doesn't break."""
    summary = compute_summary([_fail("c0", 3.0, "boom | with pipe")])
    md = render_summary_markdown(summary)
    assert "boom \\| with pipe" in md


# ─── generate_summary_stats (orchestrator) ───────────────────────────────


def test_summary_stats_idempotent(
    tmp_path: Path,
    fake_results: list[dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    """Re-running over the same results produces a non-empty markdown."""
    out = tmp_path / "summary.md"
    # Build a tiny dataset with matching conversation_ids so by-family table fills.
    fake_dataset = []
    for r in fake_results:
        e = json.loads(json.dumps(entry))
        e["metadata"]["conversation_id"] = r["conversation_id"]
        e["metadata"]["model"] = r["conversation_id"].split("_", 2)[-1]
        fake_dataset.append(e)
    generate_summary_stats(fake_results, out, dataset=fake_dataset)
    md = out.read_text()
    assert "# G-Eval Run — Summary Statistics" in md
    assert "Spearman" in md
    assert "Failed entries" in md  # one failure in fake_results


def test_summary_stats_handles_all_failures(tmp_path: Path) -> None:
    """If every entry failed, the script still produces a markdown without crashing."""
    out = tmp_path / "summary.md"
    all_failed = [_fail(f"c_{i}", 3.0, "boom") for i in range(3)]
    generate_summary_stats(all_failed, out)
    md = out.read_text()
    assert "Successful | 0" in md
    assert "Failed | 3" in md


# ─── _evaluate_one with mocked metric (no API) ───────────────────────────


def test_evaluate_one_success(entry: dict[str, Any]) -> None:
    """A passing metric returns a fully populated result."""
    fake_metric = MagicMock()
    fake_metric.score = 0.8
    fake_metric.reason = "Direct and on-topic answer."

    result = _evaluate_one(entry, fake_metric, prompt_text="P", model="gpt-4o")

    assert result["geval_score"] == EXPECTED_RESCALE_FACTOR_AT_08
    assert result["human_score"] == EXPECTED_HUMAN_SCORE
    assert result["model_used"] == "gpt-4o"
    assert result["tokens_used"]["total"] > 0
    assert result["cost_usd"] > 0
    assert result["reason"] == "Direct and on-topic answer."
    assert result["attempts"] == 1


def test_evaluate_one_non_retryable_failure_recorded(entry: dict[str, Any]) -> None:
    """A non-retryable exception produces a documented failure result."""
    fake_metric = MagicMock()
    fake_metric.measure.side_effect = ValueError("malformed prompt")

    result = _evaluate_one(entry, fake_metric, prompt_text="P", model="gpt-4o")

    assert result["geval_score"] is None
    assert result["error"].startswith("ValueError")
    assert result["tokens_used"] is None
    assert result["cost_usd"] is None


def test_evaluate_one_retryable_failure_after_retries(entry: dict[str, Any]) -> None:
    """A retryable exception that never resolves is captured (5 attempts spent)."""
    fake_metric = MagicMock()
    fake_metric.measure.side_effect = openai.RateLimitError(
        message="429", response=MagicMock(), body=None
    )

    # Patch tenacity wait to a no-op so the test runs in milliseconds.
    with patch("run_geval.wait_exponential", return_value=lambda *_a, **_k: 0):
        result = _evaluate_one(entry, fake_metric, prompt_text="P", model="gpt-4o")

    assert result["geval_score"] is None
    assert "RateLimitError" in result["error"]


def test_evaluate_one_fatal_error_propagates(entry: dict[str, Any]) -> None:
    """AuthenticationError must NOT be swallowed — it aborts the run instead."""
    fake_metric = MagicMock()
    fake_metric.measure.side_effect = openai.AuthenticationError(
        message="invalid api key", response=MagicMock(), body=None
    )

    with pytest.raises(openai.AuthenticationError):
        _evaluate_one(entry, fake_metric, prompt_text="P", model="gpt-4o")


# ─── evaluate_dataset incremental persistence ────────────────────────────


def _build_minidata(n: int, entry_template: dict[str, Any]) -> list[dict[str, Any]]:
    """Tiny dataset for evaluate_dataset tests, with unique conversation_ids."""
    out = []
    for i in range(n):
        e = json.loads(json.dumps(entry_template))
        e["metadata"]["conversation_id"] = f"conv_{i}"
        out.append(e)
    return out


def test_evaluate_dataset_writes_results_at_each_checkpoint(
    tmp_path: Path,
    entry: dict[str, Any],
) -> None:
    """After CHECKPOINT_EVERY entries, results.json + summary.md must already
    be on disk — so a Ctrl+C never leaves the user with only a checkpoint."""
    n = CHECKPOINT_EVERY  # exactly one checkpoint cycle
    dataset = _build_minidata(n, entry)
    fake_metric = MagicMock()
    fake_metric.score = 0.5
    fake_metric.reason = "ok"

    with patch("run_geval.build_geval_metric", return_value=fake_metric):
        evaluate_dataset(
            dataset=dataset,
            prompt_text="P",
            model="gpt-4o",
            output_dir=tmp_path,
            limit=None,
            resume=False,
        )

    assert (tmp_path / "geval_results.json").exists()
    assert (tmp_path / "geval_summary_stats.md").exists()
    assert (tmp_path / ".geval_checkpoint.json").exists()
    with open(tmp_path / "geval_results.json", encoding="utf-8") as f:
        written = json.load(f)
    assert len(written) == n


def test_evaluate_dataset_persists_on_fatal_via_finally(
    tmp_path: Path,
    entry: dict[str, Any],
) -> None:
    """A FATAL_EXCEPTIONS abort must STILL produce results.json from the
    finally-block. Demonstrates the worst-case shutdown path: 2 entries
    succeed, the 3rd raises AuthenticationError."""
    dataset = _build_minidata(3, entry)
    fake_metric = MagicMock()
    fake_metric.score = 0.5
    fake_metric.reason = "ok"
    # Calls 1 and 2 succeed, call 3 raises AuthenticationError.
    fake_metric.measure.side_effect = [
        None,
        None,
        openai.AuthenticationError(message="bad key", response=MagicMock(), body=None),
    ]

    with (
        patch("run_geval.build_geval_metric", return_value=fake_metric),
        pytest.raises(openai.AuthenticationError),
    ):
        evaluate_dataset(
            dataset=dataset,
            prompt_text="P",
            model="gpt-4o",
            output_dir=tmp_path,
            limit=None,
            resume=False,
        )

    # Even though the run aborted, the finally-block persisted the 2 OK entries.
    assert (tmp_path / "geval_results.json").exists()
    with open(tmp_path / "geval_results.json", encoding="utf-8") as f:
        written = json.load(f)
    assert len(written) == 2  # noqa: PLR2004
    assert all(r["geval_score"] is not None for r in written)


# ─── CLI parser ──────────────────────────────────────────────────────────


def test_cli_default_args() -> None:
    args = parse_args([])
    assert args.limit is None
    assert args.no_resume is False
    assert args.model == "gpt-4o"


def test_cli_limit_and_no_resume() -> None:
    args = parse_args(["--limit", "5", "--no-resume"])
    assert args.limit == EXPECTED_CLI_LIMIT
    assert args.no_resume is True
