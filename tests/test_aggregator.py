"""Unit tests for src/voting/aggregator.py — pure logic, no API calls.

Covers the five mandatory scenarios from the HU-07 Definition of Done
(unanimity, majority, tie behaviour, maximum dispersion, missing score)
plus output-structure and precision tests. Fixtures use the canonical
judge names declared in ``configs/agents/agent_*.yaml``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.voting.aggregator import (
    AGREEMENT_HIGH_THRESHOLD,
    SCHEME_NAME,
    SCORE_MAX,
    SCORE_MIN,
    aggregate,
)

# ─── Expected literal constants (named to avoid magic-value comparisons) ─
# Each value is hand-computed against the formulas in
# docs/voting_scheme_analysis.md §4 and the spec thresholds, so a reader
# does not have to mentally evaluate the function under test.

EXPECTED_UNANIMOUS_SCORE: float = 4.0
EXPECTED_TYPICAL_FINAL_SCORE: float = 3.67  # round(11/3, 2)
EXPECTED_TYPICAL_MEDIAN: float = 4.0
EXPECTED_DISPERSED_FINAL_SCORE: float = 3.0
EXPECTED_DISPERSED_MEDIAN: float = 3.0
EXPECTED_MIN_FINAL_SCORE: float = 1.0
EXPECTED_MAX_FINAL_SCORE: float = 5.0
EXPECTED_MISSING_FINAL_SCORE: float = 3.5  # round((4+3)/2, 2)
SINGLE_AGENT_SCORE: float = 4.0

# Number of decimals every score in the output is rounded to.
SCORE_DECIMALS: int = 2

# Top-level keys the aggregator must always return.
REQUIRED_TOP_KEYS: tuple[str, ...] = (
    "final_score",
    "individual_scores",
    "agreement_level",
    "scheme_used",
    "metadata",
)

# Metadata keys the AC requires plus the two extras from docs/voting_scheme_analysis.md.
REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "n_agents",
    "std_deviation",
    "min_score",
    "max_score",
    "median_score",
    "agreement_continuous",
)


# ─── Fixtures (canonical judge names from configs/agents/*.yaml) ─────────
@pytest.fixture
def unanimous_scores() -> dict[str, float]:
    """All three judges return the same score."""
    return {"judge_openai": 4, "judge_google": 4, "judge_anthropic": 4}


@pytest.fixture
def typical_scores() -> dict[str, float]:
    """Two judges agree on 4, one returns 3 — typical near-consensus case."""
    return {"judge_openai": 4, "judge_google": 3, "judge_anthropic": 4}


@pytest.fixture
def dispersed_scores() -> dict[str, float]:
    """Three distinct scores spanning the full 1-5 range (maximum dispersion)."""
    return {"judge_openai": 5, "judge_google": 1, "judge_anthropic": 3}


@pytest.fixture
def scores_with_missing() -> dict[str, float | None]:
    """One judge failed to produce a score (None)."""
    return {"judge_openai": 4, "judge_google": None, "judge_anthropic": 3}


# ─── 1. Unanimity ────────────────────────────────────────────────────────
def test_unanimity_returns_high_agreement_and_same_score(
    unanimous_scores: dict[str, float],
) -> None:
    """Unanimity: final_score equals the shared value, agreement is 'high'."""
    result = aggregate(unanimous_scores)
    assert result["final_score"] == EXPECTED_UNANIMOUS_SCORE
    assert result["agreement_level"] == "high"
    assert result["metadata"]["std_deviation"] == 0.0
    assert result["metadata"]["agreement_continuous"] == 1.0


# ─── 2. Typical majority case ────────────────────────────────────────────
def test_typical_majority_case(typical_scores: dict[str, float]) -> None:
    """{4, 3, 4}: arithmetic mean 11/3 rounds to 3.67, std ≈ 0.577 → medium."""
    result = aggregate(typical_scores)
    assert result["final_score"] == EXPECTED_TYPICAL_FINAL_SCORE
    assert result["agreement_level"] == "medium"
    assert result["metadata"]["std_deviation"] > AGREEMENT_HIGH_THRESHOLD


# ─── 3. Maximum dispersion (covers "tie" semantics for the mean) ─────────
def test_maximum_dispersion_is_low_agreement(
    dispersed_scores: dict[str, float],
) -> None:
    """{5, 1, 3}: a configuration ambiguous under mode/majority is single-valued under the
    mean (final 3.0); std 2.0 > 1.0 ⇒ 'low'.
    """
    result = aggregate(dispersed_scores)
    assert result["final_score"] == EXPECTED_DISPERSED_FINAL_SCORE
    assert result["agreement_level"] == "low"


# ─── 4. Missing agent score ──────────────────────────────────────────────
def test_missing_agent_score_is_handled(
    scores_with_missing: dict[str, float | None],
) -> None:
    """None score: aggregate over the remaining agents, record the missing one."""
    result = aggregate(scores_with_missing)
    assert result["final_score"] == EXPECTED_MISSING_FINAL_SCORE
    assert "missing_agents" in result["metadata"]
    assert result["metadata"]["missing_agents"] == ["judge_google"]
    assert result["metadata"]["n_agents"] == len(scores_with_missing) - 1
    # Input is echoed unchanged for traceability, including the None entry.
    assert result["individual_scores"] == scores_with_missing


# ─── 5. Score out of range ───────────────────────────────────────────────
def test_score_out_of_range_raises() -> None:
    """A score above SCORE_MAX (or below SCORE_MIN) raises ValueError, naming the agent."""
    out_of_range = {"judge_openai": SCORE_MAX + 1, "judge_google": 3, "judge_anthropic": 4}
    with pytest.raises(ValueError, match="judge_openai"):
        aggregate(out_of_range)


# ─── 6. Empty input ──────────────────────────────────────────────────────
def test_empty_input_raises() -> None:
    """An empty dict raises ValueError with the exact message from the spec."""
    with pytest.raises(ValueError, match="No agent scores provided"):
        aggregate({})


# ─── 7. All minimum scores ───────────────────────────────────────────────
def test_all_minimum_scores() -> None:
    """All judges score SCORE_MIN: final == 1.0, agreement 'high'."""
    all_min = {"judge_openai": SCORE_MIN, "judge_google": SCORE_MIN, "judge_anthropic": SCORE_MIN}
    result = aggregate(all_min)
    assert result["final_score"] == EXPECTED_MIN_FINAL_SCORE
    assert result["agreement_level"] == "high"


# ─── 8. All maximum scores ───────────────────────────────────────────────
def test_all_maximum_scores() -> None:
    """All judges score SCORE_MAX: final == 5.0, agreement 'high'."""
    all_max = {"judge_openai": SCORE_MAX, "judge_google": SCORE_MAX, "judge_anthropic": SCORE_MAX}
    result = aggregate(all_max)
    assert result["final_score"] == EXPECTED_MAX_FINAL_SCORE
    assert result["agreement_level"] == "high"


# ─── 9. Output structure ─────────────────────────────────────────────────
def test_output_structure(typical_scores: dict[str, float]) -> None:
    """Every required top-level key and metadata key is present."""
    result = aggregate(typical_scores)
    for key in REQUIRED_TOP_KEYS:
        assert key in result
    for key in REQUIRED_METADATA_KEYS:
        assert key in result["metadata"]
    assert result["scheme_used"] == SCHEME_NAME


# ─── 10. Score precision and type ────────────────────────────────────────
def test_final_score_precision_and_type(typical_scores: dict[str, float]) -> None:
    """final_score is a float and is rounded to at most two decimal places."""
    result = aggregate(typical_scores)
    final = result["final_score"]
    assert isinstance(final, float)
    assert round(final, SCORE_DECIMALS) == final


# ─── 11. Median reported in parallel as robustness check (doc HU-05) ─────
def test_median_score_in_metadata(dispersed_scores: dict[str, float]) -> None:
    """{5, 1, 3} sorted is (1, 3, 5); median == 3.0."""
    result = aggregate(dispersed_scores)
    assert result["metadata"]["median_score"] == EXPECTED_DISPERSED_MEDIAN


# ─── 12. Continuous agreement (doc HU-05 §4) ─────────────────────────────
def test_agreement_continuous_in_range(
    typical_scores: dict[str, float],
    unanimous_scores: dict[str, float],
) -> None:
    """agreement_continuous lives in [0, 1]; unanimity reaches exactly 1.0."""
    typical = aggregate(typical_scores)["metadata"]["agreement_continuous"]
    assert typical is not None
    assert 0.0 <= typical <= 1.0
    unanimous = aggregate(unanimous_scores)["metadata"]["agreement_continuous"]
    assert unanimous == 1.0


# ─── 13. Single agent ────────────────────────────────────────────────────
def test_single_agent_returns_na_agreement() -> None:
    """One score: aggregate without error; agreement is 'n/a' and continuous is None."""
    result = aggregate({"judge_openai": SINGLE_AGENT_SCORE})
    assert result["final_score"] == SINGLE_AGENT_SCORE
    assert result["agreement_level"] == "n/a"
    assert result["metadata"]["agreement_continuous"] is None
    assert result["metadata"]["n_agents"] == 1
