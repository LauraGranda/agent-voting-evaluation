"""Unit tests for scripts/transform_dataset.py transformation functions."""

import json
from pathlib import Path
from typing import Any

import pytest
from deepeval.test_case import ConversationalTestCase
from transform_dataset import (
    build_turns,
    entry_to_test_case,
    load_dataset,
    serialize_test_case,
    validate_transform,
)

# Constants for test assertions
EXPECTED_TURN_COUNT = 3
GROUND_TRUTH_SCORE = 4.5
NEGATIVE_SCORE = 2.25
LOW_SCORE = 1.5
EXPECTED_TOTAL_ENTRIES = 900

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def single_raw_entry() -> dict[str, Any]:
    """One real entry from conv_0_ground-truth."""
    return {
        "conversation_id": "conv_0_ground-truth",
        "turns": [
            "well , how does it look ?",
            "it 's a perfect fit .",
            "let me pay for it now .",
        ],
        "response": "cash , credit card , or debit card ?",
        "model": "ground-truth",
        "human_relevance_score": 4.5,
        "raw_relevance_scores": [5, 5, 3, 5],
        "human_appropriateness_score": 4.25,
        "raw_appropriateness_scores": [4, 5, 4, 4],
    }


@pytest.fixture
def negative_sample_entry() -> dict[str, Any]:
    """Entry with low relevance score — the control case."""
    return {
        "conversation_id": "conv_0_negative-sample",
        "turns": [
            "well , how does it look ?",
            "it 's a perfect fit .",
            "let me pay for it now .",
        ],
        "response": "we have binders with local job listings or you can make use of the computers . ok ?",
        "model": "negative-sample",
        "human_relevance_score": 2.25,
        "raw_relevance_scores": [1, 4, 2, 2],
        "human_appropriateness_score": 3.0,
        "raw_appropriateness_scores": [3, 4, 3, 2],
    }


@pytest.fixture
def low_relevance_entry() -> dict[str, Any]:
    """Entry with minimum-range relevance — edge case for score preservation."""
    return {
        "conversation_id": "conv_0_GPT2_small top_temp1.0_k0_p0.9",
        "turns": [
            "well , how does it look ?",
            "it 's a perfect fit .",
            "let me pay for it now .",
        ],
        "response": "i 'm so sorry . i forgot to fill out the form .",
        "model": "GPT2_small top_temp1.0_k0_p0.9",
        "human_relevance_score": 1.5,
        "raw_relevance_scores": [1, 1, 1, 3],
        "human_appropriateness_score": 3.25,
        "raw_appropriateness_scores": [4, 4, 1, 4],
    }


# ============================================================================
# Tests for build_turns()
# ============================================================================


def test_turns_count_matches_input() -> None:
    """3 input strings → 3 Turn objects."""
    turns_text = ["turn 1", "turn 2", "turn 3"]
    result = build_turns(turns_text)
    assert len(result) == EXPECTED_TURN_COUNT


def test_turns_alternating_roles() -> None:
    """Roles must alternate user/assistant/user starting with user."""
    turns_text = ["turn 1", "turn 2", "turn 3"]
    result = build_turns(turns_text)
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[2].role == "user"


def test_turns_content_preserved() -> None:
    """Text content must match input strings exactly, no modification."""
    turns_text = ["turn 1", "turn 2", "turn 3"]
    result = build_turns(turns_text)
    for i, text in enumerate(turns_text):
        assert result[i].content == text


def test_turns_single_turn() -> None:
    """1 input string → 1 Turn with role=user."""
    result = build_turns(["single turn"])
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "single turn"


def test_turns_empty_list() -> None:
    """Empty list → empty list, no crash."""
    result = build_turns([])
    assert result == []


# ============================================================================
# Tests for entry_to_test_case()
# ============================================================================


def test_transform_ground_truth_entry(single_raw_entry: dict[str, Any]) -> None:
    """Ground-truth entry transforms without error."""
    result = entry_to_test_case(single_raw_entry)
    assert isinstance(result, ConversationalTestCase)
    assert len(result.turns) == EXPECTED_TURN_COUNT + 1  # 3 context + 1 response


def test_transform_negative_sample_entry(negative_sample_entry: dict[str, Any]) -> None:
    """Negative-sample entry transforms correctly."""
    result = entry_to_test_case(negative_sample_entry)
    assert result.additional_metadata["human_score"] == NEGATIVE_SCORE


def test_transform_low_relevance_entry(low_relevance_entry: dict[str, Any]) -> None:
    """Low relevance entry (score 1.5) transforms correctly."""
    result = entry_to_test_case(low_relevance_entry)
    assert result.additional_metadata["human_score"] == LOW_SCORE


def test_metadata_human_score_preserved(single_raw_entry: dict[str, Any]) -> None:
    """metadata['human_score'] == entry['human_relevance_score'] exactly."""
    result = entry_to_test_case(single_raw_entry)
    assert result.additional_metadata["human_score"] == single_raw_entry["human_relevance_score"]


def test_metadata_raw_scores_preserved(single_raw_entry: dict[str, Any]) -> None:
    """metadata['raw_relevance_scores'] == entry['raw_relevance_scores']."""
    result = entry_to_test_case(single_raw_entry)
    assert (
        result.additional_metadata["raw_relevance_scores"]
        == single_raw_entry["raw_relevance_scores"]
    )


def test_metadata_conversation_id_preserved(single_raw_entry: dict[str, Any]) -> None:
    """metadata['conversation_id'] matches input conversation_id."""
    result = entry_to_test_case(single_raw_entry)
    assert result.additional_metadata["conversation_id"] == single_raw_entry["conversation_id"]


def test_metadata_model_preserved(single_raw_entry: dict[str, Any]) -> None:
    """metadata['model'] matches input model name including spaces."""
    result = entry_to_test_case(single_raw_entry)
    assert result.additional_metadata["model"] == single_raw_entry["model"]


def test_input_is_last_turn(single_raw_entry: dict[str, Any]) -> None:
    """Input field should match the last user turn from original entry."""
    result = entry_to_test_case(single_raw_entry)
    serialized = serialize_test_case(result)
    # Last turn in original context should be the input prompt
    assert serialized["input"] == single_raw_entry["turns"][-1]


def test_actual_output_is_response(single_raw_entry: dict[str, Any]) -> None:
    """actual_output should be the response from the entry."""
    result = entry_to_test_case(single_raw_entry)
    serialized = serialize_test_case(result)
    assert serialized["actual_output"] == single_raw_entry["response"]


# ============================================================================
# Tests for serialize_test_case()
# ============================================================================


def test_serialized_is_json_serializable(single_raw_entry: dict[str, Any]) -> None:
    """json.dumps(serialized) must not raise — no datetime, no custom objects."""
    test_case = entry_to_test_case(single_raw_entry)
    serialized = serialize_test_case(test_case)
    json.dumps(serialized)  # Should not raise


def test_serialized_has_required_keys(single_raw_entry: dict[str, Any]) -> None:
    """Serialized dict has keys: input, actual_output, turns, metadata."""
    test_case = entry_to_test_case(single_raw_entry)
    serialized = serialize_test_case(test_case)
    required_keys = {"input", "actual_output", "turns", "metadata"}
    assert set(serialized.keys()) == required_keys


def test_serialized_metadata_has_required_keys(
    single_raw_entry: dict[str, Any],
) -> None:
    """Metadata has 6 required keys."""
    test_case = entry_to_test_case(single_raw_entry)
    serialized = serialize_test_case(test_case)
    required_metadata_keys = {
        "human_score",
        "raw_relevance_scores",
        "human_appropriateness_score",
        "raw_appropriateness_scores",
        "conversation_id",
        "model",
    }
    assert set(serialized["metadata"].keys()) == required_metadata_keys


def test_serialized_turns_are_dicts(single_raw_entry: dict[str, Any]) -> None:
    """Turns field is list of dicts with 'role' and 'content' keys."""
    test_case = entry_to_test_case(single_raw_entry)
    serialized = serialize_test_case(test_case)
    assert isinstance(serialized["turns"], list)
    for turn in serialized["turns"]:
        assert isinstance(turn, dict)
        assert "role" in turn
        assert "content" in turn


# ============================================================================
# Tests for validate_transform()
# ============================================================================


def test_validation_passes_on_valid_data(single_raw_entry: dict[str, Any]) -> None:
    """Valid input/output pair → no AssertionError raised."""
    original = [single_raw_entry]
    test_case = entry_to_test_case(single_raw_entry)
    serialized = [serialize_test_case(test_case)]
    validate_transform(original, serialized)  # Should not raise


def test_validation_fails_on_count_mismatch(single_raw_entry: dict[str, Any]) -> None:
    """900 in, 899 out → AssertionError with message mentioning count."""
    original = [single_raw_entry] * EXPECTED_TOTAL_ENTRIES
    test_case = entry_to_test_case(single_raw_entry)
    serialized = [serialize_test_case(test_case)] * (EXPECTED_TOTAL_ENTRIES - 1)
    with pytest.raises(AssertionError, match="Count mismatch"):
        validate_transform(original, serialized)


def test_validation_fails_on_score_mismatch(single_raw_entry: dict[str, Any]) -> None:
    """One entry with modified human_score → AssertionError."""
    original = [single_raw_entry]
    test_case = entry_to_test_case(single_raw_entry)
    serialized = [serialize_test_case(test_case)]
    # Modify score in serialized version
    serialized[0]["metadata"]["human_score"] = 2.0
    with pytest.raises(AssertionError, match="human_score mismatch"):
        validate_transform(original, serialized)


def test_validation_fails_on_null_response(single_raw_entry: dict[str, Any]) -> None:
    """One entry with actual_output = '' → AssertionError."""
    original = [single_raw_entry]
    test_case = entry_to_test_case(single_raw_entry)
    serialized = [serialize_test_case(test_case)]
    # Clear actual_output
    serialized[0]["actual_output"] = ""
    with pytest.raises(AssertionError, match="actual_output is null or empty"):
        validate_transform(original, serialized)


def test_validation_fails_on_wrong_conversation_id(
    single_raw_entry: dict[str, Any],
) -> None:
    """One entry with modified conversation_id → AssertionError."""
    original = [single_raw_entry]
    test_case = entry_to_test_case(single_raw_entry)
    serialized = [serialize_test_case(test_case)]
    # Modify conversation_id in serialized version
    serialized[0]["metadata"]["conversation_id"] = "wrong_id"
    with pytest.raises(AssertionError, match="conversation_id mismatch"):
        validate_transform(original, serialized)


# ============================================================================
# Tests for load_dataset()
# ============================================================================


def test_load_dataset_reads_json(tmp_path: Path) -> None:
    """load_dataset reads and returns JSON data from file."""
    test_data = [{"test": "entry"}]
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(test_data))
    result = load_dataset(json_file)
    assert result == test_data


def test_load_dataset_file_not_found() -> None:
    """load_dataset raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("nonexistent.json"))


# ============================================================================
# Integration test
# ============================================================================


def test_integration_full_pipeline(single_raw_entry: dict[str, Any]) -> None:
    """Full pipeline: entry → test_case → serialized → validation."""
    original = [single_raw_entry]
    test_case = entry_to_test_case(single_raw_entry)
    serialized = [serialize_test_case(test_case)]
    validate_transform(original, serialized)
    # Verify output structure
    assert serialized[0]["input"] == single_raw_entry["turns"][-1]
    assert serialized[0]["actual_output"] == single_raw_entry["response"]
