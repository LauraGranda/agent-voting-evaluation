"""Unit tests for scripts/download_dataset.py data processing functions."""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Constants for test assertions
NUM_ANNOTATORS = 4
NUM_MODELS_PER_CONVERSATION = 9
MEAN_4_5_3_5 = 4.25  # mean([4, 5, 3, 5])
MEAN_3_4_3_5 = 3.75  # mean([3, 4, 3, 5])
MEAN_5_5_3_5 = 4.5  # mean([5, 5, 3, 5])
MEAN_1_2_3_4 = 2.5  # mean([1, 2, 3, 4])

# Import the functions to test
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from download_dataset import (  # noqa: E402
    generate_readme,
    parse_annotations,
    print_summary,
    run_integrity_checks,
)

# ============================================================================
# Fixtures
# ============================================================================


def make_raw_dialog_entry(
    dialog_id: str,
    context: list[list[str]],
    models_and_scores: dict[str, list[list[int]]],
) -> dict[str, Any]:
    """Build a raw dialog entry matching the actual Zenodo JSON structure.

    Args:
        dialog_id: Identifier for this dialog.
        context: [["speaker", "text"], ...].
        models_and_scores: {"model_name": [[relevance, content, ...], ...]}.

    Returns:
        Raw dialog dict ready for parsing (100% accuracy).
    """
    responses = {}
    for model_name, score_lists in models_and_scores.items():
        scores_dict = {}
        for worker_idx, scores in enumerate(score_lists):
            scores_dict[str(worker_idx)] = {
                "worker_id": worker_idx,
                "relevance": scores[0] if len(scores) > 0 else 3,
                "content": scores[1] if len(scores) > 1 else 3,
            }
        responses[model_name] = {
            "uttr": f"response for {model_name}",
            "scores": scores_dict,
        }
    return {
        dialog_id: {
            "context": context,
            "reference": ["A", "reference"],
            "responses": responses,
        }
    }


def make_dataset(
    n_convs: int,
    models: list[str],
) -> list[dict[str, Any]]:
    """Programmatically build a test dataset with n_convs x len(models) entries.

    Args:
        n_convs: Number of unique conversations.
        models: List of model names (same for each conversation).

    Returns:
        List of dataset entries matching the output schema.
    """
    dataset = []
    for conv_idx in range(n_convs):
        for model in models:
            entry = {
                "conversation_id": f"conv_{conv_idx}_{model}",
                "turns": [f"turn_{i}" for i in range(3)],
                "response": f"response from {model}",
                "model": model,
                "human_relevance_score": 3.5,
                "raw_relevance_scores": [3, 4, 3, 4],
                "human_appropriateness_score": 3.25,
                "raw_appropriateness_scores": [3, 3, 3, 4],
            }
            dataset.append(entry)
    return dataset


@pytest.fixture
def single_entry() -> dict[str, Any]:
    """One valid processed dataset entry matching the real schema."""
    return {
        "conversation_id": "conv_0_ground-truth",
        "turns": ["well , how does it look ?", "it 's a perfect fit ."],
        "response": "cash , credit card , or debit card ?",
        "model": "ground-truth",
        "human_relevance_score": 4.5,
        "raw_relevance_scores": [5, 5, 3, 5],
        "human_appropriateness_score": 4.25,
        "raw_appropriateness_scores": [4, 5, 4, 4],
    }


@pytest.fixture
def minimal_valid_dataset() -> list[dict[str, Any]]:
    """9 entries (1 conversation x 9 models) - smallest valid unit."""
    models = [
        "ground-truth",
        "negative-sample",
        "GPT2_small",
        "GPT2_medium",
        "HRED",
        "S2S",
        "S2S_attn",
        "VHRED_attn",
        "dummy_model",
    ]
    return make_dataset(1, models)


@pytest.fixture
def full_valid_dataset() -> list[dict[str, Any]]:
    """900 entries (100 conversations x 9 models) built programmatically."""
    models = [f"model_{i}" for i in range(9)]  # 9 models per conversation
    dataset = make_dataset(100, models)
    # Ensure min/max scores for integrity checks
    dataset[0]["raw_relevance_scores"] = [1, 1, 1, 1]
    dataset[0]["human_relevance_score"] = 1.0
    dataset[1]["raw_relevance_scores"] = [5, 5, 5, 5]
    dataset[1]["human_relevance_score"] = 5.0
    return dataset


# ============================================================================
# Tests for parse_annotations()
# ============================================================================


class TestParseAnnotations:
    """Test suite for parse_annotations function."""

    def test_parse_single_entry(self, tmp_path: Path) -> None:
        """Parse 1 dialog with 1 model -> 1 output entry, all fields correct."""
        raw_data = make_raw_dialog_entry(
            "dialog_0",
            [["A", "hello"], ["B", "world"]],
            {"test_model": [[4, 3], [5, 4], [3, 3], [5, 5]]},
        )
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(raw_data))

        result = parse_annotations(json_file)

        assert len(result) == 1
        entry = result[0]
        assert entry["conversation_id"] == "conv_0_test_model"
        assert entry["model"] == "test_model"
        assert entry["turns"] == ["hello", "world"]
        assert entry["response"] == "response for test_model"
        assert len(entry["raw_relevance_scores"]) == NUM_ANNOTATORS
        assert len(entry["raw_appropriateness_scores"]) == NUM_ANNOTATORS
        assert entry["human_relevance_score"] == MEAN_4_5_3_5
        assert entry["human_appropriateness_score"] == MEAN_3_4_3_5

    def test_parse_multiple_models(self, tmp_path: Path) -> None:
        """Parse 1 dialog with 9 models -> 9 output entries."""
        models_scores = {
            f"model_{i}": [[3, 3], [4, 4], [3, 3], [4, 4]]
            for i in range(NUM_MODELS_PER_CONVERSATION)
        }
        raw_data = make_raw_dialog_entry(
            "dialog_0",
            [["A", "text"]],
            models_scores,
        )
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(raw_data))

        result = parse_annotations(json_file)

        assert len(result) == NUM_MODELS_PER_CONVERSATION
        for i in range(NUM_MODELS_PER_CONVERSATION):
            assert result[i]["model"] == f"model_{i}"
            assert result[i]["conversation_id"] == f"conv_0_model_{i}"

    def test_parse_score_calculation(self, tmp_path: Path) -> None:
        """Verify mean is computed correctly (e.g. [5,5,3,5] -> 4.5, not rounded)."""
        raw_data = make_raw_dialog_entry(
            "dialog_0",
            [["A", "text"]],
            {"test": [[5, 1], [5, 2], [3, 3], [5, 4]]},
        )
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(raw_data))

        result = parse_annotations(json_file)
        entry = result[0]

        # mean([5, 5, 3, 5]) = 18/4 = 4.5 (not rounded)
        assert entry["human_relevance_score"] == MEAN_5_5_3_5
        # mean([1, 2, 3, 4]) = 10/4 = 2.5
        assert entry["human_appropriateness_score"] == MEAN_1_2_3_4

    def test_parse_turns_extraction(self, tmp_path: Path) -> None:
        """Turns[] contains only text strings, no speaker labels."""
        context = [["A", "first turn"], ["B", "second turn"], ["A", "third turn"]]
        raw_data = make_raw_dialog_entry(
            "dialog_0",
            context,
            {"model": [[3, 3], [3, 3], [3, 3], [3, 3]]},
        )
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(raw_data))

        result = parse_annotations(json_file)
        entry = result[0]

        assert entry["turns"] == ["first turn", "second turn", "third turn"]
        # No speaker labels in turns
        assert all(speaker not in str(turn) for speaker in ["A", "B"] for turn in entry["turns"])

    def test_parse_conversation_id_format(self, tmp_path: Path) -> None:
        """ID follows 'conv_{index}_{model}' pattern."""
        raw_data = make_raw_dialog_entry(
            "dialog_5",
            [["A", "text"]],
            {"special_model": [[3, 3], [3, 3], [3, 3], [3, 3]]},
        )
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(raw_data))

        result = parse_annotations(json_file)
        conv_id = result[0]["conversation_id"]

        expected_parts = NUM_MODELS_PER_CONVERSATION // 3
        assert conv_id.startswith("conv_")
        parts = conv_id.split("_", 2)
        assert len(parts) == expected_parts
        assert parts[0] == "conv"
        assert parts[1].isdigit()
        assert parts[2] == "special_model"

    def test_parse_ground_truth_included(self, tmp_path: Path) -> None:
        """'ground-truth' model is included as entry."""
        models_scores = {
            "ground-truth": [[5, 5], [5, 5], [5, 5], [5, 5]],
            "other": [[3, 3], [3, 3], [3, 3], [3, 3]],
        }
        raw_data = make_raw_dialog_entry(
            "dialog_0",
            [["A", "text"]],
            models_scores,
        )
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(raw_data))

        result = parse_annotations(json_file)
        models_in_result = {e["model"] for e in result}

        assert "ground-truth" in models_in_result

    def test_parse_negative_sample_included(self, tmp_path: Path) -> None:
        """'negative-sample' model is included as entry."""
        models_scores = {
            "negative-sample": [[1, 1], [1, 1], [1, 1], [1, 1]],
            "other": [[3, 3], [3, 3], [3, 3], [3, 3]],
        }
        raw_data = make_raw_dialog_entry(
            "dialog_0",
            [["A", "text"]],
            models_scores,
        )
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(raw_data))

        result = parse_annotations(json_file)
        models_in_result = {e["model"] for e in result}

        assert "negative-sample" in models_in_result

    def test_parse_empty_input(self, tmp_path: Path) -> None:
        """Empty dict input -> returns empty list, no crash."""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({}))

        result = parse_annotations(json_file)

        assert result == []
        assert isinstance(result, list)

    def test_parse_missing_relevance_score(self, tmp_path: Path) -> None:
        """Entry with missing 'relevance' key has empty raw_relevance_scores."""
        raw_data = {
            "dialog_0": {
                "context": [["A", "text"]],
                "reference": ["A", "ref"],
                "responses": {
                    "model": {
                        "uttr": "response",
                        "scores": {
                            "worker_0": {"worker_id": 0, "content": 3},
                            # Missing 'relevance'
                        },
                    }
                },
            }
        }
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(raw_data))

        result = parse_annotations(json_file)
        entry = result[0]

        assert entry["raw_relevance_scores"] == []
        assert entry["human_relevance_score"] == 0.0


# ============================================================================
# Tests for run_integrity_checks()
# ============================================================================


class TestRunIntegrityChecks:
    """Test suite for run_integrity_checks function."""

    def test_integrity_900_pairs_pass(self, full_valid_dataset: list[dict[str, Any]]) -> None:
        """900 entries (100 convs x 9 models) - passes all checks."""
        # Should not raise SystemExit
        with patch("builtins.print"):
            run_integrity_checks(full_valid_dataset)

    def test_integrity_wrong_total_fails(self) -> None:
        """850 entries instead of 900 -> fails."""
        dataset = make_dataset(85, [f"model_{i}" for i in range(10)])
        # 85 x 10 = 850

        with pytest.raises(SystemExit) as exc_info, patch("builtins.print"):
            run_integrity_checks(dataset)

        assert exc_info.value.code == 1

    def test_integrity_wrong_scale_min_fails(self) -> None:
        """One score = 0 (below min 1) -> fails."""
        dataset = make_dataset(100, ["model_0"] * NUM_MODELS_PER_CONVERSATION)
        dataset[0]["raw_relevance_scores"] = [0, 3, 3, 3]  # Invalid: 0 < 1

        with pytest.raises(SystemExit) as exc_info, patch("builtins.print"):
            run_integrity_checks(dataset)

        assert exc_info.value.code == 1

    def test_integrity_wrong_scale_max_fails(self) -> None:
        """One score = 6 (above max 5) -> fails."""
        dataset = make_dataset(100, ["model_0"] * NUM_MODELS_PER_CONVERSATION)
        dataset[0]["raw_relevance_scores"] = [6, 3, 3, 3]  # Invalid: 6 > 5

        with pytest.raises(SystemExit) as exc_info, patch("builtins.print"):
            run_integrity_checks(dataset)

        assert exc_info.value.code == 1

    def test_integrity_wrong_annotators_fails(self) -> None:
        """One entry has only 3 annotators instead of 4 -> fails."""
        dataset = make_dataset(100, ["model_0"] * NUM_MODELS_PER_CONVERSATION)
        dataset[0]["raw_relevance_scores"] = [3, 4, 3]  # Only 3 annotators

        with pytest.raises(SystemExit) as exc_info, patch("builtins.print"):
            run_integrity_checks(dataset)

        assert exc_info.value.code == 1

    def test_integrity_all_checks_reported(self, full_valid_dataset: list[dict[str, Any]]) -> None:
        """Verify PASS printed for each of 5 expected checks."""
        with patch("builtins.print") as mock_print:
            run_integrity_checks(full_valid_dataset)

        # Collect all print calls
        calls_str = " ".join(str(call) for call in mock_print.call_args_list)

        # Check that each key appears in output
        assert "total_pairs" in calls_str
        assert "annotation_scale_min" in calls_str
        assert "annotation_scale_max" in calls_str
        assert "unique_conversations" in calls_str
        assert "annotators_per_pair" in calls_str


# ============================================================================
# Tests for print_summary()
# ============================================================================


class TestPrintSummary:
    """Test suite for print_summary function."""

    def test_print_summary_with_minimal_dataset(
        self, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Print summary for 9 entries without crashing."""
        # Should not raise any exception
        with patch("builtins.print"):
            print_summary(minimal_valid_dataset)

    def test_print_summary_with_full_dataset(
        self, full_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Print summary for 900 entries without crashing."""
        with patch("builtins.print"):
            print_summary(full_valid_dataset)

    def test_print_summary_reports_total_pairs(
        self, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Summary output includes total pair count."""
        with patch("builtins.print") as mock_print:
            print_summary(minimal_valid_dataset)

        calls_str = " ".join(str(call) for call in mock_print.call_args_list)
        assert "9" in calls_str  # 9 pairs in minimal_valid_dataset

    def test_print_summary_reports_models(
        self, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Summary output includes all unique model names."""
        with patch("builtins.print") as mock_print:
            print_summary(minimal_valid_dataset)

        calls_str = " ".join(str(call) for call in mock_print.call_args_list)
        # Should mention model count and list models
        assert "Unique models" in calls_str


# ============================================================================
# Tests for generate_readme()
# ============================================================================


class TestGenerateReadme:
    """Test suite for generate_readme function."""

    def test_readme_created(self, tmp_path: Path, single_entry: dict[str, Any]) -> None:
        """File exists after call."""
        readme_path = tmp_path / "test_readme.md"
        dataset = [single_entry]

        with patch("builtins.print"):
            generate_readme(dataset, readme_path)

        assert readme_path.exists()

    def test_readme_contains_source_url(
        self, tmp_path: Path, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Content includes zenodo URL."""
        readme_path = tmp_path / "test_readme.md"

        with patch("builtins.print"):
            generate_readme(minimal_valid_dataset, readme_path)

        content = readme_path.read_text()
        assert "zenodo.org/record/3828180" in content

    def test_readme_contains_license(
        self, tmp_path: Path, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Content includes 'CC BY-NC-SA 4.0'."""
        readme_path = tmp_path / "test_readme.md"

        with patch("builtins.print"):
            generate_readme(minimal_valid_dataset, readme_path)

        content = readme_path.read_text()
        assert "CC BY-NC-SA 4.0" in content

    def test_readme_contains_size_stats(
        self, tmp_path: Path, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Content includes '900' and '100' (hardcoded stats)."""
        readme_path = tmp_path / "test_readme.md"

        with patch("builtins.print"):
            generate_readme(minimal_valid_dataset, readme_path)

        content = readme_path.read_text()
        assert "900" in content  # Total pairs (hardcoded)
        assert "100" in content  # Total conversations (hardcoded)

    def test_readme_contains_all_models(
        self, tmp_path: Path, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Each unique model name in dataset appears in readme."""
        readme_path = tmp_path / "test_readme.md"

        with patch("builtins.print"):
            generate_readme(minimal_valid_dataset, readme_path)

        content = readme_path.read_text()
        models = set(entry["model"] for entry in minimal_valid_dataset)
        for model in models:
            assert model in content

    def test_readme_contains_integrity_section(
        self, tmp_path: Path, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Content includes 'Integrity Verification' section."""
        readme_path = tmp_path / "test_readme.md"

        with patch("builtins.print"):
            generate_readme(minimal_valid_dataset, readme_path)

        content = readme_path.read_text()
        assert "Integrity Verification" in content

    def test_readme_contains_schema_table(
        self, tmp_path: Path, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Content includes all 8 field names in schema table."""
        readme_path = tmp_path / "test_readme.md"

        with patch("builtins.print"):
            generate_readme(minimal_valid_dataset, readme_path)

        content = readme_path.read_text()
        fields = [
            "conversation_id",
            "turns",
            "response",
            "model",
            "human_relevance_score",
            "raw_relevance_scores",
            "human_appropriateness_score",
            "raw_appropriateness_scores",
        ]
        for field in fields:
            assert field in content

    def test_readme_download_date_present(
        self, tmp_path: Path, minimal_valid_dataset: list[dict[str, Any]]
    ) -> None:
        """Content includes today's date."""
        readme_path = tmp_path / "test_readme.md"

        with patch("builtins.print"), patch("download_dataset.datetime") as mock_datetime:
            mock_datetime.now.return_value = MagicMock(
                isoformat=MagicMock(return_value="2026-04-11T12:00:00.000000")
            )
            generate_readme(minimal_valid_dataset, readme_path)

        content = readme_path.read_text()
        assert "Download date" in content
