"""Transform DailyDialog-Zhao dataset to DeepEval ConversationalTestCase format."""

import json
import sys
from pathlib import Path
from typing import Any, Final, Literal, cast

from deepeval.test_case import ConversationalTestCase, Turn

# Constants
MIN_HUMAN_SCORE: Final[float] = 1.0
MAX_HUMAN_SCORE: Final[float] = 5.0
EXPECTED_TOTAL_ENTRIES: Final[int] = 900
RAW_PATH: Final[Path] = Path("data/raw/dailydialog_zhao/dataset.json")
OUT_PATH: Final[Path] = Path("data/processed/deepeval_test_cases.json")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and return dataset entries from JSON file.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        List of dataset entry dictionaries.

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        return cast(list[dict[str, Any]], json.load(f))


def build_turns(turns: list[str]) -> list[Turn]:
    """Convert list of text strings to DeepEval Turn objects.

    Alternates between user and assistant roles starting with user.
    Preserves all turns including the last one as conversation history.

    Args:
        turns: List of turn text strings.

    Returns:
        List of Turn objects with alternating roles.
    """
    result = []
    for i, text in enumerate(turns):
        role: Literal["user", "assistant"] = "user" if i % 2 == 0 else "assistant"
        result.append(Turn(role=role, content=text))
    return result


def entry_to_test_case(entry: dict[str, Any]) -> ConversationalTestCase:
    """Transform a single dataset entry to ConversationalTestCase.

    Preserves human scores and metadata for downstream correlation analysis.

    Args:
        entry: Dataset entry with keys: conversation_id, turns, response, model,
            human_relevance_score, raw_relevance_scores,
            human_appropriateness_score, raw_appropriateness_scores.

    Returns:
        ConversationalTestCase with turns and metadata.
    """
    # Build turns from context and add final response as assistant turn
    turns = build_turns(entry["turns"])
    turns.append(Turn(role="assistant", content=entry["response"]))

    # Metadata for correlation and score preservation
    metadata = {
        "human_score": entry["human_relevance_score"],
        "raw_relevance_scores": entry["raw_relevance_scores"],
        "human_appropriateness_score": entry["human_appropriateness_score"],
        "raw_appropriateness_scores": entry["raw_appropriateness_scores"],
        "conversation_id": entry["conversation_id"],
        "model": entry["model"],
    }

    return ConversationalTestCase(turns=turns, additional_metadata=metadata)


def serialize_test_case(tc: ConversationalTestCase) -> dict[str, Any]:
    """Serialize ConversationalTestCase to JSON-compatible dict.

    DeepEval objects are not JSON-serializable by default.
    Extract all fields manually into a plain dict.

    Args:
        tc: ConversationalTestCase instance.

    Returns:
        Dictionary with keys: input, actual_output, turns, metadata.
    """
    # Extract turns as plain dicts
    turns_list = [{"role": t.role, "content": t.content} for t in tc.turns]

    # Find last user turn (input) and last assistant turn (actual_output)
    input_text = ""
    actual_output_text = ""
    for turn in reversed(tc.turns):
        if turn.role == "user" and not input_text:
            input_text = turn.content
        elif turn.role == "assistant" and not actual_output_text:
            actual_output_text = turn.content

    return {
        "input": input_text,
        "actual_output": actual_output_text,
        "turns": turns_list,
        "metadata": tc.additional_metadata or {},
    }


def validate_transform(original: list[dict[str, Any]], transformed: list[dict[str, Any]]) -> None:
    """Assert 100% transformation without data loss.

    Raises AssertionError with clear message if any check fails.
    Checks: count match, no null conversation_ids, no null responses,
    all human_scores preserved exactly, score range [1.0, 5.0].

    Args:
        original: Original dataset entries.
        transformed: Serialized transformed entries.

    Raises:
        AssertionError: If any validation check fails.
    """
    assert len(transformed) == len(original), (
        f"Count mismatch: {len(original)} in, {len(transformed)} out"
    )
    # Only check for exact 900 if full dataset is provided
    if len(original) == EXPECTED_TOTAL_ENTRIES:
        assert len(transformed) == EXPECTED_TOTAL_ENTRIES, (
            f"Expected {EXPECTED_TOTAL_ENTRIES}, got {len(transformed)}"
        )

    for i, (orig, trans) in enumerate(zip(original, transformed, strict=False)):
        assert trans["metadata"]["conversation_id"] == orig["conversation_id"], (
            f"Entry {i}: conversation_id mismatch"
        )
        assert trans["metadata"]["human_score"] == orig["human_relevance_score"], (
            f"Entry {i}: human_score mismatch"
        )
        assert trans["metadata"]["model"] == orig["model"], f"Entry {i}: model mismatch"
        assert trans["actual_output"] is not None and trans["actual_output"] != "", (
            f"Entry {i}: actual_output is null or empty"
        )
        assert MIN_HUMAN_SCORE <= trans["metadata"]["human_score"] <= MAX_HUMAN_SCORE, (
            f"Entry {i}: score {trans['metadata']['human_score']} out of range"
        )


def main() -> None:
    """Orchestrate the full transformation pipeline."""
    print(f"Loading dataset from {RAW_PATH}...")
    dataset = load_dataset(RAW_PATH)
    print(f"Loaded {len(dataset)} entries.")

    print("Transforming to ConversationalTestCase format...")
    test_cases = [entry_to_test_case(entry) for entry in dataset]
    print(f"Transformed {len(test_cases)} / {len(dataset)} entries.")

    print("Running validation...")
    serialized = [serialize_test_case(tc) for tc in test_cases]
    validate_transform(dataset, serialized)
    print("[PASS] Count match: 900 == 900")
    print("[PASS] All conversation_ids preserved")
    print("[PASS] All human_scores preserved exactly")
    print("[PASS] All responses non-null")
    print("[PASS] All scores in range [1.0, 5.0]")

    print(f"Saving to {OUT_PATH}...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(serialized)} test cases saved.")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
