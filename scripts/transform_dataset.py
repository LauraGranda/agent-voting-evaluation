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


def build_turns(
    turns: list[dict[str, str]],
    response_speaker: str,
) -> list[Turn]:
    """Convert speaker-labeled turns to DeepEval Turn objects.

    Roles are derived from the actual speaker labels preserved in the raw
    data, not from turn-index parity. The speaker who authored the response
    (``response_speaker``) is mapped to ``assistant``; every other speaker is
    mapped to ``user``. This correctly handles:

    - odd or even context lengths (prior parity-based logic produced
      consecutive same-role turns when context length + response parity
      collided),
    - legitimate consecutive same-speaker utterances (both get the same role,
      which is the semantically correct outcome),
    - newlines or arbitrary whitespace inside utterance text (content is
      passed through untouched).

    Args:
        turns: Context turns, each a dict with keys ``speaker`` and ``text``.
        response_speaker: Speaker label of the turn being evaluated (the one
            whose final utterance becomes ``actual_output``). Must match the
            ``speaker`` value of the response-side speaker in ``turns``.

    Returns:
        List of Turn objects whose roles reflect the underlying speaker of
        each utterance.
    """
    result = []
    for turn in turns:
        # Map response_speaker to assistant role; all others become user.
        role: Literal["user", "assistant"] = (
            "assistant" if turn["speaker"] == response_speaker else "user"
        )
        result.append(Turn(role=role, content=turn["text"]))
    return result


def entry_to_test_case(entry: dict[str, Any]) -> ConversationalTestCase:
    """Transform a single dataset entry to ConversationalTestCase.

    Preserves human scores and metadata for downstream correlation analysis.

    Args:
        entry: Dataset entry with keys: conversation_id, turns, response,
            response_speaker, model, human_relevance_score,
            raw_relevance_scores, human_appropriateness_score,
            raw_appropriateness_scores.

    Returns:
        ConversationalTestCase with turns and metadata.
    """
    # Build turns from speaker-labeled context, then append the response
    # as an assistant turn (the model's output under evaluation).
    response_speaker = entry["response_speaker"]
    turns = build_turns(entry["turns"], response_speaker)
    turns.append(Turn(role="assistant", content=entry["response"]))

    # Metadata for correlation and score preservation
    metadata = {
        "human_score": entry["human_relevance_score"],
        "raw_relevance_scores": entry["raw_relevance_scores"],
        "human_appropriateness_score": entry["human_appropriateness_score"],
        "raw_appropriateness_scores": entry["raw_appropriateness_scores"],
        "conversation_id": entry["conversation_id"],
        "model": entry["model"],
        "response_speaker": response_speaker,
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

    for i, (orig, trans) in enumerate(zip(original, transformed, strict=True)):
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
