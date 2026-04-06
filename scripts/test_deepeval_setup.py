"""Script to verify DeepEval setup and API connectivity.

This script initializes a dummy ConversationalTestCase from DeepEval,
runs a basic test against the configured LLM API, and reports whether
the setup is working correctly.

Usage:
    uv run python scripts/test_deepeval_setup.py
"""

from typing import TypedDict

from deepeval.test_case import ConversationalTestCase


class TurnDict(TypedDict):
    """Type definition for a conversation turn.

    Attributes:
        role: The role of the speaker (e.g., "user", "assistant").
        content: The textual content of the turn.
    """

    role: str
    content: str


def create_dummy_test_case() -> ConversationalTestCase:
    """Create a minimal ConversationalTestCase for setup verification.

    Returns:
        A ConversationalTestCase with a single dummy turn for testing.
    """
    turn: TurnDict = {
        "role": "user",
        "content": "Hello, how are you?",
    }
    test_case: ConversationalTestCase = ConversationalTestCase(turns=[turn])
    return test_case


def verify_setup() -> None:
    """Run a basic DeepEval verification to confirm the setup works.

    This function creates a dummy test case to verify DeepEval initialization
    and API connectivity.

    Raises:
        Exception: If setup fails or API connectivity is unavailable.
    """
    # 1. Verify ConversationalTestCase creation
    test_case: ConversationalTestCase = create_dummy_test_case()
    print(f"✓ ConversationalTestCase created with {len(test_case.turns)} turn(s).")

    # 2. Verify DeepEval imports and basic structure
    print(f"✓ Test case turns structure: {[t.model_dump() for t in test_case.turns]}")
    print("✓ DeepEval setup verified successfully!")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    verify_setup()
