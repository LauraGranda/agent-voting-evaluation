"""Test that core dependencies can be imported successfully.

This minimal test ensures that all production dependencies are installed
and can be imported without errors.
"""

import pandas as pd
import scipy
import yaml  # type: ignore[import-untyped]
from anthropic import Anthropic
from deepeval.test_case import ConversationalTestCase
from openai import OpenAI


def test_deepeval_import() -> None:
    """Verify that deepeval can be imported.

    This test ensures ConversationalTestCase is available from deepeval.test_case.
    """
    assert ConversationalTestCase is not None


def test_openai_import() -> None:
    """Verify that openai can be imported.

    This test ensures OpenAI client is available from openai package.
    """
    assert OpenAI is not None


def test_anthropic_import() -> None:
    """Verify that anthropic can be imported.

    This test ensures Anthropic client is available from anthropic package.
    """
    assert Anthropic is not None


def test_pandas_import() -> None:
    """Verify that pandas can be imported.

    This test ensures pandas data manipulation library is available.
    """
    assert pd is not None


def test_scipy_import() -> None:
    """Verify that scipy can be imported.

    This test ensures scipy scientific computing library is available.
    """
    assert scipy is not None


def test_yaml_import() -> None:
    """Verify that pyyaml can be imported.

    This test ensures yaml file processing library is available.
    """
    assert yaml is not None
