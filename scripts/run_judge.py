"""Single-call judge runner for the agentic voting system (HU-08).

Reusable module that asks one judge agent to score one conversation. It is
provider-agnostic: the dispatch picks between OpenAI, Google (via the new
``google-genai`` SDK) and Anthropic based on the ``provider`` field of the
agent YAML. The same V3 relevance prompt is used by the three judges, in
line with the methodological control fixed in HU-06.

The pilot notebook (``notebooks/03_voting_pilot.ipynb``) calls
:func:`call_agent` 60 times (20 conversations, 3 judges each). The
full-dataset runner in a later HU will reuse this exact function and
only add parallelism and retry-with-backoff around it.

Score parsing follows the strategy agreed with the user for HU-08: a
permissive regex on the first response, and one retry with a format-suffix
prompt if parsing fails. The V3 prompt itself is never modified.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Final

from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI

# ─── Per-provider list pricing, USD per million tokens ──────────────────
# Same source as the cost table in docs/agent_panel_design.md §10.
_PRICES: Final[dict[str, dict[str, float]]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

# Regex that matches both the V3 natural form ``Score = 4.`` and the
# spec's literal form ``SCORE: 4``. Whichever the model produces, we
# capture the trailing integer in [1, 5].
_SCORE_RE: Final[re.Pattern[str]] = re.compile(r"(?:SCORE|Score|score)\s*[=:]\s*([1-5])")

# Sentinel appended to the prompt on retry so the model emits an
# explicit, parseable last line. The V3 prompt itself is not modified.
_FORMAT_SUFFIX: Final[str] = "\n\nEnd your response with: SCORE: <integer 1-5>"


# ─── Prompt assembly and parsing (pure helpers) ─────────────────────────
@cache
def _load_prompt(prompt_file: str) -> str:
    """Read the prompt template from disk once and cache it for subsequent calls.

    The full panel run will invoke :func:`call_agent` ~2700 times (900 pairs
    times 3 judges), all pointing to the same V3 prompt. Reading the file on
    every invocation adds non-trivial disk I/O for no gain; the prompt is
    part of the repo and never changes mid-run, so an unbounded function
    cache keyed by the file path string is safe.
    """
    return Path(prompt_file).read_text(encoding="utf-8")


def _build_full_prompt(
    prompt_text: str,
    conversation_input: str,
    actual_output: str,
) -> str:
    """Wrap the conversation around the V3 prompt as the spec requires."""
    return (
        f"{prompt_text}\n\n"
        f"CONVERSATION HISTORY:\n{conversation_input}\n\n"
        f"RESPONSE TO EVALUATE:\n{actual_output}"
    )


def _parse_score(response_text: str) -> int | None:
    """Extract the model's final 1-5 score from the response text.

    The V3 prompt anchors three worked examples ending in ``Score = N.``,
    so a model that quotes or paraphrases the rubric mid-reasoning can
    leave several score-like tokens in its output. The verdict is always
    the **last** one (the conclusion follows after any reference to the
    anchoring examples), so we pick the last match rather than the first.
    Returns ``None`` when no token in ``[1, 5]`` is found at all.
    """
    matches = _SCORE_RE.findall(response_text)
    return int(matches[-1]) if matches else None


def _compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Map token counts to USD using the per-provider list pricing table."""
    price = _PRICES.get(model, {"input": 0.0, "output": 0.0})
    return tokens_in / 1_000_000 * price["input"] + tokens_out / 1_000_000 * price["output"]


# ─── Cached SDK clients ─────────────────────────────────────────────────
# A fresh client per call would re-do TLS, auth and connection-pool setup
# 2700 times during the full 900-pair run, and even more when the runner
# adds parallelism. ``functools.cache`` keyed by the API key keeps one
# client per credential alive for the lifetime of the process, so threads
# share the same connection pool.
@cache
def _openai_client(api_key: str) -> OpenAI:
    """Return a process-wide singleton OpenAI client for ``api_key``."""
    return OpenAI(api_key=api_key)


@cache
def _google_client(api_key: str) -> genai.Client:
    """Return a process-wide singleton Google client for ``api_key``."""
    return genai.Client(api_key=api_key)


@cache
def _anthropic_client(api_key: str) -> Anthropic:
    """Return a process-wide singleton Anthropic client for ``api_key``."""
    return Anthropic(api_key=api_key)


# ─── Provider-specific call wrappers ────────────────────────────────────
def _call_openai(
    model: str, prompt: str, temperature: float, max_tokens: int, api_key: str
) -> tuple[str, int, int]:
    """Issue one chat-completion call to OpenAI; return (text, in_tokens, out_tokens)."""
    client = _openai_client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    if usage is None:
        return text, 0, 0
    return text, usage.prompt_tokens, usage.completion_tokens


def _call_google(
    model: str, prompt: str, temperature: float, max_tokens: int, api_key: str
) -> tuple[str, int, int]:
    """Issue one generate_content call to Google; thinking is disabled for determinism.

    ``thinking_budget=0`` enforces the comparability decision from
    ``docs/agent_panel_design.md`` §4.2: the Gemini 2.5 Flash judge must
    not use extended reasoning so it stays directly comparable with the
    other two judges (which do not have a thinking mode at all).
    """
    client = _google_client(api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = response.text or ""
    usage = response.usage_metadata
    if usage is None:
        return text, 0, 0
    tokens_in = usage.prompt_token_count or 0
    tokens_out = usage.candidates_token_count or 0
    return text, tokens_in, tokens_out


def _call_anthropic(
    model: str, prompt: str, temperature: float, max_tokens: int, api_key: str
) -> tuple[str, int, int]:
    """Issue one messages.create call to Anthropic; return (text, in_tokens, out_tokens)."""
    client = _anthropic_client(api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    # ``response.content`` is a list of blocks; for plain text prompts the
    # first block is a TextBlock. Fall back to an empty string if the
    # block has no ``text`` attribute (defensive only).
    first_block = response.content[0] if response.content else None
    text = getattr(first_block, "text", "") or ""
    # Mirror the OpenAI/Google wrappers: if the SDK ever returns a
    # response without usage metadata (partial response, future schema
    # change), fall back to zero so cost reporting degrades gracefully
    # instead of raising an AttributeError mid-pilot.
    usage = response.usage
    if usage is None:
        return text, 0, 0
    return text, usage.input_tokens, usage.output_tokens


# Each caller takes (model, prompt, temperature, max_tokens, api_key) and
# returns (text, tokens_in, tokens_out). Typing the registry with the exact
# Callable shape (instead of `Any`) lets mypy reject any future provider
# implementation whose signature drifts from this contract.
_CallerFn = Callable[[str, str, float, int, str], tuple[str, int, int]]

_CALLERS: Final[dict[str, _CallerFn]] = {
    "openai": _call_openai,
    "google": _call_google,
    "anthropic": _call_anthropic,
}


# ─── Public entry point ─────────────────────────────────────────────────
def call_agent(
    agent_config: dict[str, Any],
    conversation_input: str,
    actual_output: str,
) -> dict[str, Any]:
    """Run one judge agent on one conversation-response pair.

    Reads the prompt and connection parameters from ``agent_config`` (the
    YAML loaded by the notebook), wraps the conversation in the
    ``CONVERSATION HISTORY:`` / ``RESPONSE TO EVALUATE:`` blocks, calls
    the appropriate provider SDK with the temperature and token cap from
    the YAML, parses the score with the permissive regex and, if needed,
    retries once with an explicit format suffix.

    Parameters
    ----------
    agent_config
        Parsed YAML of the form produced by ``configs/agents/agent_*.yaml``.
        Required fields: ``name``, ``model``, ``provider``, ``api_key_env``,
        ``temperature``, ``max_tokens``, ``prompt_file``.
    conversation_input
        Conversation history already formatted as one line per turn,
        ``[Turn N] User: ...`` for user turns and ``[Turn N] Assistant: ...``
        for assistant turns, joined by newlines. The notebook produces
        exactly this format from ``entry["turns"][:-1]``; the final
        assistant turn (which equals ``actual_output``) is excluded.
    actual_output
        The response under evaluation, exactly as it appears in the
        dataset.

    Returns
    -------
    dict
        Result schema, one row per agent call:

        - ``agent`` (str): the YAML ``name``.
        - ``model`` (str): the YAML ``model``.
        - ``score`` (int | None): parsed integer 1-5, or ``None`` when
          both the first attempt and the retry failed to produce a
          parseable line.
        - ``reasoning`` (str): the full model response. If a retry was
          issued, the first response, a marker line and the second
          response are concatenated in that order.
        - ``tokens_used`` (int): total tokens charged (input + output,
          summed across the original call and the retry if any).
        - ``tokens_in`` (int), ``tokens_out`` (int): broken-down counts.
        - ``cost_usd`` (float): list-price cost in USD, six decimals.
        - ``timestamp`` (str): ISO-8601 UTC.
        - ``retry_used`` (bool): whether the format-suffix retry fired.
        - ``error`` (str | None): ``"score not parseable"`` when score
          is None, ``None`` otherwise.
    """
    provider = agent_config["provider"]
    if provider not in _CALLERS:
        raise ValueError(f"Unknown provider {provider!r} in agent config")

    api_key = os.getenv(agent_config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key {agent_config['api_key_env']} not set in environment")

    prompt_text = _load_prompt(agent_config["prompt_file"])
    full_prompt = _build_full_prompt(prompt_text, conversation_input, actual_output)

    caller = _CALLERS[provider]
    model = agent_config["model"]
    temperature = agent_config["temperature"]
    max_tokens = agent_config["max_tokens"]

    text, tokens_in, tokens_out = caller(model, full_prompt, temperature, max_tokens, api_key)
    score = _parse_score(text)
    retry_used = False

    # Single retry with an explicit format suffix when the first parse
    # fails. The V3 prompt is not modified; only the suffix is added.
    if score is None:
        retry_used = True
        retry_prompt = full_prompt + _FORMAT_SUFFIX
        text2, tokens_in2, tokens_out2 = caller(
            model, retry_prompt, temperature, max_tokens, api_key
        )
        score = _parse_score(text2)
        text = f"{text}\n\n[RETRY]\n{text2}"
        tokens_in += tokens_in2
        tokens_out += tokens_out2

    cost = _compute_cost(model, tokens_in, tokens_out)

    return {
        "agent": agent_config["name"],
        "model": model,
        "score": score,
        "reasoning": text,
        "tokens_used": tokens_in + tokens_out,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost, 6),
        "timestamp": datetime.now(UTC).isoformat(),
        "retry_used": retry_used,
        "error": "score not parseable" if score is None else None,
    }
