"""Score aggregation for the agentic voting system.

Implements the voting scheme selected and justified in
``docs/voting_scheme_analysis.md``: arithmetic mean as the
primary aggregator over the panel of three judge agents, with the median
reported in parallel as a robustness check (exposed in ``metadata``).

The module is pure logic and standard-library only. It exposes a single
public entry point, :func:`aggregate`, which the panel runner (later HU)
calls once per conversation-response pair.

Notes
-----
The scheme's name in the thesis document is "media aritmética".
The module-level constant ``SCHEME_NAME`` uses the English snake_case
identifier ``"arithmetic_mean"`` as a portable code-side name; both refer
to the same operator.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any, Final

# ─── Module-level constants ──────────────────────────────────────────────
SCHEME_NAME: Final[str] = "arithmetic_mean"
SCORE_MIN: Final[int] = 1
SCORE_MAX: Final[int] = 5
AGREEMENT_HIGH_THRESHOLD: Final[float] = 0.5
AGREEMENT_MEDIUM_THRESHOLD: Final[float] = 1.0

# Maximum standard deviation theoretically achievable on the 1-5 scale,
# used as the denominator of the continuous agreement measure from
# docs/voting_scheme_analysis.md §4 (1 - std/std_max). Half the range is
# the dispersion of votes concentrated at the two extremes (e.g. {1, 5}).
_STD_MAX: Final[float] = (SCORE_MAX - SCORE_MIN) / 2.0

# Number of decimals every continuous score in the output is rounded to.
_SCORE_DECIMALS: Final[int] = 2

# Minimum sample size required to define std-based agreement.
_MIN_FOR_AGREEMENT: Final[int] = 2


# ─── Public API ──────────────────────────────────────────────────────────
def aggregate(scores: Mapping[str, float | None]) -> dict[str, Any]:
    """Aggregate per-agent scores into a final relevance score.

    Applies the scheme selected in ``docs/voting_scheme_analysis.md`` §4
    (arithmetic mean) and reports the median in parallel as a robustness
    check, plus an ordinal agreement level and a continuous agreement
    measure derived from the standard deviation.

    Parameters
    ----------
    scores
        Mapping ``{agent_name: score}`` where each score is either an
        integer/float in the closed interval [SCORE_MIN, SCORE_MAX] or
        ``None`` to signal that the agent did not produce a valid score
        (typically because its API call failed after retries).

    Returns
    -------
    dict
        Output schema, with the three keys required by the HU-07
        Definition of Done plus two extras kept aligned with the
        HU-05 contract:

        - ``final_score`` (float): arithmetic mean of the available
          scores, rounded to two decimals, in [SCORE_MIN, SCORE_MAX].
        - ``individual_scores`` (dict): the input dictionary echoed
          unchanged, including any ``None`` entries, for traceability.
        - ``agreement_level`` (str): ``"high"``, ``"medium"``, ``"low"``
          based on sample standard deviation, or ``"n/a"`` when fewer
          than two valid scores remain.
        - ``scheme_used`` (str): ``SCHEME_NAME``.
        - ``metadata`` (dict): ``n_agents``, ``std_deviation``,
          ``min_score``, ``max_score``, ``median_score`` (HU-05
          parallel-robustness check), ``agreement_continuous`` (HU-05
          continuous ``1 - std/std_max`` measure, ``None`` when not
          definable), and ``missing_agents`` listing the agent names
          whose score was ``None`` (key only present when there are
          missing agents).

    Raises
    ------
    ValueError
        If ``scores`` is empty, if it contains only ``None`` values
        (no valid score to aggregate), or if any non-``None`` score is
        outside [SCORE_MIN, SCORE_MAX]. The message identifies the
        offending agent when applicable.
    """
    valid_scores, missing_agents = validate_scores(scores)
    values: list[float] = list(valid_scores.values())

    final_score = _compute_final_score(values)
    median_score = round(float(statistics.median(values)), _SCORE_DECIMALS)
    std_dev = _stdev(values)
    agreement_level = compute_agreement(values)
    agreement_continuous = _agreement_continuous(values, std_dev)

    metadata: dict[str, Any] = {
        "n_agents": len(values),
        "std_deviation": std_dev,
        "min_score": float(min(values)),
        "max_score": float(max(values)),
        "median_score": median_score,
        "agreement_continuous": agreement_continuous,
    }
    if missing_agents:
        metadata["missing_agents"] = missing_agents

    return {
        "final_score": final_score,
        "individual_scores": dict(scores),
        "agreement_level": agreement_level,
        "scheme_used": SCHEME_NAME,
        "metadata": metadata,
    }


def validate_scores(
    scores: Mapping[str, float | None],
) -> tuple[dict[str, float], list[str]]:
    """Validate the input and return the cleaned scores plus missing names.

    The cleaned dict drops any agent whose score is ``None``. Their names
    are collected in the second return value so the caller can record them
    in ``metadata.missing_agents``. After dropping, every remaining score
    must lie in [SCORE_MIN, SCORE_MAX]; otherwise a :class:`ValueError` is
    raised identifying the agent.

    Parameters
    ----------
    scores
        The raw ``{agent_name: score | None}`` mapping passed to
        :func:`aggregate`.

    Returns
    -------
    tuple
        ``(cleaned, missing_agents)`` where ``cleaned`` is the dict of
        valid scores and ``missing_agents`` is a list of agent names
        whose score was ``None``.

    Raises
    ------
    ValueError
        If the input is empty, or contains only ``None`` values, or has
        an out-of-range score.
    """
    if not scores:
        raise ValueError("No agent scores provided")

    cleaned: dict[str, float] = {}
    missing_agents: list[str] = []
    for name, value in scores.items():
        if value is None:
            missing_agents.append(name)
            continue
        if value < SCORE_MIN or value > SCORE_MAX:
            raise ValueError(
                f"Score for agent '{name}' is {value}, must be in [{SCORE_MIN}, {SCORE_MAX}]"
            )
        cleaned[name] = float(value)

    if not cleaned:
        raise ValueError("No agent scores provided")

    return cleaned, missing_agents


def compute_agreement(values: list[float]) -> str:
    """Return the ordinal agreement level for the given valid scores.

    With fewer than two scores the std-based agreement is undefined and
    the function returns ``"n/a"``. Otherwise it maps sample standard
    deviation onto the three thresholds from the HU-07 spec
    (``AGREEMENT_HIGH_THRESHOLD``, ``AGREEMENT_MEDIUM_THRESHOLD``).

    Parameters
    ----------
    values
        Validated scores (already filtered, in range).

    Returns
    -------
    str
        ``"high"``, ``"medium"``, ``"low"`` or ``"n/a"``.
    """
    if len(values) < _MIN_FOR_AGREEMENT:
        return "n/a"
    std = statistics.stdev(values)
    if std <= AGREEMENT_HIGH_THRESHOLD:
        return "high"
    if std <= AGREEMENT_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


# ─── Private helpers ─────────────────────────────────────────────────────
def _compute_final_score(values: list[float]) -> float:
    """Apply the selected voting scheme (arithmetic mean) and round.

    See ``docs/voting_scheme_analysis.md`` §4: the arithmetic mean of the
    available judge scores is the primary aggregator. The result is
    rounded to two decimals and cast to ``float`` so the output is
    always a continuous-looking number, not an ``int`` even on whole
    inputs.
    """
    return round(float(statistics.fmean(values)), _SCORE_DECIMALS)


def _stdev(values: list[float]) -> float:
    """Sample standard deviation, or 0.0 when fewer than two values."""
    if len(values) < _MIN_FOR_AGREEMENT:
        return 0.0
    return round(float(statistics.stdev(values)), _SCORE_DECIMALS + 2)


def _agreement_continuous(values: list[float], std_dev: float) -> float | None:
    """Continuous agreement measure ``1 - std/std_max`` clamped to [0, 1].

    Returns ``None`` when fewer than two valid scores are available, in
    which case the measure is not defined. The clamp guards against
    rounding edge cases where ``std_dev`` exceeds ``_STD_MAX``.
    """
    if len(values) < _MIN_FOR_AGREEMENT:
        return None
    raw = 1.0 - std_dev / _STD_MAX
    bounded = max(0.0, min(1.0, raw))
    return round(bounded, _SCORE_DECIMALS + 2)
