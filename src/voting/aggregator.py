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

import math
import statistics
from collections.abc import Mapping
from typing import Any, Final

# ─── Module-level constants ──────────────────────────────────────────────
SCHEME_NAME: Final[str] = "arithmetic_mean"
SCORE_MIN: Final[int] = 1
SCORE_MAX: Final[int] = 5
AGREEMENT_HIGH_THRESHOLD: Final[float] = 0.5
AGREEMENT_MEDIUM_THRESHOLD: Final[float] = 1.0

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
        (no valid score to aggregate), if any non-``None`` score cannot
        be coerced to a real number, if any non-``None`` score is
        non-finite (NaN or ±inf), or if any non-``None`` score is outside
        [SCORE_MIN, SCORE_MAX]. The message identifies the offending
        agent when applicable.
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
        If the input is empty, if it contains only ``None`` values, if any
        score cannot be coerced to a real number, if any score is non-finite
        (NaN or ±inf), or if any score is out of the closed interval
        [SCORE_MIN, SCORE_MAX]. The message identifies the offending agent.
    """
    if not scores:
        raise ValueError("No agent scores provided")

    cleaned: dict[str, float] = {}
    missing_agents: list[str] = []
    for name, value in scores.items():
        if value is None:
            missing_agents.append(name)
            continue
        # Runtime input isn't guaranteed to match the static type hints (the
        # function may be reached from a config-driven runner). Coerce to
        # float first so the downstream finiteness and range checks operate
        # on a guaranteed real number; if coercion fails the agent name is
        # surfaced in the error message instead of a bare TypeError.
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Score for agent '{name}' is not a number ({value!r})") from exc
        # NaN and ±inf would slip past the range check below because every
        # numeric comparison with NaN is False, so reject them up-front with
        # a clear message before any downstream statistics call sees them.
        if not math.isfinite(coerced):
            raise ValueError(f"Score for agent '{name}' is not finite ({coerced!r})")
        if coerced < SCORE_MIN or coerced > SCORE_MAX:
            raise ValueError(
                f"Score for agent '{name}' is {coerced}, must be in [{SCORE_MIN}, {SCORE_MAX}]"
            )
        cleaned[name] = coerced

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


def _std_max(n: int) -> float:
    """Maximum sample standard deviation attainable for ``n`` votes in [SCORE_MIN, SCORE_MAX].

    With ``statistics.stdev`` (sample, Bessel-corrected), the dispersion is
    maximized when the votes split evenly at the two extremes of the score
    range, which yields ``(R / 2) * sqrt(n / (n - 1))`` where ``R`` is the
    range. This is the tight bound for even ``n``; for odd ``n`` it is a
    very tight upper bound that keeps ``1 - std / std_max`` inside [0, 1].
    Using a denominator that depends on ``n`` is necessary because the
    range-based constant ``R / 2`` under-estimates the achievable sample
    stdev and would push the continuous agreement below zero.
    """
    return (SCORE_MAX - SCORE_MIN) / 2.0 * math.sqrt(n / (n - 1))


def _agreement_continuous(values: list[float], std_dev: float) -> float | None:
    """Continuous agreement measure ``1 - std / std_max(n)`` clamped to [0, 1].

    Returns ``None`` when fewer than two valid scores are available, in
    which case the measure is not defined. The denominator is sized to the
    sample stdev definition used in ``compute_agreement`` (see
    :func:`_std_max`), so the clamp is only a defensive safeguard against
    floating-point edges and never fires in practice for valid inputs.
    """
    if len(values) < _MIN_FOR_AGREEMENT:
        return None
    raw = 1.0 - std_dev / _std_max(len(values))
    bounded = max(0.0, min(1.0, raw))
    return round(bounded, _SCORE_DECIMALS + 2)
