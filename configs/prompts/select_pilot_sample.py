"""Select a stratified 20-entry pilot sample from the processed DailyDialog-Zhao dataset.

The pilot covers five strata designed to stress-test a relevance evaluator:
    1. Human-written references (high relevance).
    2. Negative control samples (low relevance).
    3. IA-generated responses with high human relevance (>= 4.0).
    4. IA-generated responses with mid-range relevance (2.5 - 3.5).
    5. IA-generated responses with low relevance (<= 2.0).

The script is deterministic (seed=42) and writes to
``configs/prompts/pilot_sample.json``.

Speaker labels in the output (``metadata.response_speaker``, "A" or "B") are
the raw anonymous interlocutor identifiers inherited from the DailyDialog-Zhao
corpus — not gender or persona markers. See ``data/README.md`` for the full
schema description.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

DATA_PATH: Final[Path] = Path("data/processed/deepeval_test_cases.json")
OUT_PATH: Final[Path] = Path("configs/prompts/pilot_sample.json")
SEED: Final[int] = 42
IA_FAMILIES: Final[tuple[str, ...]] = ("GPT2", "S2S", "HRED", "VHRED")
PILOT_SIZE: Final[int] = 20
PER_STRATUM: Final[int] = 4


def load_processed_dataset(path: Path) -> list[dict[str, Any]]:
    """Load the DeepEval-format dataset from JSON.

    Args:
        path: Path to the processed JSON file.

    Returns:
        List of entry dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def get_model_family(model_name: str) -> str:
    """Map a dataset ``model`` value to one of the canonical families.

    The DailyDialog-Zhao corpus enumerates 16 IA variants (e.g. ``GPT2_small``,
    ``HRED_attn``, ``S2S_attn``, ``VHRED_attn``). Any variant is collapsed onto
    its family prefix. ``ground-truth`` and ``negative-sample`` are returned
    unchanged so the same helper can be used for all strata.

    Args:
        model_name: Raw ``metadata.model`` value.

    Returns:
        ``"GPT2"``, ``"S2S"``, ``"HRED"``, ``"VHRED"``, ``"ground-truth"``,
        ``"negative-sample"``, or the original string if nothing matches.
    """
    if model_name in ("ground-truth", "negative-sample"):
        return model_name
    for family in IA_FAMILIES:
        if model_name.startswith(family):
            return family
    return model_name


def _sorted_take(
    entries: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], float],
    n: int,
    reverse: bool,
) -> list[dict[str, Any]]:
    """Stable sort then take the top-n (tie-break by conversation_id)."""
    return sorted(
        entries,
        key=lambda e: (key(e), e["metadata"]["conversation_id"]),
        reverse=reverse,
    )[:n]


def select_evenly_across_families(
    entries: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    stratum_label: str,
) -> list[dict[str, Any]]:
    """Pick ``n`` entries, trying to take one per IA family.

    If a family has no candidate entries in the filtered pool, the slot is
    redistributed among the other families via random draws and a warning is
    printed naming the empty family.

    Args:
        entries: Already score-filtered IA entries.
        n: Number of entries to return.
        rng: Seeded ``random.Random`` for determinism.
        stratum_label: Human-readable label for warning messages.

    Returns:
        Exactly ``n`` entries (may contain duplicates of a family when another
        family is empty; never duplicates an individual entry).
    """
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_family[get_model_family(e["metadata"]["model"])].append(e)

    picked: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for family in IA_FAMILIES:
        pool = by_family.get(family, [])
        if not pool:
            print(
                f"  WARNING [{stratum_label}]: family '{family}' has no entries; "
                f"backfilling slot from other families."
            )
            continue
        pool_sorted = sorted(pool, key=lambda e: e["metadata"]["conversation_id"])
        choice_index = rng.randrange(len(pool_sorted))
        choice = pool_sorted[choice_index]
        picked.append(choice)
        leftovers.extend(pool_sorted[:choice_index] + pool_sorted[choice_index + 1 :])

    while len(picked) < n and leftovers:
        rng.shuffle(leftovers)
        picked.append(leftovers.pop())

    return picked[:n]


def select_stratum_1(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top 4 ground-truth entries by descending human_score."""
    pool = [e for e in dataset if e["metadata"]["model"] == "ground-truth"]
    return _sorted_take(
        pool, key=lambda e: e["metadata"]["human_score"], n=PER_STRATUM, reverse=True
    )


def select_stratum_2(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bottom 4 negative-sample entries by ascending human_score."""
    pool = [e for e in dataset if e["metadata"]["model"] == "negative-sample"]
    return _sorted_take(
        pool, key=lambda e: e["metadata"]["human_score"], n=PER_STRATUM, reverse=False
    )


def _ia_pool(
    dataset: list[dict[str, Any]],
    score_filter: Callable[[float], bool],
) -> list[dict[str, Any]]:
    return [
        e
        for e in dataset
        if e["metadata"]["model"] not in ("ground-truth", "negative-sample")
        and score_filter(e["metadata"]["human_score"])
    ]


def select_stratum_3(dataset: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """IA responses with human_score >= 4.0, sampled across families."""
    pool = _ia_pool(dataset, lambda s: s >= 4.0)
    return select_evenly_across_families(pool, PER_STRATUM, rng, "Stratum 3")


def select_stratum_4(dataset: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """IA responses with 2.5 <= human_score <= 3.5, sampled across families."""
    pool = _ia_pool(dataset, lambda s: 2.5 <= s <= 3.5)
    return select_evenly_across_families(pool, PER_STRATUM, rng, "Stratum 4")


def select_stratum_5(dataset: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """IA responses with human_score <= 2.0, sampled across families."""
    pool = _ia_pool(dataset, lambda s: s <= 2.0)
    return select_evenly_across_families(pool, PER_STRATUM, rng, "Stratum 5")


def select_pilot_sample(dataset: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """Run all five strata and tag each returned entry with ``stratum``.

    Args:
        dataset: Full processed dataset.
        seed: Random seed applied before any sampling.

    Returns:
        List of exactly ``PILOT_SIZE`` entries; each entry is a shallow copy of
        the source with an added ``"stratum"`` integer field in ``[1, 5]``.
    """
    random.seed(seed)
    rng = random.Random(seed)

    strata: list[list[dict[str, Any]]] = [
        select_stratum_1(dataset),
        select_stratum_2(dataset),
        select_stratum_3(dataset, rng),
        select_stratum_4(dataset, rng),
        select_stratum_5(dataset, rng),
    ]

    tagged: list[dict[str, Any]] = []
    for idx, stratum in enumerate(strata, start=1):
        for entry in stratum:
            tagged.append({**entry, "stratum": idx})
    return tagged


def verify_sample(sample: list[dict[str, Any]]) -> None:
    """Assert stratum counts and print a human-readable verification table.

    Args:
        sample: Tagged pilot sample.

    Raises:
        AssertionError: If counts do not match expectations.
    """
    assert len(sample) == PILOT_SIZE, f"Expected {PILOT_SIZE} entries, got {len(sample)}"

    by_stratum: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in sample:
        by_stratum[e["stratum"]].append(e)

    print()
    print("Stratum | Count | Score range | Models included")
    print("--------|-------|-------------|--------------------------------------------")
    total_scores: list[float] = []
    for s in sorted(by_stratum):
        entries = by_stratum[s]
        scores = [e["metadata"]["human_score"] for e in entries]
        total_scores.extend(scores)
        models = sorted({e["metadata"]["model"] for e in entries})
        print(
            f"{s:^7} | {len(entries):^5} | {min(scores):.1f} - {max(scores):.1f}   | "
            f"{', '.join(models)}"
        )
        assert len(entries) == PER_STRATUM, f"Stratum {s} has {len(entries)} entries"

    print(f"TOTAL   | {len(sample):^5} | {min(total_scores):.1f} - {max(total_scores):.1f}   |")
    print()


def main() -> None:
    """Orchestrate loading, sampling, verification, and saving."""
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_processed_dataset(DATA_PATH)
    print(f"Loaded {len(dataset)} entries.")

    print(f"Selecting pilot sample (seed={SEED})...")
    sample = select_pilot_sample(dataset, seed=SEED)

    verify_sample(sample)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sample)} entries to {OUT_PATH}")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
