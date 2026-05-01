"""Download and process DailyDialog-Zhao dataset from Zenodo.

This script downloads the annotated dialogue dataset from Zenodo
(https://zenodo.org/record/3828180), extracts the annotations, and
produces a clean JSON file suitable for relevance evaluation experiments.

Paper: Zhao et al. (2020). "Designing Precise and Robust Dialogue Response
Evaluators." Proceedings of ACL 2020, pp. 26-33.
https://aclanthology.org/2020.acl-main.4/
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import requests  # type: ignore[import-untyped]

# Project constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dailydialog_zhao"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "dailydialog_zhao"
OUTPUT_FILE = RAW_DIR / "dataset.json"
README_PATH = PROJECT_ROOT / "data" / "README.md"
ZIP_PATH = RAW_DIR / "ACL2020_data.zip"

# Expected numbers from the paper
EXPECTED_DIALOGUES = 100
EXPECTED_RESPONSES_PER_DIALOGUE = 9
EXPECTED_TOTAL_PAIRS = 900
EXPECTED_ANNOTATORS = 4
EXPECTED_SCALE_MIN = 1
EXPECTED_SCALE_MAX = 5

# Zenodo download URLs
ZENODO_URLS = [
    "https://zenodo.org/api/records/3828180/files/ACL2020_data.zip/content",
    "https://zenodo.org/records/3828180/files/ACL2020_data.zip?download=1",
]

# Size and format constants
MIN_ZIP_SIZE = 50000  # bytes
DATASET_SCHEMA_FIELDS = 9
SCORE_RANGE_MIN = 1
SCORE_RANGE_MAX = 5


def download_zip(dest_dir: Path) -> Path:
    """Download ACL2020_data.zip from Zenodo with fallback URLs.

    Args:
        dest_dir: Directory where zip will be saved.

    Returns:
        Path to the downloaded zip file.

    Raises:
        RuntimeError: If all download attempts fail.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_file = dest_dir / "ACL2020_data.zip"

    # Check if already exists and is valid
    if zip_file.exists() and zip_file.stat().st_size > MIN_ZIP_SIZE:
        print(f"✓ Zip already exists: {zip_file} ({zip_file.stat().st_size / 1024:.1f} KB)")
        return zip_file

    print("Downloading ACL2020_data.zip from Zenodo...")
    tmp_file = zip_file.with_suffix(".zip.tmp")

    for url_idx, url in enumerate(ZENODO_URLS, 1):
        try:
            print(f"  Attempt {url_idx}/{len(ZENODO_URLS)}: {url}")
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()

            with open(tmp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify it's a valid zip
            if not zipfile.is_zipfile(tmp_file):
                tmp_file.unlink()
                raise ValueError("Downloaded file is not a valid zip")  # noqa: TRY301

            # Atomic rename
            tmp_file.replace(zip_file)
            size_kb = zip_file.stat().st_size / 1024
            print(f"✓ Downloaded: {zip_file} ({size_kb:.1f} KB)")
            return zip_file  # noqa: TRY300

        except (requests.RequestException, ValueError, zipfile.BadZipFile) as e:
            if tmp_file.exists():
                tmp_file.unlink()
            if url_idx == len(ZENODO_URLS):
                raise RuntimeError(
                    f"Failed to download from all {len(ZENODO_URLS)} URLs. "
                    f"Last error: {type(e).__name__}: {e}"
                ) from e
            print(f"    Failed: {e}")

    raise RuntimeError("Unexpected error in download_zip")


def extract_and_find_json(zip_path: Path) -> Path:
    """Extract zip and locate dd_annotations.json.

    Args:
        zip_path: Path to the zip file.

    Returns:
        Path to the extracted dd_annotations.json file.

    Raises:
        FileNotFoundError: If dd_annotations.json is not found.
    """
    print(f"Extracting {zip_path.name}...")

    extract_dir = zip_path.parent / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        # Only extract dd_annotations.json to avoid clutter
        for member in z.namelist():
            if "dd_annotations.json" in member:
                z.extract(member, extract_dir)
                extracted_path = extract_dir / member
                print(f"✓ Extracted: {extracted_path}")
                return extracted_path

    raise FileNotFoundError(
        f"dd_annotations.json not found in {zip_path}. Contents: {z.namelist()}"
    )


def parse_annotations(json_path: Path) -> list[dict[str, Any]]:
    """Parse raw JSON annotations into clean dataset entries.

    Loads the annotations, extracts relevance and appropriateness (content)
    scores for each context-response pair, and creates standardized entries.

    Speaker labels from the raw data (``[speaker, text]`` tuples in ``context``
    and ``reference``) are preserved so downstream consumers can assign correct
    conversational roles without guessing by turn-index parity. The
    ``response_speaker`` field identifies which speaker authored the response,
    enabling the transformer to mark that speaker's utterances as ``assistant``
    and the other speaker's utterances as ``user`` regardless of context length
    or consecutive same-speaker turns.

    Args:
        json_path: Path to dd_annotations.json.

    Returns:
        List of dictionaries, one per response, with schema:
        {
            "conversation_id": "conv_{index}_{model}",
            "turns": [{"speaker": "A" | "B", "text": "..."}, ...],
            "response": "response text",
            "response_speaker": "A" | "B",
            "model": "model name",
            "human_relevance_score": float,
            "raw_relevance_scores": [int, ...],
            "human_appropriateness_score": float,
            "raw_appropriateness_scores": [int, ...]
        }
    """
    print(f"Parsing annotations from {json_path.name}...")

    with open(json_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = []

    for dialogue_idx, (_dialog_id, dialog_data) in enumerate(raw_data.items()):
        # Preserve speaker labels alongside text — discarding them here is what
        # caused the downstream role-assignment bug that produced consecutive
        # same-role turns in the DeepEval output.
        context = dialog_data.get("context", [])
        turns = [{"speaker": speaker, "text": text} for speaker, text in context]

        # The reference field identifies the speaker who authored the response.
        # In DailyDialog-Zhao this is always the speaker opposite to the last
        # context turn, but we read it from the data rather than inferring.
        reference = dialog_data.get("reference", ["", ""])
        response_speaker = reference[0] if reference else ""

        # Iterate over responses (models)
        for model_name, model_data in dialog_data.get("responses", {}).items():
            response_text = model_data.get("uttr", "")
            scores_dict = model_data.get("scores", {})

            # Extract individual annotator scores
            relevance_scores = []
            appropriateness_scores = []  # "content" maps to appropriateness

            for _worker_key, worker_scores in scores_dict.items():
                relevance = worker_scores.get("relevance")
                content = worker_scores.get("content")

                if relevance is not None:
                    relevance_scores.append(relevance)
                if content is not None:
                    appropriateness_scores.append(content)

            # Compute means
            human_relevance = mean(relevance_scores) if relevance_scores else 0.0
            human_appropriateness = mean(appropriateness_scores) if appropriateness_scores else 0.0

            # Build entry
            entry = {
                "conversation_id": f"conv_{dialogue_idx}_{model_name}",
                "turns": turns,
                "response": response_text,
                "response_speaker": response_speaker,
                "model": model_name,
                "human_relevance_score": human_relevance,
                "raw_relevance_scores": relevance_scores,
                "human_appropriateness_score": human_appropriateness,
                "raw_appropriateness_scores": appropriateness_scores,
            }
            dataset.append(entry)

    print(f"✓ Parsed {len(dataset)} entries")
    return dataset


def run_integrity_checks(dataset: list[dict[str, Any]]) -> None:
    """Validate dataset against expected paper values.

    Prints PASS/FAIL for each check. Raises SystemExit if any check fails.

    Args:
        dataset: List of parsed dataset entries.

    Raises:
        SystemExit: If any integrity check fails.
    """
    print("\n=== INTEGRITY CHECKS ===")

    checks = {
        "total_pairs": (len(dataset), EXPECTED_TOTAL_PAIRS),
        "annotation_scale_min": (
            min(
                (score for entry in dataset for score in entry["raw_relevance_scores"]),
                default=EXPECTED_SCALE_MIN,
            ),
            EXPECTED_SCALE_MIN,
        ),
        "annotation_scale_max": (
            max(
                (score for entry in dataset for score in entry["raw_relevance_scores"]),
                default=EXPECTED_SCALE_MAX,
            ),
            EXPECTED_SCALE_MAX,
        ),
    }

    # Unique conversations
    unique_conv_indices = set()
    for entry in dataset:
        conv_id = entry["conversation_id"]
        idx = conv_id.split("_")[1]
        unique_conv_indices.add(idx)

    checks["unique_conversations"] = (len(unique_conv_indices), EXPECTED_DIALOGUES)

    # Average annotators per response
    avg_annotators = mean(
        len(entry["raw_relevance_scores"]) for entry in dataset if entry["raw_relevance_scores"]
    )
    checks["annotators_per_pair"] = (round(avg_annotators), EXPECTED_ANNOTATORS)

    all_passed = True
    for check_name, (actual, expected) in checks.items():
        status = "PASS" if actual == expected else "FAIL"
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {check_name}: {actual} (expected {expected})")
        if status == "FAIL":
            all_passed = False

    if not all_passed:
        print("\n✗ Some integrity checks failed. Aborting.")
        raise SystemExit(1)

    print("✓ All integrity checks passed\n")


def print_summary(dataset: list[dict[str, Any]]) -> None:
    """Print dataset statistics summary.

    Args:
        dataset: List of parsed dataset entries.
    """
    print("=== DATASET SUMMARY ===")
    print(f"Total pairs: {len(dataset)}")

    # Unique models
    models = set(entry["model"] for entry in dataset)
    print(f"Unique models: {len(models)}")
    for model in sorted(models):
        count = sum(1 for entry in dataset if entry["model"] == model)
        print(f"  - {count}x {model}")

    # Relevance score distribution
    print("\nRelevance score distribution (from rounded means):")
    dist = [0] * (SCORE_RANGE_MAX + 1)  # index 0-5, we use 1-5
    for entry in dataset:
        score = round(entry["human_relevance_score"])
        if SCORE_RANGE_MIN <= score <= SCORE_RANGE_MAX:
            dist[score] += 1
    for score_val in range(SCORE_RANGE_MIN, SCORE_RANGE_MAX + 1):
        print(f"  {score_val}: {dist[score_val]:>3} pairs")

    # Check for malformed entries
    malformed = [
        entry
        for entry in dataset
        if not entry["raw_relevance_scores"] or not entry["raw_appropriateness_scores"]
    ]
    if malformed:
        print(f"\n⚠ {len(malformed)} malformed entries (missing scores)")
        for entry in malformed[:5]:
            print(f"  - {entry['conversation_id']}")
    else:
        print("\n✓ No malformed entries")


def generate_readme(dataset: list[dict[str, Any]], path: Path) -> None:
    """Generate data/README.md with auto-filled values.

    Args:
        dataset: List of parsed dataset entries.
        path: Path where README.md will be written.
    """
    models = sorted(set(entry["model"] for entry in dataset))

    # Score distribution
    dist = [0] * (SCORE_RANGE_MAX + 1)
    for entry in dataset:
        score = round(entry["human_relevance_score"])
        if SCORE_RANGE_MIN <= score <= SCORE_RANGE_MAX:
            dist[score] += 1

    readme = f"""# Dataset: DailyDialog-Zhao

## Source
- Paper: Zhao et al. (2020), ACL 2020
- URL: https://aclanthology.org/2020.acl-main.4/
- Repository: https://github.com/ZHAOTING/dialog-processing
- Zenodo Record: https://zenodo.org/record/3828180
- Download date: {datetime.now().isoformat()}

## License
CC BY-NC-SA 4.0 - academic non-commercial use only

## Size
- Total conversations: 100
- Responses per conversation: 9
- Total context-response pairs: 900
- Annotators per pair: 4
- Annotation scale: 1-5 (Likert)

## Models included
{chr(10).join(f"- {model}" for model in models)}

## Field schema (dataset.json)
| Field | Type | Description |
|---|---|---|
| conversation_id | string | Unique ID: conv_{{index}}_{{model}} |
| turns | list[dict] | Context turns in order, each {{"speaker": str, "text": str}} |
| response | string | Model-generated response |
| response_speaker | string | Speaker who authored the response ("A" or "B") |
| model | string | Name of the generative model |
| human_relevance_score | float | Mean relevance score across annotators (1-5) |
| raw_relevance_scores | list[int] | Individual annotator relevance scores |
| human_appropriateness_score | float | Mean appropriateness (content) score (1-5) |
| raw_appropriateness_scores | list[int] | Individual annotator appropriateness scores |

## Relevance Score Distribution
| Score | Count |
|---|---|
| 1 | {dist[1]} |
| 2 | {dist[2]} |
| 3 | {dist[3]} |
| 4 | {dist[4]} |
| 5 | {dist[5]} |

## Integrity Verification
- ✓ Total pairs: 900 (expected 900)
- ✓ Unique conversations: 100 (expected 100)
- ✓ Annotators per pair: 4 (expected 4)
- ✓ Annotation scale: 1-5 (expected 1-5)

## Notes
Responses were generated by generative language models (Seq2Seq, HRED, VHRED,
GPT-2 small/medium) with multiple decoding strategies, not modern conversational
agents. This dataset serves as a controlled evaluation benchmark for measuring
automatic relevance scoring methods (G-Eval vs. agentic voting system), not for
evaluating the models themselves.

The "appropriateness" dimension is derived from the "content" annotation field
in the raw data (Pearson r = 0.91 with Relevance, as reported in the paper).
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"✓ Generated {path}")


def main() -> None:
    """Orchestrate download, extraction, parsing, validation, and documentation."""
    print("=" * 70)
    print("DailyDialog-Zhao Dataset Download & Processing")
    print("=" * 70)
    print()

    try:
        # Step 1: Download
        zip_path = download_zip(RAW_DIR)
        print()

        # Step 2: Extract
        json_path = extract_and_find_json(zip_path)
        print()

        # Step 3: Parse
        dataset = parse_annotations(json_path)
        print()

        # Step 4: Verify
        run_integrity_checks(dataset)

        # Step 5: Summary
        print_summary(dataset)
        print()

        # Step 6: Save output
        print("Saving dataset.json...")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {OUTPUT_FILE}")
        print()

        # Step 7: Generate README
        generate_readme(dataset, README_PATH)
        print()

        # Final checklist
        print("=== FINAL CHECKLIST ===")
        checks = [
            ("data/raw/dailydialog_zhao/ exists", RAW_DIR.exists()),
            ("dataset.json exists", OUTPUT_FILE.exists()),
            (
                "dataset.json has 900 entries",
                len(dataset) == EXPECTED_TOTAL_PAIRS,
            ),
            (
                f"All entries have {DATASET_SCHEMA_FIELDS} fields",
                all(len(e) == DATASET_SCHEMA_FIELDS for e in dataset),
            ),
            (
                "No null values in required fields",
                all(v is not None for e in dataset for k, v in e.items() if k != "turns"),
            ),
            ("README.md exists", README_PATH.exists()),
        ]

        all_ok = True
        for check_name, result in checks:
            symbol = "✓" if result else "✗"
            print(f"{symbol} {check_name}")
            if not result:
                all_ok = False

        print()
        if all_ok:
            print("=" * 70)
            print("✓ SUCCESS: Dataset downloaded and processed")
            print("=" * 70)
        else:
            print("✗ Some checks failed")
            raise SystemExit(1) from None  # noqa: TRY301

    except Exception as e:
        print()
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
