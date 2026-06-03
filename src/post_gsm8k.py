import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional


NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
BOXED_RE = re.compile(r"(?is)\\boxed\{([^{}]*)\}")
TEXT_RE = re.compile(r"(?is)\\text\{[^{}]*\}")


def normalize_numeric_answer(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = BOXED_RE.sub(r"\1", text)
    text = TEXT_RE.sub("", text)
    text = text.replace(r"\%", "%")
    text = text.replace(r"\,", " ")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = " ".join(text.split())

    match = NUMBER_RE.search(text)
    if not match:
        return text.lower() or None

    number_text = match.group(0)
    try:
        number_value = float(number_text)
    except ValueError:
        return number_text

    if abs(number_value - round(number_value)) < 1e-9:
        return str(int(round(number_value)))

    return f"{number_value:.12f}".rstrip("0").rstrip(".")


def build_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".json":
        return input_path.with_name(f"{input_path.stem}_post.json")
    return input_path.with_name(f"{input_path.name}_post.json")


def post_process(input_path: Path, output_path: Optional[Path]) -> tuple[float, float, int, int]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])

    original_accuracy = float(payload.get("eval_result", {}).get("accuracy", 0.0))
    rescued = 0
    corrected = 0

    for record in records:
        normalized_prediction = normalize_numeric_answer(
            record.get("extracted_prediction")
        )
        normalized_ground_truth = normalize_numeric_answer(
            record.get("extracted_ground_truth")
        )
        post_is_correct = normalized_prediction == normalized_ground_truth

        if post_is_correct:
            corrected += 1
        if post_is_correct and not bool(record.get("is_correct", False)):
            rescued += 1

        record["normalized_prediction"] = normalized_prediction
        record["normalized_ground_truth"] = normalized_ground_truth
        record["post_is_correct"] = post_is_correct

    post_accuracy = corrected / len(records) if records else 0.0
    payload["post_eval_result"] = {
        "accuracy": post_accuracy,
        "rescued_count": rescued,
        "total": len(records),
    }

    if output_path is not None:
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return original_accuracy, post_accuracy, rescued, len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process GSM8K result JSON.")
    parser.add_argument("input_file", type=Path, help="Path to the original result JSON")
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Path to save the post-processed JSON",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Compute metrics only without writing a new JSON file",
    )
    args = parser.parse_args()

    output_path = None if args.no_save else (args.output_file or build_output_path(args.input_file))
    orig_acc, post_acc, rescued, total = post_process(args.input_file, output_path)

    print(f"Original accuracy: {orig_acc * 100:.4f}%")
    print(f"Post accuracy: {post_acc * 100:.4f}%")
    print(f"Rescued samples: {rescued}/{total}")
    if output_path is not None:
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
