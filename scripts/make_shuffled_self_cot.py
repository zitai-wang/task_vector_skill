import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def derangement(n: int, rng: random.Random):
    if n < 2:
        raise ValueError("Need at least 2 records to build a shuffled-trace control.")

    order = list(range(n))
    while True:
        rng.shuffle(order)
        if all(i != order[i] for i in range(n)):
            return order


def main():
    parser = argparse.ArgumentParser(description="Build a shuffled-trace Self-CoT control file.")
    parser.add_argument("--input", required=True, help="Input jsonl/json Self-CoT file.")
    parser.add_argument("--output", required=True, help="Output shuffled-trace json file.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    order = derangement(len(records), random.Random(args.seed))

    shuffled = []
    for idx, src_idx in enumerate(order):
        item = dict(records[idx])
        donor = records[src_idx]
        item["self_cot"] = donor["self_cot"]
        item["trace_control"] = "shuffled_trace"
        item["trace_donor_question"] = donor.get("question")
        shuffled.append(item)

    with output_path.open("w", encoding="utf-8") as f:
        for item in shuffled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(shuffled)} shuffled-trace records to {output_path}")


if __name__ == "__main__":
    main()
