"""Copy only human-verified feature records into the training-label file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_schema import SCHEMA_VERSION, validate_label_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Reviewed proposal JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Verified training-label JSONL")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    promoted = rejected = 0
    with args.input.open(encoding="utf-8-sig") as source, args.output.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"rejected line={line_number}: invalid JSON ({error})")
                rejected += 1
                continue
            errors = validate_label_record(record)
            if errors or record.get("review_status") != "verified":
                reason = "; ".join(errors) if errors else "review_status is not verified"
                print(f"rejected line={line_number}: {reason}")
                rejected += 1
                continue
            record["schema_version"] = SCHEMA_VERSION
            output.write(json.dumps(record, ensure_ascii=True) + "\n")
            promoted += 1
    print(f"promoted={promoted} rejected={rejected}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()