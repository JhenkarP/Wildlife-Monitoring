"""Validate LLM proposals without treating them as verified annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_schema import SCHEMA_VERSION, validate_label_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Agent proposal JSONL")
    parser.add_argument("--valid-output", type=Path, default=None)
    parser.add_argument("--error-output", type=Path, default=None)
    args = parser.parse_args()
    valid_output = args.valid_output or args.input.with_name(f"{args.input.stem}_valid.jsonl")
    error_output = args.error_output or args.input.with_name(f"{args.input.stem}_errors.jsonl")
    valid_output.parent.mkdir(parents=True, exist_ok=True)
    valid_count = error_count = 0
    with args.input.open(encoding="utf-8-sig") as source, valid_output.open("w", encoding="utf-8") as valid, error_output.open("w", encoding="utf-8") as errors:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                messages = validate_label_record(record)
            except (json.JSONDecodeError, TypeError) as error:
                record = {"raw": line.rstrip("\n")}
                messages = [f"invalid JSON: {error}"]
            if messages:
                errors.write(json.dumps({"line": line_number, "errors": messages, "record": record}, ensure_ascii=True) + "\n")
                error_count += 1
            else:
                record["schema_version"] = SCHEMA_VERSION
                record["review_status"] = "needs_review"
                valid.write(json.dumps(record, ensure_ascii=True) + "\n")
                valid_count += 1
    print(f"valid={valid_count} errors={error_count}")
    print(f"valid_output={valid_output.resolve()}")
    print(f"error_output={error_output.resolve()}")


if __name__ == "__main__":
    main()