"""Keep one image per iNaturalist observation and log removed duplicate frames."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).parents[1]
TRAINING_ROOT = ROOT / "data" / "training"


def main() -> None:
    manifest_path = TRAINING_ROOT / "manifest.csv"
    labels_path = TRAINING_ROOT / "labels.csv"
    review_path = TRAINING_ROOT / "review_log.csv"
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    kept = []
    removed = []
    seen = set()
    for row in rows:
        key = (row["label"], row["observation_id"])
        image_path = TRAINING_ROOT / row["image_path"]
        if key in seen:
            removed.append({"image_path": row["image_path"], "label": row["label"], "decision": "reject", "reason": "duplicate observation"})
            image_path.unlink(missing_ok=True)
            continue
        seen.add(key)
        kept.append(row)

    kept_by_path = {row["image_path"]: row for row in kept}
    labels = []
    for row in kept:
        labels.append({"image_path": row["image_path"], "label": row["label"], "split": "review"})
    with manifest_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)
    with labels_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(labels)
    with review_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["image_path", "label", "decision", "reason"])
        writer.writeheader()
        writer.writerows(removed)
        writer.writerows({"image_path": row["image_path"], "label": row["label"], "decision": "pending", "reason": "manual visual review required"} for row in kept)
    print(f"kept={len(kept)} removed_duplicates={len(removed)} review_log={review_path}")


if __name__ == "__main__":
    main()