from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "inat"
SOURCE_MANIFEST = DATA_ROOT / "manifest.csv"
OUTPUT_MANIFEST = DATA_ROOT / "manifest_three_species_all.csv"
LABELS = ("panthera_leo", "panthera_pardus", "panthera_tigris")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_ids(path: Path) -> tuple[str, str]:
    stem = path.stem
    observation_id, separator, photo_id = stem.rpartition("_")
    return (observation_id, photo_id) if separator else (stem, stem)


def main() -> None:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as file:
        existing_rows = {row["image_path"]: row for row in csv.DictReader(file)}

    fields = [
        "image_path", "label", "scientific_name", "observation_id", "photo_id",
        "source_url", "photo_url", "license", "attribution", "width", "height", "sha256",
    ]
    rows: list[dict[str, str]] = []
    for label in LABELS:
        folder = DATA_ROOT / label
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_path = f"{label}/{path.name}"
            existing = existing_rows.get(relative_path)
            if existing is not None:
                rows.append({field: existing.get(field, "") for field in fields})
                continue

            observation_id, photo_id = parse_ids(path)
            with Image.open(path) as image:
                width, height = image.size
            rows.append({
                "image_path": relative_path,
                "label": label,
                "scientific_name": label.replace("_", " ").title(),
                "observation_id": observation_id,
                "photo_id": photo_id,
                "source_url": "",
                "photo_url": "",
                "license": "",
                "attribution": "",
                "width": str(width),
                "height": str(height),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {label: sum(row["label"] == label for row in rows) for label in LABELS}
    print(f"Wrote {len(rows)} images to {OUTPUT_MANIFEST}")
    print(counts)


if __name__ == "__main__":
    main()