"""Replace the local Asian elephant images with an images.cv dataset."""

from __future__ import annotations

import csv
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
TRAINING_ROOT = ROOT / "data" / "training"
IMAGE_ROOT = TRAINING_ROOT / "images" / "asian_elephant"
ZIP_PATH = Path.home() / "Downloads" / "5mfburl2xe83ke9hb99o4w.zip"
SOURCE_URL = "https://images.cv/dataset/indian_elephant-image-classification-dataset"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not ZIP_PATH.exists():
        raise SystemExit(f"Missing ZIP: {ZIP_PATH}")

    labels_path = TRAINING_ROOT / "labels.csv"
    manifest_path = TRAINING_ROOT / "manifest.csv"
    review_path = TRAINING_ROOT / "review_log.csv"
    labels = read_rows(labels_path)
    manifest = read_rows(manifest_path)
    review = read_rows(review_path)

    with tempfile.TemporaryDirectory(prefix="imagescv-elephant-") as temporary_directory:
        staging_root = Path(temporary_directory) / "asian_elephant"
        staging_root.mkdir()
        with zipfile.ZipFile(ZIP_PATH) as archive:
            image_members = sorted([
                member
                for member in archive.infolist()
                if member.filename.lower().endswith((".jpg", ".jpeg", ".png"))
                and "/indian_elephant/" in member.filename
            ], key=lambda member: member.filename)
            if len(image_members) != 1295:
                raise SystemExit(f"Expected 1295 Indian elephant images, found {len(image_members)}")
            for index, member in enumerate(image_members, start=1):
                destination = staging_root / f"asian_elephant_{index:04d}.jpg"
                with archive.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

        if IMAGE_ROOT.exists():
            shutil.rmtree(IMAGE_ROOT)
        IMAGE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_root), str(IMAGE_ROOT))

    relative_paths = [
        f"images/asian_elephant/asian_elephant_{index:04d}.jpg"
        for index in range(1, 1296)
    ]
    labels = [row for row in labels if row["label"] != "asian_elephant"]
    labels.extend({"image_path": path, "label": "asian_elephant", "split": "review"} for path in relative_paths)

    manifest = [row for row in manifest if row["label"] != "asian_elephant"]
    manifest.extend(
        {
            "image_path": path,
            "label": "asian_elephant",
            "observation_id": "",
            "photo_id": "",
            "scientific_name": "Elephas maximus indicus",
            "source_url": SOURCE_URL,
            "photo_url": "",
            "license": "verify before use",
            "attribution": "images.cv; source dataset attribution requires verification",
        }
        for path in relative_paths
    )

    review = [row for row in review if row["label"] != "asian_elephant"]
    review.extend(
        {
            "image_path": path,
            "label": "asian_elephant",
            "decision": "pending",
            "reason": "manual visual review required; imported from images.cv Indian elephant dataset",
        }
        for path in relative_paths
    )

    write_rows(labels_path, ["image_path", "label", "split"], labels)
    write_rows(manifest_path, list(manifest[0]), manifest)
    write_rows(review_path, ["image_path", "label", "decision", "reason"], review)
    print(f"imported={len(relative_paths)} label={IMAGE_ROOT}")


if __name__ == "__main__":
    main()