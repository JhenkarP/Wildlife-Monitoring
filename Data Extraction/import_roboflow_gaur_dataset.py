"""Import Gaur-labeled images from a Roboflow COCO export."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
TRAINING_ROOT = ROOT / "data" / "training"
IMAGE_ROOT = TRAINING_ROOT / "images" / "gaur"
ZIP_PATH = Path.home() / "Downloads" / "gaur.v1-gaur-review-export.coco.zip"
SOURCE_URL = "https://app.roboflow.com/techmech/gaur-ptbii-pavgu/1"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_gaur_images(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for annotation_name in sorted(archive.namelist()):
        if not annotation_name.endswith("_annotations.coco.json"):
            continue
        split = Path(annotation_name).parent.name
        data = json.loads(archive.read(annotation_name))
        category_names = {
            category["id"]: category["name"].strip().lower()
            for category in data["categories"]
        }
        gaur_ids = {
            category_id
            for category_id, name in category_names.items()
            if name == "gaur"
        }
        gaur_image_ids = {
            annotation["image_id"]
            for annotation in data["annotations"]
            if annotation["category_id"] in gaur_ids
        }
        selected.extend(
            (split, image["file_name"])
            for image in data["images"]
            if image["id"] in gaur_image_ids
        )
    return sorted(selected)


def main() -> None:
    if not ZIP_PATH.exists():
        raise SystemExit(f"Missing ZIP: {ZIP_PATH}")

    labels_path = TRAINING_ROOT / "labels.csv"
    manifest_path = TRAINING_ROOT / "manifest.csv"
    review_path = TRAINING_ROOT / "review_log.csv"
    labels = read_rows(labels_path)
    manifest = read_rows(manifest_path)
    review = read_rows(review_path)

    with tempfile.TemporaryDirectory(prefix="roboflow-gaur-") as temporary_directory:
        staging_root = Path(temporary_directory) / "gaur"
        staging_root.mkdir()
        with zipfile.ZipFile(ZIP_PATH) as archive:
            selected_images = collect_gaur_images(archive)
            imported_rows: list[dict[str, str]] = []
            for index, (split, source_name) in enumerate(selected_images, start=1):
                suffix = Path(source_name).suffix.lower() or ".jpg"
                destination_name = f"gaur_{index:04d}{suffix}"
                destination = staging_root / destination_name
                member_name = f"{split}/{source_name}"
                with archive.open(member_name) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                imported_rows.append(
                    {
                        "image_path": f"images/gaur/{destination_name}",
                        "label": "gaur",
                        "photo_id": source_name,
                        "source_url": SOURCE_URL,
                        "license": "CC BY 4.0",
                    }
                )

        if IMAGE_ROOT.exists():
            shutil.rmtree(IMAGE_ROOT)
        IMAGE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_root), str(IMAGE_ROOT))

    manifest = [row for row in manifest if row["label"] != "gaur"]
    manifest.extend(
        {
            "image_path": row["image_path"],
            "label": "gaur",
            "observation_id": "",
            "photo_id": row["photo_id"],
            "scientific_name": "Bos gaurus",
            "source_url": row["source_url"],
            "photo_url": "",
            "license": row["license"],
            "attribution": "Roboflow techmech/gaur-ptbii-pavgu; CC BY 4.0",
        }
        for row in imported_rows
    )
    labels = [row for row in labels if row["label"] != "gaur"]
    labels.extend(
        {"image_path": row["image_path"], "label": "gaur", "split": "review"}
        for row in imported_rows
    )
    review = [row for row in review if row["label"] != "gaur"]
    review.extend(
        {
            "image_path": row["image_path"],
            "label": "gaur",
            "decision": "pending",
            "reason": "manual visual review required; Gaur annotation selected from Roboflow COCO export",
        }
        for row in imported_rows
    )

    write_rows(labels_path, ["image_path", "label", "split"], labels)
    write_rows(manifest_path, list(manifest[0]), manifest)
    write_rows(review_path, ["image_path", "label", "decision", "reason"], review)
    print(f"imported={len(imported_rows)} label={IMAGE_ROOT}")


if __name__ == "__main__":
    main()