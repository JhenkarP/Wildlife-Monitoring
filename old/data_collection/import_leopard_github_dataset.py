"""Replace the local leopard images with the GitHub dataset and normalize metadata."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import tempfile
import urllib.request
from urllib.error import HTTPError
from urllib.parse import quote
from pathlib import Path

import truststore


truststore.inject_into_ssl()


ROOT = Path(__file__).parents[1]
TRAINING_ROOT = ROOT / "data" / "training"
IMAGE_ROOT = TRAINING_ROOT / "images" / "leopard"
RAW_ROOT = "https://raw.githubusercontent.com/Nitish-011/Indian_leopard_dataset/main"
MAX_IMAGES = 500


def read_source_metadata() -> dict[str, dict[str, str]]:
    metadata_path = Path(tempfile.gettempdir()) / "Indian_leopard_dataset_metadata.csv"
    urllib.request.urlretrieve(f"{RAW_ROOT}/dataset_metadata.csv", metadata_path)
    metadata_by_filename: dict[str, dict[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8-sig") as source_file:
        for row in csv.DictReader(source_file):
            filename = (row.get("filename") or "").strip()
            if filename and filename not in metadata_by_filename:
                metadata_by_filename[filename] = row
    return metadata_by_filename


def download_images(metadata_by_filename: dict[str, dict[str, str]], staging_root: Path) -> list[dict[str, str]]:
    candidates = list(sorted(metadata_by_filename.items()))
    downloaded: list[tuple[str, dict[str, str], Path]] = []

    def download_one(item: tuple[int, tuple[str, dict[str, str]]]) -> tuple[str, dict[str, str], Path] | None:
        index, (source_filename, source_row) = item
        destination_name = f"leopard_{index:04d}.jpg"
        destination_path = staging_root / destination_name
        image_url = f"{RAW_ROOT}/dataset/{quote(source_filename)}"
        try:
            urllib.request.urlretrieve(image_url, destination_path)
        except HTTPError:
            destination_path.unlink(missing_ok=True)
            return None
        return source_filename, source_row, destination_path

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download_one, item)
            for item in enumerate(candidates, start=1)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                downloaded.append(result)

    downloaded.sort(key=lambda item: item[2].name)
    for _, _, extra_path in downloaded[MAX_IMAGES:]:
        extra_path.unlink(missing_ok=True)
    imported_rows = []
    for index, (source_filename, source_row, staged_path) in enumerate(downloaded[:MAX_IMAGES], start=1):
        destination_name = f"leopard_{index:04d}.jpg"
        destination_path = staging_root / destination_name
        if staged_path != destination_path:
            staged_path.rename(destination_path)
        imported_rows.append(
            {
                "image_path": f"images/leopard/{destination_name}",
                "label": "leopard",
                "source_filename": source_filename,
                "source_url": source_row.get("url", "").strip(),
                "license": "repository license unspecified; verify before redistribution",
            }
        )
        
    return imported_rows


def rewrite_project_csvs(imported_rows: list[dict[str, str]]) -> None:
    manifest_path = TRAINING_ROOT / "manifest.csv"
    labels_path = TRAINING_ROOT / "labels.csv"
    review_path = TRAINING_ROOT / "review_log.csv"

    existing_manifest = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    non_leopard_manifest = [row for row in existing_manifest if row.get("label") != "leopard"]
    manifest_fields = [
        "image_path",
        "label",
        "observation_id",
        "photo_id",
        "scientific_name",
        "source_url",
        "photo_url",
        "license",
        "attribution",
    ]
    manifest_rows = non_leopard_manifest + [
        {
            "image_path": row["image_path"],
            "label": "leopard",
            "observation_id": "",
            "photo_id": row["source_filename"],
            "scientific_name": "Panthera pardus fusca",
            "source_url": row["source_url"],
            "photo_url": "",
            "license": row["license"],
            "attribution": "Nitish-011/Indian_leopard_dataset; source attribution requires verification",
        }
        for row in imported_rows
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    existing_labels = list(csv.DictReader(labels_path.open(newline="", encoding="utf-8")))
    label_rows = [row for row in existing_labels if row.get("label") != "leopard"]
    label_rows.extend({"image_path": row["image_path"], "label": "leopard", "split": "review"} for row in imported_rows)
    with labels_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(label_rows)

    existing_review = list(csv.DictReader(review_path.open(newline="", encoding="utf-8"))) if review_path.exists() else []
    review_rows = [row for row in existing_review if row.get("label") != "leopard"]
    review_rows.extend(
        {
            "image_path": row["image_path"],
            "label": "leopard",
            "decision": "pending",
            "reason": "manual visual review required; imported from GitHub dataset",
        }
        for row in imported_rows
    )
    with review_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["image_path", "label", "decision", "reason"])
        writer.writeheader()
        writer.writerows(review_rows)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        staging_root = Path(temporary_directory) / "leopard"
        staging_root.mkdir()
        metadata_by_filename = read_source_metadata()
        imported_rows = download_images(metadata_by_filename, staging_root)
        IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        for old_image in IMAGE_ROOT.glob("*"):
            if old_image.is_file():
                old_image.unlink()
        for staged_image in staging_root.iterdir():
            shutil.copy2(staged_image, IMAGE_ROOT / staged_image.name)
        rewrite_project_csvs(imported_rows)
    print(f"imported={len(imported_rows)} destination={IMAGE_ROOT}")


if __name__ == "__main__":
    main()