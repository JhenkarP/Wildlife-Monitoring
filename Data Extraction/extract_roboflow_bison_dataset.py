"""Extract bison-only images from a Roboflow COCO export."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
IMAGE_ROOT = ROOT / "data" / "training" / "images" / "bison"
ZIP_PATH = Path.home() / "Downloads" / "gaur.v1-gaur-review-export.coco.zip"
EXPECTED_IMAGES = 787


def collect_bison_images(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
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
        bison_ids = {
            category_id
            for category_id, name in category_names.items()
            if name == "bison"
        }
        image_labels = {image["id"]: set() for image in data["images"]}
        for annotation in data["annotations"]:
            image_labels.setdefault(annotation["image_id"], set()).add(
                category_names[annotation["category_id"]]
            )
        selected.extend(
            (split, image["file_name"])
            for image in data["images"]
            if image_labels[image["id"]] == {category_names[bison_id] for bison_id in bison_ids}
        )
    return sorted(selected)


def main() -> None:
    if not ZIP_PATH.exists():
        raise SystemExit(f"Missing ZIP: {ZIP_PATH}")

    with tempfile.TemporaryDirectory(prefix="roboflow-bison-") as temporary_directory:
        staging_root = Path(temporary_directory) / "bison"
        staging_root.mkdir()
        with zipfile.ZipFile(ZIP_PATH) as archive:
            selected_images = collect_bison_images(archive)
            if len(selected_images) != EXPECTED_IMAGES:
                raise SystemExit(
                    f"Expected {EXPECTED_IMAGES} bison-only images, found {len(selected_images)}"
                )
            for index, (split, source_name) in enumerate(selected_images, start=1):
                suffix = Path(source_name).suffix.lower() or ".jpg"
                destination = staging_root / f"bison_{index:04d}{suffix}"
                with archive.open(f"{split}/{source_name}") as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

        if IMAGE_ROOT.exists():
            shutil.rmtree(IMAGE_ROOT)
        IMAGE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_root), str(IMAGE_ROOT))

    print(f"extracted={EXPECTED_IMAGES} destination={IMAGE_ROOT}")


if __name__ == "__main__":
    main()