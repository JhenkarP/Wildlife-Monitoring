"""Run localization, cropping, species identification, and Qwen feature review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from PIL import ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import classify_with_custom_model, _strongest_speciesnet_detection, _speciesnet_prediction
from src.feature_schema import FEATURE_SCHEMA
from src.feature_renderer import render_feature_boxes


STANDARD_CROP_SIZE = (640, 640)


def crop_from_detection(image: Image.Image, detection: dict) -> tuple[Image.Image, tuple[float, float, float, float]]:
    x, y, width, height = [float(value) for value in detection["bbox"]]
    left = max(0, int(x * image.width))
    top = max(0, int(y * image.height))
    right = min(image.width, int((x + width) * image.width))
    bottom = min(image.height, int((y + height) * image.height))
    if right <= left or bottom <= top:
        raise ValueError("SpeciesNet returned an invalid detection box")
    crop = image.crop((left, top, right, bottom))
    crop = ImageOps.pad(crop, STANDARD_CROP_SIZE, method=Image.Resampling.LANCZOS, color="black")
    return crop, (x, y, width, height)


def normalize_feature_boxes(record: dict, image: Image.Image) -> None:
    for value in record.get("features", {}).values():
        if not isinstance(value, dict) or not isinstance(value.get("bbox"), list) or len(value["bbox"]) != 4:
            continue
        parts = [float(part) for part in value["bbox"]]
        if max(parts) <= 1:
            continue
        left, top, right, bottom = parts
        value["bbox"] = [
            max(0.0, min(1.0, left / image.width)),
            max(0.0, min(1.0, top / image.height)),
            max(0.0, min(1.0, (right - left) / image.width)),
            max(0.0, min(1.0, (bottom - top) / image.height)),
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--species", choices=sorted(FEATURE_SCHEMA), default="bengal tiger")
    parser.add_argument("--output", type=Path, default=Path("outputs/tiger_pipeline"))
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Missing image: {args.image}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Qwen feature stage")

    species_slug = args.species.replace(" ", "_")
    original = Image.open(args.image).convert("RGB")
    prediction = _speciesnet_prediction(original)
    detection = _strongest_speciesnet_detection(prediction)
    if not detection:
        raise SystemExit("SpeciesNet found no animal detection")
    crop, bbox = crop_from_detection(original, detection)
    with TemporaryDirectory(prefix="wildlife-pipeline-") as work_dir:
        work_path = Path(work_dir)
        crop_path = work_path / f"{species_slug}_crop.jpg"
        crop.save(crop_path, quality=95)

        from annotation.propose_feature_labels import main as propose_feature_labels

        proposal_path = work_path / f"{species_slug}_features.json"
        old_argv = sys.argv
        try:
            sys.argv = [
                "propose_feature_labels.py",
                str(crop_path),
                "--species", args.species,
                "--output", str(proposal_path),
            ]
            propose_feature_labels()
        finally:
            sys.argv = old_argv

        record = json.loads(proposal_path.read_text(encoding="utf-8"))
        normalize_feature_boxes(record, crop)

    visible_count = sum(
        value.get("status") == "visible"
        for value in record.get("features", {}).values()
        if isinstance(value, dict)
    )
    category = str(visible_count) if visible_count else "none"
    output_dir = args.output / species_slug / category
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data") / "processed" / species_slug / category
    processed_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    feature_marked_path = output_dir / f"{stem}_feature_marked.jpg"
    processed_original_path = processed_dir / f"{stem}_original{args.image.suffix.lower()}"
    shutil.copy2(args.image, processed_original_path)
    render_feature_boxes(crop, record).save(feature_marked_path, quality=95)
    _, classifications = classify_with_custom_model(original)
    record.update(
        {
            "source_image": str(args.image.resolve()),
            "original_image": str(args.image.resolve()),
            "processed_original_image": str(processed_original_path.resolve()),
            "visible_feature_count": visible_count,
            "resnet18_classification": classifications,
            "marked_feature_image": str(feature_marked_path.resolve()),
        }
    )
    output_path = output_dir / f"{stem}_features.json"
    output_path.write_text(json.dumps(record, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"record": str(output_path.resolve()), "marked_image": str(feature_marked_path.resolve()), "visible_features": visible_count}, indent=2))


if __name__ == "__main__":
    main()