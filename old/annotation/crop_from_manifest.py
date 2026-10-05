"""Crop images from existing normalized boxes without loading an ML model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def crop_record(record: dict, output_dir: Path) -> dict:
    source_path = Path(record["image"])
    output_path = output_dir / record["species"] / f"{source_path.stem}.jpg"
    x, y, width, height = [float(value) for value in record["bbox"]]
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        left = max(0, min(image.width - 1, round(x * image.width)))
        top = max(0, min(image.height - 1, round(y * image.height)))
        right = max(left + 1, min(image.width, round((x + width) * image.width)))
        bottom = max(top + 1, min(image.height, round((y + height) * image.height)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(output_path, quality=95)
    return {**record, "crop": str(output_path.resolve()), "crop_source": "manifest_bbox"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="JSONL containing image, species, and normalized bbox fields")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_manifest = args.output / "manifest.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.manifest.open(encoding="utf-8-sig") as source, output_manifest.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("bbox") or not record.get("image"):
                continue
            target.write(json.dumps(crop_record(record, args.output), ensure_ascii=True) + "\n")
            count += 1
    print(f"cropped={count}")
    print(f"manifest={output_manifest.resolve()}")


if __name__ == "__main__":
    main()