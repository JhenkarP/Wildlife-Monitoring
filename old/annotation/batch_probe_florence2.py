"""Run Florence-2 open-vocabulary detection over a batch of images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from probe_florence2 import detect, load_model


DEFAULT_QUERIES = ("single horn", "armor-like skin folds", "rounded ears")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/florence2"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", default="microsoft/Florence-2-base")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    model, processor, device, dtype = load_model(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    skipped = 0
    failed: list[str] = []
    for index, image_path in enumerate(image_paths, start=1):
        output_path = args.output / f"{image_path.stem}_florence.json"
        if output_path.is_file() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(image_paths)}] skipped {image_path.name}")
            continue
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            result = {
                "model": args.model,
                "image": str(image_path.resolve()),
                "detections": {
                    query: detect(model, processor, device, dtype, image, query)
                    for query in DEFAULT_QUERIES
                },
            }
            output_path.write_text(
                json.dumps(result, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[{index}/{len(image_paths)}] {image_path.name} -> {output_path}")
        except Exception as error:
            failed.append(f"{image_path.name}: {error}")
            print(f"[{index}/{len(image_paths)}] failed {image_path.name}: {error}")

    print(f"completed={len(image_paths) - skipped - len(failed)} skipped={skipped} failed={len(failed)}")
    if failed:
        raise SystemExit("Batch completed with failures:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
