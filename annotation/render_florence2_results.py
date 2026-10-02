"""Render Florence-2 JSON detections onto their source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ANNOTATION_COLOR = "#16803c"


def get_font(image: Image.Image) -> ImageFont.ImageFont:
    size = max(18, min(28, min(image.size) // 30))
    for name in ("arialbd.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(result_path: Path, output_dir: Path) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    image_path = Path(result["image"])
    image = Image.open(image_path).convert("RGB")
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    font = get_font(image)

    for query, task_result in result.get("detections", {}).items():
        detection = task_result.get("<OPEN_VOCABULARY_DETECTION>", {})
        color = ANNOTATION_COLOR
        for box in detection.get("bboxes", []):
            if not isinstance(box, list) or len(box) != 4:
                continue
            left, top, right, bottom = [float(value) for value in box]
            left = max(0, min(image.width, left))
            top = max(0, min(image.height, top))
            right = max(left, min(image.width, right))
            bottom = max(top, min(image.height, bottom))
            label = query
            text_box = draw.textbbox((0, 0), label, font=font)
            padding = 4
            label_width = text_box[2] - text_box[0] + padding * 2
            label_height = text_box[3] - text_box[1] + padding * 2
            label_left = max(0, min(image.width - label_width, left))
            label_top = max(0, top - label_height)
            draw.rectangle((left, top, right, bottom), outline=color, width=6)
            draw.rectangle(
                (label_left, label_top, label_left + label_width, label_top + label_height),
                fill=color,
            )
        draw.text(
            (label_left + padding, label_top + padding - text_box[1]),
            label,
            fill="white",
            font=font,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_florence_marked.jpg"
    marked.save(output_path, quality=95)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_paths = sorted(args.results.glob("*_florence.json"))
    if not result_paths:
        raise SystemExit(f"No Florence JSON files found in {args.results}")
    for index, result_path in enumerate(result_paths, start=1):
        output_path = render(result_path, args.output)
        print(f"[{index}/{len(result_paths)}] {output_path.name}")


if __name__ == "__main__":
    main()
