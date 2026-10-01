"""Generate review-only feature-label proposals with Florence-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation.probe_florence2 import detect, load_model
from src.feature_schema import FEATURE_SCHEMA, SCHEMA_VERSION


DEFAULT_MODEL = "microsoft/Florence-2-base"


def _normalized_box(box: list[float], image: Image.Image) -> list[float] | None:
    if len(box) != 4:
        return None
    left, top, right, bottom = [float(value) for value in box]
    left = max(0.0, min(float(image.width), left))
    top = max(0.0, min(float(image.height), top))
    right = max(left, min(float(image.width), right))
    bottom = max(top, min(float(image.height), bottom))
    if right <= left or bottom <= top:
        return None
    return [
        round(left / image.width, 6),
        round(top / image.height, 6),
        round((right - left) / image.width, 6),
        round((bottom - top) / image.height, 6),
    ]


def _polygon_boxes(task_result: dict[str, object]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for polygon in task_result.get("polygons", []):
        if not isinstance(polygon, list) or not polygon:
            continue
        points = polygon[0] if isinstance(polygon[0], list) else polygon
        if not isinstance(points, list) or len(points) < 6 or len(points) % 2:
            continue
        coordinates = [float(value) for value in points]
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        if right - left < 2 or bottom - top < 2:
            continue
        boxes.append([left, top, right, bottom])
    return boxes


def _feature_proposal(
    model,
    processor,
    device: str,
    dtype,
    image: Image.Image,
    species: str,
    feature_name: str,
    description: str,
) -> tuple[dict[str, object], dict[str, object]]:
    feature_query = (
        feature_name.replace("_", " ")
        if "tail" in feature_name
        else description.rstrip(".")
    )
    query = feature_query
    raw = detect(model, processor, device, dtype, image, query)
    task_result = raw.get("<OPEN_VOCABULARY_DETECTION>", {})
    raw_boxes = task_result.get("bboxes", [])
    if not raw_boxes:
        raw_boxes = _polygon_boxes(task_result)
    boxes = [
        normalized
        for box in raw_boxes
        if isinstance(box, list)
        and (normalized := _normalized_box(box, image)) is not None
    ]
    if boxes:
        bbox = max(boxes, key=lambda value: value[2] * value[3])
        evidence = (
            f"Florence-2 returned {len(boxes)} candidate box(es) for '{feature_query}'. "
            "The candidate is unverified and requires human review."
        )
        status = "uncertain"
    else:
        bbox = None
        evidence = (
            f"Florence-2 returned no candidate box for '{feature_query}'; "
            "the feature is not shown."
        )
        status = "not_visible"
    proposal = {
        "status": status,
        "evidence": evidence,
        "bbox": bbox,
        "model_query": query,
        "feature_description": description,
    }
    return proposal, {"query": query, "result": raw}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--species", required=True, choices=sorted(FEATURE_SCHEMA))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("outputs/feature_proposals.jsonl"))
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Missing image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    model, processor, device, dtype = load_model(args.model)
    features: dict[str, object] = {}
    raw_detections: dict[str, object] = {}
    for name, description in FEATURE_SCHEMA[args.species]:
        proposal, raw = _feature_proposal(
            model, processor, device, dtype, image, args.species, name, description,
        )
        features[name] = proposal
        raw_detections[name] = raw

    record = {
        "species": args.species,
        "schema_version": SCHEMA_VERSION,
        "source_image": str(args.image.resolve()),
        "review_status": "needs_review",
        "model": args.model,
        "features": features,
    }
    raw_output_path = args.output.with_name(f"{args.output.stem}_raw.json")
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(json.dumps(raw_detections, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.output.suffix.lower() == ".json" else None
    with args.output.open("w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=True, indent=indent) + "\n")
    print(json.dumps(record, ensure_ascii=True, indent=2))
    print(f"raw_response={raw_output_path.resolve()}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_path = Path("outputs/proposal_error.log")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
