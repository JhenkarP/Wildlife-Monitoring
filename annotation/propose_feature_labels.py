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
from src.feature_schema import FEATURE_SCHEMA, SCHEMA_VERSION, feature_queries


DEFAULT_MODEL = "microsoft/Florence-2-base"
MAX_ACCEPTED_FEATURE_BOX_AREA = 0.25


def _feature_query(species: str, feature_name: str) -> str:
    return feature_queries(species, feature_name)[0]


def _feature_queries(species: str, feature_name: str) -> list[str]:
    return list(feature_queries(species, feature_name))


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
    queries = _feature_queries(species, feature_name)
    raw_results = []
    boxes = []
    for query in queries:
        raw = detect(model, processor, device, dtype, image, query)
        raw_results.append({"query": query, "result": raw})
        task_result = raw.get("<OPEN_VOCABULARY_DETECTION>", {})
        raw_boxes = task_result.get("bboxes", [])
        if not raw_boxes:
            raw_boxes = _polygon_boxes(task_result)
        boxes.extend(
            normalized
            for box in raw_boxes
            if isinstance(box, list)
            and (normalized := _normalized_box(box, image)) is not None
        )
    feature_query = "; ".join(queries)
    accepted_indices = [
        index for index, box in enumerate(boxes)
        if box[2] * box[3] <= MAX_ACCEPTED_FEATURE_BOX_AREA
    ]
    accepted_boxes = [boxes[index] for index in accepted_indices]
    rejected_boxes = [
        box for box in boxes
        if box[2] * box[3] > MAX_ACCEPTED_FEATURE_BOX_AREA
    ]
    if accepted_boxes:
        accepted_index = max(
            range(len(accepted_boxes)),
            key=lambda index: accepted_boxes[index][2] * accepted_boxes[index][3],
        )
        bbox = accepted_boxes[accepted_index]
        selected_index = accepted_indices[accepted_index]
        evidence = (
            f"Florence-2 accepted {len(accepted_boxes)} of {len(boxes)} candidate "
            f"box(es) for '{feature_query}'."
        )
        status = "visible"
    else:
        if rejected_boxes:
            selected_index = max(
                range(len(boxes)),
                key=lambda index: boxes[index][2] * boxes[index][3],
            )
            bbox = boxes[selected_index]
            evidence = (
                f"Florence-2 returned {len(rejected_boxes)} broad box(es) for "
                f"'{feature_query}'; shown for review but not verified."
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
        "candidate_bboxes": boxes,
        "accepted_candidate_bboxes": accepted_boxes,
        "rejected_candidate_bboxes": rejected_boxes,
        "selected_bbox_index": selected_index if boxes else None,
        "model_query": feature_query,
        "feature_description": description,
    }
    return proposal, {"queries": queries, "results": raw_results}


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
