"""Generate review-only feature-label proposals with a local Qwen VLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_schema import FEATURE_SCHEMA, SCHEMA_VERSION


def parse_json_response(text: str, species: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        record = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        record = json.loads(cleaned[start:end + 1])
    if isinstance(record, list):
        features = {
            item["feature"]: {
                "status": item["status"],
                "confidence": item["confidence"],
                "evidence": item["evidence"],
            }
            for item in record
            if isinstance(item, dict) and "feature" in item
        }
        return {"species": species, "features": features}
    if not isinstance(record, dict):
        raise ValueError("model response must be a JSON object or feature array")
    if isinstance(record.get("features"), dict):
        return record
    feature_names = {name for name, _ in FEATURE_SCHEMA[species]}
    direct_features = {name: record[name] for name in feature_names if isinstance(record.get(name), dict)}
    if direct_features:
        return {"species": species, "features": direct_features}
    if "feature" in record:
        name = record["feature"]
        return {"species": species, "features": {name: {key: record[key] for key in ("status", "confidence", "evidence")}}}
    raise ValueError("model response did not contain recognized features")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--species", required=True, choices=sorted(FEATURE_SCHEMA))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("outputs/feature_proposals.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Missing image: {args.image}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this local VLM command")

    feature_lines = "\n".join(
        f'- "{name}": {description}' for name, description in FEATURE_SCHEMA[args.species]
    )
    prompt = f"""Inspect this {args.species} crop and propose labels only for visible features.
Return one compact valid JSON object and no markdown, reasoning, list, or explanation. Use exactly these feature keys:
{feature_lines}
For each feature, return an object with status (visible, not_visible, or uncertain), confidence (0 to 1), and evidence. Use uncertain when the feature cannot be judged reliably. Do not invent details. Set review_status to needs_review."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(args.image.resolve())},
            {"type": "text", "text": prompt},
        ],
    }]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: "3GiB", "cpu": "24GiB"},
        offload_folder="outputs/qwen_offload",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
    raw_response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    raw_output_path = args.output.with_name(f"{args.output.stem}_raw.txt")
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(raw_response, encoding="utf-8")

    record = parse_json_response(raw_response, args.species)
    record["species"] = args.species
    record["schema_version"] = SCHEMA_VERSION
    record["source_image"] = str(args.image.resolve())
    record["review_status"] = "needs_review"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=True) + "\n")
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
