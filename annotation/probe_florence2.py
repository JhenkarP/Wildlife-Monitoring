"""Run Florence-2 open-vocabulary detection on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_MODEL = "microsoft/Florence-2-base"


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor, device, dtype


def detect(model, processor, device: str, dtype: torch.dtype, image: Image.Image, query: str):
    task = "<OPEN_VOCABULARY_DETECTION>"
    prompt = f"{task}{query}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3,
            use_cache=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text,
        task=task,
        image_size=image.size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("queries", nargs="+", help="Feature phrases to locate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Missing image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    model, processor, device, dtype = load_model(args.model)
    results = {
        "model": args.model,
        "image": str(args.image.resolve()),
        "detections": {
            query: detect(model, processor, device, dtype, image, query)
            for query in args.queries
        },
    }
    rendered = json.dumps(results, ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
