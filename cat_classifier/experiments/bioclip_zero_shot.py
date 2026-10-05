from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import open_clip
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "inat"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
TAXONOMY_PATH = PROJECT_ROOT / "taxonomy" / "felidae_species.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "experiments" / "bioclip_zero_shot.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_taxonomy() -> list[dict[str, str]]:
    with TAXONOMY_PATH.open(encoding="utf-8") as file:
        return json.load(file)["species"]


def build_prompts(species: list[dict[str, str]]) -> list[str]:
    prompts: list[str] = []
    for item in species:
        prompts.append(f"a wildlife camera photograph of a {item['common_name']}")
        prompts.append(f"a photograph of {item['scientific_name']}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioCLIP zero-shot Felidae classification.")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    species = load_taxonomy()
    species_ids = [item["id"] for item in species]
    prompts = build_prompts(species)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="hf-hub:imageomics/bioclip"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()
    with torch.inference_mode():
        text_features = model.encode_text(tokenizer(prompts).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.reshape(len(species), 2, -1).mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows = read_rows(SPLIT_ROOT / f"{args.split}.csv")
    predictions: list[str] = []
    actual: list[str] = []
    missing = 0
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images: list[torch.Tensor] = []
        valid_rows: list[dict[str, str]] = []
        for row in batch_rows:
            try:
                with Image.open(DATA_ROOT / row["image_path"]) as image:
                    images.append(preprocess(image.convert("RGB")))
                valid_rows.append(row)
            except (OSError, ValueError):
                missing += 1
        if not images:
            continue
        with torch.inference_mode():
            image_features = model.encode_image(torch.stack(images).to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = image_features @ text_features.T
            predictions.extend(species_ids[index] for index in scores.argmax(dim=1).cpu().tolist())
            actual.extend(row["label"] for row in valid_rows)

    per_class: dict[str, dict[str, float | int]] = {}
    for label in species_ids:
        true_positive = sum(actual_label == label and predicted_label == label for actual_label, predicted_label in zip(actual, predictions))
        actual_positive = sum(actual_label == label for actual_label in actual)
        predicted_positive = sum(predicted_label == label for predicted_label in predictions)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"images": actual_positive, "precision": precision, "recall": recall, "f1": f1}

    accuracy = sum(left == right for left, right in zip(actual, predictions)) / len(actual)
    macro_f1 = sum(float(item["f1"]) for item in per_class.values()) / len(per_class)
    result = {
        "method": "BioCLIP zero-shot prompt ensemble",
        "training": False,
        "classes": len(species_ids),
        "split": args.split,
        "images": len(actual),
        "missing_images": missing,
        "device": str(device),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Images: {len(actual)}; missing: {missing}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()