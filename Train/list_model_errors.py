from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "indian_four_animals_resnet18_finetuned.pt"
LABELS_PATH = ROOT / "data" / "training" / "four_animal_labels.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    rows = [
        row
        for row in csv.DictReader(LABELS_PATH.open(encoding="utf-8"))
        if row["label"] == args.label and row["split"] == args.split
    ]
    errors = []
    with torch.no_grad():
        for row in rows:
            image_path = ROOT / "data" / "training" / row["image_path"]
            probabilities = model(transform(Image.open(image_path).convert("RGB")).unsqueeze(0)).softmax(dim=1)[0]
            confidence, index = probabilities.max(dim=0)
            predicted = classes[int(index)]
            if predicted != args.label:
                errors.append((image_path.name, predicted, float(confidence)))

    print(f"{args.label} {args.split}: {len(rows) - len(errors)}/{len(rows)} correct")
    for filename, predicted, confidence in errors:
        print(f"{filename}\t{predicted}\t{confidence:.4f}")


if __name__ == "__main__":
    main()