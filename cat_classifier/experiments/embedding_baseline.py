from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "inat"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "experiments" / "embedding_baseline.json"


class ImageRows(Dataset[tuple[Tensor, str]]):
    def __init__(self, rows: list[dict[str, str]], transform: object) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        row = self.rows[index]
        with Image.open(DATA_ROOT / row["image_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, row["label"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def create_embeddings(model: nn.Module, loader: DataLoader[tuple[Tensor, str]], device: torch.device) -> tuple[Tensor, list[str]]:
    vectors: list[Tensor] = []
    labels: list[str] = []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            features = model(images.to(device)).flatten(1)
            vectors.append(torch.nn.functional.normalize(features, dim=1).cpu())
            labels.extend(batch_labels)
    return torch.cat(vectors), labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a no-training ResNet embedding prototype classifier.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    encoder = nn.Sequential(*list(backbone.children())[:-1]).to(device)
    transform = weights.transforms()

    train_rows = read_rows(SPLIT_ROOT / "train.csv")
    test_rows = read_rows(SPLIT_ROOT / "test.csv")
    train_loader = DataLoader(ImageRows(train_rows, transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ImageRows(test_rows, transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_vectors, train_labels = create_embeddings(encoder, train_loader, device)
    test_vectors, test_labels = create_embeddings(encoder, test_loader, device)

    class_vectors: dict[str, list[Tensor]] = defaultdict(list)
    for vector, label in zip(train_vectors, train_labels):
        class_vectors[label].append(vector)
    prototypes = torch.stack([torch.nn.functional.normalize(torch.stack(vectors).mean(dim=0), dim=0) for _, vectors in sorted(class_vectors.items())])
    labels = sorted(class_vectors)
    scores = test_vectors @ prototypes.T
    predicted = [labels[index] for index in scores.argmax(dim=1).tolist()]

    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        true_positive = sum(actual == label and guess == label for actual, guess in zip(test_labels, predicted))
        actual_positive = sum(actual == label for actual in test_labels)
        predicted_positive = sum(guess == label for guess in predicted)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"images": actual_positive, "precision": precision, "recall": recall, "f1": f1}

    accuracy = sum(actual == guess for actual, guess in zip(test_labels, predicted)) / len(test_labels)
    macro_f1 = sum(float(item["f1"]) for item in per_class.values()) / len(per_class)
    result = {
        "method": "ResNet18 ImageNet embedding with class prototypes",
        "training": False,
        "classes": len(labels),
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "device": str(device),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
