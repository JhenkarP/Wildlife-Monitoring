from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor


PROJECT_ROOT = Path(__file__).parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "inat"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "felidae_classifier.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "experiments" / "transfer_learning.json"


class ImageRows(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[dict[str, str]], labels: dict[str, int], transform: object) -> None:
        self.rows = rows
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = DATA_ROOT / image_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image), self.labels[row["label"]]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    valid_rows: list[dict[str, str]] = []
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = DATA_ROOT / image_path
        if image_path.is_file():
            valid_rows.append(row)
    return valid_rows


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, labels: list[str]) -> tuple[float, float, dict[str, dict[str, float | int]]]:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model(images.to(device))
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
            actual.extend(targets.tolist())

    accuracy = sum(left == right for left, right in zip(actual, predicted)) / len(actual)
    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        true_positive = sum(target == index and guess == index for target, guess in zip(actual, predicted))
        actual_positive = sum(target == index for target in actual)
        predicted_positive = sum(guess == index for guess in predicted)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"images": actual_positive, "precision": precision, "recall": recall, "f1": f1}
    macro_f1 = sum(float(item["f1"]) for item in per_class.values()) / len(per_class)
    return accuracy, macro_f1, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised ResNet18 Felidae classifier.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize from an existing compatible checkpoint.")
    parser.add_argument("--extra-manifest", type=Path, help="Append labeled image rows to the training split.")
    parser.add_argument("--correction-only", action="store_true", help="Train and select on extra-manifest rows only.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    weights = ResNet18_Weights.DEFAULT
    transforms = weights.transforms()
    eval_transform = transforms
    train_transform = eval_transform if args.correction_only else Compose([
        RandomResizedCrop(224, scale=(0.7, 1.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=transforms.mean, std=transforms.std),
    ])

    base_train_rows = read_rows(SPLIT_ROOT / "train.csv")
    train_rows = base_train_rows
    extra_rows: list[dict[str, str]] = []
    if args.extra_manifest:
        extra_rows = read_rows(args.extra_manifest)
        train_rows = extra_rows if args.correction_only else base_train_rows + extra_rows
    validation_rows = read_rows(SPLIT_ROOT / "validation.csv")
    test_rows = read_rows(SPLIT_ROOT / "test.csv")
    labels = sorted({row["label"] for row in base_train_rows + extra_rows})
    label_ids = {label: index for index, label in enumerate(labels)}
    loaders = {
        "train": DataLoader(ImageRows(train_rows, label_ids, train_transform), batch_size=args.batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(ImageRows(validation_rows, label_ids, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(ImageRows(test_rows, label_ids, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    if args.init_checkpoint:
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        saved_labels = saved.get("labels") if isinstance(saved, dict) else None
        state_dict = saved.get("state_dict") if isinstance(saved, dict) else saved
        if saved_labels == labels:
            model.load_state_dict(state_dict)
        elif saved_labels and set(saved_labels).issubset(labels):
            feature_state = {key: value for key, value in state_dict.items() if key not in {"fc.weight", "fc.bias"}}
            model.load_state_dict(feature_state, strict=False)
            with torch.no_grad():
                for old_index, label in enumerate(saved_labels):
                    new_index = labels.index(label)
                    model.fc.weight[new_index].copy_(state_dict["fc.weight"][old_index])
                    model.fc.bias[new_index].copy_(state_dict["fc.bias"][old_index])
            print(f"Expanded checkpoint labels from {saved_labels!r} to {labels!r}")
        else:
            raise ValueError(f"Checkpoint labels {saved_labels!r} are incompatible with split labels {labels!r}")
        print(f"Initialized from checkpoint: {args.init_checkpoint}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_validation = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    selection_loader = loaders["train"] if args.correction_only else loaders["validation"]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        validation_accuracy, validation_f1, _ = evaluate(model, loaders["validation"], device, labels)
        selection_accuracy, selection_f1, _ = evaluate(model, selection_loader, device, labels)
        epoch_result = {"epoch": epoch, "train_loss": total_loss / len(train_rows), "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_f1}
        history.append(epoch_result)
        print(f"Epoch {epoch}/{args.epochs}: loss={epoch_result['train_loss']:.4f} val_accuracy={validation_accuracy:.4f} val_macro_f1={validation_f1:.4f}")
        if selection_f1 > best_validation:
            best_validation = selection_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_accuracy, test_f1, per_class = evaluate(model, loaders["test"], device, labels)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": "resnet18", "labels": labels, "state_dict": best_state}, args.checkpoint)
    result = {
        "method": "Supervised ResNet18 transfer learning",
        "classes": len(labels),
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "test_images": len(test_rows),
        "device": str(device),
        "best_validation_macro_f1": best_validation,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_f1,
        "history": history,
        "per_class": per_class,
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_f1:.4f}")
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    main()