"""Train and evaluate a transfer-learning classifier for Indian mammals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

TARGET_CLASSES = (
    "asian_elephant",
    "greater_one_horned_rhino",
    "leopard",
    "gaur",
)


class MammalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_index: dict[str, int], transform):
        self.frame = frame.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), self.class_to_index[row["label"]]


def load_labels(labels_path: Path, minimum_per_class: int) -> tuple[pd.DataFrame, dict[str, int]]:
    if not labels_path.exists():
        raise SystemExit(f"Missing labels file: {labels_path}. Copy data/training/labels.template.csv and fill it with verified images.")
    frame = pd.read_csv(labels_path)
    required = {"image_path", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Labels file is missing columns: {', '.join(sorted(missing))}")
    frame["image_path"] = frame["image_path"].map(lambda value: str((labels_path.parent / value).resolve()))
    missing_files = frame.loc[~frame["image_path"].map(lambda value: Path(value).exists()), "image_path"].tolist()
    if missing_files:
        raise SystemExit(f"Labels reference missing files, first one: {missing_files[0]}")
    unknown = sorted(set(frame["label"]) - set(TARGET_CLASSES))
    if unknown:
        raise SystemExit(f"Unknown labels: {', '.join(unknown)}. Allowed labels: {', '.join(TARGET_CLASSES)}")
    counts = frame[frame["split"] == "train"]["label"].value_counts()
    insufficient = {label: int(counts.get(label, 0)) for label in TARGET_CLASSES if counts.get(label, 0) < minimum_per_class}
    if insufficient:
        details = ", ".join(f"{label}={count}" for label, count in insufficient.items())
        raise SystemExit(f"Not enough verified training images ({details}); need at least {minimum_per_class} per class.")
    if set(frame["split"]) - {"train", "test"}:
        raise SystemExit("split must be train or test")
    test_counts = frame[frame["split"] == "test"]["label"].value_counts()
    missing_test = [label for label in TARGET_CLASSES if test_counts.get(label, 0) == 0]
    if missing_test:
        raise SystemExit(f"Each class needs test images: {', '.join(missing_test)}")
    return frame, {label: index for index, label in enumerate(TARGET_CLASSES)}


def build_model(class_count: int, fine_tune: bool = False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def evaluate(model, loader, index_to_class):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            actual.extend(labels.tolist())
            predicted.extend(outputs.argmax(dim=1).tolist())
    names = [index_to_class[index] for index in range(len(index_to_class))]
    labels = list(range(len(names)))
    return classification_report(actual, predicted, labels=labels, target_names=names, zero_division=0, output_dict=True), confusion_matrix(actual, predicted, labels=labels).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/training/four_animal_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/indian_mammals_resnet18.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze ResNet18 layer4 while training the four animal classes.")
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame, class_to_index = load_labels(args.labels, args.minimum_per_class)
    image_size = 224
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])
    train_loader = DataLoader(MammalDataset(frame[frame["split"] == "train"], class_to_index, train_transform), batch_size=16, shuffle=True)
    test_loader = DataLoader(MammalDataset(frame[frame["split"] == "test"], class_to_index, eval_transform), batch_size=16)
    model = build_model(len(class_to_index), fine_tune=args.fine_tune)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = 0.00001 if args.fine_tune else 0.001
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1} completed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": list(class_to_index)}, args.output)
    test_metrics, confusion = evaluate(model, test_loader, {index: label for label, index in class_to_index.items()})
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps({"classes": list(class_to_index), "test_metrics": test_metrics, "confusion_matrix": confusion}, indent=2), encoding="utf-8")
    print(f"saved_model={args.output}", flush=True)
    print(f"saved_metrics={report_path}", flush=True)


if __name__ == "__main__":
    main()