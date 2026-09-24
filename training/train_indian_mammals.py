"""Train a multi-species Indian wildlife classifier from collected image folders."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


class WildlifeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_index: dict[str, int], transform):
        self.frame = frame.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), self.class_to_index[row["label"]]


def collect_images(images_root: Path) -> pd.DataFrame:
    if not images_root.exists():
        raise SystemExit(f"Missing image directory: {images_root}")
    rows = []
    for class_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
        image_paths = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        rows.extend({"image_path": str(path.resolve()), "label": class_dir.name} for path in image_paths)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No images found below {images_root}")
    return frame


def split_images(frame: pd.DataFrame, train_fraction: float, seed: int) -> pd.DataFrame:
    if not 0.5 <= train_fraction < 0.9:
        raise SystemExit("train fraction must be between 0.5 and 0.9")
    rng = random.Random(seed)
    split_rows = []
    for label, group in frame.groupby("label", sort=True):
        paths = group["image_path"].tolist()
        rng.shuffle(paths)
        total = len(paths)
        if total < 3:
            raise SystemExit(f"Class '{label}' needs at least 3 images; found {total}")
        train_count = min(total - 2, max(1, int(total * train_fraction)))
        validation_count = max(1, (total - train_count) // 2)
        for index, image_path in enumerate(paths):
            split = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
            split_rows.append({"image_path": image_path, "label": label, "split": split})
    return pd.DataFrame(split_rows)


def build_model(class_count: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer3.parameters():
        parameter.requires_grad = True
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(model.fc.in_features, class_count))
    return model


def evaluate(model, loader, index_to_class, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            actual.extend(labels.tolist())
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
    names = [index_to_class[index] for index in range(len(index_to_class))]
    labels = list(range(len(names)))
    report = classification_report(actual, predicted, labels=labels, target_names=names, zero_division=0, output_dict=True)
    matrix = confusion_matrix(actual, predicted, labels=labels).tolist()
    return report, matrix


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += int((outputs.argmax(dim=1) == labels).sum())
        total += labels.size(0)
    return total_loss / total, correct / total


def validation_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            loss = criterion(model(images.to(device)), labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    return total_loss / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, default=Path("data/training/images"))
    parser.add_argument("--output", type=Path, default=Path("models/indian_wildlife_resnet18.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, max(1, (os.cpu_count() or 2) - 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = split_images(collect_images(args.images_root), args.train_fraction, args.seed)
    classes = sorted(frame["label"].unique())
    class_to_index = {label: index for index, label in enumerate(classes)}
    index_to_class = {index: label for label, index in class_to_index.items()}
    normalize = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    train_frame = frame[frame["split"] == "train"]
    validation_frame = frame[frame["split"] == "validation"]
    test_frame = frame[frame["split"] == "test"]
    train_dataset = WildlifeDataset(train_frame, class_to_index, train_transform)
    counts = train_frame["label"].value_counts()
    sample_weights = train_frame["label"].map(lambda label: 1.0 / counts[label]).to_numpy()
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(WildlifeDataset(validation_frame, class_to_index, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(WildlifeDataset(test_frame, class_to_index, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.3)
    best_validation_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        current_validation_loss = validation_loss(model, validation_loader, criterion, device)
        scheduler.step(current_validation_loss)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} validation_loss={current_validation_loss:.4f}", flush=True)
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                print("early_stopping=true", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    report, confusion = evaluate(model, test_loader, index_to_class, device)
    split_counts = frame.groupby(["label", "split"]).size().unstack(fill_value=0).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "classes": classes, "image_size": 224, "train_fraction": args.train_fraction, "split_counts": split_counts, "test_metrics": report}, args.output)
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps({"classes": classes, "train_fraction": args.train_fraction, "split_counts": split_counts, "test_metrics": report, "confusion_matrix": confusion, "device": str(device)}, indent=2), encoding="utf-8")
    print(f"classes={len(classes)} train_images={len(train_frame)} validation_images={len(validation_frame)} test_images={len(test_frame)}", flush=True)
    print(f"saved_model={args.output}", flush=True)
    print(f"saved_metrics={report_path}", flush=True)


if __name__ == "__main__":
    main()
