"""Create a deterministic 70/30 train/test split for four Indian mammals."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


TARGET_CLASSES = (
    "asian_elephant",
    "greater_one_horned_rhino",
    "leopard",
    "gaur",
)
IMAGE_DIRECTORIES = {
    "asian_elephant": "asian elephant",
    "greater_one_horned_rhino": "greater_one_horned_rhino",
    "leopard": "indian leopard",
    "gaur": "gaur",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/training/labels.csv"), help="Master labels file to read.")
    parser.add_argument("--output", type=Path, default=Path("data/training/four_animal_labels.csv"))
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        raise SystemExit("--train-fraction must be between 0 and 1")

    frame = pd.read_csv(args.labels)
    required = {"image_path", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Labels file is missing columns: {', '.join(sorted(missing))}")
    frame = frame[frame["label"].isin(TARGET_CLASSES)].copy()
    if frame.empty:
        raise SystemExit("No target-class images found")
    frame["image_path"] = frame.apply(
        lambda row: f"images/{IMAGE_DIRECTORIES[row['label']]}/{Path(row['image_path']).name}",
        axis=1,
    )

    rng = random.Random(args.seed)
    assignments = pd.Series(index=frame.index, dtype="object")
    for label in TARGET_CLASSES:
        class_indexes = frame.index[frame["label"] == label].tolist()
        rng.shuffle(class_indexes)
        train_count = int(len(class_indexes) * args.train_fraction)
        if train_count < 1 or train_count >= len(class_indexes):
            raise SystemExit(f"Class {label} needs enough images for both splits")
        assignments.loc[class_indexes[:train_count]] = "train"
        assignments.loc[class_indexes[train_count:]] = "test"

    frame["split"] = assignments
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"saved={args.output}")
    print(frame.groupby(["label", "split"]).size().to_string())


if __name__ == "__main__":
    main()