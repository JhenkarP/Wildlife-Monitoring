from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "inat" / "manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "splits"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def existing_rows(rows: list[dict[str, str]], data_root: Path) -> list[dict[str, str]]:
    return [row for row in rows if (data_root / row["image_path"]).is_file()]


def eligible_labels(rows: list[dict[str, str]], minimum_images: int) -> set[str]:
    counts = Counter(row["label"] for row in rows)
    return {label for label, count in counts.items() if count >= minimum_images}


def split_rows(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    groups_by_label: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["observation_id"])].append(row)
    for (label, _), group in groups.items():
        groups_by_label[label].append(group)

    split_rows_by_name: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    randomizer = random.Random(seed)
    split_fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for label, label_groups in groups_by_label.items():
        randomizer.shuffle(label_groups)
        label_total = sum(len(group) for group in label_groups)
        targets = {name: max(1, label_total * fraction) for name, fraction in split_fractions.items()}
        current = Counter()
        for group in label_groups:
            target = min(split_fractions, key=lambda name: current[name] / targets[name])
            split_rows_by_name[target].extend(group)
            current[target] += len(group)
    return split_rows_by_name


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + ["split"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create observation-safe Felidae train/validation/test splits.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "inat")
    parser.add_argument("--minimum-images", type=int, default=50)
    parser.add_argument("--labels", nargs="+", help="Restrict the split to these labels.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    valid_rows = existing_rows(rows, args.data_root)
    print(f"Skipped missing image files: {len(rows) - len(valid_rows)}")
    labels = eligible_labels(valid_rows, args.minimum_images)
    if args.labels:
        labels &= set(args.labels)
    eligible_rows = [row for row in valid_rows if row["label"] in labels]
    splits = split_rows(eligible_rows, args.seed)
    for split_name, split in splits.items():
        for row in split:
            row["split"] = split_name
        write_split(args.output / f"{split_name}.csv", split)

    summary_path = args.output / "summary.txt"
    counts = {name: Counter(row["label"] for row in split) for name, split in splits.items()}
    with summary_path.open("w", encoding="utf-8") as file:
        file.write(f"seed={args.seed}\nminimum_images={args.minimum_images}\n")
        file.write(f"eligible_classes={len(labels)}\neligible_images={len(eligible_rows)}\n")
        for split_name, split in splits.items():
            file.write(f"{split_name}_images={len(split)}\n")
            file.write(f"{split_name}_classes={len(counts[split_name])}\n")
    print(f"Eligible classes: {len(labels)}")
    print(f"Eligible images: {len(eligible_rows)}")
    for split_name, split in splits.items():
        print(f"{split_name}: {len(split)} images, {len(counts[split_name])} classes")
    print(f"Wrote splits to {args.output}")


if __name__ == "__main__":
    main()
