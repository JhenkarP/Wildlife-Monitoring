from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Train.train_indian_mammals import build_model


MODEL_PATH = ROOT / "models" / "indian_four_animals_resnet18.pt"
IMAGE_DIR = ROOT / "data" / "training" / "images" / "indian leopard"
REPORT_PATH = ROOT / "data" / "processed" / "custom_model_leopard_evaluation.csv"


def main() -> None:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = build_model(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    rows = []
    with torch.no_grad():
        for start in range(0, len(image_paths), 32):
            batch_paths = image_paths[start : start + 32]
            batch = torch.stack(
                [transform(Image.open(path).convert("RGB")) for path in batch_paths]
            )
            probabilities = model(batch).softmax(dim=1)
            confidences, indices = probabilities.max(dim=1)
            for path, confidence, index in zip(batch_paths, confidences, indices):
                rows.append(
                    {
                        "filename": path.name,
                        "predicted_label": classes[int(index)],
                        "confidence": round(float(confidence), 4),
                        "is_leopard": classes[int(index)] == "leopard",
                    }
                )
            print(f"processed={min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}", flush=True)

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    leopard_rows = [row for row in rows if row["is_leopard"]]
    confidences = [row["confidence"] for row in rows]
    print(json.dumps({
        "report": str(REPORT_PATH),
        "total_images": len(rows),
        "predicted_leopard": len(leopard_rows),
        "predicted_other": len(rows) - len(leopard_rows),
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
    }))


if __name__ == "__main__":
    main()