from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load_speciesnet_classifier():
    import truststore

    truststore.inject_into_ssl()
    from speciesnet import DEFAULT_MODEL, SpeciesNet

    return SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False, multiprocessing=False)


IMAGE_DIR = ROOT / "data" / "training" / "images" / "indian leopard"
REPORT_PATH = ROOT / "data" / "processed" / "speciesnet_leopard_evaluation.csv"


def main() -> None:
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    model = load_speciesnet_classifier()
    with TemporaryDirectory(prefix="speciesnet-leopard-evaluation-") as directory:
        paths = []
        for image_path in image_paths:
            target = Path(directory) / image_path.name
            with Image.open(image_path) as source:
                source.convert("RGB").save(target, format="JPEG")
            paths.append(str(target))
        predictions = model.predict(
            filepaths=paths,
            run_mode="single_thread",
            batch_size=16,
            progress_bars=False,
        )

    rows = []
    for image_path in image_paths:
        prediction = predictions[str(Path(directory) / image_path.name)]
        if isinstance(prediction, list):
            prediction = prediction[0]
        raw_detections = prediction.get("detections", [])
        detection = max(
            raw_detections,
            key=lambda item: (
                float(item.get("conf", 0.0)),
                float(item["bbox"][2]) * float(item["bbox"][3]),
            ),
            default=None,
        )
        prediction_label = str(prediction.get("prediction", "unknown")).split(";")[-1]
        rows.append(
            {
                "filename": image_path.name,
                "predicted_label": prediction_label,
                "confidence": round(float(prediction.get("prediction_score", 0.0)), 3),
                "taxonomy": prediction.get("prediction", "unknown"),
                "box_count": "not_run",
                "bbox": "",
            }
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = Counter(row["predicted_label"] for row in rows)
    box_counts = Counter(int(row["box_count"]) for row in rows)
    confidences = [float(row["confidence"]) for row in rows]
    print(f"report={REPORT_PATH}")
    print(f"total={len(rows)}")
    print(f"labels={dict(labels)}")
    print(f"box_counts={dict(box_counts)}")
    print(f"confidence_min={min(confidences):.3f}")
    print(f"confidence_max={max(confidences):.3f}")
    print(f"confidence_mean={sum(confidences) / len(confidences):.3f}")


if __name__ == "__main__":
    main()