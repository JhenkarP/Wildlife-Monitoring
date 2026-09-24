"""Wildlife detection and species classification for uploaded images."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torchvision import models, transforms


CUSTOM_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "indian_wildlife_resnet18.pt"


ANNOTATION_COLOR = "#0B2D5C"


def _annotate_image_classification(
    image: Image.Image,
    label: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    font_size = max(18, min(width, height) // 28)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    text = f"{label} {confidence:.3f}"
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x, y, box_width, box_height = bbox
    box = (x * width, y * height, (x + box_width) * width, (y + box_height) * height)
    text_x = max(0, min(box[0], width - text_width - 8))
    text_y = box[1] - text_height - 8 if box[1] >= text_height + 8 else box[1] + 4
    draw.rectangle(box, outline=ANNOTATION_COLOR, width=4)
    draw.rectangle(
        (text_x, text_y, text_x + text_width + 8, text_y + text_height + 8),
        fill=ANNOTATION_COLOR,
    )
    draw.text((text_x + 4, text_y + 4), text, fill="white", font=font)
    return annotated


@lru_cache(maxsize=1)
def _load_custom_model():
    checkpoint = torch.load(CUSTOM_MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.25),
        torch.nn.Linear(model.fc.in_features, len(classes)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def _speciesnet_prediction(image: Image.Image):
    model = _load_speciesnet()
    with TemporaryDirectory(prefix="speciesnet-") as directory:
        image_path = f"{directory}/uploaded-image.jpg"
        image.save(image_path, format="JPEG")
        predictions = model.predict(
            filepaths=[image_path],
            run_mode="single_thread",
            batch_size=1,
            progress_bars=False,
        )
    prediction = next(iter(predictions.values()))
    return prediction[0] if isinstance(prediction, list) else prediction


def _strongest_speciesnet_detection(prediction):
    raw_detections = prediction.get("detections", [])
    if not raw_detections:
        return None
    return max(
        raw_detections,
        key=lambda item: (
            float(item.get("conf", 0.0)),
            float(item["bbox"][2]) * float(item["bbox"][3]),
        ),
    )


def classify_with_custom_model(image: Image.Image):
    """Classify an upload with the fine-tuned 13-species wildlife model."""
    model, classes = _load_custom_model()
    localization = _strongest_speciesnet_detection(_speciesnet_prediction(image))
    classification_image = image
    bbox = (0.0, 0.0, 1.0, 1.0)
    if localization:
        x, y, box_width, box_height = [float(value) for value in localization["bbox"]]
        width, height = image.size
        left = max(0, int(x * width))
        top = max(0, int(y * height))
        right = min(width, int((x + box_width) * width))
        bottom = min(height, int((y + box_height) * height))
        if right > left and bottom > top:
            classification_image = image.crop((left, top, right, bottom))
            bbox = (x, y, box_width, box_height)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(classification_image).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(image_tensor).softmax(dim=1)[0]
    index = int(probabilities.argmax())
    label = classes[index]
    confidence = round(float(probabilities[index]), 3)
    annotated = _annotate_image_classification(image, label, confidence, bbox)
    return annotated, [
        {
            "label": label,
            "confidence": confidence,
            "classification_model": "fine-tuned ResNet18",
            "localization_model": "SpeciesNet",
            "bbox": [round(value, 4) for value in bbox],
            "bbox_note": "SpeciesNet localization box used before fine-tuned ResNet18 classification.",
        }
    ]


@lru_cache(maxsize=1)
def _load_speciesnet():
    import truststore

    truststore.inject_into_ssl()
    from speciesnet import DEFAULT_MODEL, SpeciesNet

    return SpeciesNet(DEFAULT_MODEL, components="all", geofence=False, multiprocessing=False)


def classify_with_speciesnet(image: Image.Image):
    """Classify an uploaded image with the official SpeciesNet ensemble."""
    prediction = _speciesnet_prediction(image)
    prediction_label = str(prediction.get("prediction", "unknown")).split(";")[-1]
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = image.size
    font_size = max(18, min(width, height) // 28)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    raw_detections = prediction.get("detections", [])
    detections = []
    if raw_detections:
        # SpeciesNet returns one image-level species prediction; render only its
        # strongest box to avoid showing duplicate nested detections.
        detection = _strongest_speciesnet_detection(prediction)
        x, y, box_width, box_height = detection["bbox"]
        box = (x * width, y * height, (x + box_width) * width, (y + box_height) * height)
        confidence = round(float(prediction.get("prediction_score", detection["conf"])), 3)
        text = f"{prediction_label} {confidence:.3f}"
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = max(0, min(box[0], width - text_width - 8))
        text_y = box[1] - text_height - 8 if box[1] >= text_height + 8 else box[1] + 4
        draw.rectangle(box, outline=ANNOTATION_COLOR, width=4)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + 8, text_y + text_height + 8),
            fill=ANNOTATION_COLOR,
        )
        draw.text((text_x + 4, text_y + 4), text, fill="white", font=font)
        detections.append(
            {
                "label": prediction_label,
                "taxonomy": prediction.get("prediction", "unknown"),
                "confidence": confidence,
                "bbox": [round(value, 4) for value in (x, y, box_width, box_height)],
            }
        )
    if not detections:
        detections.append(
            {
                "label": prediction_label,
                "taxonomy": prediction.get("prediction", "unknown"),
                "confidence": round(float(prediction.get("prediction_score", 0.0)), 3),
            }
        )
    return annotated, detections