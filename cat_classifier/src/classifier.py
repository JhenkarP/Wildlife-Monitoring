from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    scientific_name: str | None
    confidence: float
    status: str
    bounding_box: tuple[int, int, int, int] | None = None
    localized_image: Image.Image | None = None
    pixel_width: int | None = None
    pixel_height: int | None = None


class CatClassifier:
    """Run inference with the trained ResNet18 Felidae checkpoint."""

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.labels: list[str] = []
        self.transform = ResNet18_Weights.DEFAULT.transforms()
        if self.is_ready:
            saved = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            self.labels = list(saved["labels"])
            self.model = resnet18(weights=None)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.labels))
            self.model.load_state_dict(saved["state_dict"])
            self.model.to(self.device).eval()

    @property
    def is_ready(self) -> bool:
        return self.checkpoint is not None and self.checkpoint.exists()

    def predict(self, image: Image.Image, taxonomy: dict[str, Any]) -> ClassificationResult:
        if image.width < 1 or image.height < 1:
            raise ValueError("The uploaded image is empty")
        if self.model is None:
            return ClassificationResult(
                label="Unknown / insufficient evidence",
                scientific_name=None,
                confidence=0.0,
                status="model_not_trained",
            )
        with torch.inference_mode():
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            index = int(probabilities.argmax().item())
        item = next(species for species in taxonomy["species"] if species["id"] == self.labels[index])
        return ClassificationResult(
            label=item["common_name"],
            scientific_name=item["scientific_name"],
            confidence=float(probabilities[index].item()),
            status="predicted",
        )

    def predict_with_localization(
        self,
        image: Image.Image,
        taxonomy: dict[str, Any],
        threshold: float = 0.35,
    ) -> ClassificationResult:
        """Classify the image and draw a Grad-CAM approximation of the animal region."""
        if image.width < 1 or image.height < 1:
            raise ValueError("The uploaded image is empty")
        if self.model is None:
            return self.predict(image, taxonomy)

        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def capture_activation(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            activations.append(output)
            output.register_hook(lambda gradient: gradients.append(gradient))

        target_layer = self.model.layer4[-1].conv2
        handle = target_layer.register_forward_hook(capture_activation)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            index = int(probabilities.argmax().item())
            logits[0, index].backward()
        finally:
            handle.remove()

        if not activations or not gradients:
            return self.predict(image, taxonomy)

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=rgb_image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
        cam -= cam.min()
        maximum = float(cam.max().item())
        if maximum > 0:
            cam /= maximum

        mask = cam >= threshold
        coordinates = torch.nonzero(mask, as_tuple=False)
        if coordinates.numel() == 0:
            bounding_box = None
            localized_image = rgb_image
            pixel_width = pixel_height = None
        else:
            top, left = coordinates.min(dim=0).values.tolist()
            bottom, right = coordinates.max(dim=0).values.tolist()
            bounding_box = (int(left), int(top), int(right), int(bottom))
            pixel_width = bounding_box[2] - bounding_box[0]
            pixel_height = bounding_box[3] - bounding_box[1]
            localized_image = rgb_image.copy()
            draw = ImageDraw.Draw(localized_image)
            draw.rectangle(bounding_box, outline=(255, 32, 32), width=max(3, rgb_image.width // 180))

        item = next(species for species in taxonomy["species"] if species["id"] == self.labels[index])
        return ClassificationResult(
            label=item["common_name"],
            scientific_name=item["scientific_name"],
            confidence=float(probabilities[index].item()),
            status="predicted",
            bounding_box=bounding_box,
            localized_image=localized_image,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
        )
