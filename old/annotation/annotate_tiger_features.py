"""Create bounding-box annotations for Bengal tiger visual features."""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk


FEATURES = {
    "1": ("face_muzzle", "#ff3b30"),
    "2": ("facial_stripes", "#00a8ff"),
    "3": ("ears", "#ff9500"),
    "4": ("body_shoulders", "#34c759"),
    "5": ("flank_stripes", "#af52de"),
    "6": ("tail_banding", "#5856d6"),
}


class FeatureAnnotator:
    def __init__(self, image_paths: list[Path], output_path: Path, max_size: tuple[int, int]):
        self.image_paths = image_paths
        self.output_path = output_path
        self.max_width, self.max_height = max_size
        self.index = 0
        self.annotations: dict[str, list[dict]] = self._load_annotations()
        self.current_boxes: list[dict] = []
        self.active_feature = "1"
        self.start = None
        self.scale = 1.0

        self.root = tk.Tk()
        self.root.title("Bengal Tiger Feature Annotation")
        self.root.bind("<Key>", self.on_key)
        self.canvas = tk.Canvas(self.root, cursor="crosshair", bg="#202124")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.start_box)
        self.canvas.bind("<B1-Motion>", self.drag_box)
        self.canvas.bind("<ButtonRelease-1>", self.finish_box)
        self.status = tk.Label(self.root, anchor="w", justify="left")
        self.status.pack(fill=tk.X)
        self.show_image()

    def _load_annotations(self):
        if not self.output_path.exists():
            return {}
        return {
            item["image"]: item["boxes"]
            for item in map(json.loads, self.output_path.read_text(encoding="utf-8").splitlines())
            if item.get("image")
        }

    def save(self):
        self.annotations[str(self.image_paths[self.index])] = self.current_boxes
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for image, boxes in sorted(self.annotations.items()):
                handle.write(json.dumps({"image": image, "boxes": boxes}) + "\n")

    def show_image(self):
        path = self.image_paths[self.index]
        image = Image.open(path).convert("RGB")
        self.original_size = image.size
        self.scale = min(self.max_width / image.width, self.max_height / image.height, 1.0)
        display_size = (int(image.width * self.scale), int(image.height * self.scale))
        display = image.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.config(width=display.width, height=display.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.current_boxes = list(self.annotations.get(str(path), []))
        self.draw_boxes()
        self.status.config(text=self.instructions(path))

    def instructions(self, path):
        feature_text = " | ".join(f"{key}: {name}" for key, (name, _) in FEATURES.items())
        return f"{self.index + 1}/{len(self.image_paths)}  {path.name}\n{feature_text}\nActive: {FEATURES[self.active_feature][0]} | N: next  S: save  B: back  Q: quit  R: clear image"

    def draw_boxes(self):
        for box in self.current_boxes:
            color = FEATURES[box["feature_key"]][1]
            x1, y1, x2, y2 = [value * self.scale for value in box["box"]]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)
            self.canvas.create_text(x1 + 4, y1 + 4, text=FEATURES[box["feature_key"]][0], anchor="nw", fill=color)

    def on_key(self, event):
        if event.char in FEATURES:
            self.active_feature = event.char
            self.status.config(text=self.instructions(self.image_paths[self.index]))
        elif event.char.lower() == "s":
            self.save()
        elif event.char.lower() == "n":
            self.save()
            if self.index < len(self.image_paths) - 1:
                self.index += 1
                self.show_image()
        elif event.char.lower() == "b" and self.index > 0:
            self.save()
            self.index -= 1
            self.show_image()
        elif event.char.lower() == "r":
            self.current_boxes = []
            self.show_image()
        elif event.char.lower() == "q":
            self.save()
            self.root.destroy()

    def start_box(self, event):
        self.start = (event.x, event.y)

    def drag_box(self, event):
        if self.start:
            self.canvas.delete("draft")
            color = FEATURES[self.active_feature][1]
            self.canvas.create_rectangle(*self.start, event.x, event.y, outline=color, width=3, tags="draft")

    def finish_box(self, event):
        if not self.start:
            return
        x1, y1 = self.start
        x2, y2 = event.x, event.y
        self.start = None
        self.canvas.delete("draft")
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < 8 or bottom - top < 8:
            return
        self.current_boxes.append({
            "feature_key": self.active_feature,
            "feature": FEATURES[self.active_feature][0],
            "box": [
                round(left / self.scale, 2),
                round(top / self.scale, 2),
                round(right / self.scale, 2),
                round(bottom / self.scale, 2),
            ],
        })
        self.draw_boxes()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/training/images/bengal tiger"))
    parser.add_argument("--output", type=Path, default=Path("data/feature_annotations/bengal_tiger.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.limit]
    if not paths:
        raise SystemExit(f"No images found in {args.images}")
    FeatureAnnotator(paths, args.output, (1400, 850)).run()


if __name__ == "__main__":
    main()