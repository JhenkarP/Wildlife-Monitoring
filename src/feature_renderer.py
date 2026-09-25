"""Render reusable feature annotations for any supported wildlife species."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def render_feature_boxes(image: Image.Image, record: dict) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    font_size = max(14, min(22, min(image.size) // 42))
    font = None
    for font_name in ("arialbd.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    colors = {"visible": "#16803c"}
    annotations = []
    for name, value in record.get("features", {}).items():
        if not isinstance(value, dict) or value.get("status") not in colors:
            continue
        bbox = value.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, width, height = [max(0.0, min(1.0, float(part))) for part in bbox]
        left = x * image.width
        top = y * image.height
        right = min(image.width, (x + width) * image.width)
        bottom = min(image.height, (y + height) * image.height)
        color = colors[value["status"]]
        draw.rectangle((left, top, right, bottom), outline=color, width=4)
        label = f"{name.replace('_', ' ').title()} - {value['status'].title()} - {float(value.get('confidence', 0.0)):.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        annotations.append((left, top, right, bottom, color, label, text_width, text_height, text_box[1]))

    occupied = []
    for left, top, right, bottom, color, label, text_width, text_height, text_offset in annotations:
        padding = 5
        label_width = text_width + padding * 2
        label_height = text_height + padding * 2
        gap = 7
        candidates = [
            (left, top - label_height - gap),
            (left, bottom + gap),
            (right - label_width, top - label_height - gap),
            (right - label_width, bottom + gap),
            (left - label_width - gap, top),
            (right + gap, top),
            (left, top + (bottom - top - label_height) / 2),
        ]
        label_left, label_top = candidates[-1]
        for candidate_left, candidate_top in candidates:
            candidate_left = max(0, min(image.width - label_width, candidate_left))
            candidate_top = max(0, min(image.height - label_height, candidate_top))
            candidate = (
                candidate_left,
                candidate_top,
                candidate_left + label_width,
                candidate_top + label_height,
            )
            if not any(
                candidate[0] < other[2] + 3
                and candidate[2] > other[0] - 3
                and candidate[1] < other[3] + 3
                and candidate[3] > other[1] - 3
                for other in occupied
            ):
                label_left, label_top = candidate_left, candidate_top
                break
        label_box = (label_left, label_top, label_left + label_width, label_top + label_height)
        occupied.append(label_box)
        draw.rounded_rectangle(label_box, radius=5, fill=color)
        draw.text((label_left + padding, label_top + padding - text_offset), label, fill="white", font=font)
    return marked