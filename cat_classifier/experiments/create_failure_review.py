from __future__ import annotations

import html
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "cat_classifier" / "data" / "inat"
CHECKPOINT_PATH = ROOT / "cat_classifier" / "models" / "panthera_five_species_lion_correction.pt"
OUTPUT_PATH = ROOT / "cat_classifier" / "data" / "experiments" / "current_failure_review.html"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model() -> tuple[torch.nn.Module, list[str], Compose]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    labels = list(checkpoint["labels"])
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, labels, transform


def scan_failures() -> tuple[list[dict[str, str]], list[str]]:
    model, labels, transform = load_model()
    failures: list[dict[str, str]] = []
    for true_label in labels:
        for path in sorted((DATA_ROOT / true_label).iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    tensor = transform(image.convert("RGB")).unsqueeze(0)
                with torch.inference_mode():
                    predicted_label = labels[int(model(tensor).argmax(1))]
            except Exception:
                continue
            if predicted_label != true_label:
                relative_path = path.relative_to(ROOT).as_posix()
                image_source = Path("../inat") / true_label / path.name
                failures.append(
                    {
                        "path": relative_path,
                        "image": image_source.as_posix(),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                    }
                )
    return failures, labels


def render(failures: list[dict[str, str]], labels: list[str]) -> str:
    label_options = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)}</option>'
        for label in labels
    )
    cards = []
    for index, item in enumerate(failures):
        cards.append(
            f"""
            <article class="card" data-index="{index}">
              <img src="{html.escape(item['image'])}" loading="lazy" alt="{html.escape(item['path'])}">
              <div class="details">
                <strong>{html.escape(item['path'])}</strong>
                <div>Folder label: <b>{html.escape(item['true_label'])}</b></div>
                <div>Model prediction: <b class="wrong">{html.escape(item['predicted_label'])}</b></div>
                <label>Status
                  <select class="status">
                    <option value="keep">Keep as valid hard example</option>
                    <option value="relabel">Relabel for fine-tuning</option>
                    <option value="exclude">Exclude from training</option>
                  </select>
                </label>
                <label class="corrected hidden">Correct label
                  <select class="correct-label"><option value="">Choose a label</option>{label_options}</select>
                </label>
                <textarea class="note" placeholder="Optional review note"></textarea>
              </div>
            </article>
            """
        )
    payload = json.dumps(failures)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Current Panthera Failure Review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f3f5f7; color: #17202a; }}
header {{ position: sticky; top: 0; z-index: 2; background: #17202a; color: white; padding: 16px; margin: -24px -24px 20px; }}
button {{ padding: 9px 14px; margin-right: 8px; cursor: pointer; }}
#grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
.card {{ background: white; border: 1px solid #d7dde3; border-radius: 8px; overflow: hidden; }}
.card img {{ display: block; width: 100%; height: 250px; object-fit: contain; background: #111; }}
.details {{ padding: 12px; display: grid; gap: 8px; }}
strong {{ font-size: 12px; overflow-wrap: anywhere; }}
label {{ display: grid; gap: 4px; font-size: 13px; }}
select, textarea {{ width: 100%; box-sizing: border-box; padding: 7px; }}
textarea {{ min-height: 48px; resize: vertical; }}
.wrong {{ color: #b42318; }} .hidden {{ display: none; }}
.card[data-status="exclude"] {{ opacity: .55; }}
.card[data-status="relabel"] {{ border-color: #d97706; }}
</style></head><body>
<header><h1>Current Panthera Failure Review</h1>
<div>{len(failures)} failed predictions. Review each image, then export the JSON for fine-tuning.</div>
<p><button id="export">Export review JSON</button><button id="reset">Reset choices</button></p></header>
<main id="grid">{''.join(cards)}</main>
<script>
const original = {payload};
const cards = [...document.querySelectorAll('.card')];
cards.forEach(card => {{
  const status = card.querySelector('.status');
  const corrected = card.querySelector('.corrected');
  status.addEventListener('change', () => {{
    card.dataset.status = status.value;
    corrected.classList.toggle('hidden', status.value !== 'relabel');
  }});
}});
document.getElementById('reset').onclick = () => location.reload();
document.getElementById('export').onclick = () => {{
  const review = cards.map((card, index) => ({{
    ...original[index],
    status: card.querySelector('.status').value,
    corrected_label: card.querySelector('.correct-label').value,
    note: card.querySelector('.note').value
  }}));
  const blob = new Blob([JSON.stringify(review, null, 2)], {{type: 'application/json'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'panthera_failure_review.json';
  link.click();
}};
</script></body></html>"""


if __name__ == "__main__":
    failures, labels = scan_failures()
    OUTPUT_PATH.write_text(render(failures, labels), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} with {len(failures)} failures")