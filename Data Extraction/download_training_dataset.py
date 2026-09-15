"""Collect a reproducible iNaturalist image dataset for transfer learning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import time
from pathlib import Path

import requests
import truststore
from PIL import Image
from io import BytesIO

truststore.inject_into_ssl()

ROOT = Path(__file__).parents[1]
OUTPUT_ROOT = ROOT / "data" / "training"
API_ROOT = "https://api.inaturalist.org/v1"
SPECIES = {
    "asian_elephant": "Elephas maximus",
    "greater_one_horned_rhino": "Rhinoceros unicornis",
    "leopard": "Panthera pardus",
    "gaur": "Bos gaurus",
}


def get_api_response(session: requests.Session, params: dict) -> requests.Response:
    last_error = None
    for attempt in range(3):
        try:
            response = session.get(f"{API_ROOT}/observations", params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"iNaturalist request failed after 3 attempts: {last_error}")


def split_rows(rows: list[dict], seed: int) -> None:
    random.Random(seed).shuffle(rows)
    total = len(rows)
    validation_count = max(1, round(total * 0.15))
    test_count = max(1, round(total * 0.15))
    for index, row in enumerate(rows):
        row["split"] = "test" if index < test_count else "validation" if index < test_count + validation_count else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=30, help="Number of images to collect for each species.")
    parser.add_argument("--place-id", type=int, default=None, help="Optional iNaturalist place ID, such as India.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_class < 24:
        raise SystemExit("Use at least 24 images per class so train, validation, and test splits remain meaningful.")

    image_root = OUTPUT_ROOT / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Animal-Monitoring training dataset collector"
    label_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for label, scientific_name in SPECIES.items():
        print(f"starting {label} ({scientific_name})", flush=True)
        class_dir = image_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        params = {"taxon_name": scientific_name, "quality_grade": "research", "photos": "true", "per_page": 200, "order_by": "votes", "order": "desc"}
        if args.place_id is not None:
            params["place_id"] = args.place_id
        response = get_api_response(session, params)
        rows_for_class = []
        seen_hashes = set()
        for observation in response.json().get("results", []):
            for photo in observation.get("photos") or []:
                photo_url = (photo.get("url") or "").replace("/square.", "/medium.")
                if not photo_url:
                    continue
                try:
                    filename = f"{observation['id']}_{photo.get('id', '')}.jpg"
                    image_path = class_dir / filename
                    if image_path.exists():
                        image_bytes = image_path.read_bytes()
                    else:
                        image_response = session.get(photo_url, timeout=3)
                        image_response.raise_for_status()
                        image_bytes = image_response.content
                    content_hash = hashlib.sha256(image_bytes).hexdigest()
                    if content_hash in seen_hashes:
                        continue
                    with Image.open(BytesIO(image_bytes)) as image:
                        image.verify()
                    image_path.write_bytes(image_bytes)
                    seen_hashes.add(content_hash)
                    relative_path = image_path.relative_to(OUTPUT_ROOT).as_posix()
                    row = {"image_path": relative_path, "label": label}
                    rows_for_class.append(row)
                    manifest_rows.append({
                        "image_path": relative_path,
                        "label": label,
                        "observation_id": observation.get("id", ""),
                        "photo_id": photo.get("id", ""),
                        "scientific_name": scientific_name,
                        "source_url": f"https://www.inaturalist.org/observations/{observation.get('id', '')}",
                        "photo_url": photo_url,
                        "license": photo.get("license_code") or "unknown",
                        "attribution": photo.get("attribution") or "",
                    })
                    if len(rows_for_class) >= args.per_class:
                        break
                except (requests.RequestException, OSError) as error:
                    print(f"skipping {label} photo: {error}", flush=True)
            if len(rows_for_class) >= args.per_class:
                break
        if len(rows_for_class) < args.per_class:
            raise SystemExit(f"Only collected {len(rows_for_class)} images for {label}; requested {args.per_class}.")
        split_rows(rows_for_class, args.seed)
        label_rows.extend(rows_for_class)
        print(f"{label}: collected {len(rows_for_class)}", flush=True)

    with (OUTPUT_ROOT / "labels.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(label_rows)
    with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = list(manifest_rows[0])
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"saved labels: {OUTPUT_ROOT / 'labels.csv'}", flush=True)
    print(f"saved manifest: {OUTPUT_ROOT / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()