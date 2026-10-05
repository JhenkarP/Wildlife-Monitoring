"""Download a quality-filtered Asiatic lion photo set from iNaturalist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import truststore
from PIL import Image

truststore.inject_into_ssl()


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.inaturalist.org/v1/observations"
TAXON_ID = 557485
INDIA_PLACE_ID = 6681
OUTPUT_ROOT = ROOT / "data" / "images" / "asiatic_lion"


def image_quality(image_bytes: bytes) -> dict[str, float | int]:
    with Image.open(BytesIO(image_bytes)) as source:
        width, height = source.size
        image = source.convert("L")
    image.thumbnail((512, 512))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    gradient_x = np.diff(pixels, axis=1)
    gradient_y = np.diff(pixels, axis=0)
    sharpness = float((gradient_x ** 2).mean() + (gradient_y ** 2).mean())
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    return {
        "width": width,
        "height": height,
        "sharpness": round(sharpness, 6),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
    }


def fetch_page(session: requests.Session, page: int) -> dict:
    params = {
        "taxon_id": TAXON_ID,
        "place_id": INDIA_PLACE_ID,
        "quality_grade": "research",
        "photos": "true",
        "per_page": 200,
        "page": page,
        "order_by": "votes",
        "order": "desc",
    }
    for attempt in range(3):
        try:
            response = session.get(API_ROOT, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--min-width", type=int, default=800)
    parser.add_argument("--min-height", type=int, default=600)
    args = parser.parse_args()

    image_dir = OUTPUT_ROOT
    image_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Animal-Monitoring Asiatic lion dataset collector"
    manifest_rows: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    seen_photos: set[int] = set()

    for page in range(1, 30):
        payload = fetch_page(session, page)
        observations = payload.get("results", [])
        if not observations:
            break
        for observation in observations:
            for photo in observation.get("photos") or []:
                photo_id = photo.get("id")
                license_code = photo.get("license_code") or "unknown"
                if not photo_id or photo_id in seen_photos:
                    continue
                photo_url = (photo.get("url") or "").replace("/square.", "/original.")
                if not photo_url:
                    continue
                try:
                    image_response = session.get(photo_url, timeout=30)
                    image_response.raise_for_status()
                    image_bytes = image_response.content
                    quality = image_quality(image_bytes)
                    if quality["width"] < args.min_width or quality["height"] < args.min_height:
                        continue
                    if quality["sharpness"] < 0.002 or quality["brightness"] < 0.12 or quality["contrast"] < 0.08:
                        continue
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    if digest in seen_hashes:
                        continue
                    filename = f"{observation['id']}_{photo_id}.jpg"
                    image_path = image_dir / filename
                    image_path.write_bytes(image_bytes)
                    seen_hashes.add(digest)
                    seen_photos.add(photo_id)
                    manifest_rows.append({
                        "image_path": image_path.relative_to(OUTPUT_ROOT).as_posix(),
                        "label": "asiatic_lion",
                        "observation_id": observation["id"],
                        "photo_id": photo_id,
                        "taxon": observation.get("taxon", {}).get("name", "Panthera leo leo"),
                        "place_guess": observation.get("place_guess", ""),
                        "source_url": f"https://www.inaturalist.org/observations/{observation['id']}",
                        "photo_url": photo_url,
                        "license": license_code,
                        "attribution": photo.get("attribution", ""),
                        **quality,
                    })
                    print(f"accepted={len(manifest_rows)}/{args.count}", flush=True)
                    if len(manifest_rows) >= args.count:
                        break
                except (requests.RequestException, OSError, ValueError) as error:
                    print(f"skipped observation={observation.get('id')} photo={photo_id}: {error}", flush=True)
            if len(manifest_rows) >= args.count:
                break
        if len(manifest_rows) >= args.count:
            break

    if len(manifest_rows) < args.count:
        raise SystemExit(f"Only collected {len(manifest_rows)} quality images; requested {args.count}.")

    manifest_path = OUTPUT_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"saved={len(manifest_rows)} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()