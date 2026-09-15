"""Append India-filtered Indian rhinoceros images for manual review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from io import BytesIO
from pathlib import Path

import requests
import truststore
from PIL import Image

truststore.inject_into_ssl()

ROOT = Path(__file__).parents[1]
OUTPUT_ROOT = ROOT / "data" / "training"
API_ROOT = "https://api.inaturalist.org/v1/observations"
LABEL = "greater_one_horned_rhino"
SCIENTIFIC_NAME = "Rhinoceros unicornis"
PLACE_ID = 6681


def request_json(session: requests.Session, params: dict) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            response = session.get(API_ROOT, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"iNaturalist request failed after 3 attempts: {last_error}")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=130, help="Target total number of rhino images.")
    parser.add_argument("--seed-page", type=int, default=1, help="First iNaturalist result page to inspect.")
    args = parser.parse_args()
    if args.total < 1:
        raise SystemExit("--total must be positive")

    labels_path = OUTPUT_ROOT / "labels.csv"
    manifest_path = OUTPUT_ROOT / "manifest.csv"
    review_path = OUTPUT_ROOT / "review_log.csv"
    labels = read_rows(labels_path)
    manifest = read_rows(manifest_path)
    review_log = read_rows(review_path)

    rhino_labels = [row for row in labels if row.get("label") == LABEL]
    if len(rhino_labels) >= args.total:
        print(f"{LABEL}: already has {len(rhino_labels)} images; nothing to download")
        return

    known_paths = {row.get("image_path", "") for row in labels}
    known_observations = {
        row.get("observation_id", "")
        for row in manifest
        if row.get("label") == LABEL
    }
    known_photos = {
        row.get("photo_id", "")
        for row in manifest
        if row.get("label") == LABEL
    }
    known_hashes = set()
    for row in rhino_labels:
        image_path = OUTPUT_ROOT / row["image_path"]
        if image_path.exists():
            known_hashes.add(hashlib.sha256(image_path.read_bytes()).hexdigest())

    image_root = OUTPUT_ROOT / "images" / LABEL
    image_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Animal-Monitoring rhino review dataset collector"
    target_new = args.total - len(rhino_labels)
    new_labels = []
    new_manifest = []
    new_review = []
    page = args.seed_page

    while len(new_labels) < target_new:
        payload = request_json(session, {
            "taxon_name": SCIENTIFIC_NAME,
            "quality_grade": "research",
            "photos": "true",
            "place_id": PLACE_ID,
            "per_page": 200,
            "page": page,
            "order_by": "votes",
            "order": "desc",
        })
        observations = payload.get("results", [])
        if not observations:
            break
        for observation in observations:
            observation_id = str(observation.get("id", ""))
            if not observation_id or observation_id in known_observations:
                continue
            photos = observation.get("photos") or []
            if not photos:
                continue
            photo = photos[0]
            photo_id = str(photo.get("id", ""))
            photo_url = (photo.get("url") or "").replace("/square.", "/medium.")
            if not photo_id or not photo_url or photo_id in known_photos:
                continue
            try:
                response = session.get(photo_url, timeout=30)
                response.raise_for_status()
                image_bytes = response.content
                content_hash = hashlib.sha256(image_bytes).hexdigest()
                if content_hash in known_hashes:
                    continue
                with Image.open(BytesIO(image_bytes)) as image:
                    image.verify()
                filename = f"{observation_id}_{photo_id}.jpg"
                image_path = image_root / filename
                image_path.write_bytes(image_bytes)
            except (requests.RequestException, OSError) as error:
                print(f"skipping observation {observation_id}: {error}", flush=True)
                continue

            relative_path = image_path.relative_to(OUTPUT_ROOT).as_posix()
            label_row = {"image_path": relative_path, "label": LABEL, "split": "review"}
            manifest_row = {
                "image_path": relative_path,
                "label": LABEL,
                "observation_id": observation_id,
                "photo_id": photo_id,
                "scientific_name": SCIENTIFIC_NAME,
                "source_url": f"https://www.inaturalist.org/observations/{observation_id}",
                "photo_url": photo_url,
                "license": photo.get("license_code") or "unknown",
                "attribution": photo.get("attribution") or "",
            }
            new_labels.append(label_row)
            new_manifest.append(manifest_row)
            new_review.append({"image_path": relative_path, "label": LABEL, "decision": "pending", "reason": "awaiting visual and license review"})
            known_paths.add(relative_path)
            known_observations.add(observation_id)
            known_photos.add(photo_id)
            known_hashes.add(content_hash)
            print(f"downloaded {len(new_labels)}/{target_new}: {relative_path}", flush=True)
            if len(new_labels) >= target_new:
                break
        page += 1
        if page > 20:
            break

    if len(new_labels) < target_new:
        raise SystemExit(f"Only downloaded {len(new_labels)} new images; needed {target_new}.")

    labels.extend(new_labels)
    manifest.extend(new_manifest)
    review_log.extend(new_review)
    write_rows(labels_path, ["image_path", "label", "split"], labels)
    write_rows(manifest_path, ["image_path", "label", "observation_id", "photo_id", "scientific_name", "source_url", "photo_url", "license", "attribution"], manifest)
    write_rows(review_path, ["image_path", "label", "decision", "reason"], review_log)
    print(f"downloaded {len(new_labels)} new images; {LABEL} total is {len(rhino_labels) + len(new_labels)}")


if __name__ == "__main__":
    main()