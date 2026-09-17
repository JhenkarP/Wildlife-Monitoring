"""Download Nilgai photos observed in India from iNaturalist."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests
import truststore

truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.inaturalist.org/v1/observations"
TAXON_ID = 42426
INDIA_PLACE_ID = 6681
OUTPUT_ROOT = ROOT / "data" / "training" / "images" / "nilgai"


def fetch_page(session: requests.Session, page: int) -> dict:
    params = {"taxon_id": TAXON_ID, "place_id": INDIA_PLACE_ID, "photos": "true", "per_page": 200, "page": page, "order_by": "votes", "order": "desc"}
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
    parser.add_argument("--max-pages", type=int, default=30)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Animal-Monitoring Nilgai India dataset collector"
    manifest_rows: list[dict[str, object]] = []
    seen_photos: set[int] = set()
    for page in range(1, args.max_pages + 1):
        observations = fetch_page(session, page).get("results", [])
        if not observations:
            break
        for observation in observations:
            for photo in observation.get("photos") or []:
                photo_id = photo.get("id")
                if not photo_id or photo_id in seen_photos:
                    continue
                photo_url = (photo.get("url") or "").replace("/square.", "/original.")
                if not photo_url:
                    continue
                image_path = OUTPUT_ROOT / f"{observation['id']}_{photo_id}.jpg"
                try:
                    if not image_path.exists():
                        image_response = session.get(photo_url, timeout=30)
                        image_response.raise_for_status()
                        image_path.write_bytes(image_response.content)
                    seen_photos.add(photo_id)
                    manifest_rows.append({"image_path": image_path.relative_to(OUTPUT_ROOT).as_posix(), "label": "nilgai", "observation_id": observation["id"], "photo_id": photo_id, "taxon": observation.get("taxon", {}).get("name", "Boselaphus tragocamelus"), "place_guess": observation.get("place_guess", ""), "source_url": f"https://www.inaturalist.org/observations/{observation['id']}", "photo_url": photo_url, "license": photo.get("license_code") or "unknown", "attribution": photo.get("attribution", "")})
                    print(f"downloaded={len(manifest_rows)}/{args.count}", flush=True)
                    if len(manifest_rows) >= args.count:
                        break
                except (requests.RequestException, OSError) as error:
                    print(f"skipped observation={observation.get('id')} photo={photo_id}: {error}", flush=True)
            if len(manifest_rows) >= args.count:
                break
        if len(manifest_rows) >= args.count:
            break
    if manifest_rows:
        with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"saved={len(manifest_rows)} manifest={OUTPUT_ROOT / 'manifest.csv'}", flush=True)
    if len(manifest_rows) < args.count:
        raise SystemExit(f"Only collected {len(manifest_rows)} images; requested {args.count}.")


if __name__ == "__main__":
    main()
