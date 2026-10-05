from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).parents[1]
TAXONOMY_PATH = PROJECT_ROOT / "taxonomy" / "felidae_species.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "inat"
API_ROOT = "https://api.inaturalist.org/v1/observations"
TAXA_API_ROOT = "https://api.inaturalist.org/v1/taxa"


def load_species() -> list[dict[str, Any]]:
    with TAXONOMY_PATH.open(encoding="utf-8") as file:
        return json.load(file)["species"]


def resolve_taxon_id(session: requests.Session, scientific_name: str) -> int | None:
    response = session.get(
        TAXA_API_ROOT,
        params={"q": scientific_name, "per_page": 20},
        timeout=30,
    )
    response.raise_for_status()
    for taxon in response.json().get("results", []):
        if taxon.get("name") == scientific_name:
            return int(taxon["id"])
    return None


def fetch_observations(session: requests.Session, taxon_id: int, page: int) -> list[dict[str, Any]]:
    response = session.get(
        API_ROOT,
        params={
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",
            "per_page": 200,
            "page": page,
            "order_by": "votes",
            "order": "desc",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def image_url(photo: dict[str, Any]) -> str:
    return (photo.get("url") or "").replace("/square.", "/original.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download licensed research-grade Felidae photos from iNaturalist.")
    parser.add_argument("--per-species", type=int, default=100, help="Maximum accepted images per species.")
    parser.add_argument("--species-id", help="Download only the taxonomy entry with this ID.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-width", type=int, default=400)
    parser.add_argument("--min-height", type=int, default=300)
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between API requests in seconds.")
    args = parser.parse_args()
    if args.per_species < 1:
        raise SystemExit("--per-species must be at least 1")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    fields = ["image_path", "label", "scientific_name", "observation_id", "photo_id", "source_url", "photo_url", "license", "attribution", "width", "height", "sha256"]
    session = requests.Session()
    session.headers["User-Agent"] = "FelidaeClassifier/1.0 research dataset collector"
    seen_hashes: set[str] = set()
    seen_photo_ids: set[str] = set()
    seen_observation_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    excluded_hashes: set[str] = set()
    excluded_photo_ids: set[str] = set()
    excluded_observation_ids: set[str] = set()
    excluded_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    excluded_path = args.output / "excluded_images.csv"
    if excluded_path.exists():
        with excluded_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("sha256"):
                    excluded_hashes.add(row["sha256"])
                if row.get("photo_id"):
                    excluded_photo_ids.add(str(row["photo_id"]))
                if row.get("observation_id"):
                    excluded_observation_ids.add(str(row["observation_id"]))
                if row.get("observation_id") and row.get("photo_id"):
                    excluded_pairs.add((str(row["observation_id"]), str(row["photo_id"])))

    registry_path = args.output / "image_registry.csv"
    if registry_path.exists():
        with registry_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("sha256"):
                    seen_hashes.add(row["sha256"])
                if row.get("photo_id"):
                    seen_photo_ids.add(str(row["photo_id"]))
                if row.get("observation_id"):
                    seen_observation_ids.add(str(row["observation_id"]))
                if row.get("observation_id") and row.get("photo_id"):
                    seen_pairs.add((str(row["observation_id"]), str(row["photo_id"])))

    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                image_path = args.output / row.get("image_path", "")
                row_pair = (str(row.get("observation_id", "")), str(row.get("photo_id", "")))
                if (
                    image_path.is_file()
                    and row.get("sha256") not in excluded_hashes
                    and str(row.get("photo_id", "")) not in excluded_photo_ids
                    and str(row.get("observation_id", "")) not in excluded_observation_ids
                    and row_pair not in excluded_pairs
                ):
                    rows.append(row)
                    if row.get("sha256"):
                        seen_hashes.add(row["sha256"])
                    if row.get("photo_id"):
                        seen_photo_ids.add(str(row["photo_id"]))
                    if row.get("observation_id"):
                        seen_observation_ids.add(str(row["observation_id"]))
                    if row.get("observation_id") and row.get("photo_id"):
                        seen_pairs.add((str(row["observation_id"]), str(row["photo_id"])))

    def save_manifest() -> None:
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    species_list = load_species()
    if args.species_id:
        species_list = [species for species in species_list if species["id"] == args.species_id]
        if not species_list:
            raise SystemExit(f"Unknown species ID: {args.species_id}")

    for species in species_list:
        existing_rows = [row for row in rows if row.get("label") == species["id"]]
        accepted = len(existing_rows)
        seen_observations: set[int] = set()
        species_dir = args.output / species["id"]
        species_dir.mkdir(parents=True, exist_ok=True)
        if accepted >= args.per_species:
            print(f"skipping {species['common_name']}: {accepted}/{args.per_species} images", flush=True)
            continue
        try:
            taxon_id = resolve_taxon_id(session, species["scientific_name"])
        except requests.RequestException:
            taxon_id = None
        if taxon_id is None:
            print(f"could not resolve iNaturalist taxon: {species['scientific_name']}", flush=True)
            continue

        for page in range(1, 51):
            observations = fetch_observations(session, taxon_id, page)
            if not observations:
                break
            for observation in observations:
                observation_id = observation.get("id")
                if not observation_id or observation_id in seen_observations:
                    continue
                seen_observations.add(observation_id)
                for photo in observation.get("photos") or []:
                    if accepted >= args.per_species:
                        break
                    photo_id = photo.get("id")
                    url = image_url(photo)
                    if not photo_id or not url:
                        continue
                    if (
                        str(observation_id) in excluded_observation_ids
                        or str(photo_id) in excluded_photo_ids
                        or (str(observation_id), str(photo_id)) in excluded_pairs
                        or str(observation_id) in seen_observation_ids
                        or str(photo_id) in seen_photo_ids
                        or (str(observation_id), str(photo_id)) in seen_pairs
                    ):
                        continue
                    try:
                        response = session.get(url, timeout=45)
                        response.raise_for_status()
                        image_bytes = response.content
                        with Image.open(BytesIO(image_bytes)) as image:
                            width, height = image.size
                            image.verify()
                        if width < args.min_width or height < args.min_height:
                            continue
                        digest = hashlib.sha256(image_bytes).hexdigest()
                        if digest in seen_hashes or digest in excluded_hashes:
                            continue
                        image_path = species_dir / f"{observation_id}_{photo_id}.jpg"
                        image_path.write_bytes(image_bytes)
                        seen_hashes.add(digest)
                        seen_photo_ids.add(str(photo_id))
                        seen_observation_ids.add(str(observation_id))
                        seen_pairs.add((str(observation_id), str(photo_id)))
                        accepted += 1
                        rows.append({
                            "image_path": image_path.relative_to(args.output).as_posix(),
                            "label": species["id"],
                            "scientific_name": species["scientific_name"],
                            "observation_id": observation_id,
                            "photo_id": photo_id,
                            "source_url": f"https://www.inaturalist.org/observations/{observation_id}",
                            "photo_url": url,
                            "license": photo.get("license_code") or "unknown",
                            "attribution": photo.get("attribution") or "",
                            "width": width,
                            "height": height,
                            "sha256": digest,
                        })
                        print(f"{species['common_name']}: {accepted}/{args.per_species}", flush=True)
                    except (requests.RequestException, OSError, ValueError):
                        continue
                if accepted >= args.per_species:
                    break
            if accepted >= args.per_species:
                break
            time.sleep(args.delay)
        print(f"completed {species['common_name']}: {accepted} images", flush=True)

        save_manifest()

    save_manifest()
    print(f"Saved {len(rows)} images and manifest at {manifest_path}")


if __name__ == "__main__":
    main()
