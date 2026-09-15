"""Download one research-grade iNaturalist photo for 15 India-occurring mammals."""

from __future__ import annotations

import csv
from pathlib import Path

import requests

ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "data" / "india_mammals"
SPECIES = [
    ("tiger", "Panthera tigris"),
    ("leopard", "Panthera pardus"),
    ("asian-elephant", "Elephas maximus"),
    ("greater-one-horned-rhino", "Rhinoceros unicornis"),
    ("gaur", "Bos gaurus"),
    ("sloth-bear", "Melursus ursinus"),
    ("dhole", "Cuon alpinus"),
    ("chital", "Axis axis"),
    ("sambar", "Rusa unicolor"),
    ("wild-boar", "Sus scrofa"),
    ("rhesus-macaque", "Macaca mulatta"),
    ("hanuman-langur", "Semnopithecus entellus"),
    ("bengal-fox", "Vulpes bengalensis"),
    ("asiatic-lion", "Panthera leo"),
    ("striped-hyena", "Hyaena hyaena"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in OUTPUT_DIR.glob("*.jpg"):
        image_path.unlink()
    rows = []
    for filename, species in SPECIES:
        row = {"file": "", "reference_species": species, "observation_id": "", "source_label": "", "photo_url": ""}
        try:
            response = requests.get(
                "https://api.inaturalist.org/v1/observations",
                params={"taxon_name": species, "quality_grade": "research", "photos": "true", "per_page": 30, "order": "random"},
                timeout=30,
            )
            response.raise_for_status()
            for observation in response.json().get("results", []):
                photos = observation.get("photos") or []
                if not photos or not photos[0].get("url"):
                    continue
                photo_url = photos[0]["url"].replace("/square.", "/medium.")
                image_response = requests.get(photo_url, timeout=60)
                image_response.raise_for_status()
                image_path = OUTPUT_DIR / f"{filename}.jpg"
                image_path.write_bytes(image_response.content)
                taxon = observation.get("taxon") or {}
                row.update(
                    file=str(image_path.relative_to(ROOT)),
                    observation_id=str(observation.get("id", "")),
                    source_label=taxon.get("name", ""),
                    photo_url=photo_url,
                )
                break
        except requests.RequestException as error:
            row["error"] = str(error)
        rows.append(row)
        print(f"{species}: {'downloaded' if row['file'] else 'not downloaded'}", flush=True)
    with (OUTPUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = ["file", "reference_species", "observation_id", "source_label", "photo_url", "error"]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()