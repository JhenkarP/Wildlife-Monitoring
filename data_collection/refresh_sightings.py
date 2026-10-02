"""Refresh the atlas database with identifiable GBIF records from India."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.contract import normalize_observations
from src.database import append_observations, replace_observations


SUPPORTED_SPECIES = {
    "Asian elephant": "Elephas maximus",
    "Asiatic lion": "Panthera leo",
    "Barasingha": "Rucervus duvaucelii",
    "Bengal tiger": "Panthera tigris",
    "Chital": "Axis axis",
    "Dhole": "Cuon alpinus",
    "Gaur": "Bos gaurus",
    "Greater one horned rhino": "Rhinoceros unicornis",
    "Hanuman langur": "Semnopithecus entellus",
    "Indian leopard": "Panthera pardus",
    "Nilgai": "Boselaphus tragocamelus",
    "Sambar": "Rusa unicolor",
    "Sloth bear": "Melursus ursinus",
    "Striped hyena": "Hyaena hyaena",
}

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "wildlife.sqlite"
GBIF_URL = "https://api.gbif.org/v1"


def resolve_taxon(scientific_name: str) -> int:
    response = requests.get(
        f"{GBIF_URL}/species/match",
        params={"name": scientific_name},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    key = result.get("speciesKey") or result.get("usageKey")
    if not key or result.get("matchType") == "NONE":
        raise RuntimeError(f"GBIF could not resolve {scientific_name!r}: {result}")
    return int(key)


def fetch_species(common_name: str, scientific_name: str, start_date: date, end_date: date, limit: int) -> pd.DataFrame:
    taxon_key = resolve_taxon(scientific_name)
    rows: list[dict] = []
    offset = 0
    while offset < limit:
        page_size = min(300, limit - offset)
        response = requests.get(
            f"{GBIF_URL}/occurrence/search",
            params={
                "taxonKey": taxon_key,
                "country": "IN",
                "hasCoordinate": "true",
                "eventDate": f"{start_date.isoformat()},{end_date.isoformat()}",
                "occurrenceStatus": "present",
                "limit": page_size,
                "offset": offset,
            },
            timeout=60,
        )
        response.raise_for_status()
        page = response.json().get("results", [])
        for record in page:
            event_date = record.get("eventDate") or record.get("year")
            latitude = record.get("decimalLatitude")
            longitude = record.get("decimalLongitude")
            if not event_date or latitude is None or longitude is None:
                continue
            media = record.get("media") or []
            rows.append(
                {
                    "species": common_name,
                    "taxonomic_group": "Mammal",
                    "taxonomic_class": record.get("class"),
                    "taxonomic_order": record.get("order"),
                    "observed_on": event_date,
                    "latitude": latitude,
                    "longitude": longitude,
                    "state": record.get("stateProvince"),
                    "district": record.get("county"),
                    "protected_area": record.get("locality"),
                    "iucn_status": record.get("iucnRedListCategory"),
                    "source": "GBIF",
                    "dataset_name": record.get("datasetName"),
                    "dataset_key": record.get("datasetKey"),
                    "publisher": record.get("publishingOrgKey"),
                    "occurrence_id": f"gbif:{record.get('key') or record.get('gbifID')}",
                    "photo_url": media[0].get("identifier") if media else None,
                }
            )
        if not page or len(page) < page_size:
            break
        offset += len(page)
    frame = normalize_observations(pd.DataFrame(rows))
    frame["extracted_on"] = date.today().isoformat()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date.today() - timedelta(days=365))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--per-species", type=int, default=300)
    parser.add_argument("--replace", action="store_true", help="Replace the database instead of appending records.")
    args = parser.parse_args()

    frames = [
        fetch_species(common_name, scientific_name, args.start_date, args.end_date, args.per_species)
        for common_name, scientific_name in SUPPORTED_SPECIES.items()
    ]
    observations = normalize_observations(pd.concat(frames, ignore_index=True))
    observations = observations[observations["species"].isin(SUPPORTED_SPECIES)]
    observations = observations.dropna(subset=["occurrence_id", "observed_on", "latitude", "longitude"])
    count = replace_observations(DB_PATH, observations) if args.replace else append_observations(DB_PATH, observations)
    action = "replaced" if args.replace else "appended"
    print(f"{action} {count} records in {DB_PATH}")
    print(observations["species"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()