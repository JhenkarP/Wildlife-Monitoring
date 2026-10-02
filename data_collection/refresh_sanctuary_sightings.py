"""Fetch GBIF sightings for six named Indian wildlife sanctuaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
import truststore

from src.contract import normalize_observations
from src.database import replace_observations
from data_collection.refresh_sightings import SUPPORTED_SPECIES

truststore.inject_into_ssl()

SANCTUARIES = {
    "Gir Wildlife Sanctuary": "70.4,20.7,71.3,21.6",
    "Wayanad Wildlife Sanctuary": "75.9,11.5,76.4,11.9",
    "Bhadra Wildlife Sanctuary": "75.4,13.5,75.8,13.8",
    "Dandeli Wildlife Sanctuary": "74.3,14.8,74.8,15.4",
    "Kumbhalgarh Wildlife Sanctuary": "73.3,25.0,73.7,25.4",
    "Gahirmatha Marine Wildlife Sanctuary": "86.8,20.3,87.2,20.8",
}

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "wildlife_sanctuaries.sqlite"
GBIF_URL = "https://api.gbif.org/v1"


def resolve_taxon(scientific_name: str) -> int:
    response = requests.get(
        f"{GBIF_URL}/species/match",
        params={"name": scientific_name},
        timeout=90,
    )
    response.raise_for_status()
    result = response.json()
    key = result.get("speciesKey") or result.get("usageKey")
    if not key or result.get("matchType") == "NONE":
        raise RuntimeError(f"GBIF could not resolve {scientific_name!r}: {result}")
    return int(key)


def fetch_sanctuary(
    sanctuary: str,
    taxon_keys: dict[int, str],
    scientific_names: dict[str, str],
    bounds: str,
    max_records: int,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while offset < max_records:
        page_size = min(300, max_records - offset)
        params: list[tuple[str, object]] = [
            ("country", "IN"),
            ("classKey", 359),
            ("geometry", f"POLYGON(({bounds.split(',')[0]} {bounds.split(',')[1]},{bounds.split(',')[0]} {bounds.split(',')[3]},{bounds.split(',')[2]} {bounds.split(',')[3]},{bounds.split(',')[2]} {bounds.split(',')[1]},{bounds.split(',')[0]} {bounds.split(',')[1]}))"),
            ("hasCoordinate", "true"),
            ("occurrenceStatus", "present"),
            ("limit", page_size),
            ("offset", offset),
        ]
        response = requests.get(
            f"{GBIF_URL}/occurrence/search",
            params=params,
            timeout=90,
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
            occurrence_id = record.get("key") or record.get("gbifID")
            species_key = record.get("speciesKey") or record.get("acceptedSpeciesKey")
            common_name = taxon_keys.get(species_key)
            if not common_name:
                scientific_name = f"{record.get('genus', '')} {record.get('specificEpithet', '')}".strip().lower()
                common_name = scientific_names.get(scientific_name)
            if not common_name:
                continue
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
                    "protected_area": sanctuary,
                    "iucn_status": record.get("iucnRedListCategory"),
                    "source": "GBIF",
                    "dataset_name": record.get("datasetName"),
                    "dataset_key": record.get("datasetKey"),
                    "publisher": record.get("publishingOrgKey"),
                    "occurrence_id": f"gbif:{occurrence_id}" if occurrence_id else None,
                    "photo_url": media[0].get("identifier") if media else None,
                }
            )
        if not page or len(page) < page_size:
            break
        offset += len(page)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-records-per-query", type=int, default=3000)
    parser.add_argument("--database", type=Path, default=DB_PATH)
    args = parser.parse_args()

    taxon_keys = {resolve_taxon(scientific_name): common_name for common_name, scientific_name in SUPPORTED_SPECIES.items()}
    scientific_names = {scientific_name.lower(): common_name for common_name, scientific_name in SUPPORTED_SPECIES.items()}
    rows: list[dict] = []
    for sanctuary, bounds in SANCTUARIES.items():
        print(f"Fetching all configured animals / {sanctuary} ...")
        rows.extend(fetch_sanctuary(sanctuary, taxon_keys, scientific_names, bounds, args.max_records_per_query))

    observations = normalize_observations(pd.DataFrame(rows))
    observations = observations.dropna(subset=["occurrence_id", "observed_on", "latitude", "longitude"])
    observations = observations.drop_duplicates(subset=["occurrence_id"])
    count = replace_observations(args.database, observations)
    print(f"replaced {count} records in {args.database}")
    print("\nRecords by sanctuary:")
    print(observations.groupby("protected_area").size().sort_values(ascending=False).to_string())
    print("\nRecords by species:")
    print(observations.groupby("species").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
