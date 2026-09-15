"""Historical wildlife observation extraction from GBIF."""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests
import truststore

from .contract import normalize_observations

truststore.inject_into_ssl()

HISTORICAL_DATASETS = {
    "eBird Observation Dataset": {
        "key": "4fa7b334-ce0d-4e88-aaae-2e0c138d049e",
        "publisher": "Cornell Lab of Ornithology",
        "description": "Large-scale historical bird observations; useful for seasonal distribution signals.",
    },
    "North-East India bird occurrences": {
        "key": "6115006c-ec2f-4bfd-9315-115879785446",
        "publisher": "Wildlife Institute of India",
        "description": "Literature-based bird occurrences from a region important for migratory birds.",
    },
    "Anamalai Hills mammal occurrences": {
        "key": "b045f965-8a06-4b9c-a1b1-a9b5a1628b52",
        "publisher": "Nature Conservation Foundation",
        "description": "Mammal records from the Western Ghats and Anamalai Tiger Reserve.",
    },
    "Dampa Tiger Reserve mammals": {
        "key": "f896ef0c-7fad-44eb-bf56-b5dbe821fb5c",
        "publisher": "Nature Conservation Foundation",
        "description": "Mammal detections from transect surveys in Mizoram.",
    },
    "Pallikaranai wetland aquatic birds": {
        "key": "f6339d85-4261-4b93-a81d-1632ce210e33",
        "publisher": "Nature Mates-Nature Club",
        "description": "Aquatic bird observations from a Chennai wetland case study.",
    },
}


def _taxonomic_group(taxon_class: str | None) -> str | None:
    groups = {
        "mammalia": "Mammal", "aves": "Bird", "reptilia": "Reptile",
        "amphibia": "Amphibian", "actinopterygii": "Fish", "insecta": "Insect",
        "arachnida": "Arachnid", "gastropoda": "Mollusc", "malacostraca": "Crustacean",
    }
    return groups.get(str(taxon_class or "").strip().lower(), "Other" if taxon_class else None)


def fetch_unfiltered_regional_gbif(
    regions: dict[str, dict[str, str]],
    records_per_region: int = 10_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fetch paginated coordinate records using only the supplied polygons."""
    rows = []
    available = {}
    for region_name, region in regions.items():
        west, south, east, north = region["bounds"].split(",")
        geometry = f"POLYGON(({west} {south},{west} {north},{east} {north},{east} {south},{west} {south}))"
        offset = 0
        target = max(0, records_per_region)
        total = 0
        while offset < target:
            page_size = min(300, target - offset)
            response = requests.get(
                "https://api.gbif.org/v1/occurrence/search",
                params={"geometry": geometry, "limit": page_size, "offset": offset},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            total = payload.get("count", 0)
            page = payload.get("results", [])
            for record in page:
                media = record.get("media") or []
                rows.append(
                    {
                        "species": record.get("vernacularName") or record.get("species") or record.get("scientificName"),
                        "taxonomic_group": _taxonomic_group(record.get("class")),
                        "taxonomic_class": record.get("class"),
                        "taxonomic_order": record.get("order"),
                        "observed_on": record.get("eventDate") or record.get("year"),
                        "latitude": record.get("decimalLatitude"),
                        "longitude": record.get("decimalLongitude"),
                        "state": record.get("stateProvince"),
                        "district": record.get("county"),
                        "protected_area": record.get("locality"),
                        "source": "GBIF",
                        "dataset_name": record.get("datasetName"),
                        "dataset_key": record.get("datasetKey"),
                        "publisher": record.get("publishingOrgKey"),
                        "monitoring_region": region_name,
                        "occurrence_id": record.get("key") or record.get("gbifID"),
                        "photo_url": media[0].get("identifier") if media else None,
                    }
                )
            offset += len(page)
            if not page or len(page) < page_size:
                break
        available[region_name] = total
    return normalize_observations(pd.DataFrame(rows)), available


def _fetch_inaturalist_fallback(start_year: int, end_year: int, limit: int) -> pd.DataFrame:
    response = requests.get(
        "https://api.inaturalist.org/v1/observations",
        params={
            "swlng": 68.1, "swlat": 6.5, "nelng": 97.4, "nelat": 35.7,
            "d1": f"{start_year}-01-01", "d2": f"{end_year}-12-31",
            "taxon_id": 1, "without_taxon_id": 47158, "photos": "true",
            "quality_grade": "research,needs_id", "per_page": min(limit, 200),
            "order": "desc", "order_by": "observed_on",
        },
        timeout=45,
    )
    response.raise_for_status()
    rows = []
    for record in response.json().get("results", []):
        taxon = record.get("taxon") or {}
        if "1" not in taxon.get("ancestry", "").split("/"):
            continue
        if "47158" in taxon.get("ancestry", "").split("/"):
            continue
        point = record.get("geojson") or {}
        coordinates = point.get("coordinates") or [None, None]
        photos = record.get("photos") or []
        rows.append(
            {
                "species": taxon.get("preferred_common_name") or taxon.get("name"),
                "observed_on": record.get("observed_on"),
                "latitude": coordinates[1], "longitude": coordinates[0],
                "state": record.get("place_guess"),
                "protected_area": record.get("place_guess"),
                "source": "iNaturalist historical fallback",
                "dataset_name": "iNaturalist historical fallback",
                "dataset_key": pd.NA,
                "publisher": "iNaturalist",
                "occurrence_id": record.get("id"),
                "photo_url": photos[0].get("url") if photos else None,
            }
        )
    return normalize_observations(pd.DataFrame(rows))


def fetch_historical_india(
    start_year: int,
    end_year: int,
    limit: int = 300,
    dataset_key: str | None = None,
    require_media: bool = False,
    region_bounds: str | None = None,
) -> pd.DataFrame:
    """Fetch a curated or broad GBIF India dataset, excluding Insecta."""
    selected_dataset = next(
        (item for item in HISTORICAL_DATASETS.values() if item["key"] == dataset_key),
        None,
    )
    selected_dataset_name = next(
        (name for name, item in HISTORICAL_DATASETS.items() if item["key"] == dataset_key),
        "GBIF occurrence search",
    )
    params = {
        "country": "IN",
        "kingdomKey": 1,
        "eventDate": f"{start_year},{end_year}",
        "has_coordinate": "true",
        "occurrence_status": "present",
        **({"datasetKey": dataset_key} if dataset_key else {}),
        **({"media_type": "StillImage"} if require_media else {}),
        "limit": min(limit, 300),
        "offset": 0,
    }
    if region_bounds:
        west, south, east, north = region_bounds.split(",")
        params["geometry"] = f"POLYGON(({west} {south},{west} {north},{east} {north},{east} {south},{west} {south}))"
    response = requests.get(
        "https://api.gbif.org/v1/occurrence/search",
        params=params,
        timeout=45,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        if response.status_code >= 500 and dataset_key is None:
            return _fetch_inaturalist_fallback(start_year, end_year, limit)
        raise error
    rows = []
    for record in response.json().get("results", []):
        if str(record.get("class", "")).lower() == "insecta":
            continue
        media = record.get("media") or []
        rows.append(
            {
                "species": record.get("vernacularName") or record.get("species") or record.get("scientificName"),
                "taxonomic_group": _taxonomic_group(record.get("class")),
                "taxonomic_class": record.get("class"),
                "taxonomic_order": record.get("order"),
                "observed_on": record.get("eventDate") or record.get("year"),
                "latitude": record.get("decimalLatitude"),
                "longitude": record.get("decimalLongitude"),
                "state": record.get("stateProvince"),
                "district": record.get("county"),
                "protected_area": record.get("locality"),
                "iucn_status": record.get("iucnRedListCategory"),
                "source": "GBIF",
                "dataset_name": record.get("datasetName") or selected_dataset_name,
                "dataset_key": record.get("datasetKey") or dataset_key,
                "publisher": (selected_dataset or {}).get("publisher"),
                "occurrence_id": record.get("key") or record.get("gbifID"),
                "photo_url": media[0].get("identifier") if media else None,
            }
        )
    frame = normalize_observations(pd.DataFrame(rows))
    if frame.empty and dataset_key is None:
        return _fetch_inaturalist_fallback(start_year, end_year, limit)
    frame["extracted_on"] = date.today().isoformat()
    return frame