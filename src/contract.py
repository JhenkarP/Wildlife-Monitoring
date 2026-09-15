"""Data contract and normalization for historical wildlife observations."""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

CANONICAL_COLUMNS: Final = [
    "species", "taxonomic_group", "taxonomic_class", "taxonomic_order",
    "observed_on", "latitude", "longitude", "state", "district",
    "protected_area", "migration_role", "season", "seasonal_density",
    "iucn_status", "source", "dataset_name", "dataset_key", "publisher",
    "monitoring_region",
    "occurrence_id", "photo_url",
]

ALIASES: Final = {
    "scientific_name": "species", "common_name": "species", "date": "observed_on",
    "observation_date": "observed_on", "year": "observed_on", "lat": "latitude",
    "lon": "longitude", "lng": "longitude", "region": "state", "park": "protected_area",
    "reserve": "protected_area", "image_url": "photo_url",
}


def _clean_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, analysis-ready frame without inventing missing data."""
    normalized = frame.copy()
    normalized.columns = [_clean_name(column) for column in normalized.columns]
    normalized = normalized.rename(columns={key: value for key, value in ALIASES.items() if key in normalized})
    for column in CANONICAL_COLUMNS:
        if column not in normalized:
            normalized[column] = pd.NA
    normalized = normalized[CANONICAL_COLUMNS]
    normalized["observed_on"] = pd.to_datetime(normalized["observed_on"], errors="coerce")
    for column in ("latitude", "longitude", "seasonal_density"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in CANONICAL_COLUMNS:
        if column not in {"observed_on", "latitude", "longitude", "seasonal_density"}:
            normalized[column] = normalized[column].astype("string").str.strip()
    return normalized


def load_csv(path: str) -> pd.DataFrame:
    return normalize_observations(pd.read_csv(path))