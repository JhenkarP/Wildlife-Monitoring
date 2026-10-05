"""SQLite persistence for normalized wildlife observations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .contract import CANONICAL_COLUMNS


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, species TEXT,
                taxonomic_group TEXT, taxonomic_class TEXT, taxonomic_order TEXT,
                observed_on TEXT,
                latitude REAL, longitude REAL, state TEXT, district TEXT,
                protected_area TEXT, migration_role TEXT, season TEXT,
                seasonal_density REAL, iucn_status TEXT, source TEXT,
                dataset_name TEXT, dataset_key TEXT, publisher TEXT,
                monitoring_region TEXT,
                occurrence_id TEXT, photo_url TEXT, imported_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(observations)")}
        for column in (
            "taxonomic_group", "taxonomic_class", "taxonomic_order",
            "dataset_name", "dataset_key", "publisher", "monitoring_region",
        ):
            if column not in columns:
                connection.execute(f"ALTER TABLE observations ADD COLUMN {column} TEXT")


def replace_observations(path: Path, observations: pd.DataFrame) -> int:
    initialize_database(path)
    stored = observations.copy()
    stored["observed_on"] = stored["observed_on"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM observations")
        stored[CANONICAL_COLUMNS].to_sql("observations", connection, if_exists="append", index=False)
    return len(stored)


def append_observations(path: Path, observations: pd.DataFrame) -> int:
    initialize_database(path)
    stored = observations.copy()
    stored["observed_on"] = stored["observed_on"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(path) as connection:
        existing = pd.read_sql_query("SELECT occurrence_id FROM observations", connection)
        known_ids = set(existing["occurrence_id"].dropna().astype(str))
        new_records = stored[~stored["occurrence_id"].astype(str).isin(known_ids)].drop_duplicates(
            subset=["occurrence_id"]
        )
        if not new_records.empty:
            new_records[CANONICAL_COLUMNS].to_sql("observations", connection, if_exists="append", index=False)
    return len(new_records)


def read_observations(path: Path) -> pd.DataFrame:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query("SELECT * FROM observations ORDER BY observed_on", connection)