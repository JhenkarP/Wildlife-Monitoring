from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TAXONOMY_PATH = Path(__file__).parents[1] / "taxonomy" / "felidae_species.json"


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        taxonomy = json.load(file)
    species = taxonomy.get("species", [])
    if len(species) != taxonomy.get("species_count"):
        raise ValueError("Taxonomy species_count does not match the species list")
    if sum(item["wild_species"] for item in species) != taxonomy.get("wild_species_count"):
        raise ValueError("Taxonomy wild_species_count does not match the species list")
    return taxonomy


def species_by_id(taxonomy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in taxonomy["species"]}


def grouped_species(taxonomy: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for species in taxonomy["species"]:
        grouped.setdefault(species["subfamily"], {}).setdefault(species["lineage"], []).append(species)
    return grouped
