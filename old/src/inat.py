"""Small iNaturalist observations adapter for a future live-data workflow."""

from __future__ import annotations

from datetime import date, timedelta

import requests
import truststore

from .contract import normalize_observations

truststore.inject_into_ssl()


def fetch_recent_observations(bounds: str, days: int = 7, per_page: int = 100) -> list[dict]:
    """Fetch recent photo observations inside a W,S,E,N bounding box."""
    start = date.today() - timedelta(days=days)
    west, south, east, north = bounds.split(",")
    response = requests.get(
        "https://api.inaturalist.org/v1/observations",
        params={
            "quality_grade": "research,needs_id", "photos": "true",
            "swlng": west, "swlat": south, "nelng": east, "nelat": north,
            "d1": start.isoformat(), "per_page": per_page, "order": "desc",
            "order_by": "observed_on", "taxon_id": 1, "without_taxon_id": 47158,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def observations_to_frame(results: list[dict]):
    """Convert iNaturalist results into the application's canonical observation frame."""
    rows = []
    for result in results:
        taxon = result.get("taxon") or {}
        ancestry = taxon.get("ancestry", "").split("/")
        if "1" not in ancestry and str(taxon.get("taxon_id")) != "1":
            continue
        if "47158" in ancestry or str(taxon.get("taxon_id")) == "47158":
            continue
        photos = result.get("photos") or []
        point = result.get("geojson") or {}
        coordinates = point.get("coordinates") or [None, None]
        rows.append(
            {
                "species": taxon.get("preferred_common_name") or taxon.get("name"),
                "observed_on": result.get("observed_on"),
                "latitude": coordinates[1],
                "longitude": coordinates[0],
                "state": (result.get("place_guess") or "").split(",")[-1].strip(),
                "protected_area": result.get("place_guess"),
                "source": "iNaturalist",
                "occurrence_id": str(result.get("id")),
                "photo_url": photos[0].get("url") if photos else None,
            }
        )
    return normalize_observations(__import__("pandas").DataFrame(rows))