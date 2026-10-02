"""HTTP API used by the React wildlife analysis frontend."""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.database import read_observations
from src.detector import classify_with_custom_model, describe_with_florence, localize_features


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed" / "wildlife.sqlite"
SANCTUARY_DB_PATH = ROOT / "data" / "processed" / "wildlife_sanctuaries.sqlite"
app = FastAPI(title="Wildlife Monitoring API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _species_key(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ")
    aliases = {"lion": "asiatic lion", "asiatic_lion": "asiatic lion"}
    return aliases.get(normalized, normalized)


def _feature_response(record: dict) -> list[dict]:
    features = []
    for name, proposal in record.get("features", {}).items():
        bbox = proposal.get("bbox") or [0, 0, 0, 0]
        features.append(
            {
                "name": name.replace("_", " ").title(),
                "status": "uncertain" if proposal.get("status") == "uncertain" else "visible",
                "description": proposal.get("feature_description", "Configured Florence feature proposal."),
                "box": bbox,
            }
        )
    return features


def _record_url(source: object, occurrence_id: object) -> str | None:
    if str(source).upper() != "GBIF" or occurrence_id is None:
        return None
    identifier = str(occurrence_id).removeprefix("gbif:").strip()
    return f"https://www.gbif.org/occurrence/{identifier}" if identifier else None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> dict:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    try:
        frame = Image.open(io.BytesIO(await image.read())).convert("RGB")
        _, detections = classify_with_custom_model(frame)
        detection = detections[0]
        species = _species_key(str(detection["label"]))
        _, record = localize_features(frame, species)
        description = describe_with_florence(frame, species, record)
    except KeyError as error:
        raise HTTPException(status_code=422, detail=f"Species is not in the feature schema: {error.args[0]}") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "species": species.title(),
        "confidence": float(detection.get("confidence", 0)),
        "features": _feature_response(record),
        "description": description,
        "model": "fine-tuned ResNet18 + microsoft/Florence-2-base",
    }


@app.get("/api/sightings")
def sightings(species: str | None = None, days: int = 365) -> dict:
    observations = read_observations(DB_PATH)
    if observations.empty:
        return {"species": species, "days": days, "records": [], "regions": []}
    cutoff = date.today() - timedelta(days=max(1, min(days, 3650)))
    observations["observed_on"] = observations["observed_on"].astype(str)
    observations = observations[observations["observed_on"] >= cutoff.isoformat()]
    if species:
        observations = observations[observations["species"].astype(str).str.contains(species, case=False, na=False)]
    records = observations[["species", "observed_on", "latitude", "longitude", "state", "district", "protected_area", "source", "dataset_name", "publisher", "occurrence_id", "photo_url"]].dropna(subset=["latitude", "longitude"])
    record_payload = records.to_dict(orient="records")
    for record in record_payload:
        record["record_url"] = _record_url(record.get("source"), record.get("occurrence_id"))
    region_counts = records.groupby("state", dropna=True).size().sort_values(ascending=False)
    return {
        "species": species,
        "days": days,
        "records": record_payload,
        "regions": [{"name": str(name), "count": int(count)} for name, count in region_counts.items()],
    }


@app.get("/api/sanctuaries")
def sanctuaries(days: int = 730) -> dict:
    observations = read_observations(SANCTUARY_DB_PATH)
    if observations.empty:
        return {"days": days, "total_records": 0, "sanctuaries": [], "match_method": "GBIF occurrences spatially matched to approximate sanctuary bounding boxes"}
    observations["observed_on"] = observations["observed_on"].astype(str)
    if days and days > 0:
        cutoff = date.today() - timedelta(days=min(days, 3650))
        observations = observations[observations["observed_on"] >= cutoff.isoformat()]
    usable = observations.copy()
    usable["protected_area"] = usable["protected_area"].astype("string").str.strip()
    usable = usable.dropna(subset=["protected_area", "species"])
    usable = usable[usable["protected_area"] != ""]
    total_records = len(usable)
    if not total_records:
        return {"days": days, "total_records": 0, "sanctuaries": [], "match_method": "GBIF occurrences spatially matched to approximate sanctuary bounding boxes"}
    sanctuary_counts = usable.groupby("protected_area").size().sort_values(ascending=False).head(7)
    result = []
    for name, count in sanctuary_counts.items():
        sanctuary_records = usable[usable["protected_area"] == name]
        species_counts = sanctuary_records.groupby("species").size().sort_values(ascending=False)
        leading_species = str(species_counts.index[0])
        leading_count = int(species_counts.iloc[0])
        result.append(
            {
                "name": str(name),
                "count": int(count),
                "percentage": round(int(count) / total_records * 100, 1),
                "leading_species": leading_species,
                "leading_species_count": leading_count,
                "leading_species_percentage": round(leading_count / int(count) * 100, 1),
            }
        )
    return {
        "days": days,
        "total_records": total_records,
        "sanctuaries": result,
        "match_method": "GBIF occurrences spatially matched to approximate sanctuary bounding boxes",
    }


@app.get("/api/sanctuaries/{sanctuary_name}")
def sanctuary_report(sanctuary_name: str, days: int = 730) -> dict:
    observations = read_observations(SANCTUARY_DB_PATH)
    if observations.empty:
        raise HTTPException(status_code=404, detail="Sanctuary database is empty")
    observations["observed_on"] = observations["observed_on"].astype(str)
    cutoff = date.today() - timedelta(days=min(days, 3650))
    observations = observations[observations["observed_on"] >= cutoff.isoformat()]
    observations["protected_area"] = observations["protected_area"].astype("string").str.strip()
    records = observations[observations["protected_area"].str.casefold() == sanctuary_name.casefold()].copy()
    if records.empty:
        raise HTTPException(status_code=404, detail="Sanctuary not found")

    records["observed_on"] = records["observed_on"].astype(str)
    species_counts = records.groupby("species").size().sort_values(ascending=False)
    composition = []
    for species, count in species_counts.items():
        species_records = records[records["species"] == species]
        composition.append(
            {
                "species": str(species),
                "count": int(count),
                "percentage": round(int(count) / len(records) * 100, 1),
                "photo_count": int(species_records["photo_url"].notna().sum()),
            }
        )

    source_records = []
    for _, record in records.sort_values("observed_on", ascending=False).head(12).iterrows():
        payload = record.to_dict()
        payload["record_url"] = _record_url(payload.get("source"), payload.get("occurrence_id"))
        source_records.append(payload)

    return {
        "name": str(records["protected_area"].iloc[0]),
        "total_records": len(records),
        "species_count": len(composition),
        "date_range": {"from": records["observed_on"].min(), "to": records["observed_on"].max()},
        "photo_count": int(records["photo_url"].notna().sum()),
        "georeferenced_count": int(records[["latitude", "longitude"]].notna().all(axis=1).sum()),
        "composition": composition,
        "source_records": source_records,
        "match_method": "GBIF occurrences spatially matched to approximate sanctuary bounding boxes",
    }