"""HTTP API used by the React wildlife analysis frontend."""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.database import read_observations
from src.detector import classify_with_custom_model, localize_features


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed" / "wildlife.sqlite"
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
    except KeyError as error:
        raise HTTPException(status_code=422, detail=f"Species is not in the feature schema: {error.args[0]}") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "species": species.title(),
        "confidence": float(detection.get("confidence", 0)),
        "features": _feature_response(record),
        "description": f"{species.title()} identified in the uploaded frame. Florence-2 localized the configured visual features for review.",
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
    records = observations[["species", "observed_on", "latitude", "longitude", "state", "district", "protected_area"]].dropna(subset=["latitude", "longitude"])
    region_counts = records.groupby("state", dropna=True).size().sort_values(ascending=False)
    return {
        "species": species,
        "days": days,
        "records": records.to_dict(orient="records"),
        "regions": [{"name": str(name), "count": int(count)} for name, count in region_counts.items()],
    }