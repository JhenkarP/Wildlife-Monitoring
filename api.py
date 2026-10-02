"""HTTP API used by the React wildlife analysis frontend."""

from __future__ import annotations

import io
from calendar import month_name
from datetime import date, timedelta
from statistics import mean
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import requests

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

SPECIES_ECOLOGY = {
    "Asian elephant": ("herbivore", "forest, grassland and river corridors", "18-35 C", "large water and forage requirements"),
    "Asiatic lion": ("large carnivore", "dry deciduous forest and open scrub", "20-40 C", "open woodland supports prey and den cover"),
    "Barasingha": ("ungulate herbivore", "floodplain grassland and wet meadow", "15-32 C", "wet grasslands provide seasonal forage"),
    "Bengal tiger": ("apex carnivore", "forests, wetlands and tall grass", "15-35 C", "dense cover, water and prey availability"),
    "Chital": ("herbivore ungulate", "open woodland, grassland and forest edge", "18-35 C", "abundant grass and mixed cover"),
    "Dhole": ("social carnivore", "deciduous forest and scrub", "15-35 C", "connected cover and medium-sized prey"),
    "Gaur": ("large herbivore", "moist and dry forest with grass clearings", "15-32 C", "forest forage and mineral-rich clearings"),
    "Greater one horned rhino": ("megaherbivore", "riverine grassland, marsh and floodplain", "20-35 C", "water, wallows and tall grasses"),
    "Hanuman langur": ("arboreal primate", "woodland, scrub and rocky forest", "18-38 C", "tree cover, fruit and adaptable edge habitat"),
    "Indian leopard": ("generalist carnivore", "forest, scrub, farmland edge and rocky terrain", "18-40 C", "prey-rich habitat with cover and escape routes"),
    "Nilgai": ("large herbivore", "dry scrub, grassland and agricultural edge", "20-40 C", "browse, open visibility and heat tolerance"),
    "Sambar": ("browser herbivore", "moist forest, bamboo and stream margins", "15-35 C", "dense cover, browse and reliable water"),
    "Sloth bear": ("omnivore", "dry deciduous forest and scrub", "20-38 C", "termite-rich soil, fruiting trees and den sites"),
    "Striped hyena": ("scavenging carnivore", "arid scrub, grassland and rocky terrain", "20-42 C", "open dry habitat and carrion availability"),
}

SANCTUARY_ECOLOGY = {
    "gir wildlife sanctuary": ("hot semi-arid to dry tropical", "dry deciduous forest, thorn scrub and seasonal streams", "warm dry conditions and open woodland support ungulates and large carnivores"),
    "wayanad wildlife sanctuary": ("warm humid tropical", "moist deciduous forest, bamboo and grassland", "high rainfall and dense canopy support browsers, elephants and predators"),
    "bhadra wildlife sanctuary": ("warm seasonal tropical", "moist deciduous forest, river valleys and grass openings", "seasonal rain, water and layered forest cover support diverse herbivores"),
    "dandeli wildlife sanctuary": ("warm humid monsoon", "moist deciduous forest, river corridor and evergreen patches", "monsoon rainfall maintains dense vegetation and year-round water"),
    "kumbhalgarh wildlife sanctuary": ("hot semi-arid", "dry deciduous forest, thorn scrub and rocky hills", "seasonal water and rugged cover suit heat-tolerant ungulates and carnivores"),
    "gahirmatha marine wildlife sanctuary": ("humid tropical coastal", "coastal waters, mudflats and sandy nesting beaches", "warm shallow seas and productive coastal food webs support marine fauna"),
}

SANCTUARY_LOCATIONS = {
    "gir wildlife sanctuary": {"latitude": 21.124, "longitude": 70.824, "zoom": 9},
    "wayanad wildlife sanctuary": {"latitude": 11.685, "longitude": 76.338, "zoom": 10},
    "bhadra wildlife sanctuary": {"latitude": 13.646, "longitude": 75.635, "zoom": 10},
    "dandeli wildlife sanctuary": {"latitude": 15.247, "longitude": 74.618, "zoom": 10},
    "kumbhalgarh wildlife sanctuary": {"latitude": 25.145, "longitude": 73.583, "zoom": 10},
    "gahirmatha marine wildlife sanctuary": {"latitude": 20.775, "longitude": 87.066, "zoom": 10},
}


def _climate_observation(location: dict) -> dict:
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    source_url = "https://archive-api.open-meteo.com/v1/archive"
    try:
        response = requests.get(
            source_url,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max",
                "timezone": "auto",
            },
            timeout=8,
        )
        response.raise_for_status()
        daily = response.json().get("daily", {})
        means = [value for value in daily.get("temperature_2m_mean", []) if value is not None]
        minimums = [value for value in daily.get("temperature_2m_min", []) if value is not None]
        maximums = [value for value in daily.get("temperature_2m_max", []) if value is not None]
        if not means:
            raise ValueError("No daily temperature values returned")
        mean_c = round(mean(means), 1)
        minimum_c = round(min(minimums), 1) if minimums else None
        maximum_c = round(max(maximums), 1) if maximums else None
        range_text = f"{minimum_c}-{maximum_c} C" if minimum_c is not None and maximum_c is not None else "range unavailable"
        return {
            "status": "available",
            "mean_c": mean_c,
            "min_c": minimum_c,
            "max_c": maximum_c,
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "source": "Open-Meteo historical weather API / ERA5 reanalysis",
            "source_url": f"{source_url}?latitude={location['latitude']}&longitude={location['longitude']}&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}&daily=temperature_2m_mean,temperature_2m_min,temperature_2m_max&timezone=auto",
            "message": f"30-day external climate context: {mean_c} C mean, daily range {range_text}. This is reanalysis data for the sanctuary location, not a direct wildlife observation.",
        }
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return {
            "status": "unavailable",
            "mean_c": None,
            "min_c": None,
            "max_c": None,
            "period": None,
            "source": "Open-Meteo historical weather API / ERA5 reanalysis",
            "source_url": source_url,
            "message": "Climate service unavailable right now; retry to load external temperature observations.",
        }

SPECIES_SOURCES = {
    "Sambar": [
        {"label": "GBIF species records", "url": "https://www.gbif.org/species/search?q=Sambar"},
        {"label": "Animal Diversity Web", "url": "https://animaldiversity.org/accounts/Rusa_unicolor/"},
    ],
}


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

    region_field = "state" if records["state"].notna().any() else "district"
    region_series = records[region_field].astype("string").str.strip()
    region_series = region_series[region_series.notna() & (region_series != "")]
    region_counts = region_series.value_counts()
    month_series = records["observed_on"].str[:7]
    month_counts = month_series.value_counts().sort_index()
    month_totals = {}
    for period, count in month_counts.items():
        month_number = int(period[-2:])
        month_totals[month_name[month_number]] = int(count)
    peak_month = max(month_totals.items(), key=lambda item: item[1]) if month_totals else (None, 0)
    leading_species = composition[0] if composition else None
    ecology = []
    for item in composition:
        group, habitat, temperature, reason = SPECIES_ECOLOGY.get(
            str(item["species"]),
            ("wildlife", "natural habitat", "not available", "ecology profile not configured"),
        )
        ecology.append(
            {
                "species": item["species"],
                "group": group,
                "habitat": habitat,
                "suitable_temperature": temperature,
                "reason_present": reason,
                "share": item["percentage"],
                "photo_url": next((str(value) for value in records[records["species"] == item["species"]]["photo_url"].dropna().tolist() if str(value).strip()), None),
                "source_url": next((_record_url(row.get("source"), row.get("occurrence_id")) for _, row in records[records["species"] == item["species"]].iterrows() if _record_url(row.get("source"), row.get("occurrence_id"))), None),
                "sources": SPECIES_SOURCES.get(str(item["species"]), [{"label": "GBIF species records", "url": f"https://www.gbif.org/species/search?q={str(item['species']).replace(' ', '%20')}"}]),
            }
        )
    climate_type, habitat, climate_reason = SANCTUARY_ECOLOGY.get(
        sanctuary_name.casefold(),
        ("not classified", "not classified", "A sanctuary climate profile is not configured for this location."),
    )
    location = SANCTUARY_LOCATIONS.get(sanctuary_name.casefold(), {"latitude": None, "longitude": None, "zoom": 8})
    temperature = _climate_observation(location) if location["latitude"] is not None else {
        "status": "unavailable",
        "mean_c": None,
        "min_c": None,
        "max_c": None,
        "period": None,
        "source": "Open-Meteo historical weather API / ERA5 reanalysis",
        "source_url": "https://open-meteo.com/en/docs/historical-weather-api",
        "message": "Climate coordinates are unavailable for this sanctuary.",
    }
    analysis = {
        "leading_species": leading_species["species"] if leading_species else None,
        "leading_species_share": leading_species["percentage"] if leading_species else 0,
        "most_observed_region": region_counts.index[0] if len(region_counts) else None,
        "most_observed_region_count": int(region_counts.iloc[0]) if len(region_counts) else 0,
        "region_field": region_field,
        "regions_observed": int(region_counts.size),
        "peak_month": peak_month[0],
        "peak_month_count": peak_month[1],
        "monthly_counts": month_totals,
        "ecology": ecology,
        "climate": {"type": climate_type, "habitat": habitat, "reason": climate_reason},
        "location": location,
        "temperature": temperature,
    }

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
        "analysis": analysis,
        "source_records": source_records,
        "match_method": "GBIF occurrences spatially matched to approximate sanctuary bounding boxes",
    }