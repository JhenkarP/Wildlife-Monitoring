# Wildlife Monitoring and Conservation System

A Python foundation for wildlife observation management, migration analysis, and conservation decision support.

## Scope

The first implementation phase focuses on the verified project objective:

- Import historical wildlife observations from CSV.
- Store normalized observations in SQLite.
- Analyze species, season, location, and migration patterns.
- Display an interactive dashboard.
- Provide an iNaturalist API adapter for recent observations.
- Classify uploaded images with the fine-tuned 13-species ResNet18 model, with SpeciesNet available for comparison.

The repository currently does not include a wildlife dataset. Place the historical CSV at `data/raw/observations.csv` when available.

See [FEATURES.md](FEATURES.md) for the current implemented, partial, and planned feature list.
See [WORKFLOW.md](WORKFLOW.md) for the stage-by-stage code layout.

## Start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The dashboard also starts without a dataset and explains the expected CSV fields.

## Refresh the sightings atlas

The atlas database is generated locally from identifiable GBIF occurrence records. The refresh is restricted to India, the 14 species supported by the atlas, records with coordinates and event dates, and present occurrences. Records retain their GBIF occurrence ID, dataset, publisher key, location, date, and optional media URL.

```powershell
python -m data_collection.refresh_sightings --start-year 2000 --per-species 300 --replace
```

The command writes `data/processed/wildlife.sqlite`. Run it again without `--replace` to append only new occurrence IDs. GBIF is the historical atlas source; iNaturalist is a separate recent community-observation source and must be labelled as supplemental rather than camera-trap data. GBIF records and media remain subject to their publisher licenses and should be cited through the occurrence or dataset metadata.

## Expected historical CSV fields

The importer accepts these canonical fields when available:

`species`, `observed_on`, `latitude`, `longitude`, `state`, `district`, `protected_area`, `migration_role`, `season`, `seasonal_density`, `iucn_status`, `source`, `occurrence_id`, `photo_url`

Common alternate names such as `scientific_name`, `date`, `lat`, `lon`, and `image_url` are normalized automatically.

## Project stages

1. Historical CSV audit and migration analytics.
2. SQLite observation repository and dashboard filters.
3. iNaturalist recent-observation ingestion for selected protected areas.
4. Fine-tuned 13-species classification with optional SpeciesNet comparison.
5. Evaluation, reporting, and deployment hardening.
