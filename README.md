# Wildlife Monitoring and Conservation System

A Python foundation for wildlife observation management, migration analysis, and conservation decision support.

## Scope

The first implementation phase focuses on the verified project objective:

- Import historical wildlife observations from CSV.
- Store normalized observations in SQLite.
- Analyze species, season, location, and migration patterns.
- Display an interactive dashboard.
- Provide an iNaturalist API adapter for recent observations.
- Classify uploaded images with the fine-tuned four-animal ResNet18 model, with SpeciesNet available for comparison.

The repository currently does not include a wildlife dataset. Place the historical CSV at `data/raw/observations.csv` when available.

See [FEATURES.md](FEATURES.md) for the current implemented, partial, and planned feature list.

## Start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The dashboard also starts without a dataset and explains the expected CSV fields.

## Expected historical CSV fields

The importer accepts these canonical fields when available:

`species`, `observed_on`, `latitude`, `longitude`, `state`, `district`, `protected_area`, `migration_role`, `season`, `seasonal_density`, `iucn_status`, `source`, `occurrence_id`, `photo_url`

Common alternate names such as `scientific_name`, `date`, `lat`, `lon`, and `image_url` are normalized automatically.

## Project stages

1. Historical CSV audit and migration analytics.
2. SQLite observation repository and dashboard filters.
3. iNaturalist recent-observation ingestion for selected protected areas.
4. Fine-tuned four-animal classification with optional SpeciesNet comparison.
5. Evaluation, reporting, and deployment hardening.