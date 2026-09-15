# Feature List

Living implementation status for the Wildlife Monitoring and Conservation System.

Status meanings:

- **Done**: the code path is implemented and available in the application. External API success and dataset completeness are tracked separately below.
- **Partial**: a working foundation exists, but the feature still depends on missing data, external validation, or additional implementation.
- **Planned**: not implemented yet.

## Done

### Five-region pilot scope

- Concentrate live monitoring and historical extraction on five Indian wildlife regions:
	- Bandipur / Nagarhole (Western Ghats).
	- Periyar (Western Ghats).
	- Ranthambore (Aravalli / dry forest).
	- Kanha (Central Indian Highlands).
	- Kaziranga (Brahmaputra floodplain).
- Use one shared geographic configuration for both iNaturalist and GBIF extraction.
- Keep records from outside the selected region out of region-specific datasets.
- Compare species and observation counts across the five regions before making biodiversity rankings.
	- Current raw snapshot: `data/raw/five_regions_gbif_unfiltered.csv`, with a configurable practical download target of 1,000 GBIF records per region (5,000 total in the verified demo flow).
- The raw snapshot applies geographic bounds only; it does not filter taxon, species, date, image, quality, or dataset.
	- Paginate GBIF results beyond the 300-record API page size; the GBIF API still reports more available records than the practical stored snapshot.
	- Preserve GBIF source availability totals so the stored sample is not presented as the complete regional census.

### Historical occurrence census

- Normalize historical records to the canonical 21-field schema.
- Categorize each GBIF record by taxonomic group, class, and order when the source provides those values.
- Summarize records by monitoring region and taxonomic group.
- Report occurrence records, unique species, dated records, and coordinate-complete records for each region/type combination.
- Label the result as an occurrence census: it counts source records and species observations, not animal population size.

### Live wildlife data

- Fetch recent observations from the iNaturalist API.
- Query selected regions: Bandipur/Nagarhole, Periyar, Ranthambore, Kanha, and Kaziranga.
- Use one shared five-region configuration for live and historical geographic extraction.
- Keep Animalia observations only.
- Exclude Insecta observations.
- Store observations in SQLite.
- Prevent duplicate occurrence IDs.
- Preserve existing observations when new live data is appended.

### Observation records

- Store species and taxon name.
- Store observation date.
- Store latitude and longitude.
- Store place or protected-area text.
- Store location label.
- Store source and occurrence ID.
- Store photo URL.

### Dashboard

- Overview metrics for observations, species, protected areas, and dated records.
- Species frequency chart.
- Observation trend chart.
- State or region filter.
- Species filter.
- Observation records table.
- Live Explorer tab.
- No-data state with the expected CSV schema.
- Historical CSV upload from the dashboard.
- Interactive observation map from stored coordinates.
- Downloadable filtered CSV report.
- Conditional migration and conservation charts for migration role, season, seasonal density, and IUCN status fields.
- Separate Historical Datasets page on the same Streamlit port.
- Curated GBIF historical sources for eBird, Wildlife Institute of India, and regional mammal or wetland datasets.
- Restrict historical extraction to the selected five-region Indian pilot scope.
- Historical extraction controls for source dataset, India date range, image availability, and record limit.
- Historical CSV provenance includes source, dataset key, dataset name, and publisher.
- Curated GBIF requests do not silently switch to iNaturalist when the selected source is unavailable.
- Historical GBIF CSV save and download path kept separate from the live iNaturalist database.
	- Historical extraction uses the five-region geography-only workflow and displays all canonical fields, including taxonomy and provenance.

### Image detection

- Upload JPG, JPEG, or PNG wildlife images.
- Run the fine-tuned ResNet18 classifier for Asian elephant, greater one-horned rhino, leopard, and gaur.
- Keep SpeciesNet available as an optional comparison model.
- Display predicted labels and confidence scores.
- Draw a dark-blue box around the localized animal; SpeciesNet supplies localization and ResNet18 supplies the four-class prediction.

### Model training

- Train a four-class Indian mammal classifier for Asian elephant, greater one-horned rhino, leopard, and gaur.
- Split the curated dataset into 70% training and 30% held-out testing data.
- Use transfer learning with torchvision ResNet18 and ImageNet preprocessing.
- Fine-tune the final ResNet18 layer block and classifier head.
- Store the fine-tuned checkpoint at `models/indian_four_animals_resnet18_finetuned.pt`.
- Verified held-out accuracy: 96.41%.
- Connect the fine-tuned checkpoint to the Streamlit image-detection workflow.

### Data quality and operation

- Normalize common CSV column aliases.
- Convert dates and numeric coordinates safely.
- Use the Windows certificate store for secure iNaturalist HTTPS requests.
- Run through Streamlit at `http://localhost:8501`.

## Partial

### Historical dataset support

- CSV importer, dashboard upload, normalization, and SQLite storage are implemented.
- Historical GBIF extraction page and separate CSV output are implemented.
- Paginated five-region GBIF collection, taxonomy categorization, and occurrence census are implemented.
- The historical migration CSV has not yet been added to `data/raw/observations.csv`.
- Migration role, season, seasonal density, and IUCN charts render automatically when those fields are present in the uploaded dataset.
- The broad GBIF query can fall back to iNaturalist when GBIF returns HTTP 5xx; curated dataset selections report the GBIF error instead of changing source.

### Geographic analysis

- Coordinates and place text are stored.
- Basic location filtering and an interactive coordinate map are available.
- Hotspot analysis and validated protected-area boundaries are still needed.

### Reporting

- Dashboard charts and tables are available.
- Filtered CSV report download is available.
- PDF conservation reports are still needed.

## Planned

- Audit the historical migration dataset after it is provided.
- Add migration trend analysis after historical data is imported.
- Add hotspot analysis and validated protected-area boundaries.
- Add formal PDF conservation reports.
- Add manual verification and correction of detections.
- Expand evaluation with a separately annotated wildlife image set and per-image error analysis.
- Add camera-trap batch processing.
- Add model accuracy metrics such as precision, recall, and mAP.
- Add future drone, IoT, and edge-AI adapters.

## Current verified demo state

- Local Streamlit app runs at `http://localhost:8501`.
- The fine-tuned four-animal model is the default image classifier; SpeciesNet remains available for comparison.
- The dashboard currently reports that `data/raw/observations.csv` is missing; historical migration analysis is therefore not populated until that file is supplied or uploaded.
- iNaturalist and GBIF counts are request-dependent and must be recorded with the region, date window, filters, and fetch date; old demo totals are not treated as current totals.
