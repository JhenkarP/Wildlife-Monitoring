# Cat classifier

This is a separate Felidae classifier workspace. It does not modify or import the existing Wildlife Monitoring detector.

## Current phase

- Defines a stable label contract for 41 extant Felidae species.
- Groups labels by subfamily, lineage, and genus.
- Shows common and scientific names in the web interface.
- Includes an explicit unknown / insufficient-evidence result.
- Leaves model training behind a checkpoint boundary at `models/felidae_classifier.pt`.

## Run the website

From the repository root:

```powershell
pip install -r cat_classifier/requirements.txt
streamlit run cat_classifier/app.py
```

The website includes the complete taxonomy explorer, classification flowchart, and image-upload placeholder. A prediction model is not claimed until a trained checkpoint is added.

## Taxonomy

The label list follows the IUCN/SSC Cat Specialist Group's Living Species page and is cross-checked against the Wikipedia List of felids. It contains 40 wild living species plus domestic cat as a separate non-wild label. Check the source pages before retraining because accepted taxonomy can change.

The static Mermaid version of the flowchart is in `taxonomy/classification_flowchart.md`.

## Collect images for experiments

Install the separate requirements and collect a balanced starter set. The default cap is 100 accepted images per taxonomy label, with source metadata and duplicate hashes preserved:

```powershell
pip install -r cat_classifier/requirements.txt
python cat_classifier/data_collection/download_inaturalist_felidae.py --per-species 100
```

Images are written under `cat_classifier/data/inat/` and the attribution manifest is `cat_classifier/data/inat/manifest.csv`. The collector uses research-grade iNaturalist observations with photos, filters small images, and deduplicates by SHA-256. Counts in the availability snapshot are not promises that every image can be downloaded; some photos may be unavailable or have restrictive licenses, so review `license` and `attribution` before publishing a dataset.

## Create the first experiment split

The first experiment uses labels with at least 50 images and keeps all photos from the same observation in one split:

```powershell
python cat_classifier/experiments/split_dataset.py --minimum-images 50
```

This writes `cat_classifier/data/splits/train.csv`, `validation.csv`, `test.csv`, and `summary.txt`.

## Run the no-training embedding baseline

This uses pretrained ImageNet ResNet18 features and nearest class prototypes. It does not train the network:

```powershell
python cat_classifier/experiments/embedding_baseline.py --device auto
```

The result is written to `cat_classifier/data/experiments/embedding_baseline.json`.

## Run BioCLIP zero-shot evaluation

BioCLIP compares each image with prompts for every taxonomy label. It does not train on the local dataset:

```powershell
pip install -r cat_classifier/requirements.txt
python cat_classifier/experiments/bioclip_zero_shot.py --device auto
```

The result is written to `cat_classifier/data/experiments/bioclip_zero_shot.json`.
