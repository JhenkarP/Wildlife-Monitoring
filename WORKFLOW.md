# Project Workflow

The repository is organized by the order in which data moves through the system.

## 1. Data Collection

`data_collection/` contains source-specific download and import utilities. These scripts create or update files under `data/`.

## 2. Dataset Preparation

`dataset_preparation/` contains dataset cleaning utilities. Run these before training or annotation so the source images and labels are stable.

## 3. Annotation

`annotation/` contains manual feature annotation, manifest-based cropping, proposal validation, and promotion of reviewed labels. Only records marked `verified` are eligible for training.

## 4. Training

`training/` contains model-training entry points. The current classifier checkpoint is `models/indian_wildlife_resnet18.pt`.

## 5. Evaluation and Explainability

`evaluation/` contains model evaluation utilities. Evaluation outputs belong under `outputs/` and are not model source files.

## 6. Application and Core Code

- `app.py` is the Streamlit entry point.
- `pages/` contains Streamlit pages.
- `src/` contains reusable application services and inference code.

## Feature-label flow

```text
data_collection
  -> dataset_preparation
  -> existing/manual animal boxes
  -> annotation/crop_from_manifest.py
  -> annotation/validate_feature_records.py
  -> human review
  -> annotation/promote_verified_features.py
  -> training
```