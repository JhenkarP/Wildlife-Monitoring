# Felidae Monitoring Research Project

## Proposed title

**An Uncertainty-Aware, Taxonomy-Guided Vision System for Cross-Location Felidae Monitoring**

## Research question

Can a biology-aware vision model identify Felidae reliably enough for conservation monitoring when test images come from observations, cameras, habitats, and sources not represented in training data?

The project must measure reliability under distribution shift, not only accuracy on randomly mixed internet images.

## Why this project

The reviewed wildlife computer-vision literature repeatedly shows three problems:

1. Large image collections make manual review impractical.
2. Random image splits can produce optimistic results because related images, observations, or camera locations appear in both train and test data.
3. A high-confidence wrong prediction is dangerous in conservation workflows.

BioCLIP demonstrates that a biology-specific foundation model can improve fine-grained organism recognition and learn hierarchical biological structure. iWildCam demonstrates that unseen-camera evaluation is a central wildlife-recognition problem. AP-10K demonstrates that broad animal diversity improves pose generalization. Wildlife Insights and related monitoring systems show that traceable metadata and operational workflows matter alongside the model.

## Scope

### In scope

- The existing 41-label Felidae taxonomy.
- Species recognition from still images.
- Hierarchical predictions: family, subfamily, genus, and species.
- Image quality and insufficient-evidence detection.
- Open-set recognition with an explicit `Unknown` result.
- Retrieval of visually similar reference images.
- Evaluation across observation groups and source/location partitions.
- A Streamlit research dashboard with prediction evidence and provenance.

### Out of scope for the first paper

- Exact animal height or body weight from an ordinary photograph.
- Individual identity unless repeated, geographically meaningful images are available.
- Automated poaching accusations or automatic enforcement actions.
- Population-size estimation from presence-only citizen-science images.
- Claims that a prediction proves an animal is present at a location without validating the source metadata.

## Main hypotheses

### H1: Biology-aware representations improve cross-source generalization

BioCLIP zero-shot or lightly fine-tuned representations will outperform the current ImageNet ResNet18 prototype baseline on a source-held-out test set.

### H2: Observation-safe evaluation reduces apparent performance

Performance on observation-, sequence-, camera-, or location-held-out data will be lower than performance from an ordinary random split. This difference is itself a result and must be reported.

### H3: Hierarchical prediction is more useful than a flat label

A model that can correctly identify genus or lineage while abstaining at species level will be more useful and less misleading than a forced 41-way prediction.

### H4: Calibration and abstention improve operational reliability

Confidence calibration plus an `Unknown` threshold will reduce high-confidence errors at a chosen coverage level, even if raw top-1 accuracy decreases.

## System design

```text
Image
  -> provenance and quality checks
  -> animal crop or detector
  -> BioCLIP / supervised embedding
  -> taxonomy-aware species scores
  -> calibration and unknown decision
  -> retrieval evidence
  -> prediction record
  -> monitoring dashboard and export
```

Every prediction record should retain:

- image hash and source URL
- license and attribution
- observation ID, date, and location when available
- model name and checkpoint hash
- taxonomy version
- top-k labels and calibrated probabilities
- quality flags
- `predicted`, `unknown`, or `insufficient_evidence` status

## Data strategy

### Existing data

Use the current iNaturalist research-grade manifest as the initial development set. Reconcile the manifest with files on disk before any new training. Do not silently treat missing files as negative examples.

### Evaluation partitions

Create separate evaluation views:

1. **Observation split:** all images from one observation remain in one partition.
2. **Source split:** train on iNaturalist and test on a verified external source where licensing permits.
3. **Location split:** hold out geographic regions when sufficient metadata exists.
4. **Camera or sequence split:** required for camera-trap data to prevent near-duplicate leakage.
5. **Species-support split:** report results separately for high-support and low-support labels.

### Data quality rules

- Deduplicate by image hash and perceptual similarity.
- Preserve source metadata and licenses.
- Keep domestic cat separate from wild Felidae.
- Record unavailable or deleted images as missing, not as `none`.
- Freeze the evaluation manifest before model comparison.

## Experiments

### Phase 0: Reproducible baseline

- Repair and reconcile the downloader manifest.
- Freeze taxonomy version and dataset manifest.
- Re-run the current ResNet18 embedding and supervised baselines.
- Report macro-F1, balanced accuracy, per-class recall, confusion matrix, and split composition.

### Phase 1: Biology-aware zero-shot model

- Install and run BioCLIP.
- Compare common-name, scientific-name, and combined prompt templates.
- Evaluate flat and hierarchical scoring.
- Record inference speed and GPU memory use.

### Phase 2: Lightweight adaptation

Compare:

- frozen BioCLIP image embeddings plus linear classifier
- frozen embeddings plus nearest-prototype retrieval
- partial fine-tuning of the classification head
- supervised ResNet18 baseline

Use identical partitions and no test-set tuning.

### Phase 3: Unknown and calibration

Evaluate:

- maximum softmax probability
- energy score or embedding distance
- temperature scaling on validation data
- confidence threshold versus coverage
- known Felidae versus domestic cat versus non-Felidae distractors

The dashboard must be able to say `Unknown` instead of forcing a species label.

### Phase 4: Retrieval and explanation

For each result, show the nearest verified reference images and their source metadata. Retrieval is supporting evidence, not proof. Test whether retrieval improves human review time and correction accuracy in a small user study.

### Phase 5: Optional pose and geometry extension

Only after the recognition study is stable:

- use animal pose estimation for visible body landmarks
- report normalized body proportions in pixels
- estimate metric size only when a known reference, stereo/depth camera, or calibrated scene is available
- report ranges and uncertainty, never unsupported exact weight

## Evaluation protocol

Primary metrics:

- macro-F1
- balanced accuracy
- per-class recall
- top-1 and top-5 accuracy
- hierarchical accuracy at genus and species levels
- expected calibration error
- Brier score
- selective risk at fixed coverage
- unknown detection AUROC and AUPR
- cross-source and cross-location performance gap

Required comparisons:

- ImageNet ResNet18 versus BioCLIP
- flat classifier versus taxonomy-aware classifier
- forced prediction versus abstaining prediction
- random split versus observation-safe split
- in-domain versus source-held-out evaluation

Do not report a single accuracy number without the partition definition, class support, and provenance policy.

## Conservation use case

The first operational use case is **assisted review of Felidae observations**:

- a field worker uploads an image
- the system returns a species candidate, lineage, confidence, and similar references
- uncertain cases are queued for expert review
- accepted records can be exported with provenance
- aggregate reports show observations by species, date, and region

The system supports monitoring. It does not replace field confirmation or make management decisions automatically.

## Deliverables

1. Reconciled, licensed dataset manifest.
2. Frozen split definitions and leakage audit.
3. Baseline and BioCLIP evaluation reports.
4. Calibration and unknown-detection report.
5. Taxonomy-aware prediction API.
6. Streamlit review dashboard.
7. Reproducible experiment configuration and model cards.
8. Research paper draft with limitations and error analysis.

## Milestones

### Milestone 1: Data and baseline

- Repair manifest and missing files.
- Freeze taxonomy and splits.
- Reproduce current baseline.

### Milestone 2: Model comparison

- Run BioCLIP zero-shot.
- Run frozen-embedding and supervised comparisons.
- Produce cross-partition metrics.

### Milestone 3: Trust layer

- Add calibration, abstention, quality flags, and retrieval.
- Evaluate false positives and high-confidence errors.

### Milestone 4: Application

- Integrate the prediction record into Streamlit.
- Add provenance-aware export and review queue.

### Milestone 5: Paper

- Complete error analysis.
- Document the literature evidence matrix.
- Write methods, results, limitations, and conservation implications.

## Success criteria

The project succeeds if it demonstrates all of the following:

- the data split prevents observation or camera leakage
- the model is evaluated on at least one held-out source or location
- the system can abstain on unfamiliar or poor-quality images
- every prediction is traceable to model, taxonomy, source, and image metadata
- performance and failure modes are reported per species
- the application helps a reviewer make a better decision than an unassisted forced classifier

## Core references reviewed directly

- Stevens et al., **BioCLIP: A Vision Foundation Model for the Tree of Life**, CVPR 2024. https://arxiv.org/abs/2311.18803
- Beery, Cole, and Gjoka, **The iWildCam 2020 Competition Dataset**, 2020. https://arxiv.org/abs/2004.10340
- Yu et al., **AP-10K: A Benchmark for Animal Pose Estimation in the Wild**, NeurIPS 2021. https://arxiv.org/abs/2108.12617
- Schneider et al., **Wildlife Insights: A Platform to Maximize the Potential of Camera Trap and Other Passive Sensor Wildlife Data for the Planet**, 2019. https://doi.org/10.1017/S0376892919000298
- Dujon et al., **Perspectives on Individual Animal Identification from Biology and Computer Vision**, 2021. https://doi.org/10.1093/icb/icab107
- Tschanz et al., **SOCRATES: Introducing Depth in Visual Wildlife Monitoring Using Stereo Vision**, 2022. https://doi.org/10.3390/s22239082
- Gómez Villa et al., **Automated visitor and wildlife monitoring with camera traps and machine learning**, 2023. https://doi.org/10.1002/rse2.367
- **Smart camera traps and computer vision improve detections of small fauna**, 2025. https://doi.org/10.1002/ecs2.70220

## Evidence boundary

A broad candidate corpus was collected through OpenAlex and Crossref. The references above are the directly inspected anchors for this plan. The remaining candidate records must be entered into an evidence matrix and marked as either full-text reviewed, abstract reviewed, or metadata-only before the final literature-review claim is made.
