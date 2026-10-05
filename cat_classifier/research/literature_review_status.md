# Literature Review Status

Date: 2026-10-04

## Current status

A broad search was performed through OpenAlex and Crossref after Semantic Scholar rate-limited the initial bulk request. The search returned a large candidate pool, including benchmark papers, conservation platforms, camera-trap studies, pose estimation, re-identification, uncertainty, drone monitoring, and biological foundation models.

This is not yet a claim that 100 papers were fully read. The review is being tracked by evidence level:

- **Full text / primary page inspected:** methods, dataset, evaluation design, and limitations extracted.
- **Abstract inspected:** research question and reported contribution extracted, but detailed claims remain provisional.
- **Metadata only:** candidate for later screening; not used as evidence for a strong conclusion.

## Directly inspected anchors

| Paper                                                  | Main evidence                                                                                                                                         | Project implication                                                                |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| BioCLIP, 2024                                          | Biology-specific foundation model; TreeOfLife-10M; reports substantial gains on fine-grained biology classification and hierarchical representations. | Test BioCLIP before collecting thousands of additional images.                     |
| iWildCam 2020                                          | Train/test cameras and locations differ; explicitly studies unseen-camera generalization and multimodal context.                                      | Camera/location-held-out evaluation is mandatory.                                  |
| AP-10K, 2021                                           | 10,015 images, 54 species, 23 families; evaluates cross-domain and unseen-animal pose generalization.                                                 | Diversity and transfer matter more than simply adding near-duplicate images.       |
| Wildlife Insights, 2019                                | Large-scale passive-sensor data needs integrated, traceable tooling for conservation use.                                                             | Preserve metadata and build a review workflow, not only a classifier.              |
| Perspectives on Individual Animal Identification, 2021 | Reviews automated individual identification and its biological applications and limitations.                                                          | Keep species recognition separate from individual re-identification.               |
| SOCRATES, 2022                                         | Stereo/depth sensing improves wildlife monitoring and supports 3D measurement.                                                                        | Metric size requires calibrated depth or scale; ordinary RGB is insufficient.      |
| Automated visitor and wildlife monitoring, 2023        | Large cross-regional camera-trap evaluation of MegaDetector; high aggregate detection performance but systematic errors remain.                       | Report per-class and cross-region failure modes.                                   |
| Smart camera traps and computer vision, 2025           | Very large field video evaluation; strong detection can coexist with poor recall for difficult taxa.                                                  | Aggregate precision is not enough; evaluate rare and difficult classes separately. |

## Additional abstract-reviewed studies

These studies were screened through scholarly metadata and abstract records. They are useful for scope and hypothesis generation, but are not counted as full-text reads until the article itself is inspected.

| Study                                                                                                 | Abstract-level finding                                                                                                                     | Relevance                                                                                                                        |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Animal recognition and identification with deep CNNs for automated wildlife monitoring, 2017          | Uses Wildlife Spotter camera-trap data and reports strong in-dataset animal filtering and species recognition.                             | Supports automation, but motivates a stricter location-safe evaluation than a single benchmark split.                            |
| Multifactorial uncertainty assessment for monitoring population abundance using computer vision, 2015 | Frames uncertainty as a combination of ecological, data, and computer-vision factors rather than only classifier confidence.               | Supports explicit calibration and uncertainty reporting in the proposed system.                                                  |
| Visual Informatics Tools for Supporting Large-Scale Collaborative Wildlife Monitoring, 2016           | Describes an end-to-end ecosystem including image processing, species classification, biometric features, cloud data, and citizen science. | Supports building a provenance-aware review workflow instead of an isolated prediction endpoint.                                 |
| Automated Wildlife Bird Detection from Drone Footage Using Computer Vision Techniques, 2023           | Applies object detection to aerial wildlife imagery and reports strong precision on its dataset.                                           | Demonstrates the value of deployment-specific evaluation and warns against transferring aerial results directly to camera traps. |
| Accelerating ecosystem monitoring through computer vision with deep metric learning, 2024             | Motivates feature learning for high-volume ecological observations and wide-area monitoring.                                               | Supports retrieval and embedding experiments alongside closed-set classification.                                                |
| Image Analysis and Computer Vision Applications in Animal Sciences: An Overview, 2020                 | Reviews applications including identification, tracking, behavior, body weight, body condition, and gait.                                  | Confirms that recognition and physical trait estimation are distinct tasks with different data requirements.                     |

## Synthesis

The strongest project contribution is an uncertainty-aware Felidae monitoring system evaluated under source, observation, camera, and location shift. A simple 41-way classifier trained and tested on mixed web images would reproduce the main weakness identified across the literature: optimistic in-domain performance with unclear field reliability.

## Screening and extraction fields for the remaining corpus

Each candidate paper should receive:

- citation and persistent identifier
- publication type and venue
- taxa and conservation context
- sensor/source: camera trap, citizen science, drone, thermal, stereo, satellite, video
- task: detection, classification, pose, behavior, re-identification, demographic inference, disease, habitat, alerting
- dataset size and label structure
- split design and leakage controls
- model and pretraining source
- metrics and reported result
- external or field validation
- limitations and reproducibility status
- relevance to this Felidae project
- evidence level: full text, abstract, or metadata

## Exclusion rules

Exclude from strong synthesis:

- duplicate versions of the same work
- papers with no wildlife or biodiversity monitoring connection
- generic computer-vision papers returned by keyword search
- unverifiable claims without a dataset or evaluation description
- papers that report only a self-created dataset with no clear split or test protocol

## Review conclusion so far

The project should prioritize:

1. reliable cross-location evaluation
2. biology-aware pretrained models
3. calibrated confidence and `Unknown`
4. provenance and expert-review workflow
5. optional pose/depth extensions only after recognition is validated

Exact height, weight, sex, or population abundance should not be promised from ordinary photographs without task-specific labels and measurement design.
