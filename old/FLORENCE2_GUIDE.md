# Florence-2 Feature Detection Guide

Florence-2-base is a small Microsoft vision-language model designed for prompt-driven vision tasks. It has about 0.23B parameters and is suitable for lower-memory feature probing.

This repository uses it to draw feature boxes from Florence-2 detections. Florence-2 does not provide a calibrated confidence score, so returned boxes are not ranked by confidence.

## Install

The project depends on `transformers`, `torch`, `Pillow`, `einops`, and `timm` for Florence-2. From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The repository loader uses `models/florence2` when the local model is installed; otherwise, the first run downloads `microsoft/Florence-2-base` from Hugging Face.

## Run A Probe

Use a crop or source image and one or more feature phrases:

```powershell
.\.venv\Scripts\python.exe annotation\probe_florence2.py `
  data\training\images\greater\example.jpg `
  "single horn" "armor-like skin folds" "rounded ears" `
  --output outputs\florence_probe.json
```

On Windows PowerShell, keep the backtick at the end of each continued line. A one-line form also works:

```powershell
.\.venv\Scripts\python.exe annotation\probe_florence2.py data\training\images\greater_one_horned_rhino\IMAGE.jpg "single horn" "armor-like skin folds" "rounded ears"
```

The probe uses Florence-2's `<OPEN_VOCABULARY_DETECTION>` task and returns normalized image-space bounding boxes after `post_process_generation` converts the model output.

The probe forces eager attention and disables generation caching because the Florence-2 remote model code is not compatible with the SDPA and cache defaults in newer Transformers releases.

## Output Shape

The result contains one entry per query. A typical detection payload has this shape:

```json
{
  "query": {
    "<OPEN_VOCABULARY_DETECTION>": {
      "bboxes": [[10, 20, 100, 120]],
      "bboxes_labels": ["query"]
    }
  }
}
```

Florence-2 does not provide a calibrated confidence score for this task. The pipeline draws returned boxes in green and records `not_visible` only when no box is returned. Review boxes and measure precision on a labeled sample before using them as training labels.

When multiple boxes are returned, the proposal keeps every normalized box in `candidate_bboxes` and stores the largest candidate in `bbox` for compatibility. The selected index is recorded in `selected_bbox_index`; this selection still requires human review.

## Recommended Evaluation

1. Keep 20 to 50 reviewed crops per species.
2. Compare false positives, missed features, invalid boxes, and elapsed time per image.
3. Florence-2 does not provide a calibrated confidence score for this task, so proposals contain no confidence field.
4. Verify boxes manually before promoting labels into training data.
5. Once the labels are stable, train a small YOLO feature detector for the final fast batch pipeline.

Batch runs skip existing JSON files by default and continue after individual image failures. Use `--overwrite` to regenerate existing results.

In the initial rhino smoke test, Florence returned duplicate horn candidates and broad feature boxes. Treat this as grounding output for review, not as verified annotation data.

## Limitations

Florence-2-base is a grounding model, not a specialist Indian wildlife anatomy model. It may still confuse a horn, ear, body fold, or background object. Fine-tuning on reviewed project crops is the path to reliable production results.

Official references:

- [Florence-2-base model card](https://huggingface.co/microsoft/Florence-2-base)
- [Florence-2 technical report](https://arxiv.org/abs/2311.06242)
