# Pest detection

This repository contains a cleaned, reproducible preparation workflow for a COCO-style insect dataset and optional YOLO training. **The repository does not include the dataset or trained weights.** No data, labels, metrics, or model results are fabricated.

## Requirements

Use Python 3.10+ (tested with Python 3.12):

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
# Only for training/evaluation:
python -m pip install 'ultralytics>=8.3,<9'
```

## Prepare data

Set `PEST_DATASET_ROOT` to a directory containing:

- `ImageClassesCombinedWithCOCOAnnotations/coco_instances.json`
- `ImageClassesCombinedWithCOCOAnnotations/images_raw/`

Then run the notebook or call the tested helpers directly. The converter validates referenced files, category IDs, dimensions, bounding boxes, and unsafe paths before copying anything. It writes YOLO files below the output directory and creates a deterministic 80/20 split with seed 42. Exact byte-identical images are kept together to prevent an obvious leakage path.

```python
from pathlib import Path
from pest_detection_pipeline import (
    convert_coco_to_yolo,
    split_dataset,
    validate_dataset_inputs,
    write_dataset_yaml,
)

root = Path("/path/to/dataset")
annotations = root / "ImageClassesCombinedWithCOCOAnnotations/coco_instances.json"
images = root / "ImageClassesCombinedWithCOCOAnnotations/images_raw"
out = Path("/path/to/generated/yolo_dataset")
validate_dataset_inputs(annotations, images)
convert_coco_to_yolo(annotations, images, out)
split_dataset(out, val_fraction=0.20, seed=42)
# Use the names from the validated COCO categories in your dataset.
write_dataset_yaml(out, class_names=["..."])
```

If a required file is absent or invalid, the workflow stops with an actionable `DataUnavailableError`; it does not silently continue or create placeholder data.

## Optional training

Training is intentionally opt-in because it downloads/loads heavy dependencies and weights:

```bash
PEST_DATASET_ROOT=/path/to/dataset PEST_RUN_TRAINING=1 jupyter lab pest-detection.ipynb
```

The notebook uses an explicit seed and deterministic data preparation. It does not install packages inside a notebook cell. Evaluation and plots require a real trained model or real results CSV; missing artifacts are reported instead of producing stale or invented metrics. Do not treat validation metrics as an unbiased test score: the current workflow has no separate held-out test set.

## Validation

```bash
python -m unittest discover -s tests -v
ruff check .
python -m json.tool pest-detection.ipynb >/dev/null
```

The tests cover missing-input errors, COCO-to-YOLO conversion, path traversal rejection, and reproducible non-overlapping splits. Full training and image corruption checks require the external dataset and optional ML dependencies, which are not bundled here.
