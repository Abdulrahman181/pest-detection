# Pest detection notebook

This repository contains a Jupyter notebook for converting COCO insect bounding-box annotations to YOLO format, splitting the images into training and validation sets, and training/evaluating a YOLOv8 model.

## Data and environment

The notebook expects the separately obtained Insect Village dataset to contain:

```
<dataset-root>/ImageClassesCombinedWithCOCOAnnotations/coco_instances.json
<dataset-root>/ImageClassesCombinedWithCOCOAnnotations/images_raw/
```

The dataset is not included in this repository. From the repository root, set `PEST_DATASET_ROOT` to the extracted dataset directory and optionally `PEST_WORK_DIR` to a location for generated data, plots, and model files. The defaults detect the original Kaggle input/working folders when available; otherwise the defaults are `data/insect-village-synthetic-dataset` and `artifacts` relative to the notebook's working directory.

Install the Python dependency with `python -m pip install -r requirements.txt`, then open `pest-detection.ipynb` and run its cells from top to bottom. The notebook uses the Ultralytics pretrained `yolov8s.pt` weights, which may need to be downloaded on first run. Training requires substantial time and compute; a GPU-capable PyTorch install may be needed for practical runtimes. Select a PyTorch build appropriate for your hardware before training if the default dependency installation is not suitable.

The COCO class names are read from the annotation file, and image splitting uses seed 42 for repeatable splits. Re-running the split cell recreates only the generated `images/train`, `images/val`, `labels/train`, and `labels/val` folders under the configured output dataset directory. The notebook no longer deletes the entire Kaggle working directory.

## Validation and limitations

No dataset or trained model is included, and this repository change does not run model training or independently validate accuracy. Metrics printed by the notebook are results from the user's own data and runtime, not a performance guarantee. Confirm dataset licensing and annotation quality before use.
