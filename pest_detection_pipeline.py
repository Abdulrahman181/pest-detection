"""Safe, deterministic data preparation helpers for the pest detector.

The module deliberately does not download data, invent annotations, or train a model.
Missing inputs fail with :class:`DataUnavailableError` and an actionable message.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DataUnavailableError(RuntimeError):
    """Raised when a required dataset or artifact is unavailable or invalid."""


def require_existing_path(path: Path | str, description: str, *, directory: bool = False) -> Path:
    """Return an existing path or raise a clear, non-silent error."""

    candidate = Path(path).expanduser()
    valid = candidate.is_dir() if directory else candidate.is_file()
    expected = "directory" if directory else "file"
    if not valid:
        raise DataUnavailableError(
            f"Required {description} {expected} was not found: {candidate}. "
            "Provide the dataset/artifact explicitly; no data is generated automatically."
        )
    return candidate.resolve()


def _safe_relative_name(name: str) -> Path:
    """Validate a COCO file name before using it as a local path."""

    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise DataUnavailableError(f"Unsafe image file_name in COCO JSON: {name!r}")
    if not name or relative.name in {"", "."}:
        raise DataUnavailableError(f"Empty image file_name in COCO JSON: {name!r}")
    return relative


def load_coco(coco_json: Path | str) -> dict[str, Any]:
    """Load and minimally validate a COCO annotations document."""

    path = require_existing_path(coco_json, "COCO annotations")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataUnavailableError(f"Could not read valid COCO JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataUnavailableError(f"COCO JSON must be an object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise DataUnavailableError(f"COCO JSON field {key!r} must be a list: {path}")
    if not payload["categories"]:
        raise DataUnavailableError(f"COCO JSON contains no categories: {path}")
    return payload


def validate_dataset_inputs(coco_json: Path | str, image_dir: Path | str) -> dict[str, int]:
    """Validate input paths, image references, categories, and bounding boxes."""

    image_root = require_existing_path(image_dir, "image root", directory=True)
    payload = load_coco(coco_json)
    category_ids = [category.get("id") for category in payload["categories"]]
    category_names = [category.get("name") for category in payload["categories"]]
    if any(value is None for value in category_ids + category_names):
        raise DataUnavailableError("Every COCO category must contain id and name")
    if len(set(category_ids)) != len(category_ids) or len(set(category_names)) != len(category_names):
        raise DataUnavailableError("COCO category ids and names must be unique")

    image_ids: set[int] = set()
    image_names: set[Path] = set()
    image_dimensions: dict[int, tuple[float, float]] = {}
    missing: list[str] = []
    for image in payload["images"]:
        image_id = image.get("id")
        name = image.get("file_name")
        width, height = image.get("width"), image.get("height")
        if image_id is None or not isinstance(name, str) or not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise DataUnavailableError("Every COCO image must contain id, file_name, numeric width, and height")
        if image_id in image_ids:
            raise DataUnavailableError(f"Duplicate COCO image id: {image_id}")
        if width <= 0 or height <= 0:
            raise DataUnavailableError(f"Image {name!r} has non-positive dimensions")
        relative = _safe_relative_name(name)
        if relative in image_names:
            raise DataUnavailableError(f"Duplicate COCO file_name: {name!r}")
        image_ids.add(image_id)
        image_names.add(relative)
        image_dimensions[image_id] = (float(width), float(height))
        source = (image_root / relative).resolve()
        if image_root not in source.parents and source != image_root:
            raise DataUnavailableError(f"Image resolves outside image root: {name!r}")
        if not source.is_file():
            missing.append(name)

    if missing:
        shown = ", ".join(repr(item) for item in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise DataUnavailableError(f"COCO references {len(missing)} missing image(s): {shown}{suffix}")

    category_id_set = set(category_ids)
    for annotation in payload["annotations"]:
        if annotation.get("image_id") not in image_ids:
            raise DataUnavailableError(f"Annotation references unknown image_id: {annotation.get('image_id')!r}")
        if annotation.get("category_id") not in category_id_set:
            raise DataUnavailableError(f"Annotation references unknown category_id: {annotation.get('category_id')!r}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
            raise DataUnavailableError("Every annotation bbox must contain four numeric values")
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise DataUnavailableError(f"Annotation has non-positive bbox: {bbox!r}")
        width, height = image_dimensions[annotation["image_id"]]
        x, y, box_width, box_height = (float(value) for value in bbox)
        if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
            raise DataUnavailableError(f"Annotation bbox exceeds image bounds: {bbox!r}")

    return {
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
        "categories": len(payload["categories"]),
    }


def find_corrupt_images(image_dir: Path | str) -> list[Path]:
    """Return unreadable image files using Pillow, without copying or deleting data."""

    root = require_existing_path(image_dir, "image root", directory=True)
    try:
        from PIL import Image
    except ImportError as exc:
        raise DataUnavailableError("Pillow is required for image validation; install requirements.txt") from exc

    corrupt: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, SyntaxError, ValueError):
                corrupt.append(path)
    return corrupt


def convert_coco_to_yolo(
    coco_json: Path | str,
    image_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, int]:
    """Copy images and convert COCO boxes to YOLO labels after full validation."""

    image_root = require_existing_path(image_dir, "image root", directory=True)
    payload = load_coco(coco_json)
    validate_dataset_inputs(coco_json, image_root)
    destination = Path(output_dir).expanduser().resolve()
    images_out, labels_out = destination / "images", destination / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    category_index = {category["id"]: index for index, category in enumerate(payload["categories"])}
    image_by_id = {image["id"]: image for image in payload["images"]}
    annotations: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations[annotation["image_id"]].append(annotation)

    for image in payload["images"]:
        relative = _safe_relative_name(image["file_name"])
        source = image_root / relative
        image_destination = images_out / relative
        label_destination = labels_out / relative.with_suffix(".txt")
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_destination)
        width, height = float(image["width"]), float(image["height"])
        lines: list[str] = []
        for annotation in annotations.get(image["id"], []):
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            x_center = (x + box_width / 2) / width
            y_center = (y + box_height / 2) / height
            lines.append(
                f"{category_index[annotation['category_id']]} {x_center:.6f} "
                f"{y_center:.6f} {box_width / width:.6f} {box_height / height:.6f}"
            )
        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {"images": len(image_by_id), "annotations": len(payload["annotations"]), "categories": len(category_index)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_dataset(output_dir: Path | str, *, val_fraction: float = 0.2, seed: int = 42) -> dict[str, int]:
    """Create deterministic train/validation copies while keeping exact duplicates together."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    root = require_existing_path(output_dir, "converted dataset", directory=True)
    images_root, labels_root = root / "images", root / "labels"
    require_existing_path(images_root, "converted images", directory=True)
    require_existing_path(labels_root, "converted labels", directory=True)
    files = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.relative_to(images_root).parts[0] not in {"train", "val"}
    )
    if len(files) < 2:
        raise DataUnavailableError("At least two images are required for a train/validation split")

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[_sha256(path)].append(path)
    if len(groups) == 1:
        raise DataUnavailableError("All images are byte-identical; refusing a leaky train/validation split")

    target_val = max(1, min(len(files) - 1, round(len(files) * val_fraction)))
    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)
    val_groups: list[list[Path]] = []
    val_count = 0
    for group in group_list:
        if val_count < target_val and val_count + len(group) < len(files):
            val_groups.append(group)
            val_count += len(group)
    if not val_groups:
        val_groups.append(group_list[0])
        val_count = len(group_list[0])
    val_set = {path for group in val_groups for path in group}
    if len(val_set) == len(files):
        group = val_groups.pop()
        val_set.difference_update(group)
        val_count -= len(group)
    if not val_set:
        raise DataUnavailableError("Could not create a non-empty validation split without duplicate leakage")

    for split in ("train", "val"):
        for directory in (images_root / split, labels_root / split):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)

    counts = {"train": 0, "val": 0}
    for image_path in files:
        split = "val" if image_path in val_set else "train"
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        if not label_path.is_file():
            raise DataUnavailableError(f"Missing YOLO label for image: {relative}")
        target_image = images_root / split / relative
        target_label = labels_root / split / relative.with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)
        counts[split] += 1
    return counts


def write_dataset_yaml(output_dir: Path | str, class_names: list[str]) -> Path:
    """Write a YOLO data config from explicit class names."""

    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be a non-empty list of unique names")
    root = require_existing_path(output_dir, "converted dataset", directory=True)
    config_path = root / "dataset.yaml"
    try:
        import yaml
    except ImportError as exc:
        raise DataUnavailableError("PyYAML is required to write dataset.yaml; install requirements.txt") from exc
    payload = {"path": str(root), "train": "images/train", "val": "images/val", "nc": len(class_names), "names": class_names}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def artifact_path(value: Path | str, description: str) -> Path:
    """Require a model/metrics artifact instead of silently plotting stale output."""

    return require_existing_path(value, description)


def training_seed(seed: int) -> None:
    """Set Python and hash seeds for callers that opt into model training."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
