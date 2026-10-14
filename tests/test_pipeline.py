import json
import tempfile
import unittest
from pathlib import Path

from pest_detection_pipeline import (
    DataUnavailableError,
    convert_coco_to_yolo,
    load_coco,
    split_dataset,
    validate_dataset_inputs,
)


class PipelineTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> tuple[Path, Path]:
        image_dir = root / "images"
        image_dir.mkdir()
        # Valid file names and deterministic bytes are sufficient for conversion tests.
        (image_dir / "a.jpg").write_bytes(b"image-a")
        (image_dir / "b.jpg").write_bytes(b"image-b")
        coco = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 100, "height": 80},
                {"id": 2, "file_name": "b.jpg", "width": 100, "height": 80},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 20, 30, 40]},
            ],
            "categories": [{"id": 7, "name": "beetle"}],
        }
        coco_path = root / "annotations.json"
        coco_path.write_text(json.dumps(coco), encoding="utf-8")
        return coco_path, image_dir

    def test_missing_input_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(DataUnavailableError, "was not found"):
            load_coco("/definitely/missing/annotations.json")

    def test_conversion_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coco_path, image_dir = self.make_dataset(Path(directory))
            self.assertEqual(validate_dataset_inputs(coco_path, image_dir)["images"], 2)
            output = Path(directory) / "yolo"
            result = convert_coco_to_yolo(coco_path, image_dir, output)
            self.assertEqual(result["annotations"], 1)
            self.assertEqual(
                (output / "labels" / "a.txt").read_text(encoding="utf-8"),
                "0 0.250000 0.500000 0.300000 0.500000\n",
            )
            self.assertEqual((output / "labels" / "b.txt").read_text(encoding="utf-8"), "")

    def test_unsafe_file_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coco_path, image_dir = self.make_dataset(Path(directory))
            payload = json.loads(coco_path.read_text(encoding="utf-8"))
            payload["images"][0]["file_name"] = "../outside.jpg"
            coco_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DataUnavailableError, "Unsafe image file_name"):
                validate_dataset_inputs(coco_path, image_dir)

    def test_split_is_reproducible_and_non_leaky_for_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coco_path, image_dir = self.make_dataset(root)
            # Add a third distinct file so both partitions are possible.
            (image_dir / "c.jpg").write_bytes(b"image-c")
            payload = json.loads(coco_path.read_text(encoding="utf-8"))
            payload["images"].append({"id": 3, "file_name": "c.jpg", "width": 100, "height": 80})
            coco_path.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "yolo"
            convert_coco_to_yolo(coco_path, image_dir, output)
            first = split_dataset(output, val_fraction=1 / 3, seed=123)
            train_first = sorted(p.name for p in (output / "images" / "train").iterdir())
            val_first = sorted(p.name for p in (output / "images" / "val").iterdir())
            second = split_dataset(output, val_fraction=1 / 3, seed=123)
            self.assertEqual(first, second)
            self.assertEqual(train_first, sorted(p.name for p in (output / "images" / "train").iterdir()))
            self.assertEqual(val_first, sorted(p.name for p in (output / "images" / "val").iterdir()))
            self.assertEqual(set(train_first) & set(val_first), set())


if __name__ == "__main__":
    unittest.main()
