# Engineering review — pest-detection

**Repository:** `Abdulrahman181/pest-detection`
**Scope:** the repository at `/home/ubuntu/github_priority/pest-detection` only
**Review date:** 2026-10-14
**Baseline:** one `pest-detection.ipynb` file, with no dataset, weights, requirements file, tests, or README.

## Executive summary

The original notebook was a Kaggle-only, stateful experiment rather than a portable project. It embedded `/kaggle/input` and `/kaggle/working` paths, installed `ultralytics` from inside a cell, used an unseeded random split, stored 587 historical output objects, hard-coded ten classes, and assumed that a prior in-memory `model` and training artifacts existed. It did not enforce safe COCO paths or validate bounding boxes before conversion. No trained result is claimed by this review.

The notebook is now a clean, six-cell workflow that requires an explicitly supplied dataset, validates it, performs deterministic preparation, keeps exact duplicate images together when splitting, and makes training/evaluation opt-in. The reusable validation/conversion/splitting logic is in `pest_detection_pipeline.py`; it fails clearly instead of inventing data or silently using stale artifacts.

## Findings and remediations

### Data and reproducibility

- **Hard-coded local paths:** removed Kaggle and `/content` paths from project logic. `PEST_DATASET_ROOT` and `PEST_OUTPUT_DIR` are explicit inputs.
- **Missing data behavior:** missing COCO JSON, image directories, referenced files, malformed JSON, duplicate IDs/names, invalid dimensions, unknown annotation references, unsafe `..`/absolute paths, and invalid bounding boxes raise `DataUnavailableError` with an actionable message.
- **Split reproducibility:** the original `random.sample` had no seed. The replacement uses seed `42` by default and records the split configuration in the notebook output.
- **Exact duplicate leakage:** split groups are based on SHA-256 image bytes, so exact duplicates cannot land in both partitions. If all images are identical, the split is refused. This does not detect perceptual or near-duplicate images; a separate dataset-level audit is still needed for that threat.
- **Validation protocol:** the project still has only train/validation data. Validation metrics must not be presented as an unbiased test result; the notebook explicitly states that a held-out test set is absent.
- **Labels/classes:** the original YAML hard-coded `nc: 10` and names independently of the COCO document. The new workflow derives names from the validated COCO categories and writes `nc` consistently.
- **Corrupt files:** image validation now checks actual image suffixes with Pillow and stops with the paths of corrupt files. The old loop attempted to open every file, so it could report non-image files such as JSON as “corrupted images.”

### Correctness and static quality

- Removed the duplicated `model.save` cell and stateful assumptions about a pre-existing `model`/`metrics` object.
- Removed fragile plotting cells that depended on stale, inconsistent artifact paths (`/kaggle/working/...` versus relative `runs/...`) and could fail with missing CSV columns.
- Removed `!pip install ultralytics` from the notebook; dependencies are declared in `requirements.txt` and optional `requirements-ml.txt`.
- Added AST parsing for every Python file and every notebook code cell, plus Ruff checks. No syntax errors or Ruff findings remain.
- Added unit tests for missing-input errors, COCO-to-YOLO conversion, path traversal rejection, and deterministic non-overlapping splitting.

### Security and artifact handling

- COCO `file_name` values are rejected if absolute or containing parent traversal, and resolved source paths are constrained to the declared image root.
- Generated datasets, model weights, runs, and local environments are ignored by Git; no data or model artifact was added to the repository.
- The training cell is opt-in (`PEST_RUN_TRAINING=1`) and does not claim success when `ultralytics`, weights, or the dataset are unavailable.

## Files changed

- `pest-detection.ipynb` — replaced the stateful notebook; all old outputs and execution counts removed.
- `pest_detection_pipeline.py` — validated COCO loading, image checks, safe conversion, duplicate-aware deterministic split, YAML writing, and explicit artifact errors.
- `tests/test_pipeline.py` — four unit tests covering core safety and reproducibility behavior.
- `README.md` — setup, input contract, operation, limitations, and validation commands.
- `requirements.txt` — core Pillow and PyYAML dependencies.
- `requirements-ml.txt` — optional Ultralytics training dependency.
- `.gitignore` — ignores local datasets and generated artifacts.
- `ENGINEERING_REVIEW.md` — this report.

## Validation performed

- `python3 -m unittest discover -s tests -v` — **4 tests passed**.
- `ruff check .` — **passed**.
- `ruff check /tmp/pest_detection_notebook.py` after extracting notebook code — **passed**.
- Python `ast.parse` on every repository Python file and notebook code cell — **passed**.
- `python3 -m json.tool pest-detection.ipynb` — **passed**.
- Notebook inspection — **6 cells, 0 outputs, all code execution counts null**.
- Text scan — no Kaggle/content paths, embedded secrets, or package-install commands remain in executable project code.
- Full model training/evaluation — **not run**: the repository contains no dataset or weights, and the heavy training dependency is optional. No metrics are reported.

## Remaining limitations

1. The external dataset is not included, so end-to-end conversion, corruption scanning over the real corpus, training, and evaluation could not be completed honestly.
2. No independent test split exists in the source project; add one before publishing generalization claims.
3. Byte hashes catch exact duplicates only. For synthetic or augmented image families, add provenance/group IDs or perceptual duplicate detection before splitting.
4. Training determinism depends on the installed Ultralytics/PyTorch stack and hardware in addition to the explicit seed; the data preparation itself is deterministic.
5. No web/API service exists in this repository, so there was no HTTP API surface to test.

A local commit is created after these checks. Nothing is pushed to GitHub.
