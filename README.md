# yolov3-car-detection

[![ci](https://github.com/willvc7/yolov3-car-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/willvc7/yolov3-car-detection/actions)
Car object detection practice with YOLOv3 (Kaggle `sshikamaru/car-object-detection`).

Single-class `car` detector trained from scratch (355 annotated street-view
images, 8:2 image-level split). One YAML per experiment drives the full
pipeline (`python -m car_tools.run --exp <yaml> --stage all`):
data prep -> train -> validate -> sweep thresholds -> record metrics.

## Results (val: 71 images / 107 instances)

| exp | epochs | lr0 | P | R | mAP50 | mAP50-95 | train time (h) |
|---|---|---|---|---|---|---|---|
| `car_640_e12` | 12 | 0.1 | 0.0128 | 0.9626 | 0.4157 | 0.1122 | 0.085 |
| `car_640_e50` | 50 | 0.1 | 0.9808 | 0.9572 | 0.988 | 0.5779 | 0.354 |

See `docs/results.md` for the honest log, including the failed 12-epoch
baseline (71/71 images with no detections at `conf>=0.15` - diagnosed as
undertraining, fixed by epochs 12 -> 50 with nothing else changed).

## Reproduce (Colab, Tesla T4)

1. `scripts/colab_bootstrap.sh` - clones the pinned training engine, installs deps.
2. `scripts/fetch_data.sh` - downloads the Kaggle set (needs `~/.kaggle/kaggle.json`).
3. Open `notebooks/YOLOv3_car.ipynb`, set `EXP_YAML` to one of
   `configs/experiments/*.yaml`, Run-all. Each experiment is a 2-line diff
   from the previous one; metrics land in the record workbook + `docs/results.md`.

## Layout

- `tools/car_tools/` - experiment tooling (AGPL-3.0, see LICENSE)
- `tests/` - CPU-only pytest suite (`python -m pytest -q`)
- `configs/` - base hyperparams + per-experiment yamls
- `notebooks/` - thin caller notebook
- `scripts/` - Colab bootstrap + data fetch
- `docs/` - tuning table + honest results

Training engine (`ws6125/yolov3_pytorch`, fork of `ultralytics/yolov3`) stays a
pinned runtime dependency - it is cloned by `scripts/colab_bootstrap.sh`,
never vendored. Dataset and weights are never committed (see `.gitignore`).

## Notebook convention

- Committed notebooks carry code only: `pre-commit` runs nbstripout on every
  commit (all outputs stripped, including images).
- Executed notebooks (with outputs) live outside this repo, e.g. alongside
  the course notes under `YOLO/output/` - download them from Colab after
  each run. They are execution evidence, not source.
- Sample visuals for the portfolio go to `docs/images/` with EXP_NAME + conf
  in the caption, not into notebook outputs.

## Contribution flow (local + Colab, both may commit)

- `main` is protected: every change lands via pull request, CI (`pytest`)
  must be green before merge. Direct pushes to `main` are rejected.
- Local: edit -> `pytest -q` green -> commit (pre-commit strips notebook
  outputs automatically) -> push to a branch -> open PR -> merge after green.
- Colab: run-only by default. If a notebook/config fix must be committed
  from Colab, strip first in a cell (`!pip install -q nbstripout` once,
  then `!nbstripout notebooks/YOLOv3_car.ipynb`), commit to a branch
  (never to `main`), open a PR, and merge only when CI is green.
- Program changes originate locally; Colab changes runtime parameters only.
  Code fixed in Colab must be ported back locally with tests before commit,
  or the two sides diverge and the next pull conflicts.
- Red CI means the merge is blocked, not the experiment: read the failing
  test, fix, push again to the same branch (the PR updates itself).