# yolov3-car-detection

Car object detection practice with YOLOv3 (Kaggle `sshikamaru/car-object-detection`).

Status: Phase 1a/1b done (`tools/`, `configs/`, tests green). `notebooks/YOLOv3_car.ipynb`
calls `python -m car_tools.run --exp <yaml> --stage <stage>`; replay chain
`car_640_e12.yaml` -> `car_640_e50.yaml` -> `car_640_e50_lr002.yaml` reproduces
run1 -> run2 -> run3 from scratch. Full plan: `YOLO/REPO_BUILD_PLAN.md`
(in the course notes repo, not committed here).

## Layout

- `tools/car_tools/` - experiment tooling (AGPL-3.0, see LICENSE)
- `tests/` - CPU-only pytest suite (`python -m pytest -q`)
- `configs/` - base hyperparams + per-experiment yamls (Phase 1b)
- `notebooks/` - thin caller notebook (Phase 2)
- `scripts/` - Colab bootstrap + data fetch (Phase 1b)
- `docs/` - tuning table + honest results (Phase 2)

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
