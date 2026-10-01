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
