# Tuning map (YOLOv3 car project)

One variable per experiment. Each row: where to set it in `configs/experiments/*.yaml`,
what it does, suggested range, and whether changing it requires retraining.

## Tier 1 - TA directions (start here)

| Direction | Config key | Effect | Suggested range | Retrain? |
|---|---|---|---|---|
| 學習率 | `hyp_override.lr0` | step size; too big oscillates, too small stalls | 0.01–0.02 (base 0.1 too big after warmup) | yes |
| 終點倍率 | `hyp_override.lrf` | final lr = lr0 × lrf | 0.01–0.1 | yes |
| 匹配閾值 | `hyp_override.iou_t` | GT-anchor match strictness; too high starves positives | 0.15–0.25 | yes |
| 訓練輪數 | `train.epochs` | more epochs until mAP plateaus | 12 → 50 → 80 | yes |
| 信心閾值 | `detect.conf` + `sweep.conf` | higher = fewer, surer boxes (P up, R down) | 0.15 / 0.25 / 0.4 | no (sweep only) |
| NMS 閾值 | `val.iou` + `sweep.iou` | overlap boxes kept vs merged | 0.5 / 0.6 / 0.65 | no (sweep only) |

## Tier 2 - common (add when Tier 1 saturates)

| Key | Effect | Suggested range | Retrain? |
|---|---|---|---|
| `train.img` | input resolution; bigger helps small cars, costs VRAM | 640 → 768 (1024 risks OOM on T4) | yes |
| `train.batch` | batch size; smaller if OOM | 16 → 8 | yes |
| `train.optimizer` | SGD / Adam / AdamW | SGD default; Adam converges faster, generalizes worse | yes |
| `train.patience` | early-stopping patience (epochs without improvement) | 50–100 | yes |
| `train.workers` | dataloader workers | 8 (Colab) | no effect on metrics |
| `hyp_override.momentum` | SGD momentum | 0.9–0.937 | yes |
| `hyp_override.warmup_epochs` | warmup length (obj loss spike here is normal) | 3 | yes |
| `hyp_override.box/cls/obj` | loss weights: localization / classification / objectness | obj 0.7→1.0 if recall low; cls 0.3→0.5 if class confusion | yes |
| `hyp_override.degrees` | rotation aug; street-view cars tolerate ~10 | 0 → 10 | yes |
| `hyp_override.mosaic` | mosaic aug strength; lower if small objects suffer | 1.0 → 0.5 | yes |
| `hyp_override.scale` | scale jitter | 0.9 | yes |

## Tier 3 - advanced (documented, default untouched)

| Key | Notes |
|---|---|
| `models/yolov3.yaml` anchors | leave to AutoAnchor (BPR=1.000 on car data, verified on the 12-epoch baseline run) |
| `train.evolve` | hyperparameter evolution; expensive, last resort |
| `train.single-cls/rect/multi-scale/image-weights/freeze/label-smoothing` | niche; change only with a hypothesis |
| `train.cos-lr` | cosine schedule alternative to linear `lrf` decay |

## Extension rule

New key = one row here + one allowlist entry in `tools/car_tools/run.py` +
one test in `tests/test_run_cmd.py`. Unknown keys fail fast with suggestions.

## Smart re-run (--skip-done)

`python -m car_tools.run --exp <yaml> --stage all --skip-done` skips stages
whose products are up to date (prints `SKIP <stage>: <reason>` per stage):

- train: `best.pt` + `results.csv` epochs == yaml + snapshot == current yaml
- val: `val.txt` + `val.meta.json` (weights fingerprint / iou / config)
- sweep: per-item cache (always skip-aware, no flag needed)
- detect: images + `meta.json` (conf / weights fingerprint / config)
- convert (notebook cell): `manifest.json` (csv fingerprint + seed + tool version)
- record: always runs (seconds, idempotent)

Rules: any yaml word change invalidates dependents (fail-open to RUN, never
stale-SKIP); single `--stage X` without the flag forces that stage.
