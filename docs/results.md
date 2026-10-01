# Results (honest log)

Source of truth: `數據紀錄表.xlsx` 實驗總表 (values below copied verbatim;
failed baselines are kept, not beautified). Val = 71 images / 107 instances,
single class `car`, `val.py --iou 0.65`.

| exp | epochs | lr0 | P | R | mAP50 | mAP50-95 | detect @conf 0.25 | train time (T4) |
|---|---|---|---|---|---|---|---|---|
| `car_640_e12` (run1) | 12 | 0.1 | 0.0128 | 0.9626 | 0.4157 | 0.1122 | 71/71 no detections | ~5 min |
| `car_640_e50` (run2) | 50 | 0.1 | 0.9808 | 0.9572 | 0.988 | 0.5779 | boxes appear | 0.354 h (~21 min) |
| `car_640_e50_lr002` (run3) | 50 | 0.02 | TBD | TBD | TBD | TBD | TBD | TBD |

Config files: `configs/experiments/car_640_e12.yaml`,
`configs/experiments/car_640_e50.yaml`,
`configs/experiments/car_640_e50_lr002.yaml`
(diff between consecutive runs is exactly the single changed variable).

## Reading notes

- run1 (e12): ultra-low precision + ultra-high recall at `conf_thres=0.001`
  means the model sprayed low-confidence boxes everywhere; at `conf>=0.15`
  nothing passed, hence empty output images. Diagnosis: undertraining
  (mAP50 took off only in the last 2 epochs: 0.060 -> 0.416), not a
  data/label/pipeline bug. Fixed by epochs 12 -> 50, nothing else changed.
- run2 (e50): mAP50 saturated (~0.98-0.99 over epochs 45-50) while
  mAP50-95 still climbed (0.504 -> 0.578). Next direction (run3): lower lr0
  to tighten boxes, epochs held at 50 (single variable).
- Honesty footnote: train/val were randomly split from the same video
  frames, so val is easier than a true holdout set; 0.988 flatters.
  Qualitative check on `testing_images/` (175 unlabeled frames) recommended
  before claiming generalization.

## Sample images

`docs/images/` holds 2-4 picked contrasts (no-box e12 vs boxed e50 at
conf 0.25), each captioned with EXP_NAME + conf. (To be added after replay.)
