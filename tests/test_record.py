# -*- coding: utf-8 -*-
"""record schema parity + train-time column behavior (tmp workbooks, no Drive)."""

import json
from pathlib import Path

from car_tools import record as R

NODRIVE = "D:\\__no_drive__"


def _wb(exp, tmp_path, **kw):
    runs = Path(tmp_path) / "runs"
    return R.update_workbook(exp, runs_root=str(runs),
                             cloud_xlsx=str(runs / "cloud.xlsx"),
                             local_xlsx=str(runs / "rec.xlsx"),
                             drive_marker=NODRIVE, **kw)


def test_total_h_g_parity():
    assert len(R.TOTAL_H) == len(R.TOTAL_G) == 37, (len(R.TOTAL_H), len(R.TOTAL_G))
    assert R.TOTAL_H.index("訓練耗時(h)") == R.TOTAL_H.index("epochs") + 1
    # group spans must tile exactly: recompute run-lengths of TOTAL_G
    runs, start = [], 0
    for i in range(1, len(R.TOTAL_G) + 1):
        if i == len(R.TOTAL_G) or R.TOTAL_G[i] != R.TOTAL_G[start]:
            runs.append((R.TOTAL_G[start], start, i - 1))
            start = i
    assert sum(e - s + 1 for _, s, e in runs) == len(R.TOTAL_H)
    by_group = {g: [R.TOTAL_H[c] for c in range(s, e + 1)] for g, s, e in runs}
    assert by_group["訓練設定"] == ["epochs", "訓練耗時(h)", "batch", "imgsz"]
    assert by_group["驗證推論條件"] == ["驗證iou", "推論conf"]
    assert len(R.CLS_H) == len(R.CLS_G) == 8


def test_train_time_rounding(tmp_path):
    runs = tmp_path / "runs"
    (runs / "train" / "exp_T").mkdir(parents=True)
    (runs / "train" / "exp_T" / "train_time.json").write_text(
        json.dumps({"seconds": 1274.4}), encoding="utf-8")
    row = _wb("exp_T", tmp_path)
    assert row["訓練耗時(h)"] == 0.354


def test_train_time_missing_blank(tmp_path):
    row = _wb("exp_T", tmp_path)
    assert row.get("訓練耗時(h)") is None


def test_train_time_manual_preserved(tmp_path):
    runs = tmp_path / "runs"
    (runs / "train" / "exp_T").mkdir(parents=True)
    (runs / "train" / "exp_T" / "train_time.json").write_text(
        json.dumps({"seconds": 7200}), encoding="utf-8")
    _wb("exp_T", tmp_path)  # writes 2.0
    (runs / "train" / "exp_T" / "train_time.json").unlink()  # json gone
    row = _wb("exp_T", tmp_path)  # must keep 2.0, not blank it
    assert row["訓練耗時(h)"] == 2.0


def test_train_time_new_value_overwrites(tmp_path):
    runs = tmp_path / "runs"
    (runs / "train" / "exp_T").mkdir(parents=True)
    (runs / "train" / "exp_T" / "train_time.json").write_text(
        json.dumps({"seconds": 7200}), encoding="utf-8")
    _wb("exp_T", tmp_path)
    (runs / "train" / "exp_T" / "train_time.json").write_text(
        json.dumps({"seconds": 3600}), encoding="utf-8")
    row = _wb("exp_T", tmp_path)
    assert row["訓練耗時(h)"] == 1.0
