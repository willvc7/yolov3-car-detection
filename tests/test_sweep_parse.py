# -*- coding: utf-8 -*-
"""all_row + detect-summary regex against real log shapes + practical val grid."""

import json
import re

from car_tools.sweep import all_row

VAL_LINE = "                   all         71        107     0.0106      0.972      0.397      0.111"
DET_HIT = "image 1/71 /content/datasets/car/images/val/vid_4_10040.jpg: 384x640 1 car, 53.0ms"
DET_MISS = "image 1/71 /content/datasets/car/images/val/vid_4_10040.jpg: 384x640 (no detections), 53.0ms"
DET_RE = re.compile(r"\d+x\d+\s+(.+),\s+[\d.]+ms")


def test_all_row_parses():
    assert all_row(VAL_LINE) == [0.0106, 0.972, 0.397, 0.111]


def test_all_row_garbage_returns_nones():
    assert all_row("no detections here") == [None] * 4


def test_detect_summary_hit():
    m = DET_RE.search(DET_HIT)
    assert m and m.group(1).strip() == "1 car"


def test_detect_summary_miss():
    m = DET_RE.search(DET_MISS)
    assert m and m.group(1).strip() == "(no detections)"


def test_val_once_passes_conf_thres_and_caches(tmp_path, monkeypatch):
    """practical 格傳 --conf-thres，檔名/meta/rows 自洽，第二次沿用快取。"""
    from car_tools import sweep as S

    calls = []

    def fake_run(cmd, timeout=3600):
        calls.append(cmd)
        assert "--conf-thres" in cmd and "0.25" in cmd
        assert "--iou" in cmd and "0.65" in cmd
        return VAL_LINE + "\n"

    monkeypatch.setattr(S, "run", fake_run)
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights-v1")
    from car_tools.sweep import weights_key
    wk = weights_key(w)
    out1 = S._run_val_once(tmp_path / "runs", str(w), "data/car.yaml",
                           640, 16, "0.65", "0.25", "exp_T",
                           "val-iou-practical", "0.65@conf0.25", wk)
    assert out1 == [0.0106, 0.972, 0.397, 0.111]
    assert (tmp_path / "runs" / "val" / "exp_T_iou065_c025" / "val.txt").exists()
    meta = json.loads((tmp_path / "runs" / "val" / "exp_T_iou065_c025" / "meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["kind"] == "val-iou-practical" and meta["param"] == "0.65@conf0.25"
    out2 = S._run_val_once(tmp_path / "runs", str(w), "data/car.yaml",
                           640, 16, "0.65", "0.25", "exp_T",
                           "val-iou-practical", "0.65@conf0.25", wk)
    assert out2 == out1 and len(calls) == 1  # 第二次命中快取，不再執行


def test_val_once_default_conf_has_no_conf_flag(tmp_path, monkeypatch):
    """傳統 val-iou 格（conf=None）不傳 --conf-thres，走引擎預設。"""
    from car_tools import sweep as S

    def fake_run(cmd, timeout=3600):
        assert "--conf-thres" not in cmd
        return VAL_LINE + "\n"

    monkeypatch.setattr(S, "run", fake_run)
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights-v1")
    from car_tools.sweep import weights_key
    out = S._run_val_once(tmp_path / "runs", str(w), "data/car.yaml",
                          640, 16, "0.65", None, "exp_T",
                          "val-iou", "0.65", weights_key(w))
    assert out == [0.0106, 0.972, 0.397, 0.111]
    assert (tmp_path / "runs" / "val" / "exp_T_iou065" / "val.txt").exists()