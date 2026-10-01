# -*- coding: utf-8 -*-
"""all_row + detect-summary regex against real log shapes."""

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