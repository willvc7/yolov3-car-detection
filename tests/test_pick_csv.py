# -*- coding: utf-8 -*-
"""pick_train_csv: submission template must lose, training CSV must win."""

from car_tools.convert import pick_train_csv

TRAIN_ROWS = """image,xmin,ymin,xmax,ymax
vid_4_1000.jpg,281.25,187.03,327.72,223.22
vid_4_10000.jpg,15.16,187.03,120.32,236.43
"""
SAMPLE_ROWS = """image,bounds
vid_5_26560.jpg,"0.0 0.0 1.0 1.0"
"""


def _layout(tmp_path):
    raw = tmp_path / "car-raw" / "data"
    (raw / "training_images").mkdir(parents=True)
    (raw / "training_images" / "vid_4_1000.jpg").write_bytes(b"fake")
    (raw / "sample_submission.csv").write_text(SAMPLE_ROWS, encoding="utf-8")
    # space + parens in name, like the real Kaggle duplicate download
    (raw / "train_solution_bounding_boxes (1).csv").write_text(TRAIN_ROWS, encoding="utf-8")
    return raw


def test_picks_training_csv_over_submission(tmp_path):
    raw = _layout(tmp_path)
    picked = pick_train_csv(raw)
    assert picked.name == "train_solution_bounding_boxes (1).csv"


def test_no_box_columns_raises(tmp_path):
    raw = tmp_path / "car-raw"
    raw.mkdir()
    (raw / "sample_submission.csv").write_text(SAMPLE_ROWS, encoding="utf-8")
    try:
        pick_train_csv(raw)
    except AssertionError as e:
        assert "xmin" in str(e)
    else:
        raise AssertionError("expected AssertionError for submission-only dir")


def test_no_csv_raises(tmp_path):
    raw = tmp_path / "empty"
    raw.mkdir()
    try:
        pick_train_csv(raw)
    except AssertionError as e:
        assert "找不到 csv" in str(e)
    else:
        raise AssertionError("expected AssertionError for csv-less dir")