# -*- coding: utf-8 -*-
"""car.yaml write + read-back assertions (mirrors Cell29 checks)."""

import yaml


def test_car_yaml_roundtrip(tmp_path):
    yaml_file = tmp_path / "car.yaml"
    payload = {
        "path": "../datasets/car",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["car"],
    }
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=False, default_flow_style=False)
    with open(yaml_file, encoding="utf-8") as f:
        check = yaml.safe_load(f)
    assert check.get("nc") == 1 and check.get("names") == ["car"]
    assert check.get("train") == "images/train" and check.get("val") == "images/val"