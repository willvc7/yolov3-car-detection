# -*- coding: utf-8 -*-
"""skip_done predicates: one hit + misses per stage (tmp dirs, no engine/GPU)."""

import json

from car_tools import run

REPO = run.REPO_ROOT
EXP50 = REPO / "configs" / "experiments" / "car_640_e50.yaml"
EXP = "exp_T"


def _cfg():
    cfg = run.load_experiment(EXP50)
    cfg["_cfg_path"] = str(EXP50)
    return cfg


def _mk_weights(path, content=b"weights-v1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _mk_train_done(eng, epochs=50):
    base = eng / "runs" / "train" / EXP
    _mk_weights(base / "weights" / "best.pt")
    lines = ["epoch,metrics/mAP_0.5"] + ["0,0.9"] * epochs
    (base / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (base / "exp.snapshot.yaml").write_bytes(EXP50.read_bytes())


def test_config_hash_stable_and_sensitive():
    a = run._cfg_hash(_cfg(), ("train", "hyp_override"))
    b = run._cfg_hash(_cfg(), ("train", "hyp_override"))
    assert a == b and len(a) == 16
    cfg = _cfg()
    cfg["train"] = dict(cfg["train"])
    cfg["train"]["epochs"] = 80
    assert run._cfg_hash(cfg, ("train", "hyp_override")) != a


def test_train_hit(tmp_path):
    _mk_train_done(tmp_path, 50)
    ok, why = run.train_up_to_date(_cfg(), EXP, tmp_path / "runs", EXP50)
    assert ok, why
    assert "50/50" in why and "snapshot match" in why


def test_train_miss_no_best(tmp_path):
    ok, why = run.train_up_to_date(_cfg(), EXP, tmp_path / "runs", EXP50)
    assert not ok and "best.pt" in why


def test_train_miss_short_results(tmp_path):
    _mk_train_done(tmp_path, 30)  # crashed at epoch 30
    ok, why = run.train_up_to_date(_cfg(), EXP, tmp_path / "runs", EXP50)
    assert not ok and "30" in why and "50" in why


def test_train_miss_config_changed(tmp_path):
    _mk_train_done(tmp_path, 50)
    (tmp_path / "runs" / "train" / EXP / "exp.snapshot.yaml").write_text("exp: other\n", encoding="utf-8")
    ok, why = run.train_up_to_date(_cfg(), EXP, tmp_path / "runs", EXP50)
    assert not ok and "config changed" in why


def _mk_val_done(eng, weights_content=b"weights-v1"):
    w = _mk_weights(eng / "runs" / "train" / EXP / "weights" / "best.pt", weights_content)
    base = eng / "runs" / "val" / EXP
    base.mkdir(parents=True, exist_ok=True)
    (base / "val.txt").write_text("all 71 107 0.98 0.95 0.98 0.57\n", encoding="utf-8")
    from car_tools import sweep as _sw
    (base / "val.meta.json").write_text(json.dumps({
        "weights_key": _sw.weights_key(w), "iou": "0.65",
        "config_hash": run._cfg_hash(_cfg(), ("val", "data")),
    }), encoding="utf-8")
    return w


def _weights(eng):
    return eng / "runs" / "train" / EXP / "weights" / "best.pt"


def test_val_hit(tmp_path):
    _mk_val_done(tmp_path)
    ok, why = run.val_up_to_date(_cfg(), EXP, tmp_path / "runs", _weights(tmp_path))
    assert ok, why


def test_val_miss_no_meta(tmp_path):
    base = tmp_path / "runs" / "val" / EXP
    base.mkdir(parents=True)
    _mk_weights(tmp_path / "runs" / "train" / EXP / "weights" / "best.pt")
    (base / "val.txt").write_text("x\n", encoding="utf-8")
    ok, why = run.val_up_to_date(_cfg(), EXP, tmp_path / "runs", _weights(tmp_path))
    assert not ok and "meta" in why


def test_val_miss_weights_retrained(tmp_path):
    _mk_val_done(tmp_path)
    _mk_weights(_weights(tmp_path), b"weights-v2-retrained-longer-bytes")
    ok, why = run.val_up_to_date(_cfg(), EXP, tmp_path / "runs", _weights(tmp_path))
    assert not ok and "weights changed" in why


def test_val_miss_iou_changed(tmp_path):
    _mk_val_done(tmp_path)
    cfg = _cfg()
    cfg["val"] = dict(cfg["val"])
    cfg["val"]["iou"] = 0.5
    ok, why = run.val_up_to_date(cfg, EXP, tmp_path / "runs", _weights(tmp_path))
    assert not ok and "iou changed" in why


def _mk_detect_done(eng, weights_content=b"weights-v1"):
    w = _mk_weights(eng / "runs" / "train" / EXP / "weights" / "best.pt", weights_content)
    base = eng / "runs" / "detect" / EXP
    base.mkdir(parents=True, exist_ok=True)
    (base / "a.jpg").write_bytes(b"fake-jpg")
    from car_tools import sweep as _sw
    (base / "meta.json").write_text(json.dumps({
        "weights_key": _sw.weights_key(w), "conf": "0.25",
        "config_hash": run._cfg_hash(_cfg(), ("detect", "data")),
    }), encoding="utf-8")
    return w


def _dweights(eng):
    return eng / "runs" / "train" / EXP / "weights" / "best.pt"


def test_detect_hit(tmp_path):
    _mk_detect_done(tmp_path)
    ok, why = run.detect_up_to_date(_cfg(), EXP, tmp_path / "runs", _dweights(tmp_path))
    assert ok and "1 images" in why


def test_detect_miss_conf_changed(tmp_path):
    _mk_detect_done(tmp_path)
    cfg = _cfg()
    cfg["detect"] = dict(cfg["detect"])
    cfg["detect"]["conf"] = 0.4
    ok, why = run.detect_up_to_date(cfg, EXP, tmp_path / "runs", _dweights(tmp_path))
    assert not ok and "conf changed" in why


def test_detect_miss_images_deleted(tmp_path):
    _mk_detect_done(tmp_path)
    for p in (tmp_path / "runs" / "detect" / EXP).glob("*.jpg"):
        p.unlink()
    ok, why = run.detect_up_to_date(_cfg(), EXP, tmp_path / "runs", _dweights(tmp_path))
    assert not ok and "no detect images" in why


def test_manifest_hit_skips_conversion(tmp_path):
    from car_tools import convert as C
    raw = tmp_path / "car-raw" / "data"
    (raw / "training_images").mkdir(parents=True)
    (raw / "training_images" / "a.jpg").write_bytes(b"x")
    csv = raw / "train_solution_bounding_boxes (1).csv"
    csv.write_text("image,xmin,ymin,xmax,ymax\na.jpg,10,10,50,50\n", encoding="utf-8")
    yolo = tmp_path / "car"
    (yolo / "images" / "train").mkdir(parents=True)
    st = csv.stat()
    import json as J
    (yolo / "manifest.json").write_text(J.dumps({
        "csv": {"name": csv.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns},
        "seed": 42, "tool": C.TOOL_VERSION,
        "train": [1, 1, 0, 0], "val": [0, 0, 0, 0],
    }), encoding="utf-8")
    out = C.split_and_convert(raw.parent, yolo, seed=42, skip_done=True,
                              check_dir=str(tmp_path / "chk"))
    assert out == {"train": (1, 1, 0, 0), "val": (0, 0, 0, 0)}


def test_manifest_miss_on_csv_change(tmp_path):
    from car_tools import convert as C
    yolo = tmp_path / "car"
    yolo.mkdir()
    import json as J
    (yolo / "manifest.json").write_text(J.dumps({
        "csv": {"name": "other.csv", "size": 1, "mtime_ns": 2},
        "seed": 42, "tool": C.TOOL_VERSION,
        "train": [1, 1, 0, 0], "val": [0, 0, 0, 0],
    }), encoding="utf-8")
    csv = tmp_path / "train.csv"
    csv.write_text("image,xmin,ymin,xmax,ymax\na.jpg,1,1,2,2\n", encoding="utf-8")
    ok, why, _ = C.manifest_up_to_date(yolo, csv, 42)
    assert not ok and "CSV" in why
