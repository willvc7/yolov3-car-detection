# -*- coding: utf-8 -*-
"""run.py: schema validation, hyp merge, command generation, dry-run equivalence."""

import os
import re
import sys
import tempfile

import pytest
import yaml

from car_tools import run

REPO = run.REPO_ROOT
EXP50 = REPO / "configs" / "experiments" / "car_640_e50.yaml"

# run2 notebook line 1086 (Colab). $EXP_NAME varies -> normalized before compare.
EXPECTED_TRAIN = (
    "python train.py --img 640 --batch 16 --epochs 50 --weights  --cfg models/yolov3.yaml "
    "--hyp runs/EXP/hyp.used.yaml --data data/car.yaml --cache "
    '--project runs/train --name EXP --exist-ok'
)


def _norm(cmd):
    s = " ".join(cmd)
    s = re.sub(r"runs/exp_[A-Za-z0-9_.-]+", "runs/EXP", s)
    s = re.sub(r"--name \S+", "--name EXP", s)
    s = s.replace("--weights ''", "--weights ").replace("--weights  ", "--weights ")
    return re.sub(r"\s+", " ", s).strip()


def test_e50_dryrun_matches_run2_log():
    cfg = run.load_experiment(EXP50)
    fake_name = "exp_20261001_140142_car_640_e50"
    got = _norm(run.build_train_cmd(cfg, fake_name))
    want = _norm(EXPECTED_TRAIN.split())
    assert got == want, f"\n got: {got}\nwant: {want}"


def test_unknown_top_key_suggests():
    with open(EXP50, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["epocs"] = 50
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
        path = f.name
    try:
        with pytest.raises(KeyError, match="epocs"):
            run.load_experiment(path)
    finally:
        os.unlink(path)


def test_unknown_hyp_key_rejected(tmp_path):
    base = REPO / "configs" / "hyps" / "hyp.scratch-high.yaml"
    cfg = {"hyp_base": "hyps/hyp.scratch-high.yaml", "hyp_override": {"lr_0": 0.02}}
    with pytest.raises(KeyError, match="lr_0"):
        run.build_hyp_used(cfg, repo_root=REPO)


def test_hyp_merge_e12_e50_equal_base_lr002_differs_only_lr0():
    base = yaml.safe_load((REPO / "configs" / "hyps" / "hyp.scratch-high.yaml").read_text(encoding="utf-8"))
    for name in ("car_640_e12", "car_640_e50"):
        cfg = run.load_experiment(REPO / "configs" / "experiments" / f"{name}.yaml")
        assert run.build_hyp_used(cfg, repo_root=REPO) == base
    cfg = run.load_experiment(REPO / "configs" / "experiments" / "car_640_e50_lr002.yaml")
    merged = run.build_hyp_used(cfg, repo_root=REPO)
    assert merged["lr0"] == 0.02
    assert {k: v for k, v in merged.items() if k != "lr0"} == {k: v for k, v in base.items() if k != "lr0"}


def test_val_detect_cmd_shapes():
    cfg = run.load_experiment(EXP50)
    name = "exp_X_car_640_e50"
    val = run.build_val_cmd(cfg, name)
    assert val[:8] == ["python", "val.py", "--img", "640", "--batch-size", "16",
                       "--weights", f"runs/train/{name}/weights/best.pt"]
    assert "--data" in val and "data/car.yaml" in val
    assert "--iou" in val and "0.65" in val
    det = run.build_detect_cmd(cfg, name)
    assert det[:4] == ["python", "detect.py", "--weights", f"runs/train/{name}/weights/best.pt"]
    assert "--source" in det and "../datasets/car/images/val" in det
    assert "--conf" in det and "0.25" in det


def test_run_capture_tees_and_returns(tmp_path):
    log = tmp_path / "val" / "exp_x" / "val.txt"
    text = run._run_capture([sys.executable, "-c", "print('hello-val')"], log)
    assert "hello-val" in text
    assert log.read_text(encoding="utf-8").strip() == "hello-val"


def test_run_capture_raises_on_failure(tmp_path):
    with pytest.raises(RuntimeError, match="returncode="):
        run._run_capture([sys.executable, "-c", "raise SystemExit(3)"],
                         tmp_path / "val.txt")


def test_run_stage_sweep_wiring(tmp_path, monkeypatch):
    import car_tools.run as R
    captured = {}

    def fake_open(*a, **k):
        return {"exp_name": "exp_T", "backup_dir": "/drive/exp_T",
                "name_file": "x", "created": False}

    def fake_sweep(exp, weights, **k):
        captured.update(k)
        captured["exp"] = exp
        captured["weights"] = weights
        return []

    monkeypatch.setattr(R._exp, "open_experiment", fake_open)
    monkeypatch.setattr(R._sweep, "run_sweep", fake_sweep)
    monkeypatch.chdir(tmp_path)
    eng = tmp_path / "eng"
    eng.mkdir()
    out = R.run_stage(str(REPO / "configs" / "experiments" / "car_640_e50.yaml"),
                      "sweep", engine_dir=str(eng))
    assert out == {"exp": "exp_T"}
    assert captured["exp"] == "exp_T"
    assert captured["weights"] == "runs/train/exp_T/weights/best.pt"
    assert captured["backup_dir"] == "/drive/exp_T"
    assert captured["data"] == "data/car.yaml"
    assert captured["iou_list"] == ("0.5", "0.6", "0.65")
    assert captured["conf_list"] == ("0.15", "0.25", "0.4")
    assert (eng / "runs" / "exp_T" / "hyp.used.yaml").exists()
    assert (eng / "runs" / "exp_T" / "hyp_source.txt").read_text(encoding="utf-8") == "runs/exp_T/hyp.used.yaml"
    assert (eng / "runs" / "exp_T" / "exp.snapshot.yaml").exists()


def test_e12_e50_configs_differ_only_in_exp_and_epochs():
    a = yaml.safe_load((REPO / "configs" / "experiments" / "car_640_e12.yaml").read_text(encoding="utf-8"))
    b = yaml.safe_load((REPO / "configs" / "experiments" / "car_640_e50.yaml").read_text(encoding="utf-8"))
    assert a["exp"] == "car_640_e12" and b["exp"] == "car_640_e50"
    assert a["train"]["epochs"] == 12 and b["train"]["epochs"] == 50
    aa = dict(a)
    bb = dict(b)
    aa.pop("exp")
    bb.pop("exp")
    aa["train"] = {k: v for k, v in aa["train"].items() if k != "epochs"}
    bb["train"] = {k: v for k, v in bb["train"].items() if k != "epochs"}
    assert aa == bb