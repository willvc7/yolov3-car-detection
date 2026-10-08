# -*- coding: utf-8 -*-
"""方案B 回歸測試: yaml 優先 / NEW_EXP 只留 env / DRIVE_ROOT 固定 / 無靜默預設.

- open_experiment() 為純函數: 不讀 env, 不寫 env, 簽名只有 (exp_tag, *, new_exp).
- NEW_EXP 由入口 (run_stage / notebook SYNC) 從 env 讀出 bool 後傳入.
- EXP_NAME/BACKUP_DIR 寫入一律經 apply_to_env() (yaml 優先 + WARN).
- DRIVE_ROOT / DRIVE_MARKER / 表格路徑固定於 exp 模組 (測試 monkeypatch 常數).
- notebook: RsukZZ 不再衍生 (無 EXP_TAG 預設), SYNC 為唯一衍生點, viz 無 "exp" 預設.
"""

import json
import os
from pathlib import Path

import pytest

from car_tools import exp as E
from car_tools import run as R

NB = Path(__file__).resolve().parents[1] / "notebooks" / "YOLOv3_car.ipynb"
SYNC_ID = "SYNC_EXP_001A"
PREFLIGHT_ID = "RsukZZ5c457R"
VIZ_ID = "RVz_yx4sBjtk"


def _point_drive_at(tmp_path, monkeypatch):
    root = tmp_path / "drv"
    marker = tmp_path / "marker"
    root.mkdir()
    marker.mkdir()
    monkeypatch.setattr(E, "DRIVE_ROOT", str(root))
    monkeypatch.setattr(E, "DRIVE_MARKER", str(marker))
    return root


def test_open_experiment_is_pure_no_env_side_effect(tmp_path, monkeypatch):
    _point_drive_at(tmp_path, monkeypatch)
    monkeypatch.setenv("EXP_NAME", "exp_HANDSET_evil")
    monkeypatch.setenv("BACKUP_DIR", "/tmp/evil")
    info = E.open_experiment("car_640_e50", new_exp=False)
    assert info["exp_name"].endswith("_car_640_e50")
    # 純函數: 呼叫前手設的 env 原封不動 (覆寫是 apply_to_env 的事)
    assert os.environ["EXP_NAME"] == "exp_HANDSET_evil"
    assert os.environ["BACKUP_DIR"] == "/tmp/evil"


def test_open_experiment_reuses_pointer(tmp_path, monkeypatch):
    root = _point_drive_at(tmp_path, monkeypatch)
    (root / ".yolo_exp_name_car_640_e50.txt").write_text("exp_OLD_car_640_e50", encoding="utf-8")
    info = E.open_experiment("car_640_e50", new_exp=False)
    assert info["exp_name"] == "exp_OLD_car_640_e50"
    assert info["created"] is False


def test_open_experiment_new_exp_bool(tmp_path, monkeypatch):
    root = _point_drive_at(tmp_path, monkeypatch)
    (root / ".yolo_exp_name_t.txt").write_text("exp_OLD_t", encoding="utf-8")
    assert E.open_experiment("t", new_exp=False)["exp_name"] == "exp_OLD_t"
    info = E.open_experiment("t", new_exp=True)
    assert info["created"] is True and info["exp_name"] != "exp_OLD_t"


def test_apply_to_env_warns_and_overwrites(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EXP_NAME", "exp_HANDSET_evil")
    monkeypatch.setenv("BACKUP_DIR", "/tmp/evil")
    E.apply_to_env("exp_NEW_x", "/drive/exp_NEW_x")
    out = capsys.readouterr().out
    assert "[WARN]" in out and "yaml" in out
    assert os.environ["EXP_NAME"] == "exp_NEW_x"
    assert os.environ["BACKUP_DIR"] == "/drive/exp_NEW_x"


def test_apply_to_env_quiet_when_consistent(monkeypatch, capsys):
    monkeypatch.setenv("EXP_NAME", "exp_A")
    monkeypatch.setenv("BACKUP_DIR", "/drive/exp_A")
    E.apply_to_env("exp_A", "/drive/exp_A")
    assert "[WARN]" not in capsys.readouterr().out


def test_run_stage_reads_new_exp_from_env(tmp_path, monkeypatch):
    import car_tools.run as RR

    seen = {}

    def fake_open(exp_tag, *, new_exp=False):
        seen["new_exp"] = new_exp
        return {"exp_name": "exp_T", "backup_dir": "/drive/exp_T",
                "name_file": "x", "created": False}

    monkeypatch.setattr(RR._exp, "open_experiment", fake_open)
    monkeypatch.setattr(RR._exp, "apply_to_env", lambda *a: None)
    monkeypatch.setattr(RR._sweep, "run_sweep", lambda *a, **k: [])
    eng = tmp_path / "eng"
    eng.mkdir()
    exp50 = str(R.REPO_ROOT / "configs" / "experiments" / "car_640_e50.yaml")
    monkeypatch.setenv("NEW_EXP", "1")
    try:
        RR.run_stage(exp50, "sweep", engine_dir=str(eng))
    finally:
        monkeypatch.delenv("NEW_EXP", raising=False)
    assert seen["new_exp"] is True


def test_no_new_exp_or_drive_root_flags():
    import argparse

    # --new-exp / --drive-root 已刪除: 傳入必須報錯 (argparse 非零退出)
    with pytest.raises(SystemExit):
        R.main(["--exp", "x.yaml", "--stage", "train", "--new-exp"])
    with pytest.raises(SystemExit):
        R.main(["--exp", "x.yaml", "--stage", "train", "--drive-root", "/tmp/x"])


def _cell_source(cid):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    by_id = {c["metadata"].get("id"): "".join(c["source"]) for c in nb["cells"]}
    return by_id[cid]


def test_notebook_preflight_does_not_derive():
    src = _cell_source(PREFLIGHT_ID)
    assert "car_640_e12" not in src, "RsukZZ 仍有 EXP_TAG 硬編碼預設"
    assert "open_experiment" not in src, "RsukZZ 應只做 Drive 檢查，不再衍生"
    assert 'get("EXP_NAME"' not in src and "BACKUP_DIR" not in src or "SYNC" in src


def test_notebook_sync_is_single_derivation():
    src = _cell_source(SYNC_ID)
    assert "new_exp=False" not in src, "SYNC 仍寫死 new_exp=False"
    assert 'get("NEW_EXP"' in src
    assert "apply_to_env" in src
    assert "car_640_e12" not in src, "SYNC 不應有 tag 預設"


def test_notebook_viz_has_no_exp_default():
    src = _cell_source(VIZ_ID)
    assert '"exp"' not in src and "'exp'" not in src, "viz 仍有靜默預設"
    assert "EXP_NAME" in src