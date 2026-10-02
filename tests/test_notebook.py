# -*- coding: utf-8 -*-
"""Guard the committed notebook: ids unique, thin cells marked, no shell/# joins.

Addressed by id (not index): the user may add/remove markdown freely.
jNDzhAxx1B0S (old carcsv2yolo def cell) was intentionally deleted after the
logic moved into car_tools.convert; its absence is locked here.
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "YOLOv3_car.ipynb"
THIN_IDS = {"TvXGaJoO7lS1", "XdbMbQ7WE5m0", "IQLX2_AW-moK",
            "yX1zHIGMBLQ_", "yNibtqbZ4fp1", "ytsRjL9NBRPe", "9O9I6TMJAxO1",
            "SYNC_EXP_001A"}
SYNC_ID = "SYNC_EXP_001A"
EXP_YAML_CELL = "TvXGaJoO7lS1"
TRAIN_CELL = "IQLX2_AW-moK"
REPORT_CELL = "qfBqK4umVAlk"
STAGE_IDS = {"IQLX2_AW-moK": "train", "yX1zHIGMBLQ_": "val",
             "yNibtqbZ4fp1": "sweep", "ytsRjL9NBRPe": "detect",
             "9O9I6TMJAxO1": "record"}
DELETED_IDS = {"jNDzhAxx1B0S"}


def _load():
    return json.loads(NB.read_text(encoding="utf-8"))


def _by_id(nb):
    return {c["metadata"].get("id"): "".join(c["source"]) for c in nb["cells"]}


def test_ids_unique_and_required_present():
    nb = _load()
    ids = [c["metadata"].get("id") for c in nb["cells"]]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    missing = THIN_IDS - set(ids)
    assert not missing, missing
    assert not (DELETED_IDS & set(ids)), "deleted def cell resurrected"


def test_thin_cells_marked():
    by_id = _by_id(_load())
    for cid in sorted(THIN_IDS):
        assert "[TRY3]" in by_id[cid], cid


def test_no_shell_comment_join():
    # %/! lines must never contain '#': IPython passes the rest of the
    # line as arguments (this exact bug once produced `openpyxl#`).
    nb = _load()
    bad = [(c["metadata"].get("id"), l) for c in nb["cells"] if c["cell_type"] == "code"
           for l in c["source"] if l.strip().startswith(("!", "%")) and "#" in l]
    assert not bad, bad


def test_thin_stages_reference_exp_yaml():
    by_id = _by_id(_load())
    for cid, stage in STAGE_IDS.items():
        assert f'car_tools.run --exp "$EXP_YAML" --stage {stage}' in by_id[cid], cid


def test_no_outputs_committed():
    # Executed notebooks live in YOLO/output (outside this repo);
    # the committed copy must carry code only (pre-commit nbstripout
    # enforces this; visuals live in docs/images/).
    nb = _load()
    offenders = [c["metadata"].get("id") for c in nb["cells"]
                 if c["cell_type"] == "code"
                 and (c.get("outputs") or c.get("execution_count") is not None)]
    assert not offenders, offenders


def _idx(nb):
    return {c["metadata"].get("id"): i for i, c in enumerate(nb["cells"])}


def test_exp_sync_cell_order():
    # all-run 保證：EXP_YAML 指派格 < 同步格 < train 格，否則 shell EXP_NAME
    # 必與 yaml 實際實驗脫鉤（e12/e50 事故）。
    pos = _idx(_load())
    assert pos[EXP_YAML_CELL] < pos[SYNC_ID] < pos[TRAIN_CELL], pos


def test_exp_name_readers_after_sync():
    # 所有讀取 $EXP_NAME 的格都必須在同步格之後（寫入格除外）。
    nb = _load()
    pos = _idx(nb)
    readers = [c["metadata"].get("id") for c in nb["cells"]
               if c["cell_type"] == "code"
               and ("$EXP_NAME" in "".join(c["source"])
                    or 'get("EXP_NAME"' in "".join(c["source"]))]
    assert readers, "no EXP_NAME readers found (guard broken?)"
    # 同步格本身讀舊值印對帳（old -> new）是合法的；早於它的讀取才算錯。
    late = [cid for cid in readers if pos[cid] < pos[SYNC_ID]]
    assert not late, late


def test_no_relative_engine_guard():
    # engine clone 檢查必須用絕對路徑，否則 cwd 殘留會巢狀複製
    # （yolov3_pytorch/yolov3_pytorch 事故）。
    nb = _load()
    bad = [c["metadata"].get("id") for c in nb["cells"]
           if "test -d yolov3_pytorch" in "".join(c.get("source", []))]
    assert not bad, bad


def test_no_hardcoded_detect_fallback():
    # viz 格不得寫死範例圖路徑；無圖時應報 EXP_NAME/目錄現況。
    by_id = _by_id(_load())
    assert "runs/detect/exp/" not in by_id["RVz_yx4sBjtk"], "stale fallback path"


def test_preamble_hidden_by_default():
    # Report 之前的前言區（Project Part 1 及其內容）預設不展開。
    nb = _load()
    pos = _idx(nb)
    pre = [c for c in nb["cells"] if pos[c["metadata"].get("id")] < pos[REPORT_CELL]]
    assert pre, "preamble empty?"
    shown = [c["metadata"].get("id") for c in pre
             if c["metadata"].get("jupyter", {}).get("source_hidden") is not True]
    assert not shown, shown
