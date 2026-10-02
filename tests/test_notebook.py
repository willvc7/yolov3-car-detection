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
            "yX1zHIGMBLQ_", "yNibtqbZ4fp1", "ytsRjL9NBRPe", "9O9I6TMJAxO1"}
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
