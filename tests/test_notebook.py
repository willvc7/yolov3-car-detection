# -*- coding: utf-8 -*-
"""Guard the committed notebook: structure, markers, no shell/# joins."""

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "YOLOv3_car.ipynb"
THIN_CELLS = {10, 19, 26, 27, 34, 38, 40, 43, 47}
STAGES = {34: "train", 38: "val", 40: "sweep", 43: "detect", 47: "record"}


def _load():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 51
    return nb


def test_thin_cells_marked():
    nb = _load()
    for i in sorted(THIN_CELLS):
        src = "".join(nb["cells"][i]["source"])
        assert "[TRY3]" in src, f"cell {i} missing marker"


def test_no_shell_comment_join():
    # %/! lines must never contain '#': IPython passes the rest of the
    # line as arguments (this exact bug once produced `openpyxl#`).
    nb = _load()
    bad = [(ci, l) for ci, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
           for l in c["source"] if l.strip().startswith(("!", "%")) and "#" in l]
    assert not bad, bad


def test_thin_stages_reference_exp_yaml():
    nb = _load()
    for i, stage in STAGES.items():
        src = "".join(nb["cells"][i]["source"])
        assert f'car_tools.run --exp "$EXP_YAML" --stage {stage}' in src, i


def test_no_outputs_committed():
    # Executed notebooks live in YOLO/output (outside this repo);
    # the committed copy must carry code only (pre-commit nbstripout
    # enforces this; PNG visualizations are exempted by config).
    nb = _load()
    offenders = [i for i, c in enumerate(nb["cells"])
                 if c["cell_type"] == "code"
                 and (c.get("outputs") or c.get("execution_count") is not None)]
    assert not offenders, offenders