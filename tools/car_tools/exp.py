# -*- coding: utf-8 -*-
"""Experiment naming + cloud backup directory resolution.

Ported from YOLOv3_try2.ipynb Cell10. Statements unchanged;
EXP_TAG/DRIVE_ROOT/NEW_EXP became function parameters.
"""

import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

TAIPEI = timezone(timedelta(hours=8))
DEFAULT_ROOT = "/content/drive/MyDrive/YOLO_Experiments"


def check_exp_name(v):
    """EXP_NAME 僅允許英數字與 _ . -（防 shell 注入與非法路徑）"""
    v = (v or "").strip()
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", v or ""), (
        f"EXP_NAME 非法：{v!r}（僅允許英數字與 _ . -，且以英數字開頭）"
    )
    return v


def safe_tag(exp_tag):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", exp_tag).strip("._") or "exp"


def open_experiment(exp_tag, drive_root=DEFAULT_ROOT,
                    new_exp=False, set_env=True, drive_marker="/content/drive/MyDrive"):
    """Resolve (exp_name, backup_dir).

    new_exp=False + pointer file exists -> reuse existing EXP_NAME.
    Otherwise create exp_<timestamp>_<tag> and (over)write the pointer.
    Overwriting never touches existing experiment backups; the old name
    stays recoverable from the backup dir listing / record workbook.
    """
    assert Path(drive_marker).exists(), "請先執行 drive.mount('/content/drive')"
    tag = safe_tag(exp_tag)
    name_file = Path(f"{drive_root}/.yolo_exp_name_{tag}.txt")
    if name_file.exists() and not new_exp:
        exp_name = check_exp_name(name_file.read_text(encoding="utf-8").strip())
        print(f"沿用既有 EXP_NAME={exp_name}")
        created = False
    else:
        if new_exp and name_file.exists():
            print(f"NEW_EXP=1：將覆寫 {name_file} 的舊指標並新建 EXP_NAME（舊實驗備份不受影響）")
        timestamp = datetime.now(TAIPEI).strftime("%Y%m%d_%H%M%S")
        exp_name = check_exp_name(f"exp_{timestamp}_{tag}")
        name_file.parent.mkdir(parents=True, exist_ok=True)
        name_file.write_text(exp_name, encoding="utf-8")
        print(f"新建 EXP_NAME={exp_name}")
        created = True
    backup_dir = f"{drive_root}/{exp_name}"
    if set_env:
        os.environ["EXP_NAME"] = exp_name
        os.environ["BACKUP_DIR"] = backup_dir
    print("BACKUP_DIR =", backup_dir)
    return {"exp_name": exp_name, "backup_dir": backup_dir,
            "name_file": str(name_file), "created": created}