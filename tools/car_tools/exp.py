# -*- coding: utf-8 -*-
"""Experiment naming + cloud backup directory resolution.

Precedence contract (方案B, DRIVE_ROOT 固定):
- yaml 唯一真相: EXP_YAML -> yaml[exp] -> tag -> pointer 檔 -> EXP_NAME.
- os.environ 的 EXP_NAME/BACKUP_DIR 只是快取, 由呼叫方經 apply_to_env()
  顯式寫入; 若與衍生值不一致, 印出 WARN 後以 yaml 衍生值為準.
- DRIVE_ROOT / DRIVE_MARKER / 雲端表格路徑固定於此, 不接受經由 env、
  yaml 或參數自訂 (測試以 monkeypatch 改模組常數指向 tmp 目錄).
- NEW_EXP 只留 env 通道: NEW_EXP=1 即開新實驗. 本模組不讀 env,
  由入口 (notebook cell / run.run_stage / CLI) 讀出 bool 後傳入.
"""

import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

TAIPEI = timezone(timedelta(hours=8))
DRIVE_ROOT = "/content/drive/MyDrive/YOLO_Experiments"
DRIVE_MARKER = "/content/drive/MyDrive"
CLOUD_XLSX = f"{DRIVE_ROOT}/數據紀錄表.xlsx"
LOCAL_XLSX = "數據紀錄表.xlsx"
# 舊別名: 正式流程一律用 DRIVE_ROOT, 保留僅為相容外部引用.
DEFAULT_ROOT = DRIVE_ROOT


def check_exp_name(v):
    """EXP_NAME 僅允許英數字與 _ . -（防 shell 注入與非法路徑）"""
    v = (v or "").strip()
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", v or ""), (
        f"EXP_NAME 非法：{v!r}（僅允許英數字與 _ . -，且以英數字開頭）"
    )
    return v


def safe_tag(exp_tag):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", exp_tag).strip("._") or "exp"


def pointer_file(exp_tag):
    """此 tag 的 pointer 檔路徑 (DRIVE_ROOT 固定)."""
    return Path(f"{DRIVE_ROOT}/.yolo_exp_name_{safe_tag(exp_tag)}.txt")


def open_experiment(exp_tag, *, new_exp=False):
    """Resolve (exp_name, backup_dir). 純函數: 不讀 env, 不寫 env.

    new_exp=False + pointer file exists -> reuse existing EXP_NAME.
    Otherwise create exp_<timestamp>_<tag> and (over)write the pointer.
    Overwriting never touches existing experiment backups; the old name
    stays recoverable from the backup dir listing / record workbook.
    測試以 monkeypatch 改 DRIVE_ROOT / DRIVE_MARKER 指向 tmp 目錄.
    """
    assert Path(DRIVE_MARKER).exists(), "請先執行 drive.mount('/content/drive')"
    tag = safe_tag(exp_tag)
    name_file = pointer_file(tag)
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
    backup_dir = f"{DRIVE_ROOT}/{exp_name}"
    print("BACKUP_DIR =", backup_dir)
    return {"exp_name": exp_name, "backup_dir": backup_dir,
            "name_file": str(name_file), "created": created}


def apply_to_env(exp_name, backup_dir):
    """把 yaml 衍生值寫入 os.environ. 舊手設值不一致時印 WARN (yaml 優先)."""
    old_n, old_b = os.environ.get("EXP_NAME", ""), os.environ.get("BACKUP_DIR", "")
    if old_n and old_n != exp_name:
        print(f"[WARN] os.environ EXP_NAME={old_n} 與 yaml 衍生值不一致，"
              f"已依 yaml 覆寫 -> {exp_name}（yaml 優先，手設值不生效）")
    if old_b and old_b != backup_dir:
        print(f"[WARN] os.environ BACKUP_DIR={old_b} 與 yaml 衍生值不一致，"
              f"已依 yaml 覆寫 -> {backup_dir}（yaml 優先，手設值不生效）")
    os.environ["EXP_NAME"] = exp_name
    os.environ["BACKUP_DIR"] = backup_dir
