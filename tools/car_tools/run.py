# -*- coding: utf-8 -*-
"""Single-entry experiment runner: experiment yaml -> train/val/sweep/detect/record.

Path contract (fixed, see docs/tuning.md):
- <repo>/configs/experiments/<exp>.yaml is the only hand-edited file.
- hyp_base is resolved under <repo>/configs/.
- All engine-relative paths (data/*.yaml, runs/...) assume commands
  execute with cwd == engine_dir; run.py chdirs there before delegating.
- Base hyp file is read-only; overrides land in runs/<EXP_NAME>/hyp.used.yaml.
- Precedence (方案B, DRIVE_ROOT 固定): EXP_YAML -> yaml[exp] -> pointer
  檔 -> EXP_NAME/BACKUP_DIR. 父 kernel 手設的 EXP_NAME/BACKUP_DIR 由
  exp.apply_to_env() 顯式覆寫 (WARN + yaml 優先).
- NEW_EXP 只留 env 通道: NEW_EXP=1 即開新實驗. 讀取只發生在 run_stage /
  CLI 入口, 不在 library 深處 (--new-exp / --drive-root 已刪除,
  傳入即 argparse 報錯).
"""

import argparse
import difflib
import hashlib
import os
import subprocess
from pathlib import Path

import yaml

from . import exp as _exp
from . import record as _record
from . import sweep as _sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

TOP_KEYS = {"exp", "data", "hyp_base", "hyp_override", "seed",
            "train", "val", "sweep", "detect"}

# (yaml_key, cli_flag, order) per section. kind: value | switch.
TRAIN_SPEC = [
    ("img", "--img", "value"), ("batch", "--batch", "value"),
    ("epochs", "--epochs", "value"), ("weights", "--weights", "value"),
    ("cfg", "--cfg", "value"),
    # hyp/data are computed (used-file / data/<name>.yaml), never taken raw.
    ("cache", "--cache", "cache"), ("project", "--project", "value"),
    ("name", "--name", "value"), ("exist-ok", "--exist-ok", "switch"),
    ("optimizer", "--optimizer", "value"), ("patience", "--patience", "value"),
    ("workers", "--workers", "value"),
]
TRAIN_EXTRA_ALLOW = [
    "rect", "resume", "nosave", "noval", "noautoanchor", "noplots",
    "evolve", "bucket", "image-weights", "device", "multi-scale",
    "single-cls", "sync-bn", "quad", "cos-lr", "label-smoothing",
    "save-period", "seed", "lrf", "momentum",
]
VAL_SPEC = [
    ("img", "--img", "value"), ("batch-size", "--batch-size", "value"),
    ("weights", "--weights", "value"), ("iou", "--iou", "value"),
    ("project", "--project", "value"), ("name", "--name", "value"),
    ("exist-ok", "--exist-ok", "switch"),
]
VAL_EXTRA_ALLOW = [
    "conf-thres", "task", "device", "workers", "single-cls", "augment",
    "save-txt", "save-hybrid", "save-conf", "save-json", "half", "dnn", "max-det",
]
DETECT_SPEC = [
    ("weights", "--weights", "value"), ("source", "--source", "value"),
    ("img", "--img", "value"), ("conf", "--conf", "value"),
    ("project", "--project", "value"), ("name", "--name", "value"),
    ("exist-ok", "--exist-ok", "switch"),
]
DETECT_EXTRA_ALLOW = [
    "iou-thres", "max-det", "device", "line-thickness", "hide-labels",
    "hide-conf", "classes", "agnostic-nms", "augment", "save-txt",
    "save-conf", "save-crop", "nosave", "view-img", "visualize", "update",
    "vid-stride", "half", "dnn",
]
SWEEP_KEYS = {"iou", "conf", "val_conf"}


def _suggest(bad, allowed):
    near = difflib.get_close_matches(bad, sorted(allowed), n=3, cutoff=0.6)
    hint = f"（你是指 {near}？）" if near else f"（合法 key：{sorted(allowed)}）"
    return hint


def load_experiment(path):
    """Load + validate. Unknown keys raise with suggestions (fail fast)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - TOP_KEYS
    if unknown:
        bad = sorted(unknown)[0]
        raise KeyError(f"未知頂層 key：{bad} {_suggest(bad, TOP_KEYS)}")
    for req in ("exp", "data", "hyp_base", "train"):
        if req not in cfg:
            raise KeyError(f"缺少必要 key：{req}")
    _check_section("train", cfg.get("train") or {},
                   {k for k, _, _ in TRAIN_SPEC} | set(TRAIN_EXTRA_ALLOW))
    _check_section("val", cfg.get("val") or {},
                   {k for k, _, _ in VAL_SPEC} | set(VAL_EXTRA_ALLOW))
    _check_section("detect", cfg.get("detect") or {},
                   {k for k, _, _ in DETECT_SPEC} | set(DETECT_EXTRA_ALLOW))
    sw = cfg.get("sweep") or {}
    unknown_sw = set(sw) - SWEEP_KEYS
    if unknown_sw:
        bad = sorted(unknown_sw)[0]
        raise KeyError(f"未知 sweep key：{bad} {_suggest(bad, SWEEP_KEYS)}")
    for k in ("hyp_override",):
        if cfg.get(k) is not None and not isinstance(cfg[k], dict):
            raise TypeError(f"{k} 必須是 mapping")
    return cfg


def _check_section(name, section, allowed):
    if not isinstance(section, dict):
        raise TypeError(f"{name} 必須是 mapping")
    unknown = set(section) - allowed
    if unknown:
        bad = sorted(unknown)[0]
        raise KeyError(f"未知 {name} key：{bad} {_suggest(bad, allowed)}")


def build_hyp_used(cfg, repo_root=REPO_ROOT):
    """Merge base + override. Base file is never modified."""
    base_path = Path(repo_root) / "configs" / cfg["hyp_base"]
    assert base_path.exists(), f"找不到 hyp base：{base_path}"
    with open(base_path, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    override = cfg.get("hyp_override") or {}
    unknown = set(override) - set(base)
    if unknown:
        bad = sorted(unknown)[0]
        raise KeyError(f"未知 hyp key：{bad} {_suggest(bad, set(base))}")
    merged = dict(base)
    merged.update(override)
    return merged


def _emit(out, flag, kind, value):
    if kind == "switch":
        if value:
            out.append(flag)
    elif kind == "cache":
        if value is True or (isinstance(value, str) and value == "ram"):
            out.append(flag)  # bare --cache == ram (matches notebook CLI)
        elif value:
            out.extend([flag, str(value)])
    else:
        out.extend([flag, str(value)])


def build_train_cmd(cfg, exp_name):
    t = dict(cfg.get("train") or {})
    hyp_rel = f"runs/{exp_name}/hyp.used.yaml"
    data_ref = f"data/{cfg['data']}.yaml"
    cmd = ["python", "train.py"]
    # fixed order mirrors the notebook CLI (dry-run equivalence baseline)
    order = ["img", "batch", "epochs", "weights", "cfg"]
    for key, flag, kind in TRAIN_SPEC:
        if key in ("img", "batch", "epochs", "weights", "cfg") and key in t:
            _emit(cmd, flag, kind, t.pop(key))
    cmd.extend(["--hyp", hyp_rel, "--data", data_ref])
    rest = {k: v for k, v in t.items()}
    params = {"project": "runs/train", "name": exp_name, "exist-ok": True}
    params.update(rest)
    for key, flag, kind in TRAIN_SPEC:
        if key in params and key not in order:
            _emit(cmd, flag, kind, params.pop(key))
    for key in TRAIN_EXTRA_ALLOW:
        if key in params:
            v = params.pop(key)
            if v is True:
                cmd.append("--" + key)
            elif v not in (False, None):
                cmd.extend(["--" + key, str(v)])
    return cmd


def build_val_cmd(cfg, exp_name, weights=None):
    v = dict(cfg.get("val") or {})
    w = weights or f"runs/train/{exp_name}/weights/best.pt"
    data_ref = f"data/{cfg['data']}.yaml"
    cmd = ["python", "val.py"]
    for key, flag, kind in VAL_SPEC:
        if key in ("img", "batch-size") and key in v:
            _emit(cmd, flag, kind, v.pop(key))
    cmd.extend(["--weights", w, "--data", data_ref])
    params = {"project": "runs/val", "name": exp_name, "exist-ok": True}
    params.update(v)
    for key, flag, kind in VAL_SPEC:
        if key in params and key not in ("img", "batch-size"):
            _emit(cmd, flag, kind, params.pop(key))
    for key in VAL_EXTRA_ALLOW:
        if key in params:
            vv = params.pop(key)
            if vv is True:
                cmd.append("--" + key)
            elif vv not in (False, None):
                cmd.extend(["--" + key, str(vv)])
    return cmd


def build_detect_cmd(cfg, exp_name, weights=None):
    d = dict(cfg.get("detect") or {})
    w = weights or f"runs/train/{exp_name}/weights/best.pt"
    data_ref = f"data/{cfg['data']}.yaml"
    cmd = ["python", "detect.py", "--weights", w]
    for key, flag, kind in DETECT_SPEC:
        if key in ("source", "img", "conf") and key in d:
            _emit(cmd, flag, kind, d.pop(key))
    # 類別名由 data yaml 來；不傳則引擎預設 coco128.yaml（舊錯：全程用錯類別名）
    cmd.extend(["--data", data_ref])
    params = {"project": "runs/detect", "name": exp_name, "exist-ok": True}
    params.update(d)
    for key, flag, kind in DETECT_SPEC:
        if key in params and key not in ("source", "img", "conf"):
            _emit(cmd, flag, kind, params.pop(key))
    for key in DETECT_EXTRA_ALLOW:
        if key in params:
            vv = params.pop(key)
            if vv is True:
                cmd.append("--" + key)
            elif vv not in (False, None):
                cmd.extend(["--" + key, str(vv)])
    return cmd


def _run_capture(cmd, log_path):
    """Run cmd, tee combined output to log_path (mirrors `| tee`), print it."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = (p.stdout or "") + ("\n" if p.stdout and p.stderr else "") + (p.stderr or "")
    log_path.write_text(text, encoding="utf-8", errors="ignore")
    print(text)
    if p.returncode != 0:
        raise RuntimeError(
            f"指令失敗 (returncode={p.returncode})：{' '.join(cmd)}\n"
            f"--- 輸出尾部 ---\n{text[-3000:]}"
        )
    return text


def _cfg_hash(cfg, keys):
    """配置哈希（只看給定節）：任一字改動即變化；順序無關。"""
    sub = {k: cfg.get(k) for k in keys}
    blob = yaml.dump(sub, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _results_epochs(results_csv):
    with open(results_csv, encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f) - 1  # 扣表頭


def train_up_to_date(cfg, exp_name, runs_dir, cfg_path):
    """(ok, reason)。best.pt＋results.csv 輪數達標＋快照與現 yaml 逐字一致才算完成。"""
    base = Path(runs_dir) / "train" / exp_name
    if not (base / "weights" / "best.pt").exists():
        return False, "missing best.pt"
    try:
        if (base / "exp.snapshot.yaml").read_bytes() != Path(cfg_path).read_bytes():
            return False, "config changed since last train"
    except OSError:
        return False, "no comparable snapshot"
    want = int((cfg.get("train") or {}).get("epochs", 0))
    res = base / "results.csv"
    if not res.exists():
        return False, "missing results.csv"
    rows = _results_epochs(res)
    if rows != want:
        return False, f"results.csv has {rows} epochs, want {want} (crashed or extended?)"
    return True, f"best.pt exists + results.csv {rows}/{want} + snapshot match"


def val_up_to_date(cfg, exp_name, runs_dir, weights_path):
    """(ok, reason)。val.txt＋val.meta.json（權重指紋/iou/配置）全吻合才算完成。"""
    import json
    base = Path(runs_dir) / "val" / exp_name
    if not (base / "val.txt").exists():
        return False, "missing val.txt"
    try:
        m = json.loads((base / "val.meta.json").read_text(encoding="utf-8"))
    except Exception:
        return False, "no val.meta.json"
    want_iou = str((cfg.get("val") or {}).get("iou", "0.65"))
    if m.get("weights_key") != _sweep.weights_key(weights_path):
        return False, "weights changed since last val"
    if str(m.get("iou")) != want_iou:
        return False, f"iou changed ({m.get('iou')}->{want_iou})"
    if m.get("config_hash") != _cfg_hash(cfg, ("val", "data")):
        return False, "val config changed"
    return True, "val.txt exists + weights/iou/config match"


def detect_up_to_date(cfg, exp_name, runs_dir, weights_path):
    """(ok, reason)。jpg＋meta.json（conf/權重指紋/配置）全吻合才算完成。"""
    import json
    base = Path(runs_dir) / "detect" / exp_name
    jpgs = sorted(base.glob("*.jpg")) if base.exists() else []
    if not jpgs:
        return False, "no detect images"
    try:
        m = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return False, "no detect meta.json"
    want_conf = str((cfg.get("detect") or {}).get("conf", "0.25"))
    if m.get("weights_key") != _sweep.weights_key(weights_path):
        return False, "weights changed since last detect"
    if str(m.get("conf")) != want_conf:
        return False, f"conf changed ({m.get('conf')}->{want_conf})"
    if m.get("config_hash") != _cfg_hash(cfg, ("detect", "data")):
        return False, "detect config changed"
    return True, f"{len(jpgs)} images + weights/conf/config match"


def resolve_name(cfg, exp_tag=None, dry_run=False, new_exp=False):
    """Resolve EXP_NAME (+BACKUP_DIR). dry-run never touches the pointer file.

    new_exp 為純 bool, 由入口從 NEW_EXP env 讀出後傳入 (library 不讀 env).
    非 dry-run 時經 exp.apply_to_env() 顯式寫入 os.environ (yaml 優先).
    """
    tag = exp_tag or cfg["exp"]
    if dry_run:
        return {"exp_name": f"exp_DRYRUN_{_exp.safe_tag(tag)}", "backup_dir": ""}
    info = _exp.open_experiment(tag, new_exp=bool(new_exp))
    _exp.apply_to_env(info["exp_name"], info["backup_dir"])
    return {"exp_name": info["exp_name"], "backup_dir": info["backup_dir"]}


def run_stage(cfg_path, stage, dry_run=False, engine_dir="yolov3_pytorch",
              repo_root=REPO_ROOT, skip_done=False):
    cfg = load_experiment(cfg_path)
    cfg["_cfg_path"] = str(cfg_path)
    new_exp = os.environ.get("NEW_EXP", "0") == "1"
    engine_dir = Path(engine_dir)
    runs_dir = engine_dir / "runs"
    ident = resolve_name(cfg, dry_run=dry_run, new_exp=new_exp)
    exp_name, backup_dir = ident["exp_name"], ident["backup_dir"]
    hyp = build_hyp_used(cfg, repo_root=repo_root)
    hyp_rel = f"runs/{exp_name}/hyp.used.yaml"

    def _write_used():
        dest = engine_dir / "runs" / exp_name / "hyp.used.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(hyp, f, sort_keys=False, default_flow_style=False)
        (engine_dir / "runs" / exp_name / "hyp_source.txt").write_text(
            hyp_rel, encoding="utf-8")  # record.py 用它還原 hyp 檔名
        snap = engine_dir / "runs" / exp_name / "exp.snapshot.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            snap.write_text(f.read(), encoding="utf-8")
        return dest

    def _write_val_meta(weights_path):
        import json
        (runs_dir / "val" / exp_name / "val.meta.json").write_text(json.dumps({
            "weights_key": _sweep.weights_key(weights_path),
            "iou": str((cfg.get("val") or {}).get("iou", "0.65")),
            "config_hash": _cfg_hash(cfg, ("val", "data")),
        }, ensure_ascii=False), encoding="utf-8")

    def _write_detect_meta(weights_path):
        import json
        (runs_dir / "detect" / exp_name / "meta.json").write_text(json.dumps({
            "weights_key": _sweep.weights_key(weights_path),
            "conf": str((cfg.get("detect") or {}).get("conf", "0.25")),
            "config_hash": _cfg_hash(cfg, ("detect", "data")),
        }, ensure_ascii=False), encoding="utf-8")

    train_cmd = build_train_cmd(cfg, exp_name)
    val_cmd = build_val_cmd(cfg, exp_name)
    detect_cmd = build_detect_cmd(cfg, exp_name)
    sw = cfg.get("sweep") or {}
    weights_rel = f"runs/train/{exp_name}/weights/best.pt"
    weights_path = runs_dir / "train" / exp_name / "weights" / "best.pt"

    decisions = {}
    if skip_done and stage in ("all", "train", "val", "detect"):
        decisions["train"], decisions["val"], decisions["detect"] = (
            train_up_to_date(cfg, exp_name, runs_dir, cfg_path),
            val_up_to_date(cfg, exp_name, runs_dir, weights_path),
            detect_up_to_date(cfg, exp_name, runs_dir, weights_path),
        )

    if dry_run:
        print("EXP =", exp_name)
        for _st, _cmd in (("train", train_cmd), ("val", val_cmd), ("detect", detect_cmd)):
            tag = ""
            if _st in decisions:
                ok, why = decisions[_st]
                tag = f" [{'SKIP' if ok else 'RUN'}: {why}]"
            print(f"{_st.upper()}:", " ".join(_cmd), tag)
        print("HYP_USED:", hyp)
        return {"exp": exp_name, "train": train_cmd, "val": val_cmd,
                "detect": detect_cmd, "hyp": hyp, "decisions": decisions}

    _write_used()
    os.chdir(engine_dir)

    def _gate(name):
        if not skip_done or name not in decisions:
            return True
        ok, why = decisions[name]
        print(f"{'SKIP' if ok else 'RUN'} {name}: {why}")
        return not ok

    if stage in ("train", "all") and _gate("train"):
        import time as _time
        _t0 = _time.monotonic()
        r = subprocess.run(train_cmd)
        _dt = _time.monotonic() - _t0
        if r.returncode != 0:
            raise RuntimeError(f"train 失敗 (returncode={r.returncode})")
        import json as _json
        (Path("runs") / "train" / exp_name / "train_time.json").write_text(
            _json.dumps({"seconds": round(_dt, 1)}), encoding="utf-8")
    if stage in ("val", "all") and _gate("val"):
        # tee to runs/val/<exp>/val.txt (record.py parses it for 類別明細)
        _run_capture(val_cmd, Path(engine_dir) / "runs" / "val" / exp_name / "val.txt")
        _write_val_meta(weights_path)
    if stage in ("sweep", "all"):
        _sweep.run_sweep(
            exp_name, f"runs/train/{exp_name}/weights/best.pt",
            data=f"data/{cfg['data']}.yaml", runs_root="runs",
            img=cfg.get("val", {}).get("img", 640),
            batch_size=cfg.get("val", {}).get("batch-size", 16),
            iou_list=tuple(str(x) for x in sw.get("iou", ("0.5", "0.6", "0.65"))),
            conf_list=tuple(str(x) for x in sw.get("conf", ("0.15", "0.25", "0.4"))),
            val_conf_list=tuple(str(x) for x in sw.get("val_conf", ("0.25",))),
            backup_dir=backup_dir,
        )
    if stage in ("detect", "all") and _gate("detect"):
        r = subprocess.run(detect_cmd)
        if r.returncode != 0:
            raise RuntimeError(f"detect 失敗 (returncode={r.returncode})")
        _write_detect_meta(weights_path)
    if stage in ("record", "all"):
        _record.update_workbook(
            exp_name, runs_root="runs",
            val_iou=float(cfg.get("val", {}).get("iou", 0.65)),
            det_conf=float(cfg.get("detect", {}).get("conf", 0.25)),
        )
    return {"exp": exp_name}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Single-entry YOLO car experiment runner")
    ap.add_argument("--exp", required=True, help="path to configs/experiments/<name>.yaml")
    ap.add_argument("--stage", default="all",
                    choices=["train", "val", "sweep", "detect", "record", "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip stages whose products are up to date (see SKIP/RUN log)")
    ap.add_argument("--engine-dir", default=".",
                    help="training engine root; commands run with cwd here (notebook already %%cd here)")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    args = ap.parse_args(argv)
    return run_stage(args.exp, args.stage, dry_run=args.dry_run,
                     engine_dir=args.engine_dir,
                     repo_root=Path(args.repo_root),
                     skip_done=args.skip_done)


if __name__ == "__main__":
    main()