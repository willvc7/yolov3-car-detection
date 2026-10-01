# -*- coding: utf-8 -*-
"""Kaggle car CSV -> YOLO format conversion.

Ported from YOLOv3_try2.ipynb Cell26 (carcsv2yolo) + Cell27
(selection / normalize / split / convert / verify). Statements
unchanged; notebook globals became function parameters.
Single deviation: PIL/tqdm/matplotlib imports are lazy (inside
functions) so CPU-only test/CI environments without those packages
can still import this module and test pick_train_csv.
"""

from pathlib import Path

import pandas as pd

SEED = 42
IMAGE_COL_CANDIDATES = ["image", "image_id", "filename", "file_name", "img", "name"]
REQUIRED_BOX_COLS = ["xmin", "ymin", "xmax", "ymax"]
RANK_NAME_KEYS = ("train", "solution", "label", "annot")


def pick_train_csv(raw_dir):
    """Pick the training annotation CSV, skipping submission templates."""
    raw_dir = Path(raw_dir)
    csvs = sorted(raw_dir.rglob("*.csv"))
    assert csvs, f"{raw_dir} 下找不到 csv，請檢查 zip 結構"
    print("[TRY] 候選 CSV：", [c.name for c in csvs])
    # 排除提交模板（image,bounds 假框）：只留含 xmin/ymin/xmax/ymax 的訓練標註
    cands = []
    for _c in csvs:
        try:
            _cols = [x.strip() for x in pd.read_csv(_c, nrows=0).columns]
        except Exception as _e:
            print(f"[TRY] 跳過 {_c.name}（讀不到 header：{_e}）")
            continue
        _low0 = {x.lower().replace(" ", "") for x in _cols}
        if set(REQUIRED_BOX_COLS) <= _low0:
            cands.append(_c)
        else:
            print(f"[TRY] 跳過 {_c.name}（欄位={_cols}，缺 xmin/ymin/xmax/ymax，疑為提交模板）")
    assert cands, "找不到含 xmin/ymin/xmax/ymax 的訓練 CSV，請檢查 zip 內容"
    # 檔名含 train/solution/label/annot 優先，否則列數最多者
    def _rank(_p):
        _n = _p.name.lower()
        _key = 0 if any(_k in _n for _k in RANK_NAME_KEYS) else 1
        _rows = sum(1 for _ in open(_p, encoding="utf-8", errors="ignore")) - 1
        return (_key, -_rows)
    csv_path = sorted(cands, key=_rank)[0]
    print("[TRY] CSV =", csv_path)
    return csv_path


def load_annotations(csv_path):
    """Read CSV + normalize column names. Returns (df, img_col)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    print("columns =", list(df.columns), "rows =", len(df))
    print("nulls =", df.isnull().sum().to_dict())
    # 欄名容錯：image / image_id / filename / file_name → 統一為 image
    for cand in IMAGE_COL_CANDIDATES:
        if cand in df.columns:
            img_col = cand
            break
    else:
        raise KeyError(f"找不到圖片檔名欄（現有欄：{list(df.columns)}）")
    if img_col != "image":
        df = df.rename(columns={img_col: "image"})
        img_col = "image"
    # 欄名大小寫容錯：XMIN/x_min 等 → xmin/ymin/xmax/ymax
    low = {c.lower().replace(" ", ""): c for c in df.columns}
    for k in REQUIRED_BOX_COLS:
        assert k in low or k in df.columns, f"CSV 缺少 {k} 欄（現有欄：{list(df.columns)}）"
        if k not in df.columns:
            df = df.rename(columns={low[k]: k})
    print(df.head(3).to_string())
    print("unique images =", df[img_col].nunique())
    return df, img_col


def locate_image_dir(raw_dir):
    """Auto-locate the training image directory (training_images first)."""
    raw_dir = Path(raw_dir)
    img_dirs = [p for p in raw_dir.rglob("*") if p.is_dir() and any(p.glob("*.*"))]
    img_dir = next((p for p in img_dirs if "train" in p.name.lower()), None) or max(
        img_dirs, key=lambda p: len(list(p.glob("*.*"))))
    print("[TRY] IMG_DIR =", img_dir, f"({len(list(img_dir.glob('*.*')))} files)")
    return img_dir


def split_images(df, img_col, seed=SEED, ratio=0.8):
    """Split by image (not by box) so one image never spans train/val."""
    import random
    imgs = sorted(df[img_col].unique())
    rnd = random.Random(seed)
    rnd.shuffle(imgs)
    cut = int(len(imgs) * ratio)
    tr_imgs, va_imgs = set(imgs[:cut]), set(imgs[cut:])
    print(f"[TRY] train imgs={len(tr_imgs)} val imgs={len(va_imgs)} (seed={seed})")
    return tr_imgs, va_imgs


def carcsv2yolo(df, img_col, img_dir, out_img_dir, out_lbl_dir):
    """df: 清洗後標註；img_col: 檔名欄；img_dir: 原圖目錄；out_*: 輸出圖/標目錄。回傳 (n_img, n_box, n_missing, n_dropped)。"""
    from PIL import Image
    from tqdm import tqdm
    import shutil
    out_img_dir = Path(out_img_dir)
    out_lbl_dir = Path(out_lbl_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    n_box, n_missing, n_dropped = 0, 0, 0
    for img_name, g in tqdm(df.groupby(img_col), desc=f"Converting {out_img_dir}"):
        src_img = Path(img_dir) / str(img_name)
        if not src_img.exists():  # 大小寫/副檔名容錯找一次
            cands = list(Path(img_dir).glob(Path(str(img_name)).stem + ".*"))
            src_img = cands[0] if cands else src_img
        if not src_img.exists():
            n_missing += 1
            continue
        with Image.open(src_img) as im:
            W, H = im.size
        lines = []
        for _, r in g.iterrows():
            try:
                xmin, ymin, xmax, ymax = (float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"]))
            except KeyError:
                raise KeyError(f"CSV 缺少 xmin/ymin/xmax/ymax 欄（現有欄：{list(df.columns)}），請回 Cell23 確認 header")
            xmin, xmax = sorted([min(max(xmin, 0), W), min(max(xmax, 0), W)])  # clip 超界
            ymin, ymax = sorted([min(max(ymin, 0), H), min(max(ymax, 0), H)])
            bw, bh = xmax - xmin, ymax - ymin
            if bw <= 0 or bh <= 0:  # 零面積丟棄（計數進報告）
                n_dropped += 1
                continue
            xc, yc = (xmin + xmax) / 2 / W, (ymin + ymax) / 2 / H
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw / W:.6f} {bh / H:.6f}\n")  # 單類 car → cls=0
            n_box += 1
        shutil.copy2(src_img, out_img_dir / src_img.name)
        (out_lbl_dir / (src_img.stem + ".txt")).write_text("".join(lines), encoding="utf-8")  # 無框允許空txt（負樣本）
    return len(list(out_img_dir.glob("*"))), n_box, n_missing, n_dropped


def verify_conversion(yolo_dir, check_dir, n=3):
    """Re-project n label files back to pixels and save preview images."""
    from PIL import Image, ImageDraw
    import shutil
    yolo_dir, check_dir = Path(yolo_dir), Path(check_dir)
    shutil.rmtree(check_dir, ignore_errors=True)
    check_dir.mkdir(parents=True)
    for p in sorted((yolo_dir / "images/train").glob("*"))[:n]:
        im = Image.open(p).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for ln in (yolo_dir / "labels/train" / (p.stem + ".txt")).read_text().splitlines():
            _, xc, yc, w, h = ln.split()
            xc, yc, w, h = map(float, (xc, yc, w, h))
            d.rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H],
                        outline="red", width=3)
        im.save(check_dir / p.name)
    print("[TRY] 抽查圖 →", check_dir.resolve(), sorted(p.name for p in check_dir.glob("*")))


def split_and_convert(raw_dir, yolo_dir, seed=SEED, check_dir="runs/convert_check"):
    """Full Cell27 flow. Returns dict with counts for train/val."""
    raw_dir, yolo_dir = Path(raw_dir), Path(yolo_dir)
    csv_path = pick_train_csv(raw_dir)
    df, img_col = load_annotations(csv_path)
    img_dir = locate_image_dir(raw_dir)
    missing = [n for n in df[img_col].unique()[:5] if not (img_dir / str(n)).exists()]
    print("[TRY] 前5檔名存在抽查，缺檔範例：", missing if missing else "全存在")
    tr_imgs, va_imgs = split_images(df, img_col, seed=seed)
    r_tr = carcsv2yolo(df[df[img_col].isin(tr_imgs)], img_col, img_dir,
                       yolo_dir / "images/train", yolo_dir / "labels/train")
    r_va = carcsv2yolo(df[df[img_col].isin(va_imgs)], img_col, img_dir,
                       yolo_dir / "images/val", yolo_dir / "labels/val")
    print(f"[TRY] train: imgs={r_tr[0]} boxes={r_tr[1]} missing={r_tr[2]} dropped={r_tr[3]}")
    print(f"[TRY] val:   imgs={r_va[0]} boxes={r_va[1]} missing={r_va[2]} dropped={r_va[3]}")
    verify_conversion(yolo_dir, check_dir)
    print("\nDone!")
    return {"train": r_tr, "val": r_va}