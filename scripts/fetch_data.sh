#!/usr/bin/env bash
# Fetch the Kaggle car dataset and verify its layout.
# Requires: kaggle CLI + ~/.kaggle/kaggle.json (see README).
# NOTE: not yet smoke-tested (no credentials in CI); first live run is the
# Phase-2 replay, which asserts the same layout independently.
set -euo pipefail

DEST="${1:-../datasets/car-raw}"
SLUG="sshikamaru/car-object-detection"

command -v kaggle >/dev/null 2>&1 || { echo "missing kaggle CLI (pip install kaggle)" >&2; exit 2; }
test -f ~/.kaggle/kaggle.json || { echo "missing ~/.kaggle/kaggle.json (see README)" >&2; exit 2; }

kaggle datasets download -d "$SLUG" -p "$DEST" --unzip

test -d "$DEST/data/training_images" || { echo "training_images missing under $DEST" >&2; exit 1; }
CSV_OK=0
for f in "$DEST"/data/*.csv; do
  if head -1 "$f" | grep -q xmin; then CSV_OK=1; fi
done
test "$CSV_OK" = 1 || { echo "no training CSV with xmin columns under $DEST/data" >&2; exit 1; }
echo "dataset ok: $DEST"