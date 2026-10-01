#!/usr/bin/env bash
# Colab bootstrap: clone the training engine at a pinned commit, then verify.
# Engine stays a runtime dependency - it is never vendored into this repo.
set -euo pipefail

ENGINE_URL="https://github.com/ws6125/yolov3_pytorch.git"
ENGINE_DIR="${ENGINE_DIR:-yolov3_pytorch}"
# Capture per REPO_BUILD_PLAN.md section 8. DO NOT invent a SHA.
ENGINE_SHA="${ENGINE_SHA:-<PINNED_SHA_TBD>}"

if [ "$ENGINE_SHA" = "<PINNED_SHA_TBD>" ]; then
  echo "ENGINE_SHA is not set. Capture it with: git -C $ENGINE_DIR rev-parse HEAD" >&2
  echo "then re-run with ENGINE_SHA=<40-hex> $0" >&2
  exit 2
fi

if [ ! -d "$ENGINE_DIR/.git" ]; then
  git clone --depth 1 "$ENGINE_URL" "$ENGINE_DIR"
fi
git -C "$ENGINE_DIR" fetch origin "$ENGINE_SHA"
git -C "$ENGINE_DIR" checkout "$ENGINE_SHA"
HEAD_SHA="$(git -C "$ENGINE_DIR" rev-parse HEAD)"
test "$HEAD_SHA" = "$ENGINE_SHA" || { echo "SHA mismatch: $HEAD_SHA" >&2; exit 1; }
echo "engine pinned at $HEAD_SHA"