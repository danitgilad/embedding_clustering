#!/usr/bin/env bash
# Bootstrap the Point-MAE encoder (vendored; custom ops + checkpoint not pip-installable).
# Idempotent: safe to re-run. Run on the GPU box after `pip install -r requirements.txt`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"
mkdir -p "$VENDOR"

if [ ! -d "$VENDOR/Point-MAE" ]; then
  git clone --depth 1 https://github.com/Pang-Yatian/Point-MAE.git "$VENDOR/Point-MAE"
fi

CKPT_DIR="$VENDOR/Point-MAE/checkpoints"
mkdir -p "$CKPT_DIR"
# Set POINT_MAE_CKPT_URL to the pretrain .pth URL from the Point-MAE repo README.
CKPT_URL="${POINT_MAE_CKPT_URL:?Set POINT_MAE_CKPT_URL to the pretrain .pth URL}"
if [ ! -f "$CKPT_DIR/pretrain.pth" ]; then
  wget -O "$CKPT_DIR/pretrain.pth" "$CKPT_URL"
fi
echo "Point-MAE ready at $VENDOR/Point-MAE"
