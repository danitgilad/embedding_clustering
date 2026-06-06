#!/usr/bin/env bash
# Fetch the Point-MAE pretrained encoder weights for Part A's 3D feature.
#
# The encoder architecture is reimplemented (CPU-friendly, no CUDA extensions) in
# src/part_a/extractors/_point_mae_backbone.py; this script only downloads the official
# ShapeNet pretrained weights that get loaded into it. No repo clone, no compilation.
# Idempotent: safe to re-run.
#
# Usage:  bash scripts/setup_encoders.sh [DEST_PATH]
#   DEST_PATH defaults to checkpoints/point_mae_pretrain.pth (matches config/default.yaml).
#   Override the source with POINT_MAE_CKPT_URL=... if the release URL ever moves.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:-$ROOT/checkpoints/point_mae_pretrain.pth}"
URL="${POINT_MAE_CKPT_URL:-https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth}"

mkdir -p "$(dirname "$CKPT")"
if [ ! -f "$CKPT" ]; then
  echo "Downloading Point-MAE pretrain checkpoint -> $CKPT"
  wget -q -O "$CKPT" "$URL"
fi
echo "Point-MAE checkpoint ready at $CKPT"
