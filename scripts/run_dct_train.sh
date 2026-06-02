#!/usr/bin/env bash
set -euo pipefail

ROOT="${CAUVIS_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CODE_DIR="$ROOT/code/Cauvis"
WORK_DIR="${CAUVIS_WORK_DIR:-$ROOT/work_dirs/dct_cauvis_train}"
DATA_ROOT="${SDGOD_DATA_ROOT:-/path/to/project/datasets/Single-DGOD}"
CKPT="${CAUVIS_PRETRAINED:-$ROOT/pretrained_weights/cauvis/cauvis_dinohead.pth}"

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Missing checkpoint: $CKPT"
  exit 1
fi

export SDGOD_DATA_ROOT="$DATA_ROOT"
export CAUVIS_PRETRAINED="$CKPT"
export PROJECT_ROOT="$ROOT"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

mkdir -p "$WORK_DIR"
cd "$CODE_DIR"
python tools/train.py \
  configs/cauvis/dct_cauvis_dinov2_dinohead_bs1x4_sdgod_project.py \
  --amp \
  --work-dir "$WORK_DIR" \
  --cfg-options "load_from=$CAUVIS_PRETRAINED"
