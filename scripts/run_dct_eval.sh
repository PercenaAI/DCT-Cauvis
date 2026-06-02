#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dct_checkpoint>"
  exit 2
fi

ROOT="${CAUVIS_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CODE_DIR="$ROOT/code/Cauvis"
WORK_DIR="${CAUVIS_WORK_DIR:-$ROOT/work_dirs/dct_cauvis_eval}"
DATA_ROOT="${SDGOD_DATA_ROOT:-/path/to/project/datasets/Single-DGOD}"
CKPT="$1"

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Missing checkpoint: $CKPT"
  exit 1
fi

export SDGOD_DATA_ROOT="$DATA_ROOT"
export PROJECT_ROOT="$ROOT"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

mkdir -p "$WORK_DIR"
out_file="$WORK_DIR/predictions.pkl"

cd "$CODE_DIR"
python tools/test.py \
  configs/cauvis/dct_cauvis_dinov2_dinohead_bs1x4_sdgod_project.py "$CKPT" \
  --work-dir "$WORK_DIR" \
  --out "$out_file"
