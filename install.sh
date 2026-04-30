#!/usr/bin/env bash
# install.sh — full environment setup for FootballEventAnticipation
# Usage:  bash install.sh
# Requires: conda environment already activated (or plain venv with pip available)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] Installing core requirements ==="
pip install -r "$REPO_ROOT/requirements.txt"

echo "=== [2/4] Installing pose requirements ==="
pip install -r "$REPO_ROOT/requirements_pose.txt"

echo "=== [3/4] Installing SAM2 real-time (segment-anything-2-real-time) ==="
SAM2_DIR="$REPO_ROOT/segment-anything-2-real-time"

if [ ! -d "$SAM2_DIR" ]; then
    git clone https://github.com/Gy920/segment-anything-2-real-time.git "$SAM2_DIR"
fi

cd "$SAM2_DIR"
pip install -e . -q
python setup.py build_ext --inplace

echo "=== [4/4] Downloading SAM2 checkpoints ==="
if [ -f "$SAM2_DIR/checkpoints/download_ckpts.sh" ]; then
    (cd "$SAM2_DIR/checkpoints" && bash download_ckpts.sh)
else
    echo "WARNING: $SAM2_DIR/checkpoints/download_ckpts.sh not found — skipping checkpoint download."
fi

cd "$REPO_ROOT"
echo ""
echo "=== Installation complete ==="
