#!/usr/bin/env bash
# Write BILO snapshot from TypeScript and verify gradients against NumPy/PyTorch.
# Run from repo root or from playground:
#   ./playground/bilo_np/verify_ts_gradients.sh
#   cd playground && ./bilo_np/verify_ts_gradients.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAYGROUND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SNAPSHOT_JSON="$SCRIPT_DIR/ts_snapshot.json"

echo "=== 1. Build TS snapshot bundle and write $SNAPSHOT_JSON ==="
cd "$PLAYGROUND_DIR"
npm run write-bilo-snapshot

echo ""
echo "=== 2. Verify gradients (NumPy + PyTorch) ==="
cd "$SCRIPT_DIR"
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate math10 2>/dev/null || true
fi
python verify_ts_gradients.py "$SNAPSHOT_JSON"
