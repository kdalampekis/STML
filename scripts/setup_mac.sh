#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon). Creates .venv, installs everything,
# checks that PyTorch can use the Apple GPU (Metal / 'mps'), and builds the game.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "Python 3 not found. Install it with: brew install python@3.12"; exit 1; }
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed (brew install python@3.12)"'
echo "Using $($PY --version) at $(command -v $PY)"

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

python - <<'PY'
import torch
print("torch", torch.__version__)
print("Metal (mps) available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(256, 256, device="mps")
    print("Metal test OK:", float((x @ x).sum()) == float((x @ x).sum()))
PY

python scripts/build_game.py
echo
echo "Done. Next: source .venv/bin/activate && python -m pytest -q"
