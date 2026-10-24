#!/bin/bash
# Solver environment on a CUDA machine (tested target: RTX 5090 / Blackwell, needs the cu128 wheels).
# Usage: tools/setup_gpu.sh   (from the repo root; creates .venv-gpu)
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.12}
$PY -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install --upgrade pip
pip install numpy numba scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
echo "next: PYTHONPATH=. python tools/gpu_smoke.py"
