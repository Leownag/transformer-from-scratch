#!/bin/bash
# Environment setup for this assignment on NEU Explorer.
#
# DO NOT run on a login node (course policy: no heavy work there). Submit it:
#   sbatch -p courses --cpus-per-task=4 --mem=12G --time=02:00:00 \
#       -o logs/setup-%j.out -e logs/setup-%j.err slurm/setup-explorer.sh
#
# Uses python -m venv rather than conda: `conda create` deadlocks on the
# /courses VAST filesystem (parallel extraction blocks forever in futex_wait,
# reproduced twice) and a venv needs far fewer files anyway.
set -euo pipefail

case "$(hostname)" in
    explorer-*|login*) echo "ERROR: on a login node. Submit with sbatch instead." >&2; exit 1 ;;
esac

CW=/courses/INFO7375.202710/students/$USER
REPO="$(cd "$(dirname "$0")/.." && pwd)"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1   # explorer = outbound proxy

# keep every cache out of /home (course quota rule)
export PIP_CACHE_DIR="$CW/pip_cache"
export HF_HOME="$CW/hf_cache"
export WANDB_DIR="$CW/wandb"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$WANDB_DIR"

echo "=== host $(hostname) / $(python -V 2>&1) ==="

if [[ ! -x "$CW/envs/llm/bin/python" ]]; then
    rm -rf "$CW/envs/llm"
    python -m venv "$CW/envs/llm"
fi
source "$CW/envs/llm/bin/activate"
python -m pip install --upgrade pip

# prefer the wheel matching the loaded cuda module; fall back to default PyPI
# (modern torch wheels bundle their own CUDA runtime) if cu121 has no build
if ! python -m pip install torch --index-url https://download.pytorch.org/whl/cu121; then
    echo "cu121 wheel unavailable for $(python -V 2>&1); falling back to default PyPI torch"
    python -m pip install torch
fi

cd "$REPO"
python -m pip install -r requirements.txt
python -m pip install -e .

if [[ ! -f data/tokens.npz ]]; then
    echo "=== downloading tokens.npz (544MB) ==="
    curl -fL https://huggingface.co/datasets/yimingzhang/llms-hw2/resolve/main/tokens.npz \
        -o data/tokens.npz.part
    mv data/tokens.npz.part data/tokens.npz
fi

echo "=== verification ==="
python - <<'PYEOF'
import torch, tiktoken, wandb, omegaconf, einops, numpy, transformers
print("python      ", __import__("sys").version.split()[0])
print("torch       ", torch.__version__, "| cuda build:", torch.version.cuda)
print("cuda avail  ", torch.cuda.is_available())
print("tiktoken/wandb/omegaconf/einops/transformers all import OK")
PYEOF
ls -lh data/tokens.npz
echo "=== setup done ==="
