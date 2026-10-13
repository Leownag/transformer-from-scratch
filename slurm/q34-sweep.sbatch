#!/bin/bash
#SBATCH --job-name=hw2-q34
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/q34-%j.out
#SBATCH --error=logs/q34-%j.err

set -euo pipefail

CW=/courses/INFO7375.202710/students/$USER

# `explorer` provides the outbound proxy; without it pip/wandb hang silently
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

export PIP_CACHE_DIR="$CW/pip_cache"
export HF_HOME="$CW/hf_cache"
export WANDB_DIR="$CW/wandb"

# The venv lives on node-local disk, not /courses. Installing torch into
# /courses crawled at ~1.6MB/min (pip stuck in D state / folio_wait_bit:
# page-cache writeback stalling against the 12G cgroup); on /tmp the same
# install takes ~4min. Node-local means it must be rebuilt each job, which is
# the cheaper trade. Only outputs are written back to /courses.
VENV="/tmp/$USER-hw2-${SLURM_JOB_ID:-manual}"
cleanup() { rm -rf "$VENV"; }
trap cleanup EXIT

echo "=== host $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip -q install --upgrade pip

# Pin cu121: CUDA 13 dropped Pascal (sm_60) and Volta (sm_70), and courses-gpu
# is P100/V100. A default-PyPI torch (cu130) fails at the first kernel launch.
python -m pip -q install torch --index-url https://download.pytorch.org/whl/cu121
cd "$CW/hw2"
python -m pip -q install -r requirements.txt
python -m pip -q install -e .

python - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability()
print("gpu", torch.cuda.get_device_name(0), "sm_%d%d" % cap)
# prove kernels actually run on this architecture before burning GPU hours
x = torch.randn(512, 512, device="cuda")
assert torch.isfinite(x @ x).all(), "matmul produced non-finite values"
print("cuda matmul smoke test OK")
PYEOF

test -s data/tokens.npz || { echo "ERROR: data/tokens.npz missing"; exit 1; }

if [[ -f "$HOME/.netrc" ]] && grep -q api.wandb.ai "$HOME/.netrc"; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
    echo "NOTE: wandb not authenticated -> offline"
fi

CFGS=(s1-e128 s2-e192 s3-e256 s4-e320 s5-e256lr)

echo "=== sweep start $(date) ==="
for cfg in "${CFGS[@]}"; do
    echo "######## $cfg ########"
    WANDB_NAME="q34-$cfg" python src/lm/train.py "configs/$cfg.yaml"
done
echo "=== sweep done $(date) ==="

echo "### RESULTS ###"
for cfg in "${CFGS[@]}"; do
    printf "%-12s %s\n" "$cfg" "$(tr -d '\n ' < "outputs/$cfg/eval.json" 2>/dev/null || echo MISSING)"
done
