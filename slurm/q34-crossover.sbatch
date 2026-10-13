#!/bin/bash
#SBATCH --job-name=hw2-cross
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cross-%j.out
#SBATCH --error=logs/cross-%j.err

set -euo pipefail

CW=/courses/INFO7375.202710/students/$USER

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

export PIP_CACHE_DIR="$CW/pip_cache"
export HF_HOME="$CW/hf_cache"
export WANDB_DIR="$CW/wandb"

# venv on node-local disk: installing torch into /courses crawls at ~1.6MB/min
VENV="/tmp/$USER-hw2-${SLURM_JOB_ID:-manual}"
cleanup() { rm -rf "$VENV"; }
trap cleanup EXIT

echo "=== host $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip -q install --upgrade pip
# cu121: CUDA 13 dropped Pascal (sm_60) / Volta (sm_70), which is all of courses-gpu
python -m pip -q install torch --index-url https://download.pytorch.org/whl/cu121
cd "$CW/hw2"
python -m pip -q install -r requirements.txt
python -m pip -q install -e .

python -c 'import torch; assert torch.cuda.is_available(); print("gpu", torch.cuda.get_device_name(0), "sm_%d%d" % torch.cuda.get_device_capability())'
test -s data/tokens.npz || { echo "ERROR: tokens.npz missing"; exit 1; }

if [[ -f "$HOME/.netrc" ]] && grep -q api.wandb.ai "$HOME/.netrc"; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi

# smaller model first: if the wall clock is hit, the cheaper point still lands
for cfg in c1-e128-1e16 c2-e256-1e16; do
    echo "######## $cfg | $(date) ########"
    WANDB_NAME="cross-$cfg" python src/lm/train.py "configs/$cfg.yaml"
    echo "---- $cfg done: $(tr -d '\n ' < outputs/$cfg/eval.json) ----"
done

echo "### CROSSOVER RESULTS ###"
for cfg in s1-e128 s3-e256 c1-e128-1e16 c2-e256-1e16; do
    printf "%-16s %s\n" "$cfg" "$(tr -d '\n ' < "outputs/$cfg/eval.json" 2>/dev/null || echo MISSING)"
done
