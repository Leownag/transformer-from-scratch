#!/bin/bash
# Q3.5 final training. Usage:  sbatch slurm/q35-final.sbatch <config-name>
#
# Resume-safe: train.py checkpoints every num_training_steps/50 into
# outputs/<cfg>/ckpt.pt and resumes from it, so if the wall clock is hit you
# just resubmit this exact command and it continues. A 1e17 run is ~5h on a
# V100, which is close to the courses-gpu limit.
#SBATCH --job-name=hw2-q35
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/q35-%j.out
#SBATCH --error=logs/q35-%j.err

set -euo pipefail

CFG="${1:?usage: sbatch slurm/q35-final.sbatch <config-name>}"
CW=/courses/INFO7375.202710/students/$USER

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

export PIP_CACHE_DIR="$CW/pip_cache"
export HF_HOME="$CW/hf_cache"
export WANDB_DIR="$CW/wandb"

VENV="/tmp/$USER-hw2-${SLURM_JOB_ID:-manual}"
cleanup() { rm -rf "$VENV"; }
trap cleanup EXIT

echo "=== host $(hostname) | config $CFG | $(date) ==="
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

if [[ -f "outputs/$CFG/ckpt.pt" ]]; then
    echo "=== existing checkpoint found, this run will resume ==="
fi

WANDB_NAME="q35-$CFG" python src/lm/train.py "configs/$CFG.yaml"

echo "### Q3.5 RESULT ###"
cat "outputs/$CFG/eval.json"
ls -lh "outputs/$CFG/model.pt"
