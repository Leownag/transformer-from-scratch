#!/bin/bash
# Re-run Q4.2 own-model generations with the final Q3.5 model (q35-e256).
# CPU partition: generating 3 prefixes x 32 tokens is trivial and this avoids
# competing for the 2-job courses-gpu limit while q35-e128 is still training.
#SBATCH --job-name=hw2-q42regen
#SBATCH --partition=courses
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/q42regen-%j.out
#SBATCH --error=logs/q42regen-%j.err

set -euo pipefail
CW=/courses/INFO7375.202710/students/$USER

module purge
module load explorer anaconda3/2024.06
export PIP_CACHE_DIR="$CW/pip_cache"

VENV="/tmp/$USER-q42-${SLURM_JOB_ID:-manual}"
cleanup() { rm -rf "$VENV"; }
trap cleanup EXIT

python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip -q install --upgrade pip
python -m pip -q install torch --index-url https://download.pytorch.org/whl/cu121
cd "$CW/hw2"
python -m pip -q install -r requirements.txt
python -m pip -q install -e .

for T in 0.01 0.7 1.0; do
    echo "######## temperature $T ########"
    python src/lm/generate.py --config configs/q35-e256.yaml \
        --prefixes data/my_prefixes.jsonl --max_new_tokens 32 --temperature "$T"
    mv outputs/q35-e256/generation.jsonl "outputs/q35-e256/generation_T$T.jsonl"
done

echo "### DONE ###"
ls -l outputs/q35-e256/generation_T*.jsonl
