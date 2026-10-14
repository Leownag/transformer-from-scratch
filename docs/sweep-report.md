# Cluster run report — iso-FLOP sweep on NEU Explorer

**Date** 2026-10-13 · **Cluster** NEU Explorer (Slurm) · **Partition** `courses-gpu`
**Hardware** Tesla V100-SXM2-32GB, compute capability 7.0, single GPU, fp32 (Volta has no bf16)
**Jobs** `11038194` (data download, CPU) · `11038277` (sweep) · `11047611` (final training)
**Wall clock** 30 min 55 s for the sweep, including ~5 min of environment build

## 1. Results

Five configurations, each trained to the same 1e15 FLOP budget. Everything except the listed
hyperparameter is held fixed: `n_positions=256`, `batch_size=64`, `seq_len=256`,
`grad_accumulation_steps=1`.

| config | changed | params | ckpt | steps | tokens seen | FLOPs | val loss | **val PPL** |
|---|---|---|---|---|---|---|---|---|
| **`s1-e128`** | n_embd 128, 4 heads, 4 layers | 7M | 29.1 MB | 1352 | 22.2M | 1.00e15 | 5.7237 | **306.04** |
| `s2-e192` | n_embd 192, 6 heads, 6 layers | 12M | 49.5 MB | 785 | 12.9M | 1.00e15 | 5.9752 | 393.56 |
| `s5-e256lr` | as s3, max_lr 3e-3 | 18M | 70.7 MB | 551 | 9.0M | 1.00e15 | 6.1681 | 477.27 |
| `s3-e256` | n_embd 256, 8 heads, 6 layers | 18M | 70.7 MB | 551 | 9.0M | 1.00e15 | 6.2495 | 517.77 |
| `s4-e320` | n_embd 320, 8 heads, 6 layers | 24M | 94.3 MB | 414 | 6.8M | 9.99e14 | 6.5172 | 676.66 |

**Perplexity increases monotonically with model size.** The smallest model wins; every widening
makes it worse. At 1e15 the budget is severely data-limited — `s1` takes 1352 steps over 22.2M
tokens while `s4` manages 414 steps over 6.8M, 3.3× fewer. All five are still on the steep part
of the loss curve, where one more step buys more than one more parameter.

**Learning rate helps, but an order of magnitude less than size.** Same architecture, `max_lr`
from 1e-3 to 3e-3: 517.77 → 477.27, a 40-point gain. The spread caused by model size is 371 points.

**Budget compliance.** All five report 1.00e15 FLOPs or below (`train.py` prints its own
accounting), checkpoints are 29.1–94.3 MB, and `MaxRSS` was 12.3 GB against a 12 GiB
request — 97.8%.

## 2. Why n_embd stops at 320

With weight tying, the parameter count is dominated by the token embedding, `50257 × n_embd`:

| n_embd | checkpoint | usable? |
|---|---|---|
| 256 | 70.7 MB | yes |
| 320 | 94.3 MB | yes, at the limit |
| 384 | ~120 MB | no — over the 100 MB submission cap |

The assignment text does not mention this, but the final model has to be uploaded, so 384 was
never an option regardless of how it would have scored.

## 3. The ranking does not survive a change of budget

The sweep answers "best configuration at 1e15". The final training run gets 100× that. Those are
not the same question.

| budget | e128 / 4 heads / 4 layers | e256 / 8 heads / 6 layers |
|---|---|---|
| 1e15 | **306.04** | 517.77 |
| 1e16 | 73.89 | **60.67** |
| 1e17 | 59.85 | **38.34** |

All three budgets were measured, not extrapolated. `e256` trails by 69% at 1e15, leads by 18% at
1e16, and leads by 36% at 1e17. **Taking the 1e15 winner at face value and training `e128` for
the final run would have landed in the half-credit band.** The 1e16 ladder cost about 1.3 hours
and changed the outcome by 10 points.

The reasoning that prompted the check: in `s1-e128`, token embeddings account for
`50257 × 128 = 6.43M` of the 7.3M parameters, so the transformer proper is only ~0.87M. At 1e17
it would run 135,252 steps over 2216M tokens — 4.3 epochs. Compute-optimal model size grows
roughly as the square root of budget, so a 100× budget points at roughly 10× the parameters,
which lands near e320–e384, not e128.

Final submission: `q35-e256`, val PPL **38.34**, checkpoint 70.7 MB, 1.00e17 train FLOPs.

## 4. Two blockers that had to be fixed first

1. **No checkpoint or resume.** `model.pt` was only written after the whole run finished, so a
   wall-clock timeout meant losing everything. At 5.5 TFLOPS measured, 1e17 FLOPs is ~5.1 hours
   against an 8-hour limit — too close to risk. Added periodic checkpointing with atomic
   tmp-then-rename and a deterministic sha1 run id so a resumed job rejoins the same wandb run.
2. **Memory was already at the ceiling.** `MaxRSS` hit 12.3 GB of a 12 GiB request. The cause was
   casting 520M `uint16` tokens to int64 up front — 4.16 GB that is never all needed at once.
   Changed to widen each batch lazily instead.

## 5. Nine problems hit, and what they turned out to be

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `rsync` could not transfer | macOS ships openrsync (protocol 29); negotiation with remote rsync 3.2.5 fails | `tar` over ssh |
| 2 | Jobs died silently, no output | Slurm opens the `--output` file *before* the script runs, so a missing `logs/` kills the job | Create the directory before submitting |
| 3 | Batch jobs hung until timeout | `wandb.init()` waits on stdin for credentials; a batch job has no stdin | Detect missing `~/.netrc`, fall back to `WANDB_MODE=offline` |
| 4 | `set -u` broke conda activate | The activate script dereferences an unset `PS1` | Moot after switching to venv |
| 5 | **conda deadlocked** | Two `conda create` runs stalled in `futex_wait_queue` with 4316 files extracted and the env directory never created. `flock` misbehaves on the VAST network filesystem, so conda's parallel extraction waits forever. **Reproduced twice** | `python -m venv`, 5 seconds |
| 6 | Caches filled `/home` | `--prefix` relocates the env but not the package cache, which still went to `~/.conda/pkgs` (274 MB) | Point `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`, `HF_HOME`, `WANDB_DIR` at the course space |
| 7 | **torch install projected at 52 hours** | Unpacking cudnn (700 MB) and cublas (400 MB) fills the page cache against a 12G cgroup; writeback stalls in `folio_wait_bit` D-state and throughput collapses to 1.6 MB/min | Build the venv on node-local `/tmp` — the same install finishes in **259 s** |
| 8 | **Default torch fails on the GPU** | PyPI now serves torch 2.14.1+cu130, and CUDA 13 dropped Pascal (sm_60) and Volta (sm_70) — which is the entire `courses-gpu` fleet | Pin `--index-url .../whl/cu121` |
| 9 | Shared environment unusable | `shared/envs/hw4` has no `bin/`, no interpreter, and holds cp311 packages while the cluster offers 3.9/3.12/3.13 | Build our own |

The cost of #7 is that `/tmp` is node-local, so the environment cannot be reused across jobs and
each one spends ~4 minutes rebuilding. Worth it: the wheels are already cached in
`$CW/pip_cache`, so nothing is re-downloaded.

**Measured filesystem behaviour** — counterintuitive enough to be worth recording:

| operation | /tmp | /scratch | /courses |
|---|---|---|---|
| create 500 small files | 0 s | 2 s | 3 s |
| sequential write, 512 MB | 8 s | — | **4 s** |

`/courses` handles large files *faster* than node-local disk (128 MB/s — the 544 MB `tokens.npz`
downloads in 37 s). The failure mode in #7 is specifically "many small files under cgroup memory
pressure", not bandwidth.

## 6. Shared-resource etiquette

| Rule | How it was met |
|---|---|
| No computation on login nodes | All compute on `c0584` / `c3014` / `d1004`; login nodes only for file copies and `sbatch` |
| Keep `/home` small | Final size 18 K. conda had leaked 274 MB there; cleaned up and redirected |
| Environments and caches off `/home` | `pip_cache`, `hf_cache`, `wandb`, `outputs`, `tokens.npz` all under the course space (5.4 G total) |
| Use `courses` / `courses-gpu` | GPU sweep on `courses-gpu`; environment build and data download on CPU `courses` |
| One GPU job at a time | All five configurations run sequentially inside a single `--gres=gpu:1` job |
| Right-size requests | `--mem=12G`, `--time=04:00:00`; actual use 31 minutes |
| Do not idle on a GPU | `sbatch`, never an interactive session; the job releases on completion |
| Load the site module | Every job does `module purge` then `module load explorer anaconda3/2024.06 cuda/12.1.1` |
| Leave no litter on shared nodes | The `/tmp` venv is removed by `trap cleanup EXIT` |
