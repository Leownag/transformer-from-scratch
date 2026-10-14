# Running the sweep on NEU Explorer

How to reproduce the iso-FLOP sweep and the final training run on Northeastern's Explorer
cluster (Slurm). Results from the run this was written for are in
[`sweep-report.md`](sweep-report.md).

## 0. Access

Explorer replaced the older Discovery cluster. Request an account through a ticket at
[rc.northeastern.edu](https://rc.northeastern.edu/) or by mailing `rchelp@northeastern.edu`;
turnaround is usually one business day.

```bash
ssh <username>@login.explorer.northeastern.edu     # Duo 2FA
```

[ood.explorer.northeastern.edu](http://ood.explorer.northeastern.edu) (Open OnDemand) gives a
browser file manager and terminal, which is gentler for a first session.

Worth asking the course staff whether the class is registered with Research Computing: that
grants the `courses-gpu` partition (4 h default, 8 h limit) instead of the general `gpu-short`
(2 h limit). The final training run needs it.

## 1. Upload

Leave the virtualenv and the corpus behind — both are faster to rebuild on the cluster than to
transfer.

```bash
tar czf - --exclude='.venv' --exclude='data/tokens.npz' --exclude='__pycache__' \
          --exclude='.pytest_cache' --exclude='wandb' --exclude='outputs' . \
  | ssh <username>@login.explorer.northeastern.edu 'mkdir -p ~/hw2 && tar xzf - -C ~/hw2'
```

macOS ships openrsync (protocol 29), which fails to negotiate with the cluster's rsync 3.2.5 —
hence `tar` over ssh rather than the obvious `rsync -av`.

## 2. Build the environment

```bash
cd ~/hw2
bash slurm/setup-explorer.sh
```

The script refuses to run on a login node, builds with `python -m venv` rather than conda, and
fetches `tokens.npz` through a `.part`-then-rename so an interrupted download cannot masquerade
as a complete one.

Check the module names before trusting them — `module avail anaconda` and `module avail cuda`
show what the cluster actually has, and the `module load` lines at the top of
`setup-explorer.sh` and each `.sbatch` may need to match. The scripts have `||` fallbacks but
those are not guaranteed to hit.

## 3. Log in to Weights & Biases

```bash
wandb login     # key from https://wandb.ai/authorize, stored in ~/.netrc
```

**Do this on the login node, before submitting anything.** A batch job has no stdin, so an
unauthenticated `wandb.init()` blocks until the wall clock runs out. The job scripts detect a
missing `~/.netrc` and fall back to `WANDB_MODE=offline`, but losing live curves is worse than
spending thirty seconds here.

## 4. Submit

```bash
mkdir -p slurm/logs            # Slurm opens --output before the script runs; without this
                               # the job dies with no diagnostics at all
sbatch slurm/q34-sweep.sbatch              # the five 1e15 configurations
sbatch slurm/q34-crossover.sbatch          # the 1e16 ladder
sbatch slurm/q35-final.sbatch q35-e256     # the final 1e17 run
```

Monitoring:

```bash
squeue -u $USER
tail -f slurm/logs/q34-<jobid>.out
scancel <jobid>
```

The sweep job prints a summary of all five `eval.json` files when it finishes.

## 5. Why these five configurations

**The binding constraint is not compute, it is the 100 MB submission cap** (`train.py` warns at
98 MB). Weight tying makes the token embedding dominate the parameter count at
`50257 × n_embd`:

| n_embd | checkpoint | usable? |
|---|---|---|
| 256 | 71 MB | yes |
| 320 | 94 MB | yes, at the limit |
| 384 | 120 MB | no |

So `n_embd` cannot exceed 320, whatever the compute budget says. The assignment text does not
mention this.

The FLOP accounting (`src/lm/train.py`) is:

```
FLOPs            = flops_per_token × num_training_steps × grad_accumulation_steps
                                   × batch_size × seq_len
flops_per_token  = 6 × params + 12 × n_layer × n_embd × n_positions
```

With `batch_size=64` and `seq_len=256` fixed, the sweep varies width alone, then revisits the
learning rate:

| config | n_embd/heads/pos/layers | params | size | steps | FLOPs |
|---|---|---|---|---|---|
| `s1-e128` | 128/4/256/4 | 7.3M | 29 MB | 1352 | 9.996e14 |
| `s2-e192` | 192/6/256/6 | 12.4M | 50 MB | 785 | 9.999e14 |
| `s3-e256` | 256/8/256/6 | 17.7M | 71 MB | 551 | 9.997e14 |
| `s4-e320` | 320/8/256/6 | 23.6M | 94 MB | 414 | 9.990e14 |
| `s5-e256lr` | 256/8/256/6, max_lr 3e-3 | 17.7M | 71 MB | 551 | 9.997e14 |

All five sit at 99.9% of the budget — nothing wasted, nothing over. `s1`–`s4` move one axis at a
time, which is what makes the result readable; `s5` then tunes the learning rate at a fixed width.

## 6. Watch out for

- **`slurm/logs/` must exist before `sbatch`.** Repeated here because the failure is silent.
- **Pin the CUDA wheel.** Default PyPI serves a cu130 build, and CUDA 13 dropped Volta (sm_70),
  which is what `courses-gpu` runs. The job scripts pin
  `--index-url https://download.pytorch.org/whl/cu121`.
- **Build the venv on node-local `/tmp`, not `/courses`.** Installing torch into `/courses`
  stalls at ~1.6 MB/min under cgroup memory pressure; on `/tmp` the same install takes ~4
  minutes. The scripts `trap cleanup EXIT` so nothing is left behind.
- **Redirect every cache.** `PIP_CACHE_DIR`, `HF_HOME`, `WANDB_DIR` all point into the course
  space, or `/home` fills up.

Each of these is explained in [`sweep-report.md`](sweep-report.md) §5, with the symptom that led to it.
