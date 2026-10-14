# transformer-from-scratch

A byte-level BPE tokenizer and a decoder-only transformer language model, written from empty
function bodies and trained to a fixed compute budget on a V100 through Slurm.

The interesting part is not the architecture. It is that **the configuration that wins at 1e15
FLOPs loses badly at 1e17**, and finding that out took one extra experiment.

| budget | e128 / 4 heads / 4 layers | e256 / 8 heads / 6 layers |
|---|---|---|
| 1e15 | **306.04** ppl | 517.77 |
| 1e16 | 73.89 | **60.67** |
| 1e17 | 59.85 | **38.34** |

All three rungs measured, not extrapolated. A sweep at 1e15 says the smallest model is best, by
69%. Taking that at face value into the final 100× run would have produced a model a third worse
than the one that shipped. [The full story.](docs/scaling-experiment.md)

## What is here

```
src/tokenizer/bpe.py   BPE merge/encode/decode, ASCII and Unicode byte-level
src/lm/model.py        multi-head attention, decoder blocks, weight-tied DecoderLM
src/lm/train.py        samplers, cosine schedule, training loop, checkpoint/resume
src/lm/generate.py     temperature sampling, batched generation
configs/               10 training configurations (1e15 sweep, 1e16 ladder, 1e17 final)
slurm/                 cluster bootstrap and every Slurm job script
results/               per-run perplexities, generation samples, test snapshot
docs/                  cluster runbook, run report, scaling experiment, provenance
```

20 tests pass, 75/75 points, 1 skipped (GPU-only). `results/pytest-results.json` is the snapshot.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest tests -v                                     # no download needed
```

Training needs the pre-tokenized corpus (544 MB, not redistributed here):

```bash
curl -L https://huggingface.co/datasets/yimingzhang/llms-hw2/resolve/main/tokens.npz \
  -o data/tokens.npz
python -m lm.train configs/GPT-tiny.yaml
python -m lm.generate --model_dir outputs/GPT-tiny --temperature 0.7
```

## On the cluster

The sweep ran on NEU Explorer: `courses-gpu` partition, one Tesla V100-SXM2-32GB, fp32 (Volta has
no bf16), 30 min 55 s wall clock for all five configurations in a single job.

[`docs/cluster-runbook.md`](docs/cluster-runbook.md) is the how-to.
[`docs/sweep-report.md`](docs/sweep-report.md) is what actually happened, including the nine
problems that had to be solved first. A few that generalise beyond this cluster:

- **The default PyTorch wheel does not work on a V100 any more.** PyPI serves cu130 builds, and
  CUDA 13 dropped Volta (sm_70) along with Pascal. Every job pins
  `--index-url https://download.pytorch.org/whl/cu121`.
- **Installing torch into the shared filesystem was projected at 52 hours.** Unpacking cudnn and
  cublas fills the page cache against a 12 G cgroup; writeback stalls in `folio_wait_bit` D-state
  and throughput collapses to 1.6 MB/min. On node-local `/tmp` the same install takes 259 seconds.
  The venv is rebuilt per job and removed by `trap cleanup EXIT`.
- **conda deadlocks on this filesystem.** Two `conda create` runs stalled in `futex_wait_queue`
  with 4316 files extracted and the environment directory never created — `flock` misbehaves on
  VAST, so conda's parallel extraction waits forever. `python -m venv` takes 5 seconds.
- **Slurm opens `--output` before your script runs**, so a missing `logs/` directory kills the
  job with no diagnostics. Hence `slurm/logs/.gitkeep`.
- **`/courses` is faster than local disk for large files** (128 MB/s; the 544 MB corpus downloads
  in 37 s) and slower for many small ones. The benchmark is in the run report.

Two constraints shaped the experiments more than compute did: the 100 MB submission cap, which
with weight tying puts a hard ceiling at `n_embd = 320`; and a 12 GiB memory request that peak
RSS was already filling to 97.8%, which is why tokens are held as int32 and widened per batch
rather than converted up front.

## Provenance

Built for INFO7375 (Semantic Intelligence in Large Language Models) at Northeastern University,
on a starter scaffold derived from CMU 11-667 and nanoGPT. The function signatures, docstrings
and test suite were provided; the implementations, configurations, cluster tooling, experiments
and write-ups are mine.

[`docs/model-card.md`](docs/model-card.md) records the trained model — architecture, training
data, evaluation, and where it fails. The weights are not distributed; the config and Slurm job
reproduce it in about five hours on one V100.

[`NOTICE`](NOTICE) lists ownership file by file. [`docs/provenance.md`](docs/provenance.md)
covers the changes I made to files I was not asked to touch — a wrong model name in the starter,
an fp16-on-MPS instability, and the two additions that made the final run survivable.

The written portions of the assignment that were literature review are not part of this
repository. Original Chinese working notes are preserved under [`docs/zh/`](docs/zh/).

Reorganised from the original assignment layout; the commit history was reconstructed from file
modification times rather than kept live during development.
