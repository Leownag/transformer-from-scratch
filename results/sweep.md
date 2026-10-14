# Measured results

Raw files: `eval/*.json` (val loss and perplexity per run), `generations/*.jsonl` (samples),
`pytest-results.json` (test snapshot). Narrative in
[`../docs/sweep-report.md`](../docs/sweep-report.md) and
[`../docs/scaling-experiment.md`](../docs/scaling-experiment.md).

## Validation perplexity

| run | budget | n_embd / heads / layers | val loss | val PPL |
|---|---|---|---|---|
| `GPT-tiny` | baseline, 2000 steps | 64 / 4 / 4 | 7.0580 | 1162.12 |
| `s1-e128` | 1e15 | 128 / 4 / 4 | 5.7237 | **306.04** |
| `s2-e192` | 1e15 | 192 / 6 / 6 | 5.9752 | 393.56 |
| `s5-e256lr` | 1e15, max_lr 3e-3 | 256 / 8 / 6 | 6.1681 | 477.27 |
| `s3-e256` | 1e15 | 256 / 8 / 6 | 6.2495 | 517.77 |
| `s4-e320` | 1e15 | 320 / 8 / 6 | 6.5172 | 676.66 |
| `q35-e256` | 1e17 | 256 / 8 / 6 | 3.6465 | **38.34** |

## The crossover

| budget | e128 / 4 / 4 | e256 / 8 / 6 |
|---|---|---|
| 1e15 | **306.04** | 517.77 |
| 1e16 | 73.89 | **60.67** |
| 1e17 | 59.85 | **38.34** |

The 1e16 and `q35-e128` runs were executed on the cluster but their output directories were not
copied back, so only their perplexities are recorded here — there is no `eval/` file for them.

## Tests

`pytest-results.json`: 20 passed, 75/75 points, 1 skipped (`test_lm_forward_on_gpu`, no CUDA
device on the machine that produced the snapshot).

## Generations

`generations/` holds three prefixes × three temperatures (0.01, 0.7, 1.0) for:

- `GPT-tiny-*` — the baseline model
- `q35-e256-*` — the final 1e17 model
- `pythia_T*` — EleutherAI Pythia-1.4B, for comparison

Perplexity probes for the Q4.3 analysis are in `../data/q43_*.jsonl`: Northeastern wiki text
against the same text with every "the" removed, and Jabberwocky against a hand-written imitation.
