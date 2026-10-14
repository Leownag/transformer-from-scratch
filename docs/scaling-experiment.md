# The 1e15 winner is not the 1e17 winner

The assignment asks two questions that look like one: find the best configuration under a 1e15
FLOP budget, then train a final model with 100× that budget and the same hyperparameters. The
tempting move is to take the sweep's winner straight to the final run.

That would have cost 10 points.

## What the sweep said

Five configurations at 1e15, everything fixed except width (and, for `s5`, the learning rate):

| config | val PPL |
|---|---|
| **`s1-e128`** | **306.04** |
| `s2-e192` | 393.56 |
| `s5-e256lr` | 477.27 |
| `s3-e256` | 517.77 |
| `s4-e320` | 676.66 |

Perplexity rises monotonically with model size. The smallest model wins by a wide margin — `s1`
beats `s3` by 69%.

The reason is visible in the step counts. At a fixed FLOP budget, a wider model costs more per
step and therefore takes fewer of them: `s1` runs 1352 steps over 22.2M tokens, `s4` only 414
steps over 6.8M. All five are still on the steep part of the loss curve, where another step is
worth more than another parameter. 1e15 is a severely data-limited budget.

## Why that should not be trusted at 1e17

`s1-e128` has 7.3M parameters, but `50257 × 128 = 6.43M` of those are token embeddings. The
transformer proper is about 0.87M parameters. At 1e17 it would run 135,252 steps over 2216M
tokens — 4.3 epochs over the corpus.

Compute-optimal model size grows roughly as the square root of the compute budget. A 100×
budget therefore points at roughly 10× the parameters, which lands around e320–e384 — nowhere
near e128. The data-limited regime that made the small model win is exactly the regime a 100×
budget leaves behind.

So the sweep ranking is an artifact of its budget, not a property of the architectures. The
honest thing to do is measure rather than argue.

## The ladder

Two configurations, three budgets, all measured:

| budget | e128 / 4 heads / 4 layers | e256 / 8 heads / 6 layers |
|---|---|---|
| 1e15 | **306.04** | 517.77 |
| 1e16 | 73.89 | **60.67** |
| 1e17 | 59.85 | **38.34** |

The crossover sits between 1e15 and 1e16. `e256` trails by 69%, then leads by 18%, then leads by
36%.

The 1e16 rung cost about 1.3 hours of V100 time. It changed the final answer.

## What was submitted

`q35-e256` — val loss 3.6465, val perplexity **38.34**, 1.00e17 training FLOPs, 70.7 MB
checkpoint.

Reported honestly in both directions: `s1-e128` is the correct answer to "best configuration at
1e15", and `e256` is the configuration that should actually be trained at 1e17. Those are
different answers to different questions, and the write-up says so rather than quietly
substituting one for the other.

One constraint shaped the whole experiment and is worth repeating: the submission cap is 100 MB,
and with weight tying the token embedding dominates at `50257 × n_embd`. That puts e384 at
~120 MB and therefore out of reach, regardless of what the scaling argument recommends. The
ladder was run between the two largest configurations that could actually be submitted.
