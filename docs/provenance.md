# What I wrote, and what I changed

The scaffold for this project was course-provided: file layout, function signatures, docstrings,
the test suite, and the Gradescope scoring plugin. [`NOTICE`](../NOTICE) lists ownership file by
file. This page covers the part that needs more than a list — the changes I made to files I was
not asked to touch, and why.

## Implemented from empty bodies

- `src/tokenizer/bpe.py` — `merge`, `encode`, `decode` for `ASCIIBPETokenizer`, and the
  regex-pre-tokenized span-wise variant in `UnicodeBPETokenizer`.
- `src/lm/model.py` — `MultiHeadAttention` (`q_kT_v`, masked `self_attention`, `projection`),
  `FeedForward`, `DecoderBlock`, and `DecoderLM` with weight-tied logits.
- `src/lm/train.py` — `random_batch_sampler`, `sequential_batch_sampler`, `cosine_lr_schedule`,
  `compute_language_modeling_loss`, and the `train` loop.
- `src/lm/generate.py` — `softmax_with_temperature` and batched autoregressive `generate`.
- `src/lm/evaluate_perplexity.py` — `compute_perplexity`.
- `src/lm/use_pythia.py` — `initialize_pythia`, `generate_pythia`.

One detail in `model.py` worth calling out: when a query row is fully masked, softmax over all
`-inf` produces `NaN`. The row is explicitly zeroed instead, which is what
`F.scaled_dot_product_attention` does and what `test_self_attention` compares against.

## Changes to files I was not asked to implement

**`use_pythia.py` and `evaluate_perplexity.py` — wrong model.** Both hard-coded
`model_name = "EleutherAI/pythia-6.9b"`. The assignment specifies Pythia-1.4B. Corrected in both.

**`use_pythia.py` — fp16 on MPS.** `from_pretrained` now passes `dtype=torch.float32`. The
checkpoint ships in fp16, and fp16 cross-entropy on Apple MPS produced unstable perplexities.

**`train.py` — checkpoint and resume.** The starter writes `model.pt` only after the final step.
A 1e17 FLOP run is roughly 5–7.5 hours on the V100 I was allocated, against an 8-hour wall clock,
so a timeout meant losing the entire run. Added `save_checkpoint()` with atomic tmp-then-rename
and periodic checkpointing, keyed to a deterministic sha1 run id so a resumed job rejoins the
same wandb run. The new arguments to `train()` are keyword-only and default to off, so the graded
call signature is unchanged.

**`train.py` `main()` — int32 tokens, widened per batch.** The starter converts all 520M tokens
to int64 up front; that copy alone is 4.16 GB and peak RSS reached 97.8% of the 12 GB cgroup.
Tokens are now held as int32 and widened one batch at a time. Numerically identical. The widening
wraps the sampler inside `main()`, so `random_batch_sampler` and `sequential_batch_sampler`
themselves are untouched and still match their tests.

No model hyperparameter was changed by any of this.

## Changes made while turning this into a repository

**`tests/test_tokenizer.py` — offline fixture.** `test_100_merges` fetched tinyshakespeare from
GitHub at test time. The expected merge rules are for the full corpus, so it is vendored at
`tests/fixtures/tinyshakespeare.txt` and the download remains only as a fallback. A dead
`import datasets as ds` left over from an earlier version was removed.

**`tests/test_env.py` — Python version.** The assertion was
`python_version.startswith("3.11")`, which fails on any other interpreter. Relaxed to 3.11 or
newer; nothing else in the check changed.

**`setup.py` — package name.** Renamed from the upstream `cmu-llms-hw2`, with the original
authors credited in [`NOTICE`](../NOTICE) rather than in package metadata.

## Tool use

I used Claude Code (Anthropic's command-line coding assistant) throughout, conversationally
rather than as autocomplete: implementing and debugging the tokenizer, the architecture, the
training loop and generation; planning which questions could be answered before the final model
existed; and running the Pythia generation and perplexity experiments. Every number reported
here came from actually executing the code, and the generated code was reviewed against the
provided unit tests.

Pythia-1.4B also appears in this project, but as the object of study the assignment requires,
not as a writing or coding aid.
