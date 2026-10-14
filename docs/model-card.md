# Model card — `q35-e256`

A record of the model this repository trains. The weights are not distributed: at 18M
parameters and 38.34 perplexity this is a teaching artifact, not something anyone should build
on. The card exists because writing one is part of the work — documenting a model honestly is a
separate skill from training it, and this is the format the field has settled on
(Mitchell et al., *Model Cards for Model Reporting*, FAT\* 2019).

Everything below is reproducible from this repository: `configs/q35-e256.yaml` plus
`slurm/q35-final.sbatch`.

## Model details

| | |
|---|---|
| Architecture | Decoder-only transformer, written from scratch (`src/lm/model.py`) |
| Parameters | 17.7M, of which `50257 × 256 = 12.9M` are tied token embeddings |
| Layers / heads / width | 6 / 8 / 256 |
| Context | 256 positions |
| Tokenizer | GPT-2 BPE via `tiktoken`, 50,257 entries (not the BPE implemented in `src/tokenizer/`) |
| Output head | Weight-tied — `logits = h @ E.T` |
| Normalization | Pre-LN residual blocks |
| Initialization | nanoGPT scheme |
| Precision | fp32 (the V100 used has no bf16) |
| Checkpoint | 70.7 MB |

The BPE tokenizer in `src/tokenizer/bpe.py` is implemented from scratch and tested, but the
language model is trained on a corpus pre-tokenized with GPT-2's vocabulary. The two are
separate exercises.

## Training data

A pre-tokenized split of **C4** (Colossal Clean Crawled Corpus, derived from Common Crawl),
distributed by the course as a single `tokens.npz`: 520M training tokens and 5M validation
tokens. It is not redistributed here; the download command is in the README.

The model saw **903M tokens — 1.74 epochs** of that split, so some data was seen more than once.

No filtering, deduplication or decontamination was applied beyond whatever produced the
provided file. Its exact derivation is not documented upstream, which is itself worth stating:
this card cannot fully characterise what the model was trained on.

## Training procedure

| | |
|---|---|
| Budget | 1.00e17 FLOPs (the fixed constraint the experiment was designed around) |
| Steps | 55,115 |
| Batch | 64 sequences × 256 tokens, no gradient accumulation |
| Schedule | Cosine, 1102 warmup steps, max_lr 1e-3, min_lr 1e-5 |
| Hardware | One Tesla V100-SXM2-32GB, `courses-gpu` partition, NEU Explorer |
| Wall clock | ~5h 23m |
| Tracking | Weights & Biases |

Trained through Slurm with checkpoint/resume, because the run is longer than a comfortable
margin under the partition's wall clock. See [`cluster-runbook.md`](cluster-runbook.md).

## Evaluation

**Validation perplexity: 38.34** (val loss 3.6465), measured on the held-out 5M-token split.

For context, all runs in this repository:

| run | budget | val PPL |
|---|---|---|
| `GPT-tiny` baseline | 2000 steps | 1162.12 |
| `s1-e128` | 1e15 | 306.04 |
| `q35-e128` | 1e17 | 59.85 |
| **`q35-e256`** | **1e17** | **38.34** |

Which configuration to train was not obvious, and the sweep's answer was wrong for this budget —
see [`scaling-experiment.md`](scaling-experiment.md).

### Compared against an open-source model

**EleutherAI Pythia-1.4B** was used as the reference point throughout: roughly 79× the parameter
count, trained on the Pile for far more tokens. It is used here as an object of study, not as a
component — nothing in this repository fine-tunes or redistributes it.

Generation perplexity at matched sampling temperature, same three prefixes:

| temperature | Pythia-1.4B | `q35-e256` |
|---|---|---|
| 0.01 | 12.65 | 83.18 |
| 1.0 | 18.80 | 298.84 |

The qualitative gap is one of degree rather than kind. `q35-e256` produces grammatical English
and has clearly learned corpus structure — given a forum prefix it continues with
`ReplyDelete / Replies / Reply / Anonymous August 10, 2012 at 12:29 PM`, reproducing
blog-comment boilerplate from the training distribution. What it does not do is stay on the
prefix's topic, where Pythia continues a passage about maple sap into sap collection and boiling.

At T≈0 both models collapse into verbatim repetition (`a good idea to be a good idea to be...`).
At T=1.0 both become more varied at the cost of coherence: Pythia drifts off-topic, this model
degrades into locally plausible but globally meaningless text.

Samples: [`../results/generations/`](../results/generations/).

### Two probes worth recording

Both run against Pythia-1.4B, as the assignment specifies, using `src/lm/evaluate_perplexity.py`.

**Function words carry predictability.** Northeastern's Wikipedia paragraphs score 7.09.
Removing every instance of "the" raises that to 9.28 — partly because "the" is among the most
predictable tokens in English and each occurrence was contributing a very low loss term, and
partly because its removal leaves ungrammatical noun phrases that make the *surrounding* tokens
harder to predict.

**Low perplexity on nonsense means memorization, not comprehension.** Jabberwocky scores 1.58.
A hand-written imitation preserving its meter, rhyme and syntactic skeleton but substituting
fresh nonsense words (`brillig → snurglic`, `slithy → vorghy`, `Jabberwock → Snatterwock`)
scores 19.39. Jabberwocky is public-domain and appears many times in the Pile, so Pythia
predicts even its invented words almost deterministically. The substituted words are genuinely
novel token sequences. Perplexity on a famous text measures recall, not language ability.

## Intended use and limitations

**Intended use:** none beyond study and reproduction. This is a demonstration that the pipeline
— tokenizer, architecture, training loop, cluster orchestration, evaluation — works end to end
under a fixed compute budget.

**Not suitable for** any downstream or production use. At 18M parameters and PPL 38 it produces
fluent-looking but unreliable text, and it is dramatically outperformed by any open model of
comparable or larger size that is actually distributed for use.

**Known limitations:**

- 256-token context, so it cannot model anything longer.
- Trained on web crawl with no additional filtering, deduplication or decontamination. It will
  reproduce whatever bias, boilerplate and unpleasantness survived in that split — visible in
  the comment-spam boilerplate it generates unprompted.
- No safety tuning, instruction tuning or alignment of any kind. It is a raw next-token
  predictor.
- 1.74 epochs means some training data was seen twice, so validation perplexity may be mildly
  optimistic relative to a single-epoch run.
- English only in practice.

**Why the weights are not published:** the architecture is a custom `DecoderLM`, not a
`transformers` class, so `AutoModel.from_pretrained` would not load it. A checkpoint without a
working load path is not a useful artifact. The configuration and job scripts here reproduce it
in about five hours on one V100.
