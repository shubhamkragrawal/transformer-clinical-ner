# Design decisions

This document explains the reasoning behind every non-obvious choice in this project — both architectural decisions and real bugs hit during implementation. The goal is to be able to defend each one, not just state it.

## Data

**Why BC5CDR via CoNLL files instead of a HuggingFace dataset.** `tner/bc5cdr`, `ncbi/ncbi_disease`, and even `conll2003` all failed to load under `datasets==5.0.0` (`RuntimeError: Dataset scripts are no longer supported`) — modern `datasets` versions dropped support for script-based dataset loaders entirely. Rather than pin an old `datasets` version (which created its own dependency conflicts with `transformers` and failed to build `tokenizers` from source on Python 3.13), the data was downloaded directly as CoNLL-format text files and parsed with a custom loader. This also makes the project depend on one fewer moving part — no HF dataset wrapper, no script trust issues, just plain text files and a parser anyone can read.

**Why `-100` as the ignore label.** This is PyTorch's documented default `ignore_index` for `CrossEntropyLoss`. Using it during label alignment means padding, `[CLS]`/`[SEP]`, and non-first-subword positions are automatically excluded from both the loss and its gradient — no manual masking logic needed at train time. The same `-100` sentinel is reused in evaluation (`decode_predictions`) for consistency: positions never trained on are never scored either.

**Why only the first subword of a word gets the real label.** When a word like "naloxone" splits into `na`, `##lo`, `##xon`, `##e`, only `na` carries the entity label; the rest get `-100`. This is the standard convention for token classification with subword tokenizers (matches HuggingFace's own NER examples) — labeling every subword would let the model "cheat" by predicting the same label for trivial continuations, inflating apparent performance without real understanding.

**Truncation behavior.** Sentences longer than `max_seq_len=128` get cut off mid-word by the tokenizer. Verified this doesn't silently misalign labels — the longest sentence in training data (138 word-tokens) truncates cleanly to exactly 128 subword-tokens with consistent shapes. Some entity information is lost at the tail of long sentences; an acceptable tradeoff at this sequence length for a project of this scope.

## Architecture

**Why scale embeddings by `sqrt(d_model)`.** Embedding values are initialized small (~N(0,1)). Without scaling, the positional encoding (values bounded in [-1, 1]) would dominate the token embedding signal when summed. Scaling balances their magnitudes before addition.

**Why sinusoidal positional encoding instead of learned.** Fixed sinusoids let the model generalize to sequence lengths unseen during training, and the relative-position structure (different frequencies per dimension pair) gives a consistent way to compute relative offsets via linear combinations. Learned positional embeddings (BERT's approach) work fine too — this was a deliberate choice to use the original Vaswani et al. design rather than a default.

**Why divide attention scores by `sqrt(d_k)`.** Without scaling, dot products grow large in magnitude as `d_k` increases (variance scales with dimension). Large values pushed into softmax saturate it, causing vanishing gradients. Dividing by `sqrt(d_k)` keeps score variance roughly constant regardless of dimension.

**Why mask before softmax, not after.** Softmax of `-inf` is exactly `0`. Masking before softmax guarantees masked positions get zero probability mass and don't distort normalization of the unmasked positions. Masking after softmax would zero them out but leave the remaining probabilities not summing to 1 — confirmed by direct test: `attn_weights` rows sum to exactly 1.0 in eval mode with masking applied before softmax.

**Why multiple attention heads instead of one large attention computation.** Each head can specialize — attend to different relationship types in parallel subspaces. Splitting `d_model` into `num_heads` smaller computations gives more representational flexibility than one large attention over the full dimension.

**Why `.contiguous()` before `view()` when merging heads.** `transpose()` only changes the tensor's view into underlying memory, leaving it non-contiguous. `view()` requires contiguous memory layout, so `.contiguous()` forces a real memory copy before reshaping.

**Why one combined `W_q`/`W_k`/`W_v` projection instead of separate per-head projections.** Mathematically equivalent to separate per-head projections concatenated, but a single larger matmul is faster on GPU than many small kernel launches.

**Why pre-LayerNorm instead of post-LayerNorm (the original 2017 design).** Post-LN (`LayerNorm(x + sublayer(x))`) rescales the residual output at every layer, which compounds across a deep stack and destabilizes training. Pre-LN normalizes the *input* to each sublayer instead, keeping the residual path (`x = x + ...`) clean and letting gradients flow through unscaled. This is why GPT-2 onward and most production transformers use pre-LN. A final `LayerNorm` after the last encoder layer compensates for the fact that pre-LN never normalizes the final residual output directly.

**Why LayerNorm instead of BatchNorm.** BatchNorm's statistics depend on batch size and become noisy with padded, variable-length sequences. LayerNorm normalizes across the feature dimension independently per token, making it indifferent to batch size and padding.

**Why ReLU instead of GELU in the feed-forward block.** GELU (used by BERT/GPT) tends to perform marginally better, but ReLU is simpler to reason about and explain, and the difference is small at this model scale. A deliberate simplicity-over-marginal-gains tradeoff, not an oversight.

**Why a single linear layer for the classification head.** After 4 layers of self-attention, each token's representation already contains rich contextual information. A linear projection to label space is standard practice for token classification (this is exactly what BERT-for-NER does) — added head complexity rarely helps once the encoder has done the work.

## Training

**Why warmup before linear decay, not decay from step 0.** At the start of training, AdamW's running gradient mean/variance estimates are near zero and unstable. A linear warmup ramps the learning rate up slowly while these estimates stabilize, avoiding destabilizing updates in the first few hundred steps.

**Why clip gradients by global norm, not by value.** Clipping each gradient value independently would distort the gradient vector's direction. Clipping by global norm scales the entire vector down proportionally when it exceeds a threshold, preserving direction while controlling magnitude.

**Validation loss and entity-level F1 don't agree on the "best" epoch.** Across multiple training runs, val_loss bottomed out around epoch 6 (~0.243) and rose steadily afterward — classic overfitting by the loss metric. But entity-level F1 kept slowly improving through epoch 19 (0.662) even as loss worsened. This is because cross-entropy loss rewards calibrated probabilities at *every* token position, while F1 only cares about the final hard argmax decision at entity boundaries — a model can become less calibrated overall while still getting more entity spans exactly right. Checkpointing was changed to track best F1 rather than best val_loss, since F1 is the metric that actually reflects deployment-relevant performance for NER. Both metrics are still logged every epoch for transparency.

**Why both val_loss and F1 are computed every epoch despite the added cost.** Computing F1 every epoch (a second pass over validation data plus seqeval scoring) adds real time per epoch, but the loss-vs-F1 divergence above would have gone unnoticed without it. The cost was worth the signal.

## Evaluation

**Why entity-level F1 instead of token-level accuracy.** Most tokens are `O` (not part of any entity), so a model that predicts `O` everywhere achieves high token accuracy while being useless. Entity-level F1 (via seqeval) requires getting the entire span right — correct boundaries and correct type — which is what actually reflects whether the model found real drug/disease mentions.

**Why the BERT baseline is a fair comparison.** Both models share the exact same `BC5CDRDataset`, label alignment, train/val split, optimizer/scheduler setup, and evaluation logic (`decode_predictions` reused identically for both). The only difference is the encoder itself: BERT pretrained on a large general corpus and fine-tuned for 5 epochs, versus a from-scratch transformer trained from random initialization on this dataset alone for 20 epochs. That isolates pretraining as the variable being measured.

**Why the BERT baseline uses 5 epochs while the from-scratch model uses 20.** BERT is already pretrained; fine-tuning typically converges in a handful of epochs (confirmed here — train loss collapsed from 0.24 to 0.007 in 5 epochs). Running it for 20 epochs would mostly add overfitting time, not signal. The from-scratch model, training from random initialization, needs the full 20 epochs to reach a reasonable F1.

## Results

| Model | Parameters | Epochs | Precision | Recall | F1 |
|---|---|---|---|---|---|
| From-scratch transformer | 9.9M | 20 | 0.6306 | 0.6967 | 0.6620 |
| BERT-base (fine-tuned) | 110M | 5 | 0.8678 | 0.8780 | 0.8728 |

The ~21-point F1 gap is attributable to pretraining: BERT entered fine-tuning already understanding English syntax and some biomedical vocabulary from a massive general corpus, while the from-scratch model learned everything — including basic language structure — from ~3,942 training sentences alone. The gap is the evidence, not a shortcoming to explain away.

## Bugs hit and fixed during implementation

**YAML scientific notation parsed as a string, not a float.** `learning_rate: 3e-4` in `config.yaml` was read back by `yaml.safe_load` as the string `"3e-4"`, not the float `0.0003`, because PyYAML's safe loader follows the YAML 1.1 spec, which requires an explicit sign or decimal point to recognize scientific notation as a float. This caused a `TypeError` deep inside AdamW's internals (`'<=' not supported between float and str`) only when the optimizer actually tried to use the learning rate — not at config load time. Fixed by explicitly casting `learning_rate` and `weight_decay` to `float()` inside `load_config()`, rather than relying on YAML's type inference.

**`CrossEntropyLoss` returns `nan`, not `0.0`, when every position in a batch is masked.** Initially assumed a fully-masked batch (all labels `-100`) would produce zero loss. In fact, `nn.CrossEntropyLoss` with `ignore_index` and default mean reduction divides by the count of non-ignored elements — zero valid elements means division by zero, yielding `nan`. This is correct PyTorch behavior, not a bug; the test was updated to assert `torch.isnan(loss)` instead of `loss == 0.0`.

**`torch.load` rejected checkpoints containing numpy scalars.** Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`, refusing to unpickle arbitrary objects for security. seqeval's scoring functions return `np.float64`, not plain Python `float`, and these ended up embedded in the checkpoint's `metrics` dict, triggering `UnpicklingError: Unsupported global: numpy._core.multiarray.scalar`. Fixed two ways: immediately, by loading with `weights_only=False` (acceptable only because the checkpoint was self-created and trusted); going forward, by casting all metrics to plain `float()` before saving, so checkpoints remain loadable under the safer default.

**Attention weight rows don't sum to exactly 1.0 during training.** Observed `attn_weights.sum(dim=-1)` returning values like `1.11` instead of `1.0` when dropout was active. This is expected: dropout zeroes some attention weights and scales survivors by `1/(1-p)` to preserve expected value — not to preserve the sum-to-1 property. Confirmed by re-running in `.eval()` mode (dropout disabled), where sums returned to exactly `1.0`. Not a bug; documented so it isn't mistaken for one later.

**MPS device detection.** `torch.cuda.is_available()` is always `False` on Apple Silicon. Device selection checks `torch.backends.mps.is_available()` first, falling back to `cuda` then `cpu`. Seeding was explicitly verified to produce identical model initialization on MPS (`torch.manual_seed(42)` twice → identical weights) rather than assumed, since MPS doesn't guarantee the same bit-for-bit determinism guarantees as CUDA for every operation.

## What I'd do differently

- Stop training around epoch 8-10 rather than 20 — F1 gains after that point are marginal (0.65 → 0.66 over 10+ epochs) while val_loss actively worsens, suggesting diminishing returns not worth the extra compute.
- Try GELU instead of ReLU to match BERT/GPT convention and see if it closes any of the F1 gap.
- Add a CRF layer on top of the classification head, which typically improves entity boundary precision in NER, deliberately skipped here to keep the from-scratch architecture easier to reason about and explain.
- Track F1 per-epoch from the start rather than retrofitting it after noticing the val_loss/F1 divergence — would have caught the discrepancy a training run earlier.