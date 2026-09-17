# tinygpt-rs — decoder-only transformer from scratch in Rust/candle

## Goal

Build a GPT-style decoder-only transformer entirely by hand on top of `candle`'s
tensor/autograd primitives — no pretrained weights, no `candle-transformers`
model blocks. The point is to demonstrate real understanding of transformer
internals (attention, KV-cache, training loop) rather than wrapping an
existing model, and to end up with a working, trainable, GPU-accelerated demo
on the author's own hardware (RTX 4050 Laptop, 6GB VRAM).

## Non-goals (v1)

- No pretrained model loading / fine-tuning of existing checkpoints.
- No RoPE, no BPE tokenizer, no custom CUDA kernels — these are v2 stretch
  goals (see below) and must not block v1 completion.
- No distributed/multi-GPU training.
- No web UI — a CLI is sufficient for the demo.

## Architecture

### Model (hand-composed, using `candle-core` + `candle-nn` primitives only)

- **Tokenizer**: char-level for v1. Vocab = distinct bytes/chars in the
  training corpus. Simple, deterministic, no external dependency.
- **Embeddings**: learned token embedding + learned positional embedding
  (added), both `candle_nn::Embedding`.
- **Decoder block** (repeated N times):
  - RMSNorm → multi-head causal self-attention → residual add
  - RMSNorm → SwiGLU MLP → residual add
- **Self-attention**, composed by hand (not a prebuilt attention module):
  - Q, K, V via three `candle_nn::Linear` projections
  - Reshape to `(batch, heads, seq, head_dim)`
  - Scaled dot-product: `softmax((QK^T) / sqrt(head_dim) + causal_mask) @ V`
  - Causal mask: precomputed upper-triangular `-inf` mask, added before softmax
  - Output projection via another `Linear`, then merge heads
- **MLP**: SwiGLU (`Linear` up-proj ×2 with SiLU gate, `Linear` down-proj)
- **Final**: RMSNorm → `Linear` vocab projection → logits

### Model size

Target 10–30M parameters (exact config chosen during implementation to fit
comfortably in 6GB VRAM with room for activations/optimizer state at the
chosen batch size/seq length). Example starting point: 6 layers, 6 heads,
384 hidden dim, 1024 seq len — to be tuned empirically.

### Training

- `candle-nn`'s `VarBuilder` + `VarMap` for parameter management and autograd.
- Optimizer: AdamW (candle built-in), cosine LR schedule with warmup, gradient
  clipping (by global norm).
- Loss: cross-entropy over next-token prediction.
- Backend: CUDA (`candle-core` with `cuda` feature), falling back to CPU if
  CUDA is unavailable at build/run time (feature-gated, not a runtime branch
  that silently degrades — CPU is a separate build profile for portability).
- Dataset: tiny-Shakespeare (public domain, ~1MB, standard nanoGPT-style
  benchmark corpus — fast to iterate on, and loss curves are easy to sanity
  check against well-known reference behavior).
- Batching: fixed-length causal LM batches sampled randomly from the corpus
  (standard random-offset chunking, no padding needed at char level).

### Inference

- Autoregressive generation loop.
- **KV-cache implemented by hand**: per-layer K/V tensors grown across
  generation steps, so each new token only computes attention against the
  cached prefix instead of recomputing the full sequence. This is the piece
  that most directly demonstrates mechanism understanding vs. library usage.
- Sampling: temperature scaling, top-k, and top-p (nucleus), all implemented
  directly against candle tensors.

## Demo centerpiece

1. **Live training run**: CLI prints loss per step/epoch and tokens/sec
   (throughput flex stat), ideally with the loss curve also dumped to a CSV
   for a quick plot.
2. **Ablation**: run the same trained-shape model with (a) causal mask
   disabled and (b) attention head count changed, and show the qualitative
   difference in generated output — proof the author understands what each
   architectural piece is doing, not just that the code runs.

## v2 / stretch goals (explicitly out of scope for "done")

- RoPE positional embeddings (replacing learned positional embeddings).
- BPE tokenizer via the `tokenizers` crate (replacing char-level).
- Hand-written CUDA kernel via `cudarc` for one hot path (fused QKV
  projection or attention softmax), benchmarked against candle's default GPU
  op to quantify the difference.

## Environment / prerequisites

- Rust/Cargo: already installed (1.98.1).
- GPU: RTX 4050 Laptop, 6GB VRAM, driver 610.57.04 — confirmed present.
- **CUDA toolkit (`nvcc`) is not yet installed** on this machine (only the
  NVIDIA driver stack is present). Required for candle's `cuda` build
  feature. Install via `sudo pacman -S --needed cuda` — done by the user
  directly (requires a password prompt this session can't satisfy).
- New standalone repo at `~/tinygpt-rs` (git-initialized, not nested inside
  the `~/` home-directory repo), so the project has clean, honest history
  from the first commit — every commit authored by the project's actual
  author.

## Testing strategy

- Unit tests for shape correctness at each stage (embeddings, attention
  output shape, block output shape == input shape).
- A causal-mask correctness test: verify a token's attention weights over
  future positions are exactly zero after softmax.
- A KV-cache correctness test: generation with cache enabled must produce
  logits numerically equivalent (within float tolerance) to a full
  recomputation without cache, for the same prefix.
- End-to-end smoke test: a handful of training steps on a tiny synthetic
  corpus must reduce loss (sanity check the training loop actually learns).

## Success criteria

- `cargo run --release --features cuda -- train` trains on tiny-Shakespeare
  on the RTX 4050 and shows a decreasing loss curve.
- `cargo run --release --features cuda -- generate` produces coherent-ish
  Shakespeare-like text using the hand-written KV-cache path.
- The ablation demo runs and shows a visible quality difference.
- Repo is public, MIT-licensed, with a README describing the architecture
  and what was hand-implemented vs. what candle provided.
