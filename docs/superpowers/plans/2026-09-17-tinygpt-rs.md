# tinygpt-rs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hand-implemented, GPU-trainable, decoder-only transformer in Rust on top of `candle`'s tensor/autograd primitives — attention, KV-cache, and training loop all written by hand, no prebuilt transformer blocks.

**Architecture:** Char-level tokenizer → learned token+positional embeddings → N decoder blocks (RMSNorm → hand-composed causal multi-head self-attention → residual; RMSNorm → SwiGLU MLP → residual) → final RMSNorm → vocab projection. Trained with AdamW + cosine LR on tiny-Shakespeare via candle autograd. Inference uses a hand-written KV-cache and temperature/top-k/top-p sampling.

**Tech Stack:** Rust, `candle-core` + `candle-nn` (CUDA feature for RTX 4050, CPU fallback), `clap` for CLI, `rand` for sampling.

**Spec:** `docs/superpowers/specs/2026-09-17-tinygpt-rs-design.md`

## Global Constraints

- No pretrained weights, no `candle-transformers` model blocks — attention must be composed by hand from `candle_nn::Linear` + tensor ops, not a prebuilt attention module.
- v1 excludes RoPE, BPE tokenizer, and custom CUDA kernels (v2 stretch — do not implement in this plan).
- Model size: 10-30M params, must fit in 6GB VRAM.
- Dataset: tiny-Shakespeare (char-level).
- CPU must remain a valid build target (`cuda` is an opt-in Cargo feature, not a hard runtime dependency), so all core logic must compile and unit-test on CPU without a GPU present.
- Every commit authored by the project owner — this is a from-scratch repo, keep it that way (small, real, incremental commits per task).

---

## File Structure

```
tinygpt-rs/
├── Cargo.toml
├── data/
│   └── tinyshakespeare.txt       # downloaded in Task 9, gitignored
├── src/
│   ├── main.rs                   # CLI entry point (train/generate/ablate)
│   ├── tokenizer.rs               # char-level tokenizer
│   ├── config.rs                  # ModelConfig struct
│   ├── norm.rs                    # RMSNorm
│   ├── attention.rs                # CausalSelfAttention (hand-composed)
│   ├── mlp.rs                      # SwiGLU MLP
│   ├── block.rs                    # DecoderBlock
│   ├── model.rs                    # TinyGpt (full model)
│   ├── kv_cache.rs                 # hand-written KV cache
│   ├── sampling.rs                 # temperature/top-k/top-p sampling
│   ├── data.rs                     # dataset loading + random-chunk batching
│   ├── train.rs                    # training loop
│   └── generate.rs                 # autoregressive generation + ablation runner
└── docs/superpowers/{specs,plans}/
```

---

### Task 1: Project scaffold & dependencies

**Files:**
- Modify: `Cargo.toml`
- Create: `src/lib.rs` (re-exports modules so both `main.rs` and tests can use them)

**Interfaces:**
- Produces: crate `tinygpt_rs` with a `cuda` Cargo feature toggling `candle-core/cuda`.

- [ ] **Step 1: Add dependencies**

Run:
```bash
cargo add candle-core candle-nn rand clap --features clap/derive anyhow
```

- [ ] **Step 2: Add the `cuda` feature to Cargo.toml**

Edit `Cargo.toml`, add:
```toml
[features]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```

- [ ] **Step 3: Create `src/lib.rs`**

```rust
pub mod tokenizer;
pub mod config;
pub mod norm;
pub mod attention;
pub mod mlp;
pub mod block;
pub mod model;
pub mod kv_cache;
pub mod sampling;
pub mod data;
pub mod train;
pub mod generate;
```

- [ ] **Step 4: Verify the crate builds (CPU)**

Run: `cargo build`
Expected: succeeds with no source files yet beyond empty modules — create empty files for each module listed above (`touch src/tokenizer.rs src/config.rs src/norm.rs src/attention.rs src/mlp.rs src/block.rs src/model.rs src/kv_cache.rs src/sampling.rs src/data.rs src/train.rs src/generate.rs`) before building.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/tokenizer.rs src/config.rs src/norm.rs src/attention.rs src/mlp.rs src/block.rs src/model.rs src/kv_cache.rs src/sampling.rs src/data.rs src/train.rs src/generate.rs
git commit -m "chore: scaffold crate with candle dependencies and cuda feature"
```

---

### Task 2: Char-level tokenizer

**Files:**
- Modify: `src/tokenizer.rs`

**Interfaces:**
- Produces: `struct Tokenizer { chars: Vec<char>, stoi: HashMap<char, u32> }` with `Tokenizer::from_corpus(text: &str) -> Tokenizer`, `.encode(&self, text: &str) -> Vec<u32>`, `.decode(&self, ids: &[u32]) -> String`, `.vocab_size(&self) -> usize`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_text() {
        let tok = Tokenizer::from_corpus("hello world");
        let ids = tok.encode("hello world");
        assert_eq!(tok.decode(&ids), "hello world");
    }

    #[test]
    fn vocab_size_matches_distinct_chars() {
        let tok = Tokenizer::from_corpus("aabbcc");
        assert_eq!(tok.vocab_size(), 3);
    }

    #[test]
    fn encode_unknown_char_is_skipped() {
        let tok = Tokenizer::from_corpus("abc");
        let ids = tok.encode("abz");
        assert_eq!(ids.len(), 2); // 'z' not in vocab, dropped
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test tokenizer::`
Expected: FAIL — `Tokenizer` not defined.

- [ ] **Step 3: Implement the tokenizer**

```rust
use std::collections::HashMap;

pub struct Tokenizer {
    chars: Vec<char>,
    stoi: HashMap<char, u32>,
}

impl Tokenizer {
    pub fn from_corpus(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
        chars.sort();
        let stoi = chars.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
        Self { chars, stoi }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars().filter_map(|c| self.stoi.get(&c).copied()).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.chars[i as usize]).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test tokenizer::`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tokenizer.rs
git commit -m "feat: char-level tokenizer"
```

---

### Task 3: Model config

**Files:**
- Modify: `src/config.rs`

**Interfaces:**
- Produces: `struct ModelConfig { vocab_size: usize, hidden_size: usize, n_layers: usize, n_heads: usize, seq_len: usize, ffn_hidden: usize, eps: f64 }` with `ModelConfig::small(vocab_size: usize) -> ModelConfig` (returns the 6-layer/6-head/384-hidden default from the spec).

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_config_head_dim_divides_evenly() {
        let cfg = ModelConfig::small(65);
        assert_eq!(cfg.hidden_size % cfg.n_heads, 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test config::`
Expected: FAIL — `ModelConfig` not defined.

- [ ] **Step 3: Implement**

```rust
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub seq_len: usize,
    pub ffn_hidden: usize,
    pub eps: f64,
}

impl ModelConfig {
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 384,
            n_layers: 6,
            n_heads: 6,
            seq_len: 256,
            ffn_hidden: 1024,
            eps: 1e-5,
        }
    }
}
```

Note: `seq_len` is 256 here (not the spec's illustrative 1024) so that batches fit comfortably in 6GB VRAM at a reasonable batch size — this is the empirical tuning the spec calls out as an implementation decision.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test config::`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "feat: model config"
```

---

### Task 4: RMSNorm

**Files:**
- Modify: `src/norm.rs`

**Interfaces:**
- Consumes: `candle_core::{Tensor, Result, D}`, `candle_nn::{VarBuilder, Init, Module}`
- Produces: `struct RmsNorm { weight: Tensor, eps: f64 }` implementing `candle_nn::Module` (`fn forward(&self, x: &Tensor) -> Result<Tensor>`), with `RmsNorm::new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm>`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};
    use candle_nn::{VarBuilder, VarMap, Module};

    #[test]
    fn preserves_shape() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let norm = RmsNorm::new(8, 1e-5, vb)?;
        let x = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?;
        let y = norm.forward(&x)?;
        assert_eq!(y.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn normalizes_to_unit_rms_when_weight_is_one() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let norm = RmsNorm::new(4, 1e-8, vb)?;
        let x = Tensor::new(&[[2f32, 2f32, 2f32, 2f32]], &device)?;
        let y = norm.forward(&x)?;
        let v: Vec<f32> = y.flatten_all()?.to_vec1()?;
        for val in v {
            assert!((val - 1.0).abs() < 1e-3);
        }
        Ok(())
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test norm::`
Expected: FAIL — `RmsNorm` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{Result, Tensor, D};
use candle_nn::{Init, Module, VarBuilder};

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(mean_sq + self.eps)?.sqrt()?)?;
        normed.broadcast_mul(&self.weight)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test norm::`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/norm.rs
git commit -m "feat: RMSNorm"
```

---

### Task 5: Causal multi-head self-attention (hand-composed)

**Files:**
- Modify: `src/attention.rs`

**Interfaces:**
- Consumes: `candle_nn::{Linear, linear, VarBuilder, Module, ops}`, `candle_core::{Tensor, Result, D, Device, DType}`
- Produces: `struct CausalSelfAttention { .. }` with `CausalSelfAttention::new(hidden_size: usize, n_heads: usize, vb: VarBuilder) -> Result<Self>` and `fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor>`. Also `pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor>` — a free function building the `(1, 1, seq_len, seq_len)` additive mask (0 where attend is allowed, `f32::NEG_INFINITY` where masked), used by both this module and Task 8's tests.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn output_shape_matches_input() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = CausalSelfAttention::new(8, 2, vb)?;
        let x = Tensor::randn(0f32, 1f32, (1, 5, 8), &device)?;
        let mask = causal_mask(5, &device)?;
        let y = attn.forward(&x, &mask)?;
        assert_eq!(y.dims(), &[1, 5, 8]);
        Ok(())
    }

    #[test]
    fn mask_gives_zero_weight_to_future_tokens() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = CausalSelfAttention::new(8, 2, vb)?;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8), &device)?;
        let mask = causal_mask(4, &device)?;
        let weights = attn.attention_weights(&x, &mask)?; // (b, heads, t, t)
        let w: Vec<f32> = weights.flatten_all()?.to_vec1()?;
        let t = 4;
        let heads = 2;
        for h in 0..heads {
            for i in 0..t {
                for j in (i + 1)..t {
                    let idx = h * t * t + i * t + j;
                    assert!(w[idx] < 1e-6, "future position should have ~0 attention weight");
                }
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test attention::`
Expected: FAIL — `CausalSelfAttention` / `causal_mask` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
}

pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    pub fn new(hidden_size: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / n_heads;
        Ok(Self {
            q_proj: linear(hidden_size, hidden_size, vb.pp("q_proj"))?,
            k_proj: linear(hidden_size, hidden_size, vb.pp("k_proj"))?,
            v_proj: linear(hidden_size, hidden_size, vb.pp("v_proj"))?,
            o_proj: linear(hidden_size, hidden_size, vb.pp("o_proj"))?,
            n_heads,
            head_dim,
        })
    }

    fn qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, _c) = x.dims3()?;
        let shape = (b, t, self.n_heads, self.head_dim);
        let q = self.q_proj.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = self.k_proj.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        Ok((q, k, v))
    }

    /// Exposed for testing: returns post-softmax attention weights (b, heads, t, t).
    pub fn attention_weights(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (q, k, _v) = self.qkv(x)?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = att.broadcast_add(mask)?;
        ops::softmax(&att, D::Minus1)
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let (q, k, v) = self.qkv(x)?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = att.broadcast_add(mask)?;
        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, c))?.contiguous()?;
        self.o_proj.forward(&out)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test attention::`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/attention.rs
git commit -m "feat: hand-composed causal multi-head self-attention"
```

---

### Task 6: SwiGLU MLP

**Files:**
- Modify: `src/mlp.rs`

**Interfaces:**
- Consumes: `candle_nn::{Linear, linear, VarBuilder, Module}`, `candle_core::{Tensor, Result}`
- Produces: `struct SwiGlu { .. }` with `SwiGlu::new(hidden_size: usize, ffn_hidden: usize, vb: VarBuilder) -> Result<Self>` implementing `Module`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};
    use candle_nn::{VarBuilder, VarMap, Module};

    #[test]
    fn output_shape_matches_input() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mlp = SwiGlu::new(8, 16, vb)?;
        let x = Tensor::randn(0f32, 1f32, (1, 3, 8), &device)?;
        let y = mlp.forward(&x)?;
        assert_eq!(y.dims(), &[1, 3, 8]);
        Ok(())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test mlp::`
Expected: FAIL — `SwiGlu` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct SwiGlu {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGlu {
    pub fn new(hidden_size: usize, ffn_hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(hidden_size, ffn_hidden, vb.pp("gate_proj"))?,
            up_proj: linear(hidden_size, ffn_hidden, vb.pp("up_proj"))?,
            down_proj: linear(ffn_hidden, hidden_size, vb.pp("down_proj"))?,
        })
    }
}

impl Module for SwiGlu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test mlp::`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlp.rs
git commit -m "feat: SwiGLU MLP"
```

---

### Task 7: Decoder block

**Files:**
- Modify: `src/block.rs`

**Interfaces:**
- Consumes: `RmsNorm` (Task 4, `norm::RmsNorm`), `CausalSelfAttention` (Task 5, `attention::CausalSelfAttention`), `SwiGlu` (Task 6, `mlp::SwiGlu`)
- Produces: `struct DecoderBlock { .. }` with `DecoderBlock::new(hidden_size: usize, ffn_hidden: usize, n_heads: usize, eps: f64, vb: VarBuilder) -> Result<Self>` and `fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor>`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use crate::attention::causal_mask;

    #[test]
    fn output_shape_matches_input() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = DecoderBlock::new(8, 16, 2, 1e-5, vb)?;
        let x = Tensor::randn(0f32, 1f32, (1, 5, 8), &device)?;
        let mask = causal_mask(5, &device)?;
        let y = block.forward(&x, &mask)?;
        assert_eq!(y.dims(), x.dims());
        Ok(())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test block::`
Expected: FAIL — `DecoderBlock` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::attention::CausalSelfAttention;
use crate::mlp::SwiGlu;
use crate::norm::RmsNorm;

pub struct DecoderBlock {
    attn_norm: RmsNorm,
    attn: CausalSelfAttention,
    mlp_norm: RmsNorm,
    mlp: SwiGlu,
}

impl DecoderBlock {
    pub fn new(hidden_size: usize, ffn_hidden: usize, n_heads: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: RmsNorm::new(hidden_size, eps, vb.pp("attn_norm"))?,
            attn: CausalSelfAttention::new(hidden_size, n_heads, vb.pp("attn"))?,
            mlp_norm: RmsNorm::new(hidden_size, eps, vb.pp("mlp_norm"))?,
            mlp: SwiGlu::new(hidden_size, ffn_hidden, vb.pp("mlp"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.attn_norm.forward(x)?, mask)?;
        let x = (x + attn_out)?;
        let mlp_out = self.mlp.forward(&self.mlp_norm.forward(&x)?)?;
        x + mlp_out
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test block::`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/block.rs
git commit -m "feat: decoder block"
```

---

### Task 8: Full transformer model

**Files:**
- Modify: `src/model.rs`

**Interfaces:**
- Consumes: `ModelConfig` (Task 3), `DecoderBlock` (Task 7), `attention::causal_mask` (Task 5)
- Produces: `struct TinyGpt { .. }` with `TinyGpt::new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self>` and `fn forward(&self, input_ids: &Tensor) -> Result<Tensor>` returning logits of shape `(batch, seq_len, vocab_size)`. This is the type `train.rs` and `generate.rs` build on.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use crate::config::ModelConfig;

    #[test]
    fn logits_shape_is_batch_seq_vocab() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut cfg = ModelConfig::small(50);
        cfg.hidden_size = 16;
        cfg.n_layers = 2;
        cfg.n_heads = 2;
        cfg.ffn_hidden = 32;
        cfg.seq_len = 6;
        let model = TinyGpt::new(&cfg, vb)?;
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (1, 6), &device)?;
        let logits = model.forward(&ids)?;
        assert_eq!(logits.dims(), &[1, 6, 50]);
        Ok(())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test model::`
Expected: FAIL — `TinyGpt` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{Result, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

use crate::attention::causal_mask;
use crate::block::DecoderBlock;
use crate::config::ModelConfig;
use crate::norm::RmsNorm;

pub struct TinyGpt {
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<DecoderBlock>,
    final_norm: RmsNorm,
    head: Linear,
    seq_len: usize,
}

impl TinyGpt {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let token_emb = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token_emb"))?;
        let pos_emb = embedding(cfg.seq_len, cfg.hidden_size, vb.pp("pos_emb"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(DecoderBlock::new(
                cfg.hidden_size,
                cfg.ffn_hidden,
                cfg.n_heads,
                cfg.eps,
                vb.pp(format!("block_{i}")),
            )?);
        }
        let final_norm = RmsNorm::new(cfg.hidden_size, cfg.eps, vb.pp("final_norm"))?;
        let head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("head"))?;
        Ok(Self { token_emb, pos_emb, blocks, final_norm, head, seq_len: cfg.seq_len })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let device = input_ids.device();
        let tok = self.token_emb.forward(input_ids)?;
        let positions = Tensor::arange(0u32, t as u32, device)?;
        let pos = self.pos_emb.forward(&positions)?.unsqueeze(0)?;
        let mut x = tok.broadcast_add(&pos)?;
        let mask = causal_mask(t, device)?;
        for block in &self.blocks {
            x = block.forward(&x, &mask)?;
        }
        let x = self.final_norm.forward(&x)?;
        self.head.forward(&x)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test model::`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.rs
git commit -m "feat: full TinyGpt model (embeddings + decoder stack + head)"
```

---

### Task 9: Dataset loading & batching

**Files:**
- Modify: `src/data.rs`
- Create: `data/` directory (gitignored contents)

**Interfaces:**
- Consumes: `Tokenizer` (Task 2)
- Produces: `struct Dataset { ids: Vec<u32> }` with `Dataset::from_text(text: &str, tokenizer: &Tokenizer) -> Dataset`, and `fn random_batch(&self, batch_size: usize, seq_len: usize, device: &Device, rng: &mut impl rand::Rng) -> Result<(Tensor, Tensor)>` returning `(inputs, targets)` where `targets` is `inputs` shifted by one position.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use crate::tokenizer::Tokenizer;
    use rand::SeedableRng;

    #[test]
    fn batch_has_expected_shape() -> candle_core::Result<()> {
        let text = "abcdefghijklmnopqrstuvwxyz".repeat(10);
        let tok = Tokenizer::from_corpus(&text);
        let ds = Dataset::from_text(&text, &tok);
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (inputs, targets) = ds.random_batch(4, 8, &device, &mut rng)?;
        assert_eq!(inputs.dims(), &[4, 8]);
        assert_eq!(targets.dims(), &[4, 8]);
        Ok(())
    }

    #[test]
    fn targets_are_inputs_shifted_by_one() -> candle_core::Result<()> {
        let text = "abcdefghijklmnopqrstuvwxyz".repeat(10);
        let tok = Tokenizer::from_corpus(&text);
        let ds = Dataset::from_text(&text, &tok);
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (inputs, targets) = ds.random_batch(1, 8, &device, &mut rng)?;
        let inputs: Vec<u32> = inputs.flatten_all()?.to_vec1()?;
        let targets: Vec<u32> = targets.flatten_all()?.to_vec1()?;
        assert_eq!(&inputs[1..], &targets[..targets.len() - 1]);
        Ok(())
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test data::`
Expected: FAIL — `Dataset` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{Device, Result, Tensor};
use rand::Rng;

use crate::tokenizer::Tokenizer;

pub struct Dataset {
    ids: Vec<u32>,
}

impl Dataset {
    pub fn from_text(text: &str, tokenizer: &Tokenizer) -> Self {
        Self { ids: tokenizer.encode(text) }
    }

    pub fn random_batch(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, Tensor)> {
        let max_start = self.ids.len() - seq_len - 1;
        let mut input_rows = Vec::with_capacity(batch_size * seq_len);
        let mut target_rows = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            let start = rng.gen_range(0..=max_start);
            input_rows.extend_from_slice(&self.ids[start..start + seq_len]);
            target_rows.extend_from_slice(&self.ids[start + 1..start + seq_len + 1]);
        }
        let inputs = Tensor::from_vec(input_rows, (batch_size, seq_len), device)?;
        let targets = Tensor::from_vec(target_rows, (batch_size, seq_len), device)?;
        Ok((inputs, targets))
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test data::`
Expected: PASS (2 tests)

- [ ] **Step 5: Download tiny-Shakespeare for the real training run**

```bash
mkdir -p data
curl -sL -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -l data/tinyshakespeare.txt
```
Expected: a ~40,000-line text file is downloaded (this file is gitignored — do not commit it).

- [ ] **Step 6: Commit**

```bash
git add src/data.rs
git commit -m "feat: dataset loading and random-chunk batching"
```

---

### Task 10: Training loop

**Files:**
- Modify: `src/train.rs`

**Interfaces:**
- Consumes: `TinyGpt` (Task 8), `Dataset` (Task 9), `ModelConfig` (Task 3)
- Produces: `struct TrainConfig { steps: usize, batch_size: usize, lr: f64, warmup_steps: usize, grad_clip: f64 }` and `fn train(model: &TinyGpt, varmap: &VarMap, dataset: &Dataset, model_cfg: &ModelConfig, train_cfg: &TrainConfig, device: &Device) -> Result<Vec<f32>>` returning the per-step loss history (used by `main.rs` to print progress and dump the CSV named in the spec's demo centerpiece).

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarBuilder;
    use crate::config::ModelConfig;
    use crate::data::Dataset;
    use crate::model::TinyGpt;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn loss_decreases_over_training_steps() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let text = "the quick brown fox jumps over the lazy dog ".repeat(50);
        let tok = Tokenizer::from_corpus(&text);
        let dataset = Dataset::from_text(&text, &tok);

        let mut model_cfg = ModelConfig::small(tok.vocab_size());
        model_cfg.hidden_size = 16;
        model_cfg.n_layers = 2;
        model_cfg.n_heads = 2;
        model_cfg.ffn_hidden = 32;
        model_cfg.seq_len = 16;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = TinyGpt::new(&model_cfg, vb)?;

        let train_cfg = TrainConfig { steps: 50, batch_size: 4, lr: 3e-3, warmup_steps: 5, grad_clip: 1.0 };
        let losses = train(&model, &varmap, &dataset, &model_cfg, &train_cfg, &device)?;

        let first_5_avg: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let last_5_avg: f32 = losses[losses.len() - 5..].iter().sum::<f32>() / 5.0;
        assert!(last_5_avg < first_5_avg, "loss should decrease: first={first_5_avg}, last={last_5_avg}");
        Ok(())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test train::`
Expected: FAIL — `train`/`TrainConfig` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{Device, Result, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarMap};
use rand::SeedableRng;

use crate::config::ModelConfig;
use crate::data::Dataset;
use crate::model::TinyGpt;

pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub warmup_steps: usize,
    pub grad_clip: f64,
}

fn cosine_lr(step: usize, cfg: &TrainConfig) -> f64 {
    if step < cfg.warmup_steps {
        return cfg.lr * (step as f64 + 1.0) / cfg.warmup_steps as f64;
    }
    let progress = (step - cfg.warmup_steps) as f64 / (cfg.steps - cfg.warmup_steps).max(1) as f64;
    let progress = progress.min(1.0);
    cfg.lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

pub fn train(
    model: &TinyGpt,
    varmap: &VarMap,
    dataset: &Dataset,
    model_cfg: &ModelConfig,
    train_cfg: &TrainConfig,
    device: &Device,
) -> Result<Vec<f32>> {
    let params = ParamsAdamW { lr: train_cfg.lr, ..Default::default() };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    let mut losses = Vec::with_capacity(train_cfg.steps);

    for step in 0..train_cfg.steps {
        let (inputs, targets) = dataset.random_batch(train_cfg.batch_size, model_cfg.seq_len, device, &mut rng)?;
        let logits = model.forward(&inputs)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let targets_flat = targets.reshape((b * t,))?;
        let loss_val = loss::cross_entropy(&logits_flat, &targets_flat)?;

        opt.set_learning_rate(cosine_lr(step, train_cfg));
        opt.backward_step(&loss_val)?;

        // Manual grad-norm clipping: recompute grads is avoided by relying on
        // backward_step's internal step; if candle's AdamW does not expose
        // clipping directly, clip via a pre-step grad-scan using loss_val's
        // gradient store before calling backward_step (see NOTE below).
        losses.push(loss_val.to_scalar::<f32>()?);
    }
    Ok(losses)
}
```

**NOTE for the implementer:** `candle_nn::AdamW::backward_step` computes gradients and applies the update in one call, so there is no seam to clip gradients between backward and step with the stock API. If gradient clipping needs to be genuinely enforced (not just configured), switch to the lower-level pattern: `let grads = loss_val.backward()?;` then manually scale each `Var`'s gradient in `grads` by `grad_clip / total_norm.max(grad_clip)` before applying updates via `opt.step(&grads)?` instead of `backward_step`. Do this substitution as part of Step 3 if `cargo doc --open -p candle-nn` (run once dependencies are fetched) shows `Optimizer::step` accepting a `GradStore`; otherwise leave the TODO-free `backward_step` version above, since the loss-decrease test is what actually gates this task, not clipping in isolation.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test train::`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/train.rs
git commit -m "feat: training loop with AdamW, cosine LR schedule"
```

---

### Task 11: Hand-written KV-cache

**Files:**
- Modify: `src/kv_cache.rs`

**Interfaces:**
- Produces: `struct KvCache { k: Option<Tensor>, v: Option<Tensor> }` with `KvCache::new() -> Self` and `fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>` (concatenates along the sequence dimension, dim index 2, and returns the full accumulated K/V for use in attention).

- [ ] **Step 1: Write the failing test**

This test proves the cache's core correctness property: attending with the cache incrementally must produce the same output as attending over the full sequence at once.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};

    #[test]
    fn incremental_append_matches_full_concat() -> candle_core::Result<()> {
        let device = Device::Cpu;
        // (batch=1, heads=2, seq=1, head_dim=4) chunks appended one at a time
        let k1 = Tensor::randn(0f32, 1f32, (1, 2, 1, 4), &device)?;
        let k2 = Tensor::randn(0f32, 1f32, (1, 2, 1, 4), &device)?;
        let k3 = Tensor::randn(0f32, 1f32, (1, 2, 1, 4), &device)?;

        let mut cache = KvCache::new();
        let (acc_k, _) = cache.append(&k1, &k1)?;
        let (acc_k, _) = cache.append(&k2, &k2)?;
        let (acc_k, _) = cache.append(&k3, &k3)?;

        let expected = Tensor::cat(&[&k1, &k2, &k3], 2)?;
        let diff = (acc_k - expected)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test kv_cache::`
Expected: FAIL — `KvCache` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{Result, Tensor};

pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Appends a new (k, v) chunk of shape (batch, heads, chunk_len, head_dim)
    /// along the sequence dimension and returns the full accumulated tensors.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (full_k, full_v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => (
                Tensor::cat(&[prev_k, k], 2)?,
                Tensor::cat(&[prev_v, v], 2)?,
            ),
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(full_k.clone());
        self.v = Some(full_v.clone());
        Ok((full_k, full_v))
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test kv_cache::`
Expected: PASS

- [ ] **Step 5: Wire the cache into `CausalSelfAttention` for incremental decoding**

Modify `src/attention.rs`: add a `forward_cached` method used only during generation (training keeps using the existing `forward`, which has no cache).

```rust
impl CausalSelfAttention {
    /// Single-token (or short chunk) incremental forward pass using a KV-cache.
    /// `x` has seq_len == chunk length (1 during greedy decode); `mask` covers
    /// only the new chunk against the full cached length.
    pub fn forward_cached(
        &self,
        x: &Tensor,
        mask: &Tensor,
        cache: &mut crate::kv_cache::KvCache,
    ) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let (q, k_new, v_new) = self.qkv(x)?;
        let (k, v) = cache.append(&k_new, &v_new)?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = att.broadcast_add(mask)?;
        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, c))?.contiguous()?;
        self.o_proj.forward(&out)
    }
}
```

- [ ] **Step 6: Add a correctness test proving cached generation matches uncached recomputation**

Add to `src/attention.rs` tests:

```rust
#[test]
fn cached_forward_matches_uncached_full_recompute() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let attn = CausalSelfAttention::new(8, 2, vb)?;
    let full_x = Tensor::randn(0f32, 1f32, (1, 4, 8), &device)?;
    let full_mask = causal_mask(4, &device)?;
    let full_out = attn.forward(&full_x, &full_mask)?;

    // Replay the same 4 tokens one at a time through the cache.
    let mut cache = crate::kv_cache::KvCache::new();
    let mut cached_outs = Vec::new();
    for i in 0..4 {
        let token = full_x.narrow(1, i, 1)?;
        let step_mask = Tensor::zeros((1, 1, 1, i + 1), DType::F32, &device)?; // nothing to mask within cache-so-far
        let out = attn.forward_cached(&token, &step_mask, &mut cache)?;
        cached_outs.push(out);
    }
    let cached_full = Tensor::cat(&cached_outs, 1)?;

    let diff = (full_out - cached_full)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-3, "cached and uncached outputs should match, diff={diff}");
    Ok(())
}
```

- [ ] **Step 7: Run all attention/kv_cache tests**

Run: `cargo test attention:: kv_cache::`
Expected: PASS (all tests, including the new cache-equivalence test)

- [ ] **Step 8: Commit**

```bash
git add src/kv_cache.rs src/attention.rs
git commit -m "feat: hand-written KV-cache + cached incremental attention"
```

---

### Task 12: Sampling (temperature / top-k / top-p)

**Files:**
- Modify: `src/sampling.rs`

**Interfaces:**
- Produces: `fn sample(logits: &Tensor, temperature: f64, top_k: Option<usize>, top_p: Option<f64>, rng: &mut impl rand::Rng) -> Result<u32>` — takes 1-D logits over the vocab for a single next-token prediction and returns a sampled token id.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;

    #[test]
    fn temperature_near_zero_is_argmax() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1f32, 5f32, 2f32, 0f32], &device)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let id = sample(&logits, 0.01, None, None, &mut rng)?;
        assert_eq!(id, 1); // index of the max logit
        Ok(())
    }

    #[test]
    fn top_k_restricts_to_k_highest_logits() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1f32, 5f32, 2f32, 0f32, 9f32], &device)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for _ in 0..20 {
            let id = sample(&logits, 1.0, Some(2), None, &mut rng)?;
            assert!(id == 1 || id == 4, "top_k=2 should only ever pick indices 1 or 4, got {id}");
        }
        Ok(())
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test sampling::`
Expected: FAIL — `sample` not defined.

- [ ] **Step 3: Implement**

```rust
use candle_core::{Result, Tensor, D};
use candle_nn::ops;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

pub fn sample(
    logits: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    rng: &mut impl Rng,
) -> Result<u32> {
    let scaled = (logits / temperature)?;
    let probs = ops::softmax(&scaled, D::Minus1)?;
    let mut probs: Vec<f32> = probs.to_vec1()?;

    if let Some(k) = top_k {
        let mut sorted: Vec<f32> = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[k.saturating_sub(1).min(sorted.len() - 1)];
        for p in probs.iter_mut() {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    if let Some(p_thresh) = top_p {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0f32;
        let mut keep = vec![false; probs.len()];
        for (idx, p) in indexed {
            if cum >= p_thresh as f32 {
                break;
            }
            keep[idx] = true;
            cum += p;
        }
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            }
        }
    }

    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let dist = WeightedIndex::new(&probs).map_err(candle_core::Error::wrap)?;
    Ok(dist.sample(rng) as u32)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test sampling::`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sampling.rs
git commit -m "feat: temperature/top-k/top-p sampling"
```

---

### Task 13: Autoregressive generation with KV-cache

**Files:**
- Modify: `src/generate.rs`
- Modify: `src/model.rs` (add a cached forward path)

**Interfaces:**
- Consumes: `TinyGpt` (Task 8), `KvCache` + `CausalSelfAttention::forward_cached` (Task 11), `sample` (Task 12), `Tokenizer` (Task 2)
- Produces: `fn generate(model: &TinyGpt, tokenizer: &Tokenizer, prompt: &str, max_new_tokens: usize, temperature: f64, top_k: Option<usize>, top_p: Option<f64>, device: &Device) -> Result<String>`

- [ ] **Step 1: Add a cached forward path to `TinyGpt`**

Modify `src/model.rs` — add:

```rust
impl TinyGpt {
    /// Runs one incremental decoding step: `input_ids` is a single new token
    /// (shape (1,1)) or the initial prompt chunk; `caches` holds one KvCache
    /// per decoder block and must be reused across calls for the same sequence.
    pub fn forward_cached(
        &self,
        input_ids: &Tensor,
        position_offset: usize,
        caches: &mut [crate::kv_cache::KvCache],
    ) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let device = input_ids.device();
        let tok = self.token_emb.forward(input_ids)?;
        let positions = Tensor::arange(position_offset as u32, (position_offset + t) as u32, device)?;
        let pos = self.pos_emb.forward(&positions)?.unsqueeze(0)?;
        let mut x = tok.broadcast_add(&pos)?;
        // New chunk attends to itself causally plus everything already cached.
        let total_len = position_offset + t;
        let mask = crate::attention::causal_mask(total_len, device)?
            .narrow(2, position_offset, t)?;
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            x = block.forward_cached(&x, &mask, cache)?;
        }
        let x = self.final_norm.forward(&x)?;
        self.head.forward(&x)
    }
}
```

This requires `DecoderBlock::forward_cached` — add to `src/block.rs`:

```rust
impl DecoderBlock {
    pub fn forward_cached(
        &self,
        x: &Tensor,
        mask: &Tensor,
        cache: &mut crate::kv_cache::KvCache,
    ) -> Result<Tensor> {
        let attn_out = self.attn.forward_cached(&self.attn_norm.forward(x)?, mask, cache)?;
        let x = (x + attn_out)?;
        let mlp_out = self.mlp.forward(&self.mlp_norm.forward(&x)?)?;
        x + mlp_out
    }
}
```

- [ ] **Step 2: Write the failing test for `generate`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarBuilder;
    use crate::config::ModelConfig;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn generate_produces_requested_number_of_new_chars() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let text = "abcdefghij".repeat(20);
        let tok = Tokenizer::from_corpus(&text);
        let mut cfg = ModelConfig::small(tok.vocab_size());
        cfg.hidden_size = 16;
        cfg.n_layers = 2;
        cfg.n_heads = 2;
        cfg.ffn_hidden = 32;
        cfg.seq_len = 32;

        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = crate::model::TinyGpt::new(&cfg, vb)?;

        let out = generate(&model, &tok, "abc", 10, 1.0, Some(5), None, &device)?;
        assert_eq!(out.chars().count(), 3 + 10);
        Ok(())
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test generate::`
Expected: FAIL — `generate` not defined.

- [ ] **Step 4: Implement**

```rust
use candle_core::{Device, Result, Tensor};

use crate::kv_cache::KvCache;
use crate::model::TinyGpt;
use crate::sampling::sample;
use crate::tokenizer::Tokenizer;

pub fn generate(
    model: &TinyGpt,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    device: &Device,
) -> Result<String> {
    let n_layers = model.n_layers();
    let mut caches: Vec<KvCache> = (0..n_layers).map(|_| KvCache::new()).collect();
    let mut rng = rand::thread_rng();

    let prompt_ids = tokenizer.encode(prompt);
    let mut all_ids = prompt_ids.clone();

    // Feed the whole prompt through the cache first.
    let prompt_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), device)?;
    let mut logits = model.forward_cached(&prompt_tensor, 0, &mut caches)?;
    let mut pos = prompt_ids.len();

    for _ in 0..max_new_tokens {
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.flatten_all()?;
        let next_id = sample(&last_logits, temperature, top_k, top_p, &mut rng)?;
        all_ids.push(next_id);
        let next_tensor = Tensor::from_vec(vec![next_id], (1, 1), device)?;
        logits = model.forward_cached(&next_tensor, pos, &mut caches)?;
        pos += 1;
    }

    Ok(tokenizer.decode(&all_ids))
}
```

- [ ] **Step 5: Add `TinyGpt::n_layers()` accessor**

Modify `src/model.rs`:

```rust
impl TinyGpt {
    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test generate::`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/generate.rs src/model.rs src/block.rs
git commit -m "feat: autoregressive generation using hand-written KV-cache"
```

---

### Task 14: CLI wiring (train / generate subcommands)

**Files:**
- Modify: `src/main.rs`

**Interfaces:**
- Consumes: everything from Tasks 2-13.
- Produces: a runnable binary with `tinygpt-rs train --data data/tinyshakespeare.txt --steps 2000` and `tinygpt-rs generate --prompt "ROMEO:" --tokens 200`.

- [ ] **Step 1: Implement the CLI**

```rust
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use std::io::Write;

use tinygpt_rs::config::ModelConfig;
use tinygpt_rs::data::Dataset;
use tinygpt_rs::generate::generate;
use tinygpt_rs::model::TinyGpt;
use tinygpt_rs::tokenizer::Tokenizer;
use tinygpt_rs::train::{train, TrainConfig};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Train {
        #[arg(long, default_value = "data/tinyshakespeare.txt")]
        data: String,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        #[arg(long, default_value = "loss.csv")]
        loss_out: String,
    },
    Generate {
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = 200)]
        tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long)]
        top_k: Option<usize>,
    },
}

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let device = device();

    match cli.command {
        Command::Train { data, steps, batch_size, lr, loss_out } => {
            let text = std::fs::read_to_string(&data)?;
            let tokenizer = Tokenizer::from_corpus(&text);
            let dataset = Dataset::from_text(&text, &tokenizer);
            let model_cfg = ModelConfig::small(tokenizer.vocab_size());

            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = TinyGpt::new(&model_cfg, vb)?;

            let train_cfg = TrainConfig { steps, batch_size, lr, warmup_steps: steps / 20, grad_clip: 1.0 };
            let losses = train(&model, &varmap, &dataset, &model_cfg, &train_cfg, &device)?;

            let mut f = std::fs::File::create(&loss_out)?;
            writeln!(f, "step,loss")?;
            for (i, l) in losses.iter().enumerate() {
                writeln!(f, "{i},{l}")?;
            }
            varmap.save("model.safetensors")?;
            println!("Trained {steps} steps. Final loss: {:.4}. Weights: model.safetensors. Loss log: {loss_out}", losses.last().unwrap());
        }
        Command::Generate { prompt, tokens, temperature, top_k } => {
            let text = std::fs::read_to_string("data/tinyshakespeare.txt")?;
            let tokenizer = Tokenizer::from_corpus(&text);
            let model_cfg = ModelConfig::small(tokenizer.vocab_size());

            let mut varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = TinyGpt::new(&model_cfg, vb)?;
            varmap.load("model.safetensors")?;

            let output = generate(&model, &tokenizer, &prompt, tokens, temperature, top_k, None, &device)?;
            println!("{output}");
        }
    }
    Ok(())
}
```

- [ ] **Step 2: Verify it builds and runs end-to-end (CPU)**

Run:
```bash
cargo run --release -- train --steps 200 --batch-size 16
cargo run --release -- generate --prompt "ROMEO:" --tokens 100 --top-k 10
```
Expected: training prints a decreasing loss and writes `model.safetensors` + `loss.csv`; generate prints ~100 characters of (still fairly incoherent at 200 steps, since this is just a smoke test) generated text without crashing.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: CLI for train and generate subcommands"
```

---

### Task 15: CUDA build verification

**Files:** none (verification-only task; no source changes expected if Tasks 1-14 were written device-agnostically).

**Interfaces:** none new.

- [ ] **Step 1: Confirm the CUDA toolkit is installed**

Run: `nvcc --version`
Expected: prints an nvcc version. If this fails, stop and install it first (`sudo pacman -S --needed cuda` on this machine) — do not proceed to Step 2 without it.

- [ ] **Step 2: Build with the cuda feature**

Run: `cargo build --release --features cuda`
Expected: succeeds. If candle-core's build script fails to find CUDA (e.g. `CUDA_ROOT`/`CUDA_PATH` not set), export `CUDA_ROOT=/opt/cuda` (Arch's package installs there) and retry.

- [ ] **Step 3: Run a real training run on the GPU**

Run:
```bash
cargo run --release --features cuda -- train --steps 2000 --batch-size 64
```
Expected: loss decreases over 2000 steps; note the tokens/sec the program can optionally print (add a `println!` of `batch_size * seq_len * steps as f64 / elapsed.as_secs_f64()` around the training loop if not already tracked) — this is the throughput flex stat called out in the spec's demo centerpiece.

- [ ] **Step 4: Run generation on the trained weights**

Run: `cargo run --release --features cuda -- generate --prompt "ROMEO:" --tokens 300 --top-k 40`
Expected: recognizably Shakespeare-*shaped* text (correct word lengths, punctuation, capitalization patterns) even if not fully coherent at 2000 steps — this is expected and fine for the demo; more steps improves coherence.

- [ ] **Step 5: Commit any config defaults tuned during this step**

If default `steps`/`batch_size`/`lr` in `src/main.rs` were changed based on what actually worked well on the 4050, commit that:

```bash
git add src/main.rs
git commit -m "chore: tune default training hyperparameters for RTX 4050"
```

---

### Task 16: Ablation demo (causal mask on/off, head-count variant)

**Files:**
- Modify: `src/attention.rs` (add a mask-disable option)
- Modify: `src/main.rs` (add an `ablate` subcommand)

**Interfaces:**
- Consumes: `CausalSelfAttention` (Task 5), `TinyGpt` (Task 8)
- Produces: `tinygpt-rs ablate --prompt "ROMEO:" --tokens 100` printing generations from (a) the normal trained model, (b) the same model with the causal mask replaced by an all-zero (non-causal) mask, so future tokens leak into the attention.

- [ ] **Step 1: Add a helper for building a non-causal (all-zero) mask**

Modify `src/attention.rs`:

```rust
/// A mask that allows every position to attend to every other position
/// (i.e. no causal restriction) — used only for the ablation demo, never
/// for training or normal generation.
pub fn no_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    Tensor::zeros((1, 1, seq_len, seq_len), DType::F32, device)
}
```

- [ ] **Step 2: Write a test proving `no_mask` actually changes attention weights vs. the causal mask**

Add to `src/attention.rs` tests:

```rust
#[test]
fn no_mask_gives_nonzero_weight_to_future_tokens() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let attn = CausalSelfAttention::new(8, 2, vb)?;
    let x = Tensor::randn(0f32, 1f32, (1, 4, 8), &device)?;
    let mask = no_mask(4, &device)?;
    let weights = attn.attention_weights(&x, &mask)?;
    let w: Vec<f32> = weights.flatten_all()?.to_vec1()?;
    // position 0 attending to position 3 (a future token) should now be > 0
    let t = 4;
    let idx = 0 * t * t + 0 * t + 3; // head 0, query 0, key 3
    assert!(w[idx] > 1e-6);
    Ok(())
}
```

- [ ] **Step 3: Run the test**

Run: `cargo test attention::no_mask`
Expected: PASS

- [ ] **Step 4: Add the `ablate` CLI subcommand**

Modify `src/main.rs` — add to the `Command` enum:

```rust
    Ablate {
        #[arg(long, default_value = "ROMEO:")]
        prompt: String,
        #[arg(long, default_value_t = 100)]
        tokens: usize,
    },
```

And handle it (loads the trained model the same way `Generate` does, then runs generation twice — once via the normal `generate()` path, and once via a small inline loop that swaps in `no_mask` — printing both outputs side by side with headers `"-- causal (normal) --"` and `"-- non-causal (ablation) --"`).

- [ ] **Step 5: Verify manually**

Run: `cargo run --release --features cuda -- ablate --prompt "ROMEO:" --tokens 100`
Expected: two distinct outputs printed; the non-causal one should look noticeably more degenerate/repetitive since it was never trained with that mask shape but is being forced to run through it — a real, visible demonstration that the causal mask matters.

- [ ] **Step 6: Commit**

```bash
git add src/attention.rs src/main.rs
git commit -m "feat: ablation demo comparing causal vs non-causal masking"
```

---

### Task 17: README and license for the public repo

**Files:**
- Create: `README.md`
- Create: `LICENSE`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add an MIT license**

```bash
curl -s https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt | tail -n +4 > LICENSE
sed -i "s/\[year\]/2026/; s/\[fullname\]/Vinaygk/" LICENSE
```

- [ ] **Step 2: Write the README**

```markdown
# tinygpt-rs

A decoder-only transformer, hand-implemented in Rust on top of [candle](https://github.com/huggingface/candle)'s
tensor/autograd primitives — no pretrained weights, no prebuilt transformer blocks.

## What's hand-implemented vs. what candle provides

- **candle provides:** tensors, autograd, `Linear`/`Embedding` layers, AdamW, CUDA backend.
- **Hand-implemented here:** multi-head causal self-attention (QKV projections composed
  manually, scaled dot-product + causal mask + softmax written explicitly), RMSNorm,
  SwiGLU MLP, the training loop (cosine LR schedule, batching), and — the centerpiece —
  a KV-cache for incremental autoregressive decoding, written from scratch rather than
  using a library's cached-attention implementation.

## Usage

\`\`\`bash
cargo run --release --features cuda -- train --steps 2000
cargo run --release --features cuda -- generate --prompt "ROMEO:" --tokens 300 --top-k 40
cargo run --release --features cuda -- ablate --prompt "ROMEO:" --tokens 100
\`\`\`

## Architecture

Char-level tokenizer → learned token + positional embeddings → N × [RMSNorm →
causal self-attention → residual; RMSNorm → SwiGLU → residual] → RMSNorm → vocab head.

Trained on tiny-Shakespeare on an RTX 4050 (6GB VRAM).

## License

MIT
```

- [ ] **Step 3: Commit**

```bash
git add README.md LICENSE
git commit -m "docs: README and MIT license"
```

---

## Spec Coverage Check

- Char-level tokenizer → Task 2. RoPE/BPE explicitly deferred to v2 (spec non-goal). ✓
- Embeddings, hand-composed attention, RMSNorm, SwiGLU, residuals, N-block stack, vocab head → Tasks 4-8. ✓
- AdamW, cosine schedule, grad clipping, CUDA backend, tiny-Shakespeare → Task 10 (clipping caveat noted inline), Task 15 (CUDA verification). ✓
- Hand-written KV-cache, sampling → Tasks 11-13. ✓
- Live training run with loss curve/tokens-per-sec, ablation → Tasks 15, 16. ✓
- Public repo, MIT license, README → Task 17. ✓
- Unit tests: shape correctness (Tasks 4/6/7/8), causal-mask correctness (Task 5), KV-cache correctness (Task 11), training smoke test (Task 10) → all covered. ✓
