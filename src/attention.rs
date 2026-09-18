use candle_core::{Device, Result, Tensor, D};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};

/// Builds the additive causal attention mask of shape `(1, 1, seq_len, seq_len)`.
///
/// Entry `(i, j)` is `0.0` when position `i` is allowed to attend to position `j`
/// (i.e. `j <= i`), and `f32::NEG_INFINITY` when `j` is in the future relative to `i`
/// (i.e. `j > i`). Adding this mask to raw attention scores before softmax drives the
/// softmax weight for future positions to zero.
pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
}

/// Hand-composed causal multi-head self-attention.
///
/// Q, K, V are each produced by their own `Linear` projection (no fused QKV or
/// prebuilt attention op). Attention is computed manually as
/// `softmax((Q K^T) / sqrt(head_dim) + mask) @ V`, followed by a final output
/// projection.
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

    /// Projects `x` into per-head Q, K, V tensors of shape `(b, n_heads, t, head_dim)`.
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
}
