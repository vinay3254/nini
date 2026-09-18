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
        Ok(Self { token_emb, pos_emb, blocks, final_norm, head })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let mask = causal_mask(t, input_ids.device())?;
        self.forward_with_mask(input_ids, &mask)
    }

    /// Runs a full forward pass using an explicit attention mask instead of
    /// building the standard causal mask internally. Used only by the
    /// `ablate` CLI demo to show what happens when the trained model is
    /// forced through a non-causal mask (e.g. `attention::no_mask`) —
    /// never used for training or normal generation.
    pub fn forward_with_mask(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let device = input_ids.device();
        let tok = self.token_emb.forward(input_ids)?;
        let positions = Tensor::arange(0u32, t as u32, device)?;
        let pos = self.pos_emb.forward(&positions)?.unsqueeze(0)?;
        let mut x = tok.broadcast_add(&pos)?;
        for block in &self.blocks {
            x = block.forward(&x, mask)?;
        }
        let x = self.final_norm.forward(&x)?;
        self.head.forward(&x)
    }

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
        let mask = causal_mask(total_len, device)?
            .narrow(2, position_offset, t)?;
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            x = block.forward_cached(&x, &mask, cache)?;
        }
        let x = self.final_norm.forward(&x)?;
        self.head.forward(&x)
    }

    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }
}

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

    /// `forward_with_mask` given the standard causal mask must reproduce
    /// `forward`'s output exactly, since `forward` is just `forward_with_mask`
    /// with the causal mask built internally. This guards the `ablate` CLI's
    /// non-causal path against silently diverging from the real forward pass.
    #[test]
    fn forward_with_mask_matches_forward_when_given_causal_mask() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut cfg = ModelConfig::small(50);
        cfg.hidden_size = 16;
        cfg.n_layers = 2;
        cfg.n_heads = 2;
        cfg.ffn_hidden = 32;
        cfg.seq_len = 10;
        let model = TinyGpt::new(&cfg, vb)?;
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device)?;
        let expected = model.forward(&ids)?;
        let mask = crate::attention::causal_mask(4, &device)?;
        let actual = model.forward_with_mask(&ids, &mask)?;
        let diff = (expected - actual)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "forward_with_mask(causal) should match forward exactly, diff={diff}");
        Ok(())
    }

    /// Full-model regression test for the KV-cache path: `forward_cached`,
    /// replayed across a prefill chunk plus several single-token steps at
    /// incrementing `position_offset`, must reproduce `forward`'s logits for
    /// the same sequence. This is the highest-risk composition in the
    /// KV-cache/generation feature — it's where `position_offset`-based
    /// position embeddings and the narrowed causal mask
    /// (`causal_mask(total_len).narrow(2, offset, t)`) actually get exercised
    /// together, and `CausalSelfAttention::forward_cached` alone (tested in
    /// attention.rs) doesn't cover either of those.
    #[test]
    fn forward_cached_matches_forward_full_recompute() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut cfg = ModelConfig::small(50);
        cfg.hidden_size = 16;
        cfg.n_layers = 2;
        cfg.n_heads = 2;
        cfg.ffn_hidden = 32;
        cfg.seq_len = 10;
        let model = TinyGpt::new(&cfg, vb)?;

        let ids_vec = vec![1u32, 2, 3, 4, 5, 6];
        let ids = Tensor::from_vec(ids_vec.clone(), (1, ids_vec.len()), &device)?;
        let expected = model.forward(&ids)?; // (1, 6, vocab)

        let mut caches: Vec<crate::kv_cache::KvCache> =
            (0..model.n_layers()).map(|_| crate::kv_cache::KvCache::new()).collect();

        // Prefill with the first 3 tokens at position_offset=0.
        let prefill_ids = Tensor::from_vec(ids_vec[..3].to_vec(), (1, 3), &device)?;
        let mut all_logits = vec![model.forward_cached(&prefill_ids, 0, &mut caches)?];

        // Feed the remaining tokens one at a time at the correct incrementing offset.
        for (i, &id) in ids_vec[3..].iter().enumerate() {
            let position_offset = 3 + i;
            let token = Tensor::from_vec(vec![id], (1, 1), &device)?;
            all_logits.push(model.forward_cached(&token, position_offset, &mut caches)?);
        }

        let cached_full = Tensor::cat(&all_logits, 1)?; // (1, 6, vocab)
        assert_eq!(cached_full.dims(), expected.dims());
        let diff = (&expected - &cached_full)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "forward_cached should match forward exactly, diff={diff}");
        Ok(())
    }

    /// Regression test for the actual-length-vs-configured-length distinction:
    /// `cfg.seq_len` is set much larger than the real input length, so if `forward`
    /// ever sized positions/mask off `cfg.seq_len` instead of the actual input
    /// length (e.g. via a cached mask/positions built at construction time), the
    /// output shape here would be wrong (or the call would panic/error).
    #[test]
    fn logits_shape_uses_actual_length_not_configured_seq_len() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut cfg = ModelConfig::small(50);
        cfg.hidden_size = 16;
        cfg.n_layers = 2;
        cfg.n_heads = 2;
        cfg.ffn_hidden = 32;
        cfg.seq_len = 10;
        let model = TinyGpt::new(&cfg, vb)?;
        let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device)?;
        let logits = model.forward(&ids)?;
        assert_eq!(logits.dims(), &[1, 3, 50]);
        Ok(())
    }
}
