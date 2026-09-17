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
