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

    /// Single-token (or short chunk) incremental forward pass using a KV-cache.
    /// See `CausalSelfAttention::forward_cached` for the cache contract.
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
