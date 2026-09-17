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
