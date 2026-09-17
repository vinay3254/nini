use candle_core::{Result, Tensor, D};
use candle_nn::{Init, Module, VarBuilder};

/// Root Mean Square Layer Normalization.
///
/// Normalizes the last dimension of the input by its root-mean-square,
/// then scales elementwise by a learned weight vector. Unlike LayerNorm,
/// there is no mean-centering step and no bias term.
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
