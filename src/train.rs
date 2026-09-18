use candle_core::backprop::GradStore;
use candle_core::{Device, Result, Tensor, Var};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarMap};
use rand::SeedableRng;

use crate::config::ModelConfig;
use crate::data::Dataset;
use crate::model::TinyGpt;

/// Hyper-parameters for the training loop.
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub warmup_steps: usize,
    pub grad_clip: f64,
}

/// Linear warmup followed by cosine decay, both scaled by `cfg.lr`.
fn cosine_lr(step: usize, cfg: &TrainConfig) -> f64 {
    if step < cfg.warmup_steps {
        return cfg.lr * (step as f64 + 1.0) / cfg.warmup_steps as f64;
    }
    let progress = (step - cfg.warmup_steps) as f64 / (cfg.steps - cfg.warmup_steps).max(1) as f64;
    let progress = progress.min(1.0);
    cfg.lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Global L2 norm over all parameter gradients present in `grads`.
///
/// Accumulates the sum-of-squares as a `Tensor` across all vars (keeping the
/// reduction on-device) and only calls `.to_scalar::<f32>()` once at the end,
/// rather than once per parameter tensor — with ~60 parameter tensors in the
/// default model config, a per-tensor `.to_scalar()` call is a GPU->CPU sync
/// that stalls the CUDA pipeline once per tensor per step.
fn grad_total_norm(grads: &GradStore, vars: &[Var]) -> Result<f64> {
    let mut sum_sq: Option<Tensor> = None;
    for var in vars {
        if let Some(g) = grads.get(var) {
            let s = g.sqr()?.sum_all()?;
            sum_sq = Some(match sum_sq {
                Some(acc) => (acc + s)?,
                None => s,
            });
        }
    }
    let total = match sum_sq {
        Some(t) => t.to_scalar::<f32>()? as f64,
        None => 0.0,
    };
    Ok(total.sqrt())
}

/// Train `model` in place on `dataset`, returning the loss at every step.
///
/// Uses AdamW with a linear-warmup + cosine-decay learning-rate schedule and
/// global-norm gradient clipping.
pub fn train(
    model: &TinyGpt,
    varmap: &VarMap,
    dataset: &Dataset,
    model_cfg: &ModelConfig,
    train_cfg: &TrainConfig,
    device: &Device,
) -> Result<Vec<f32>> {
    let params = ParamsAdamW { lr: train_cfg.lr, ..Default::default() };
    let vars = varmap.all_vars();
    let mut opt = AdamW::new(vars.clone(), params)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    let mut losses = Vec::with_capacity(train_cfg.steps);

    for step in 0..train_cfg.steps {
        let (inputs, targets) =
            dataset.random_batch(train_cfg.batch_size, model_cfg.seq_len, device, &mut rng)?;
        let logits = model.forward(&inputs)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let targets_flat = targets.reshape((b * t,))?;
        let loss_val = loss::cross_entropy(&logits_flat, &targets_flat)?;

        opt.set_learning_rate(cosine_lr(step, train_cfg));

        let mut grads = loss_val.backward()?;
        let total_norm = grad_total_norm(&grads, &vars)?;
        if train_cfg.grad_clip > 0.0 && total_norm > train_cfg.grad_clip {
            let scale = train_cfg.grad_clip / total_norm;
            for var in vars.iter() {
                let scaled = match grads.get(var) {
                    Some(g) => (g * scale)?,
                    None => continue,
                };
                grads.insert(var, scaled);
            }
        }
        opt.step(&grads)?;

        losses.push(loss_val.to_scalar::<f32>()?);
    }
    Ok(losses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use crate::config::ModelConfig;
    use crate::data::Dataset;
    use crate::model::TinyGpt;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn cosine_lr_warms_up_then_decays() {
        let cfg = TrainConfig { steps: 100, batch_size: 1, lr: 1.0, warmup_steps: 10, grad_clip: 1.0 };
        assert_eq!(cosine_lr(0, &cfg), 0.1);
        assert_eq!(cosine_lr(9, &cfg), 1.0);
        assert_eq!(cosine_lr(10, &cfg), 1.0);
        for step in 11..100 {
            assert!(cosine_lr(step, &cfg) < cosine_lr(step - 1, &cfg), "lr must decay at step {step}");
        }
        assert!(cosine_lr(99, &cfg) < 0.01);
    }

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
