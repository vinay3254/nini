//! Visualizes backpropagation + gradient descent + the cosine LR schedule
//! by training a small fresh model for a handful of steps on CPU and
//! printing, per step: loss, learning rate (linear warmup -> cosine decay),
//! and the global L2 gradient norm before/after clipping.
use candle_core::{DType, Device, Result};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::SeedableRng;

use tinygpt_rs::config::ModelConfig;
use tinygpt_rs::data::Dataset;
use tinygpt_rs::model::TinyGpt;
use tinygpt_rs::tokenizer::Tokenizer;

const STEPS: usize = 40;
const WARMUP: usize = 8;
const PEAK_LR: f64 = 3e-3;
const GRAD_CLIP: f64 = 1.0;

fn cosine_lr(step: usize) -> f64 {
    if step < WARMUP {
        return PEAK_LR * (step as f64 + 1.0) / WARMUP as f64;
    }
    let progress = ((step - WARMUP) as f64 / (STEPS - WARMUP).max(1) as f64).min(1.0);
    PEAK_LR * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn bar(value: f64, max: f64, width: usize) -> String {
    let filled = ((value / max).clamp(0.0, 1.0) * width as f64).round() as usize;
    format!("[{}{}]", "#".repeat(filled), " ".repeat(width - filled))
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let text = std::fs::read_to_string("data/tinyshakespeare.txt").expect("read corpus");
    let tok = Tokenizer::from_corpus(&text);
    let dataset = Dataset::from_text(&text, &tok);

    // Small config so backprop is fast enough to run live for a demo.
    let mut cfg = ModelConfig::small(tok.vocab_size());
    cfg.hidden_size = 64;
    cfg.n_layers = 4;
    cfg.n_heads = 4;
    cfg.ffn_hidden = 128;
    cfg.seq_len = 64;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TinyGpt::new(&cfg, vb)?;

    let vars = varmap.all_vars();
    let n_params: usize = vars.iter().map(|v| v.elem_count()).sum();
    println!("Model: {} layers, hidden={}, {} trainable tensors, {} total params\n", cfg.n_layers, cfg.hidden_size, vars.len(), n_params);

    let params = ParamsAdamW { lr: PEAK_LR, ..Default::default() };
    let mut opt = AdamW::new(vars.clone(), params)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    println!(
        "{:>4} {:>10} {:>10} {:>12} {:>12}  gradient norm",
        "step", "loss", "lr", "grad_norm", "clipped_to"
    );

    for step in 0..STEPS {
        let (inputs, targets) = dataset.random_batch(8, cfg.seq_len, &device, &mut rng)?;
        let logits = model.forward(&inputs)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let targets_flat = targets.reshape((b * t,))?;
        let loss_val = loss::cross_entropy(&logits_flat, &targets_flat)?;
        let loss_scalar = loss_val.to_scalar::<f32>()?;

        let lr = cosine_lr(step);
        opt.set_learning_rate(lr);

        // --- Backpropagation: compute d(loss)/d(every parameter) ---
        let mut grads = loss_val.backward()?;

        // --- Gradient descent: global-norm clip, then AdamW step ---
        let mut sum_sq = 0f64;
        for var in &vars {
            if let Some(g) = grads.get(var) {
                sum_sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            }
        }
        let total_norm = sum_sq.sqrt();
        let clipped_norm = if total_norm > GRAD_CLIP {
            let scale = GRAD_CLIP / total_norm;
            for var in &vars {
                if let Some(g) = grads.get(var) {
                    let scaled = (g * scale)?;
                    grads.insert(var, scaled);
                }
            }
            GRAD_CLIP
        } else {
            total_norm
        };
        opt.step(&grads)?;

        println!(
            "{:>4} {:>10.4} {:>10.6} {:>12.4} {:>12.4}  {}",
            step, loss_scalar, lr, total_norm, clipped_norm,
            bar(lr, PEAK_LR, 24)
        );
    }

    println!("\nLR schedule shape (warmup={WARMUP} steps, then cosine decay to {STEPS}):");
    for step in (0..STEPS).step_by(2) {
        println!("{:>4}  {}", step, bar(cosine_lr(step), PEAK_LR, 40));
    }

    Ok(())
}
