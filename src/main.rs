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
        #[arg(long, default_value_t = 24)]
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
            let start = std::time::Instant::now();
            let losses = train(&model, &varmap, &dataset, &model_cfg, &train_cfg, &device)?;
            let elapsed = start.elapsed();

            let mut f = std::fs::File::create(&loss_out)?;
            writeln!(f, "step,loss")?;
            for (i, l) in losses.iter().enumerate() {
                writeln!(f, "{i},{l}")?;
            }
            varmap.save("model.safetensors")?;
            let tokens_per_sec = batch_size as f64 * model_cfg.seq_len as f64 * steps as f64 / elapsed.as_secs_f64();
            println!("Trained {steps} steps. Final loss: {:.4}. Weights: model.safetensors. Loss log: {loss_out}", losses.last().unwrap());
            println!("Elapsed: {:.1}s. Throughput: {:.0} tokens/sec.", elapsed.as_secs_f64(), tokens_per_sec);
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
