//! Computes cosine similarity between learned character-embedding vectors
//! (token_emb.weight rows) from the trained model, to visualize what the
//! model learned about which characters are "similar" in embedding space.
use candle_core::{DType, Device, Result};
use std::collections::HashMap;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-8)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let data = std::fs::read_to_string("data/tinyshakespeare.txt").expect("read corpus");
    let mut chars: Vec<char> = data.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
    chars.sort();
    let stoi: HashMap<char, u32> = chars.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();

    let weights = candle_core::safetensors::load("model.safetensors", &device)?;
    let token_emb = weights.get("token_emb.weight").expect("token_emb.weight").to_dtype(DType::F32)?;
    let rows: Vec<Vec<f32>> = token_emb.to_vec2()?;

    // Pick a readable, meaningful subset of characters to compare.
    let sample: Vec<char> = "RJMEOabcdeqz .,!?:;\n".chars().filter(|c| stoi.contains_key(c)).collect();
    let mut seen = std::collections::HashSet::new();
    let sample: Vec<char> = sample.into_iter().filter(|c| seen.insert(*c)).collect();

    let label = |c: char| match c {
        '\n' => "\\n".to_string(),
        ' ' => "spc".to_string(),
        other => other.to_string(),
    };

    print!("{:>6}", "");
    for &c in &sample {
        print!("{:>6}", label(c));
    }
    println!();

    for &r in &sample {
        print!("{:>6}", label(r));
        let rv = &rows[stoi[&r] as usize];
        for &c in &sample {
            let cv = &rows[stoi[&c] as usize];
            let sim = cosine_sim(rv, cv);
            print!("{sim:>6.2}");
        }
        println!();
    }

    // Rank the most-similar *other* character for each sampled character.
    println!("\nNearest neighbor (by cosine similarity) for each character:");
    let all_ids: Vec<(char, u32)> = chars.iter().map(|&c| (c, stoi[&c])).collect();
    for &c in &sample {
        let cv = &rows[stoi[&c] as usize];
        let mut best: Option<(char, f32)> = None;
        for &(other, id) in &all_ids {
            if other == c {
                continue;
            }
            let sim = cosine_sim(cv, &rows[id as usize]);
            if best.is_none_or(|(_, best_sim)| sim > best_sim) {
                best = Some((other, sim));
            }
        }
        if let Some((other, sim)) = best {
            println!("  {:>4}  ->  {:>4}   cos_sim = {:.3}", label(c), label(other), sim);
        }
    }

    Ok(())
}
