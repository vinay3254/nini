//! Traces real matrix multiplications inside block_0's causal self-attention,
//! using the actual trained weights in model.safetensors. Prints every stage
//! of softmax((Q K^T) / sqrt(head_dim) + causal_mask) @ V for a short prompt,
//! one head at a time, so the numbers are small enough to read on screen.
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use std::collections::HashMap;

const HIDDEN: usize = 384;
const N_HEADS: usize = 6;
const HEAD_DIM: usize = HIDDEN / N_HEADS;
const EPS: f64 = 1e-5;

fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&(mean_sq + EPS)?.sqrt()?)?;
    normed.broadcast_mul(weight)
}

fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    // w is stored (out, in) like candle_nn::Linear, so x @ w^T + b.
    x.broadcast_matmul(&w.t()?)?.broadcast_add(b)
}

fn print_matrix(name: &str, t: &Tensor, chars: &[char]) -> Result<()> {
    let t = t.to_dtype(DType::F32)?;
    println!("\n{name}  shape={:?}", t.dims());
    let rows: Vec<Vec<f32>> = t.to_vec2()?;
    print!("        ");
    for c in chars {
        print!("{c:>8}");
    }
    println!();
    for (i, row) in rows.iter().enumerate() {
        print!("{:>7} ", chars.get(i).copied().unwrap_or('?'));
        for v in row {
            print!("{v:>8.3}");
        }
        println!();
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let data = std::fs::read_to_string("data/tinyshakespeare.txt").expect("read corpus");
    let mut chars: Vec<char> = data.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
    chars.sort();
    let stoi: HashMap<char, u32> = chars.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();

    let prompt = "ROMEO:";
    let ids: Vec<u32> = prompt.chars().map(|c| stoi[&c]).collect();
    let t = ids.len();
    println!("Prompt: {prompt:?} -> token ids {ids:?}");

    let weights = candle_core::safetensors::load("model.safetensors", &device)?;
    let get = |name: &str| weights.get(name).unwrap_or_else(|| panic!("missing tensor {name}")).clone();

    // --- Embeddings ---
    let token_emb = get("token_emb.weight"); // (vocab, hidden)
    let pos_emb = get("pos_emb.weight");
    let ids_t = Tensor::from_vec(ids.clone(), t, &device)?;
    let tok = token_emb.index_select(&ids_t, 0)?; // (t, hidden)
    let positions = Tensor::arange(0u32, t as u32, &device)?;
    let pos = pos_emb.index_select(&positions, 0)?;
    let x = (tok + pos)?; // (t, hidden)

    // --- Pre-attention RMSNorm ---
    let attn_norm_w = get("block_0.attn_norm.weight");
    let xn = rms_norm(&x, &attn_norm_w)?;

    // --- Q, K, V projections (full hidden size) ---
    let q = linear(&xn, &get("block_0.attn.q_proj.weight"), &get("block_0.attn.q_proj.bias"))?;
    let k = linear(&xn, &get("block_0.attn.k_proj.weight"), &get("block_0.attn.k_proj.bias"))?;
    let v = linear(&xn, &get("block_0.attn.v_proj.weight"), &get("block_0.attn.v_proj.bias"))?;

    // --- Split into heads, keep only head 0 for a readable trace ---
    let head = 0usize;
    let q_h = q.reshape((t, N_HEADS, HEAD_DIM))?.i((.., head, ..))?; // (t, head_dim)
    let k_h = k.reshape((t, N_HEADS, HEAD_DIM))?.i((.., head, ..))?;
    let v_h = v.reshape((t, N_HEADS, HEAD_DIM))?.i((.., head, ..))?;

    let chars_labels: Vec<char> = prompt.chars().collect();

    println!("\n================ MATRIX MULTIPLICATION: Q @ K^T ================");
    println!("Q shape (t, head_dim) = ({t}, {HEAD_DIM}), K^T shape (head_dim, t) = ({HEAD_DIM}, {t})");
    let scale = 1f64 / (HEAD_DIM as f64).sqrt();
    let raw_scores = q_h.matmul(&k_h.t()?)?; // (t, t)
    print_matrix("Raw scores  Q @ K^T", &raw_scores, &chars_labels)?;

    let scaled = (raw_scores * scale)?;
    print_matrix(&format!("Scaled scores  (Q @ K^T) / sqrt({HEAD_DIM})"), &scaled, &chars_labels)?;

    // --- Causal mask ---
    let mask_vals: Vec<f32> = (0..t).flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })).collect();
    let mask = Tensor::from_vec(mask_vals, (t, t), &device)?;
    print_matrix("Causal mask (0 = visible, -inf = blocked)", &mask, &chars_labels)?;

    let masked = (scaled + mask)?;
    print_matrix("Masked scores", &masked, &chars_labels)?;

    println!("\n================ SOFTMAX -> ATTENTION WEIGHTS ================");
    let weights_mat = candle_nn::ops::softmax(&masked, D::Minus1)?;
    print_matrix("softmax(masked scores)  <- each row sums to 1.0", &weights_mat, &chars_labels)?;

    println!("\n================ MATRIX MULTIPLICATION: weights @ V ================");
    println!("weights shape (t, t) = ({t}, {t}), V shape (t, head_dim) = ({t}, {HEAD_DIM})");
    let out = weights_mat.matmul(&v_h)?; // (t, head_dim)
    println!("\nOutput  shape={:?}  (context vector per token, head {head} only, first 8 dims shown)", out.dims());
    let out_rows: Vec<Vec<f32>> = out.narrow(1, 0, 8)?.to_vec2()?;
    for (i, row) in out_rows.iter().enumerate() {
        print!("{:>7} ", chars_labels[i]);
        for val in row {
            print!("{val:>8.3}");
        }
        println!();
    }

    println!("\nRow sums of the attention-weight matrix (should all be 1.0):");
    let sums: Vec<f32> = weights_mat.sum(D::Minus1)?.to_vec1()?;
    for (c, s) in chars_labels.iter().zip(sums) {
        println!("  {c}: {s:.6}");
    }

    Ok(())
}
