//! Traces a full decoder block (block_0) forward pass on real trained
//! weights: attention -> residual -> RMSNorm -> SwiGLU feed-forward ->
//! residual, printing the tensor shape and a value sample at every stage.
//! This is the "feed forwarding" + "dimensions" half of the pipeline.
use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;

const HIDDEN: usize = 384;
const N_HEADS: usize = 6;
const HEAD_DIM: usize = HIDDEN / N_HEADS;
const FFN_HIDDEN: usize = 1024;
const EPS: f64 = 1e-5;

fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&(mean_sq + EPS)?.sqrt()?)?;
    normed.broadcast_mul(weight)
}

fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.broadcast_matmul(&w.t()?)?.broadcast_add(b)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&candle_nn::ops::sigmoid(x)?)
}

fn stage(name: &str, t: &Tensor, preview: usize) -> Result<()> {
    let flat: Vec<f32> = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let sample: Vec<String> = flat.iter().take(preview).map(|v| format!("{v:.3}")).collect();
    println!("{:<42} shape={:<16?} first {preview}: [{}]", name, t.dims(), sample.join(", "));
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
    println!("Prompt: {prompt:?} -> token ids {ids:?}\n");

    let weights = candle_core::safetensors::load("model.safetensors", &device)?;
    let get = |name: &str| weights.get(name).unwrap_or_else(|| panic!("missing tensor {name}")).clone();

    // --- Embeddings ---
    let ids_t = Tensor::from_vec(ids.clone(), t, &device)?;
    let tok = get("token_emb.weight").index_select(&ids_t, 0)?;
    let positions = Tensor::arange(0u32, t as u32, &device)?;
    let pos = get("pos_emb.weight").index_select(&positions, 0)?;
    let x = (tok + pos)?;
    stage("1. token_emb + pos_emb", &x, 6)?;

    // --- Attention sub-layer (all 6 heads, full residual math) ---
    let xn = rms_norm(&x, &get("block_0.attn_norm.weight"))?;
    stage("2. RMSNorm(x)  [pre-attention]", &xn, 6)?;

    let q = linear(&xn, &get("block_0.attn.q_proj.weight"), &get("block_0.attn.q_proj.bias"))?;
    let k = linear(&xn, &get("block_0.attn.k_proj.weight"), &get("block_0.attn.k_proj.bias"))?;
    let v = linear(&xn, &get("block_0.attn.v_proj.weight"), &get("block_0.attn.v_proj.bias"))?;
    stage("3a. Q = xn @ Wq^T + bq", &q, 6)?;
    stage("3b. K = xn @ Wk^T + bk", &k, 6)?;
    stage("3c. V = xn @ Wv^T + bv", &v, 6)?;

    let q_h = q.reshape((t, N_HEADS, HEAD_DIM))?.transpose(0, 1)?.contiguous()?; // (heads, t, head_dim)
    let k_h = k.reshape((t, N_HEADS, HEAD_DIM))?.transpose(0, 1)?.contiguous()?;
    let v_h = v.reshape((t, N_HEADS, HEAD_DIM))?.transpose(0, 1)?.contiguous()?;
    stage("3d. reshape+transpose -> (heads, t, head_dim)", &q_h, 6)?;

    let scale = 1f64 / (HEAD_DIM as f64).sqrt();
    let scores = (q_h.matmul(&k_h.transpose(1, 2)?.contiguous()?)? * scale)?; // (heads, t, t)
    let mask_vals: Vec<f32> = (0..t).flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })).collect();
    let mask = Tensor::from_vec(mask_vals, (1, t, t), &device)?;
    let scores = scores.broadcast_add(&mask)?;
    let weights_mat = candle_nn::ops::softmax(&scores, D::Minus1)?;
    stage("3e. softmax(QK^T/sqrt(d) + mask)", &weights_mat, 6)?;

    let attn_ctx = weights_mat.matmul(&v_h)?; // (heads, t, head_dim)
    let attn_ctx = attn_ctx.transpose(0, 1)?.reshape((t, HIDDEN))?.contiguous()?; // merge heads -> (t, hidden)
    stage("3f. weights @ V, merge heads -> (t, hidden)", &attn_ctx, 6)?;

    let attn_out = linear(&attn_ctx, &get("block_0.attn.o_proj.weight"), &get("block_0.attn.o_proj.bias"))?;
    stage("3g. attn_out = attn_ctx @ Wo^T + bo", &attn_out, 6)?;

    let x2 = (&x + &attn_out)?; // residual 1
    stage("4. residual: x2 = x + attn_out", &x2, 6)?;

    // --- Feed-forward sub-layer (SwiGLU) ---
    println!("\n---------------- SwiGLU feed-forward ({HIDDEN} -> {FFN_HIDDEN} -> {HIDDEN}) ----------------");
    let xn2 = rms_norm(&x2, &get("block_0.mlp_norm.weight"))?;
    stage("5. RMSNorm(x2)  [pre-MLP]", &xn2, 6)?;

    let gate = linear(&xn2, &get("block_0.mlp.gate_proj.weight"), &get("block_0.mlp.gate_proj.bias"))?;
    stage("6a. gate = xn2 @ Wgate^T + bgate", &gate, 6)?;

    let up = linear(&xn2, &get("block_0.mlp.up_proj.weight"), &get("block_0.mlp.up_proj.bias"))?;
    stage("6b. up = xn2 @ Wup^T + bup", &up, 6)?;

    let gate_act = silu(&gate)?;
    stage("6c. SiLU(gate) = gate * sigmoid(gate)", &gate_act, 6)?;

    let gated = (gate_act * up)?;
    stage("6d. SwiGLU inner = SiLU(gate) * up", &gated, 6)?;

    let mlp_out = linear(&gated, &get("block_0.mlp.down_proj.weight"), &get("block_0.mlp.down_proj.bias"))?;
    stage("6e. mlp_out = gated @ Wdown^T + bdown", &mlp_out, 6)?;

    let x3 = (&x2 + &mlp_out)?; // residual 2
    stage("7. residual: block_0 output = x2 + mlp_out", &x3, 6)?;

    println!("\nFull dimension flow through block_0:");
    println!("  ({t}, {HIDDEN}) --attn_norm--> ({t}, {HIDDEN}) --Q/K/V--> 3x ({t}, {HIDDEN})");
    println!("  --split heads--> ({N_HEADS}, {t}, {HEAD_DIM}) --attention--> ({N_HEADS}, {t}, {HEAD_DIM})");
    println!("  --merge heads--> ({t}, {HIDDEN}) --o_proj--> ({t}, {HIDDEN}) --residual--> ({t}, {HIDDEN})");
    println!("  --mlp_norm--> ({t}, {HIDDEN}) --gate/up--> 2x ({t}, {FFN_HIDDEN}) --down--> ({t}, {HIDDEN}) --residual--> ({t}, {HIDDEN})");

    Ok(())
}
