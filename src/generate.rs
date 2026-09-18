use candle_core::{bail, Device, Result, Tensor};

use crate::kv_cache::KvCache;
use crate::model::TinyGpt;
use crate::sampling::sample;
use crate::tokenizer::Tokenizer;

/// Autoregressively generates `max_new_tokens` tokens after `prompt`, using a
/// per-layer KV-cache so each new token only recomputes attention for itself
/// against the already-cached prefix. Returns the decoded prompt + continuation.
#[allow(clippy::too_many_arguments)] // signature is the fixed task interface contract
pub fn generate(
    model: &TinyGpt,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    device: &Device,
) -> Result<String> {
    let n_layers = model.n_layers();
    let mut caches: Vec<KvCache> = (0..n_layers).map(|_| KvCache::new()).collect();
    let mut rng = rand::rng();

    let prompt_ids = tokenizer.encode(prompt);
    if prompt_ids.is_empty() {
        // `Tokenizer::encode` silently drops chars outside its vocab, so an
        // empty or all-unknown-char prompt encodes to zero tokens. Reject it
        // explicitly rather than feeding a zero-length sequence through the
        // model, which would produce zero-width logits and underflow the
        // `dim(1) - 1` indexing below on the first sampling step.
        bail!("generate: prompt encoded to zero tokens (empty or entirely out-of-vocab)");
    }
    let mut all_ids = prompt_ids.clone();

    // Feed the whole prompt through the cache first.
    let prompt_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), device)?;
    let mut logits = model.forward_cached(&prompt_tensor, 0, &mut caches)?;

    for pos in (prompt_ids.len()..).take(max_new_tokens) {
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.flatten_all()?;
        let next_id = sample(&last_logits, temperature, top_k, top_p, &mut rng)?;
        all_ids.push(next_id);
        let next_tensor = Tensor::from_vec(vec![next_id], (1, 1), device)?;
        logits = model.forward_cached(&next_tensor, pos, &mut caches)?;
    }

    Ok(tokenizer.decode(&all_ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarBuilder;
    use crate::config::ModelConfig;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn generate_produces_requested_number_of_new_chars() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let text = "abcdefghij".repeat(20);
        let tok = Tokenizer::from_corpus(&text);
        let mut cfg = ModelConfig::small(tok.vocab_size());
        cfg.hidden_size = 16;
        cfg.n_layers = 2;
        cfg.n_heads = 2;
        cfg.ffn_hidden = 32;
        cfg.seq_len = 32;

        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = crate::model::TinyGpt::new(&cfg, vb)?;

        let out = generate(&model, &tok, "abc", 10, 1.0, Some(5), None, &device)?;
        assert_eq!(out.chars().count(), 3 + 10);
        Ok(())
    }

    /// Regression test: a prompt that encodes to zero tokens (empty, or made
    /// entirely of characters outside the tokenizer's vocab) must return a
    /// clean error instead of panicking/underflowing on `dim(1) - 1` indexing
    /// during the first sampling step.
    #[test]
    fn generate_errors_cleanly_on_empty_prompt() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let text = "abcdefghij".repeat(20);
        let tok = Tokenizer::from_corpus(&text);
        let mut cfg = ModelConfig::small(tok.vocab_size());
        cfg.hidden_size = 16;
        cfg.n_layers = 2;
        cfg.n_heads = 2;
        cfg.ffn_hidden = 32;
        cfg.seq_len = 32;

        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = crate::model::TinyGpt::new(&cfg, vb)?;

        // "" is empty; "XYZ" (uppercase chars entirely outside this
        // lowercase corpus's vocab) both encode to zero tokens via
        // Tokenizer::encode.
        assert!(generate(&model, &tok, "", 5, 1.0, None, None, &device).is_err());
        assert!(generate(&model, &tok, "XYZ", 5, 1.0, None, None, &device).is_err());
        Ok(())
    }
}
