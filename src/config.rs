#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub seq_len: usize,
    pub ffn_hidden: usize,
    pub eps: f64,
}

impl ModelConfig {
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 384,
            n_layers: 6,
            n_heads: 6,
            seq_len: 256,
            ffn_hidden: 1024,
            eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_config_head_dim_divides_evenly() {
        let cfg = ModelConfig::small(65);
        assert_eq!(cfg.hidden_size % cfg.n_heads, 0);
    }
}
