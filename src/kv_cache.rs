use candle_core::{Result, Tensor};

/// Hand-written key/value cache for incremental (token-by-token) decoding.
///
/// Holds the accumulated key and value tensors of shape
/// `(batch, heads, seq_len_so_far, head_dim)`. Each call to [`KvCache::append`]
/// concatenates a new chunk onto the cached sequence dimension (dim `2`) and
/// returns the full accumulated tensors for use in attention. This is v1:
/// no RoPE/position tracking, just tensor accumulation.
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Appends a new (k, v) chunk of shape (batch, heads, chunk_len, head_dim)
    /// along the sequence dimension and returns the full accumulated tensors.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (full_k, full_v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => (
                Tensor::cat(&[prev_k, k], 2)?,
                Tensor::cat(&[prev_v, v], 2)?,
            ),
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(full_k.clone());
        self.v = Some(full_v.clone());
        Ok((full_k, full_v))
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn incremental_append_matches_full_concat() -> candle_core::Result<()> {
        let device = Device::Cpu;
        // (batch=1, heads=2, seq=1, head_dim=4) chunks appended one at a time
        let k1 = Tensor::randn(0f32, 1f32, (1, 2, 1, 4), &device)?;
        let k2 = Tensor::randn(0f32, 1f32, (1, 2, 1, 4), &device)?;
        let k3 = Tensor::randn(0f32, 1f32, (1, 2, 1, 4), &device)?;

        let mut cache = KvCache::new();
        cache.append(&k1, &k1)?;
        cache.append(&k2, &k2)?;
        let (acc_k, _) = cache.append(&k3, &k3)?;

        let expected = Tensor::cat(&[&k1, &k2, &k3], 2)?;
        let diff = (acc_k - expected)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }
}
