use candle_core::{Result, Tensor, D};
use candle_nn::ops;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::Rng;

pub fn sample(
    logits: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    rng: &mut impl Rng,
) -> Result<u32> {
    let scaled = (logits / temperature)?;
    let probs = ops::softmax(&scaled, D::Minus1)?;
    let mut probs: Vec<f32> = probs.to_vec1()?;

    if let Some(k) = top_k {
        let mut sorted: Vec<f32> = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[k.saturating_sub(1).min(sorted.len() - 1)];
        for p in probs.iter_mut() {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    if let Some(p_thresh) = top_p {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0f32;
        let mut keep = vec![false; probs.len()];
        for (idx, p) in indexed {
            if cum >= p_thresh as f32 {
                break;
            }
            keep[idx] = true;
            cum += p;
        }
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            }
        }
    }

    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let dist = WeightedIndex::new(&probs).map_err(candle_core::Error::wrap)?;
    Ok(dist.sample(rng) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;

    #[test]
    fn temperature_near_zero_is_argmax() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1f32, 5f32, 2f32, 0f32], &device)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let id = sample(&logits, 0.01, None, None, &mut rng)?;
        assert_eq!(id, 1); // index of the max logit
        Ok(())
    }

    #[test]
    fn top_k_restricts_to_k_highest_logits() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1f32, 5f32, 2f32, 0f32, 9f32], &device)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for _ in 0..20 {
            let id = sample(&logits, 1.0, Some(2), None, &mut rng)?;
            assert!(id == 1 || id == 4, "top_k=2 should only ever pick indices 1 or 4, got {id}");
        }
        Ok(())
    }
}
