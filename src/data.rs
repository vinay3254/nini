use candle_core::{Device, Result, Tensor};
use rand::Rng;

use crate::tokenizer::Tokenizer;

pub struct Dataset {
    ids: Vec<u32>,
}

impl Dataset {
    pub fn from_text(text: &str, tokenizer: &Tokenizer) -> Self {
        Self {
            ids: tokenizer.encode(text),
        }
    }

    pub fn random_batch(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, Tensor)> {
        let max_start = self.ids.len() - seq_len - 1;
        let mut input_rows = Vec::with_capacity(batch_size * seq_len);
        let mut target_rows = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            let random_num: u32 = rng.next_u32();
            let start = (random_num as usize) % (max_start + 1);
            input_rows.extend_from_slice(&self.ids[start..start + seq_len]);
            target_rows.extend_from_slice(&self.ids[start + 1..start + seq_len + 1]);
        }
        let inputs = Tensor::from_vec(input_rows, (batch_size, seq_len), device)?;
        let targets = Tensor::from_vec(target_rows, (batch_size, seq_len), device)?;
        Ok((inputs, targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use crate::tokenizer::Tokenizer;
    use rand::SeedableRng;

    #[test]
    fn batch_has_expected_shape() -> candle_core::Result<()> {
        let text = "abcdefghijklmnopqrstuvwxyz".repeat(10);
        let tok = Tokenizer::from_corpus(&text);
        let ds = Dataset::from_text(&text, &tok);
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (inputs, targets) = ds.random_batch(4, 8, &device, &mut rng)?;
        assert_eq!(inputs.dims(), &[4, 8]);
        assert_eq!(targets.dims(), &[4, 8]);
        Ok(())
    }

    #[test]
    fn targets_are_inputs_shifted_by_one() -> candle_core::Result<()> {
        let text = "abcdefghijklmnopqrstuvwxyz".repeat(10);
        let tok = Tokenizer::from_corpus(&text);
        let ds = Dataset::from_text(&text, &tok);
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (inputs, targets) = ds.random_batch(1, 8, &device, &mut rng)?;
        let inputs: Vec<u32> = inputs.flatten_all()?.to_vec1()?;
        let targets: Vec<u32> = targets.flatten_all()?.to_vec1()?;
        assert_eq!(&inputs[1..], &targets[..targets.len() - 1]);
        Ok(())
    }
}
