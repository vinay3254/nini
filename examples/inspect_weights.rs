use candle_core::{Device, Tensor};

fn stats(t: &Tensor) -> candle_core::Result<(f32, f32, f32, f32)> {
    let flat = t.flatten_all()?.to_dtype(candle_core::DType::F32)?;
    let mean = flat.mean_all()?.to_scalar::<f32>()?;
    let sq = flat.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let std = (sq - mean * mean).max(0.0).sqrt();
    let min = flat.min(0)?.to_scalar::<f32>()?;
    let max = flat.max(0)?.to_scalar::<f32>()?;
    Ok((mean, std, min, max))
}

fn main() -> candle_core::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "model.safetensors".to_string());
    let device = Device::Cpu;
    let tensors = candle_core::safetensors::load(&path, &device)?;

    let mut names: Vec<_> = tensors.keys().cloned().collect();
    names.sort();

    println!("{:<28} {:<16} {:>10} {:>10} {:>10} {:>10}", "name", "shape", "mean", "std", "min", "max");
    for name in names {
        let t = &tensors[&name];
        let (mean, std, min, max) = stats(t)?;
        println!(
            "{:<28} {:<16?} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            name,
            t.dims(),
            mean,
            std,
            min,
            max
        );
    }
    Ok(())
}
