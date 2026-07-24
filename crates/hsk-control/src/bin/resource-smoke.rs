use std::{env, fs, time::Instant};

use hsk_control::{HskControl, HskLevel};

fn main() {
    if let Err(error) = run() {
        eprintln!("resource-smoke: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let hsk_path = arguments
        .next()
        .ok_or("usage: resource-smoke HSK_JSON DICTIONARY_JSON [ITERATIONS]")?;
    let dictionary_path = arguments
        .next()
        .ok_or("usage: resource-smoke HSK_JSON DICTIONARY_JSON [ITERATIONS]")?;
    let iterations = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(10_000);

    let load_started = Instant::now();
    let control = HskControl::from_json(
        &fs::read_to_string(hsk_path)?,
        &fs::read_to_string(dictionary_path)?,
    )?;
    let load_elapsed = load_started.elapsed();

    let run_started = Instant::now();
    for _ in 0..iterations {
        let _ = control.validate("我们现在要离开。", HskLevel::SIX, &[]);
        let _ = control.lookup("我们现在要离开。", &[]);
    }
    let run_elapsed = run_started.elapsed();

    println!(
        "loaded {} HSK and {} dictionary entries in {:?}; {} validate+lookup iterations in {:?}; revision {}",
        control.hsk_dataset().entries().len(),
        control.dictionary().entries().len(),
        load_elapsed,
        iterations,
        run_elapsed,
        control.cache_revision()
    );
    Ok(())
}
