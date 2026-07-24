use std::{env, ffi::OsString, fs, path::PathBuf};

use hsk_control::TextNormalizer;

fn main() {
    if let Err(error) = run() {
        eprintln!("hsk-normalize: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse(env::args_os().skip(1))?;
    let source = fs::read_to_string(&arguments.source)?;
    let normalizer = TextNormalizer::new();
    let mut output = String::with_capacity(source.len());

    for line in source.lines() {
        output.push_str(&normalizer.normalize(line));
        output.push('\n');
    }

    fs::write(&arguments.output, output)?;
    Ok(())
}

struct Arguments {
    source: PathBuf,
    output: PathBuf,
}

impl Arguments {
    fn parse(arguments: impl Iterator<Item = OsString>) -> Result<Self, String> {
        let mut source = None;
        let mut output = None;
        let mut arguments = arguments;

        while let Some(argument) = arguments.next() {
            let flag = argument
                .to_str()
                .ok_or_else(|| "arguments must be valid Unicode".to_owned())?;
            match flag {
                "--source" => source = Some(next_path(&mut arguments, flag)?),
                "--output" => output = Some(next_path(&mut arguments, flag)?),
                "--help" | "-h" => {
                    return Err("usage: hsk-normalize --source PATH --output PATH".into());
                }
                _ => return Err(format!("unknown argument {flag:?}")),
            }
        }

        Ok(Self {
            source: source.ok_or_else(|| "--source is required".to_owned())?,
            output: output.ok_or_else(|| "--output is required".to_owned())?,
        })
    }
}

fn next_path(
    arguments: &mut impl Iterator<Item = OsString>,
    flag: &str,
) -> Result<PathBuf, String> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} requires a path"))
}
