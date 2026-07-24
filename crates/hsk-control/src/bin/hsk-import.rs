use std::{env, ffi::OsString, fs, path::PathBuf};

use hsk_control::{Delimiter, generate_hsk_artifact, parse_import_metadata};

fn main() {
    if let Err(error) = run() {
        eprintln!("hsk-import: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse(env::args_os().skip(1))?;
    let source = fs::read(&arguments.source)?;
    let metadata = parse_import_metadata(&fs::read(&arguments.metadata)?)?;
    let generated = generate_hsk_artifact(&source, &metadata, arguments.delimiter)?;
    fs::write(&arguments.output, generated)?;
    println!(
        "generated {} from audited source {}",
        arguments.output.display(),
        arguments.source.display()
    );
    Ok(())
}

struct Arguments {
    source: PathBuf,
    metadata: PathBuf,
    output: PathBuf,
    delimiter: Delimiter,
}

impl Arguments {
    fn parse(arguments: impl Iterator<Item = OsString>) -> Result<Self, String> {
        let mut source = None;
        let mut metadata = None;
        let mut output = None;
        let mut delimiter = Delimiter::Tab;
        let mut arguments = arguments;

        while let Some(argument) = arguments.next() {
            let flag = argument
                .to_str()
                .ok_or_else(|| "arguments must be valid Unicode".to_owned())?;
            match flag {
                "--source" => source = Some(next_path(&mut arguments, flag)?),
                "--metadata" => metadata = Some(next_path(&mut arguments, flag)?),
                "--output" => output = Some(next_path(&mut arguments, flag)?),
                "--delimiter" => {
                    let value = arguments
                        .next()
                        .and_then(|value| value.into_string().ok())
                        .ok_or_else(|| "--delimiter requires tab or comma".to_owned())?;
                    delimiter = match value.as_str() {
                        "tab" => Delimiter::Tab,
                        "comma" => Delimiter::Comma,
                        _ => return Err("--delimiter requires tab or comma".into()),
                    };
                }
                "--help" | "-h" => {
                    return Err(
                        "usage: hsk-import --source PATH --metadata PATH --output PATH [--delimiter tab|comma]"
                            .into(),
                    );
                }
                _ => return Err(format!("unknown argument {flag:?}")),
            }
        }

        Ok(Self {
            source: source.ok_or_else(|| "--source is required".to_owned())?,
            metadata: metadata.ok_or_else(|| "--metadata is required".to_owned())?,
            output: output.ok_or_else(|| "--output is required".to_owned())?,
            delimiter,
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
