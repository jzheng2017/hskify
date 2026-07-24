use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use browser_companion::daemon::{DaemonExit, DaemonOptions};

#[tokio::main]
async fn main() {
    match parse_options(std::env::args_os().skip(1).collect()) {
        Ok(options) => match browser_companion::daemon::run_daemon(options).await {
            Ok(DaemonExit::Idle | DaemonExit::AlreadyRunning) => {}
            Err(error) => {
                eprintln!("hsk-manga-browser-daemon: {error}");
                std::process::exit(1);
            }
        },
        Err(message) => {
            eprintln!("hsk-manga-browser-daemon: {message}");
            std::process::exit(2);
        }
    }
}

fn parse_options(arguments: Vec<OsString>) -> Result<DaemonOptions, String> {
    let mut state_dir = None;
    let mut idle_timeout = Duration::from_secs(10 * 60);
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--state-dir") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "--state-dir requires a path".to_owned())?;
                state_dir = Some(PathBuf::from(value));
            }
            Some("--idle-milliseconds") => {
                idle_timeout = parse_duration(
                    arguments.next(),
                    "--idle-milliseconds requires a positive integer",
                )?;
            }
            _ => return Err("unknown command-line argument".to_owned()),
        }
    }
    let state_dir = match state_dir {
        Some(path) => path,
        None => {
            browser_companion::discovery::default_state_dir().map_err(|error| error.to_string())?
        }
    };
    Ok(DaemonOptions {
        state_dir,
        idle_timeout,
    })
}

fn parse_duration(value: Option<OsString>, message: &str) -> Result<Duration, String> {
    let milliseconds = value
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| message.to_owned())?;
    Ok(Duration::from_millis(milliseconds))
}
