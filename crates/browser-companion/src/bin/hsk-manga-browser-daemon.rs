use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use browser_companion::daemon::{DaemonExit, DaemonOptions};

const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(2 * 60);
const INFERENCE_THREADS_ENV: &str = "KOHARU_INFERENCE_THREADS";

fn main() {
    configure_process_resources();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(4)
        .enable_all()
        .build()
        .expect("build bounded browser companion runtime");
    runtime.block_on(run());
}

async fn run() {
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

#[cfg(windows)]
fn configure_process_resources() {
    use windows_sys::Win32::System::Threading::{
        BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass, SetProcessAffinityMask,
    };

    unsafe {
        let process = GetCurrentProcess();
        SetPriorityClass(process, BELOW_NORMAL_PRIORITY_CLASS);
        let available = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let configured = std::env::var(INFERENCE_THREADS_ENV)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
        let cores = process_core_limit(available, configured);
        let mask = usize::MAX.checked_shr(usize::BITS.saturating_sub(cores as u32));
        if let Some(mask) = mask.filter(|mask| *mask != 0) {
            SetProcessAffinityMask(process, mask);
        }
    }
}

#[cfg(not(windows))]
fn configure_process_resources() {}

fn process_core_limit(available: usize, configured: Option<usize>) -> usize {
    let available = available.max(1).min(usize::BITS as usize);
    configured
        .unwrap_or_else(|| available.div_ceil(2).min(6))
        .clamp(1, available)
}

fn parse_options(arguments: Vec<OsString>) -> Result<DaemonOptions, String> {
    let mut state_dir = None;
    let mut idle_timeout = DEFAULT_IDLE_TIMEOUT;
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

#[cfg(test)]
mod tests {
    use super::process_core_limit;

    #[test]
    fn process_core_limit_is_bounded_and_overrideable() {
        assert_eq!(process_core_limit(1, None), 1);
        assert_eq!(process_core_limit(8, None), 4);
        assert_eq!(process_core_limit(32, None), 6);
        assert_eq!(process_core_limit(8, Some(2)), 2);
        assert_eq!(process_core_limit(8, Some(99)), 8);
    }
}
