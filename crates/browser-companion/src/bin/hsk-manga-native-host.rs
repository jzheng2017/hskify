#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
compile_error!("hsk-manga-native-host only supports 64-bit Windows");

use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("hsk-manga-native-host: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), browser_companion::launcher::LauncherError> {
    let current_executable = std::env::current_exe()?;
    let arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
    let state_dir = std::env::var_os("HSK_MANGA_STATE_DIR").map(PathBuf::from);
    let stdin = io::stdin();
    let stdout = io::stdout();
    browser_companion::launcher::run_native_host(
        &mut stdin.lock(),
        &mut stdout.lock(),
        &arguments,
        &current_executable,
        state_dir,
    )
}
