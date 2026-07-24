//! One-shot native launcher, caller validation, discovery, and detached spawn.

use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

use crate::contracts::{NativeHandshakeRequest, NativeReadyResponse, Validate};
use crate::discovery::{
    DaemonRecord, DiscoveryError, StatePaths, default_state_dir, prepare_state_paths,
    read_daemon_record,
};
use crate::native_framing::{NativeFrameError, read_frame, write_frame};
use crate::origin::validate_extension_origin;
use crate::{CONTROL_HEADER, FIREFOX_EXTENSION_ID, NATIVE_HOST_NAME};

const MAX_MANIFEST_BYTES: u64 = 32 * 1024;
const MAX_CONTROL_RESPONSE_BYTES: u64 = 64 * 1024;
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(6);
const CONNECT_TIMEOUT: Duration = Duration::from_millis(400);
const IO_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug, Error)]
pub enum LauncherError {
    #[error("Firefox caller arguments are missing or unauthorized")]
    UnauthorizedCaller,
    #[error("native host manifest is invalid")]
    InvalidManifest,
    #[error("native launcher I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("native launcher framing failed: {0}")]
    Frame(#[from] NativeFrameError),
    #[error("native handshake failed contract validation: {0}")]
    Contract(#[from] crate::contracts::ContractError),
    #[error("extension origin is invalid")]
    InvalidOrigin,
    #[error("daemon discovery failed: {0}")]
    Discovery(#[from] DiscoveryError),
    #[error("daemon did not become healthy before the discovery timeout")]
    DiscoveryTimeout,
    #[error("daemon returned an invalid control response")]
    InvalidControlResponse,
}

#[derive(Debug, Deserialize)]
struct NativeHostManifest {
    name: String,
    path: String,
    #[serde(rename = "type")]
    manifest_type: String,
    allowed_extensions: Vec<String>,
}

/// Firefox passes the manifest path and permanent add-on ID as the native
/// process's two arguments. Validate both, plus the manifest's executable,
/// before reading the extension-controlled message body.
pub fn validate_firefox_caller(
    arguments: &[OsString],
    current_executable: &Path,
) -> Result<(), LauncherError> {
    if arguments.len() != 2
        || arguments[1].to_str() != Some(FIREFOX_EXTENSION_ID)
        || arguments[0].is_empty()
    {
        return Err(LauncherError::UnauthorizedCaller);
    }
    let manifest_path = PathBuf::from(&arguments[0]);
    if !manifest_path.is_absolute() {
        return Err(LauncherError::InvalidManifest);
    }
    let file = File::open(&manifest_path).map_err(|_| LauncherError::InvalidManifest)?;
    let metadata = file
        .metadata()
        .map_err(|_| LauncherError::InvalidManifest)?;
    if metadata.len() == 0 || metadata.len() > MAX_MANIFEST_BYTES {
        return Err(LauncherError::InvalidManifest);
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| LauncherError::InvalidManifest)?;
    let manifest: NativeHostManifest =
        serde_json::from_slice(&bytes).map_err(|_| LauncherError::InvalidManifest)?;
    if manifest.name != NATIVE_HOST_NAME
        || manifest.manifest_type != "stdio"
        || manifest.allowed_extensions != [FIREFOX_EXTENSION_ID]
    {
        return Err(LauncherError::InvalidManifest);
    }
    let configured_path = Path::new(&manifest.path);
    if !configured_path.is_absolute() {
        return Err(LauncherError::InvalidManifest);
    }
    let configured = configured_path
        .canonicalize()
        .map_err(|_| LauncherError::InvalidManifest)?;
    let running = current_executable
        .canonicalize()
        .map_err(|_| LauncherError::InvalidManifest)?;
    if configured != running {
        return Err(LauncherError::InvalidManifest);
    }
    Ok(())
}

pub fn run_native_host<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    caller_arguments: &[OsString],
    current_executable: &Path,
    state_dir: Option<PathBuf>,
) -> Result<(), LauncherError> {
    validate_firefox_caller(caller_arguments, current_executable)?;
    let request: NativeHandshakeRequest = read_frame(reader)?;
    request.validate()?;
    validate_extension_origin(&request.extension_origin)
        .map_err(|_| LauncherError::InvalidOrigin)?;
    let state_dir = match state_dir {
        Some(path) => path,
        None => default_state_dir()?,
    };
    let paths = prepare_state_paths(state_dir)?;
    let daemon_executable = sibling_daemon_executable(current_executable);
    let ready = start_or_discover(&paths, &daemon_executable, &request.extension_origin)?;
    ready.validate()?;
    write_frame(writer, &ready)?;
    Ok(())
}

pub fn sibling_daemon_executable(native_host: &Path) -> PathBuf {
    let mut daemon = native_host.to_path_buf();
    daemon.set_file_name(format!(
        "hsk-manga-browser-daemon{}",
        std::env::consts::EXE_SUFFIX
    ));
    daemon
}

pub fn start_or_discover(
    paths: &StatePaths,
    daemon_executable: &Path,
    extension_origin: &str,
) -> Result<NativeReadyResponse, LauncherError> {
    if let Some(ready) = try_discover(paths, extension_origin) {
        return Ok(ready);
    }

    let mut spawned = spawn_detached_daemon(daemon_executable, &paths.root, None)?;
    let deadline = Instant::now() + DISCOVERY_TIMEOUT;
    loop {
        if let Some(ready) = try_discover(paths, extension_origin) {
            return Ok(ready);
        }
        if Instant::now() >= deadline {
            return Err(LauncherError::DiscoveryTimeout);
        }
        if let Some(status) = spawned.try_wait()?
            && !status.success()
        {
            // A concurrent daemon may still be starting. Keep polling the
            // authenticated state record until the common deadline.
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn try_discover(paths: &StatePaths, extension_origin: &str) -> Option<NativeReadyResponse> {
    let record = read_daemon_record(paths).ok().flatten()?;
    request_session(&record, extension_origin).ok()
}

pub fn request_session(
    record: &DaemonRecord,
    extension_origin: &str,
) -> Result<NativeReadyResponse, LauncherError> {
    record.validate()?;
    validate_extension_origin(extension_origin).map_err(|_| LauncherError::InvalidOrigin)?;
    let address = SocketAddrV4::new(Ipv4Addr::LOCALHOST, record.port);
    let mut stream = TcpStream::connect_timeout(&address.into(), CONNECT_TIMEOUT)?;
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;
    let body = serde_json::to_vec(&json!({ "extensionOrigin": extension_origin }))
        .map_err(|_| LauncherError::InvalidControlResponse)?;
    write!(
        stream,
        "POST /browser-internal/v1/session HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n{}: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        record.port,
        CONTROL_HEADER,
        record.control_secret,
        body.len()
    )?;
    stream.write_all(&body)?;
    stream.flush()?;

    let mut response = Vec::new();
    stream
        .take(MAX_CONTROL_RESPONSE_BYTES + 1)
        .read_to_end(&mut response)?;
    if response.len() as u64 > MAX_CONTROL_RESPONSE_BYTES {
        return Err(LauncherError::InvalidControlResponse);
    }
    let separator = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or(LauncherError::InvalidControlResponse)?;
    let headers = &response[..separator];
    let body = &response[separator + 4..];
    let headers =
        std::str::from_utf8(headers).map_err(|_| LauncherError::InvalidControlResponse)?;
    let mut lines = headers.lines();
    if !matches!(lines.next(), Some("HTTP/1.1 200 OK" | "HTTP/1.0 200 OK"))
        || lines.any(|line| line.to_ascii_lowercase().starts_with("transfer-encoding:"))
    {
        return Err(LauncherError::InvalidControlResponse);
    }
    let ready: NativeReadyResponse =
        serde_json::from_slice(body).map_err(|_| LauncherError::InvalidControlResponse)?;
    ready.validate()?;
    if ready.port != record.port {
        return Err(LauncherError::InvalidControlResponse);
    }
    Ok(ready)
}

pub fn spawn_detached_daemon(
    daemon_executable: &Path,
    state_dir: &Path,
    idle_timeout: Option<Duration>,
) -> io::Result<Child> {
    let mut command = Command::new(daemon_executable);
    command
        .arg("--state-dir")
        .arg(state_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(idle_timeout) = idle_timeout {
        command
            .arg("--idle-milliseconds")
            .arg(idle_timeout.as_millis().max(1).to_string());
    }
    configure_detached(&mut command);
    command.spawn()
}

#[cfg(windows)]
pub const fn windows_detached_creation_flags() -> u32 {
    use windows_sys::Win32::System::Threading::{
        CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW,
    };
    CREATE_BREAKAWAY_FROM_JOB | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
}

#[cfg(windows)]
fn configure_detached(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(windows_detached_creation_flags());
}

#[cfg(unix)]
fn configure_detached(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: this callback performs only the async-signal-safe `setsid`
    // syscall and constructs an OS error from thread-local errno.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(not(any(unix, windows)))]
fn configure_detached(_command: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_framing::write_frame;
    use serde_json::json;

    fn manifest(directory: &Path, executable: &Path) -> PathBuf {
        let path = directory.join("local.mangalations.hsk_manga.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&json!({
                "name": NATIVE_HOST_NAME,
                "description": "test",
                "path": executable,
                "type": "stdio",
                "allowed_extensions": [FIREFOX_EXTENSION_ID]
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn caller_requires_exact_firefox_id_manifest_and_executable() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory
            .path()
            .join(format!("host{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&executable, b"test").unwrap();
        let manifest = manifest(directory.path(), &executable);
        let valid = vec![
            manifest.as_os_str().to_owned(),
            OsString::from(FIREFOX_EXTENSION_ID),
        ];
        assert!(validate_firefox_caller(&valid, &executable).is_ok());
        assert!(validate_firefox_caller(&valid[..1], &executable).is_err());
        let wrong_id = vec![
            manifest.into_os_string(),
            OsString::from("attacker@example"),
        ];
        assert!(validate_firefox_caller(&wrong_id, &executable).is_err());
    }

    #[test]
    fn native_host_rejects_caller_before_consuming_frame() {
        let mut frame = Vec::new();
        write_frame(
            &mut frame,
            &json!({
                "type": "start-or-discover-daemon",
                "protocolVersion": 1,
                "extensionVersion": "0.1.0",
                "extensionOrigin": "moz-extension://00000000-0000-4000-8000-000000000001"
            }),
        )
        .unwrap();
        let mut input = io::Cursor::new(frame);
        let mut output = Vec::new();
        let position = input.position();
        let error =
            run_native_host(&mut input, &mut output, &[], Path::new("missing"), None).unwrap_err();
        assert!(matches!(error, LauncherError::UnauthorizedCaller));
        assert_eq!(input.position(), position);
        assert!(output.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn windows_spawn_requests_firefox_job_breakaway() {
        use windows_sys::Win32::System::Threading::{
            CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW,
        };
        let flags = windows_detached_creation_flags();
        assert_ne!(flags & CREATE_BREAKAWAY_FROM_JOB, 0);
        assert_ne!(flags & CREATE_NEW_PROCESS_GROUP, 0);
        assert_ne!(flags & CREATE_NO_WINDOW, 0);
    }
}
