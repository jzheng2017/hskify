//! Per-user daemon discovery record and lifetime lock.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::crypto::decode_secret;

pub const STATE_VERSION: u8 = 1;
const STATE_FILENAME: &str = "daemon-state-v1.json";
const LOCK_FILENAME: &str = "daemon-v1.lock";
const MAX_STATE_BYTES: u64 = 16 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DaemonRecord {
    pub state_version: u8,
    pub instance_id: String,
    pub pid: u32,
    pub port: u16,
    pub control_secret: String,
    pub started_at_unix_ms: u64,
}

impl DaemonRecord {
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if self.state_version != STATE_VERSION
            || Uuid::parse_str(&self.instance_id).is_err()
            || self.pid == 0
            || self.port == 0
            || self.started_at_unix_ms == 0
            || decode_secret(&self.control_secret).is_err()
        {
            return Err(DiscoveryError::InvalidRecord);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StatePaths {
    pub root: PathBuf,
    pub lock: PathBuf,
    pub record: PathBuf,
    pub cache: PathBuf,
}

impl StatePaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            lock: root.join(LOCK_FILENAME),
            record: root.join(STATE_FILENAME),
            cache: root.join("browser-cache-v1"),
            root,
        }
    }
}

#[derive(Debug)]
pub struct DaemonLock {
    _file: File,
}

#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("could not determine the current user's local data directory")]
    NoUserDataDirectory,
    #[error("daemon discovery I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("another browser daemon already holds the per-user lock")]
    AlreadyRunning,
    #[error("daemon state record is invalid")]
    InvalidRecord,
    #[error("daemon state JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn default_state_dir() -> Result<PathBuf, DiscoveryError> {
    let base = dirs::data_local_dir().ok_or(DiscoveryError::NoUserDataDirectory)?;
    Ok(base
        .join("Hskify")
        .join("HSKMangaTranslator")
        .join("browser-companion-v1"))
}

pub fn prepare_state_paths(root: impl Into<PathBuf>) -> Result<StatePaths, DiscoveryError> {
    let paths = StatePaths::new(root);
    fs::create_dir_all(&paths.root)?;
    fs::create_dir_all(&paths.cache)?;
    secure_directory(&paths.root)?;
    secure_directory(&paths.cache)?;
    Ok(paths)
}

pub fn acquire_daemon_lock(paths: &StatePaths) -> Result<DaemonLock, DiscoveryError> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&paths.lock)?;
    secure_file(&paths.lock)?;
    match fs4::FileExt::try_lock(&file) {
        Ok(()) => Ok(DaemonLock { _file: file }),
        Err(fs4::TryLockError::WouldBlock) => Err(DiscoveryError::AlreadyRunning),
        Err(fs4::TryLockError::Error(error)) => Err(DiscoveryError::Io(error)),
    }
}

pub fn read_daemon_record(paths: &StatePaths) -> Result<Option<DaemonRecord>, DiscoveryError> {
    let file = match File::open(&paths.record) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let metadata = file.metadata()?;
    if metadata.len() == 0 || metadata.len() > MAX_STATE_BYTES {
        return Err(DiscoveryError::InvalidRecord);
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_STATE_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_STATE_BYTES {
        return Err(DiscoveryError::InvalidRecord);
    }
    let record: DaemonRecord = serde_json::from_slice(&bytes)?;
    record.validate()?;
    Ok(Some(record))
}

pub fn write_daemon_record(
    paths: &StatePaths,
    record: &DaemonRecord,
) -> Result<(), DiscoveryError> {
    record.validate()?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".daemon-state-")
        .suffix(".tmp")
        .tempfile_in(&paths.root)?;
    secure_file(temporary.path())?;
    serde_json::to_writer(&mut temporary, record)?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;

    // `persist` cannot replace an existing file on Windows. The daemon holds
    // the exclusive lock while replacing this narrowly scoped state record.
    match fs::remove_file(&paths.record) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    temporary
        .persist(&paths.record)
        .map_err(|error| DiscoveryError::Io(error.error))?;
    secure_file(&paths.record)?;
    Ok(())
}

pub fn remove_record_if_instance(
    paths: &StatePaths,
    instance_id: &str,
) -> Result<(), DiscoveryError> {
    let Some(record) = read_daemon_record(paths)? else {
        return Ok(());
    };
    if record.instance_id == instance_id {
        match fs::remove_file(&paths.record) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn secure_directory(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
fn secure_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn secure_file(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn secure_file(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(instance_id: String, control_secret: String) -> DaemonRecord {
        DaemonRecord {
            state_version: STATE_VERSION,
            instance_id,
            pid: 42,
            port: 43127,
            control_secret,
            started_at_unix_ms: 1,
        }
    }

    #[test]
    fn duplicate_lock_is_prevented_and_released_on_drop() {
        let directory = tempfile::tempdir().unwrap();
        let paths = prepare_state_paths(directory.path()).unwrap();
        let first = acquire_daemon_lock(&paths).unwrap();
        assert!(matches!(
            acquire_daemon_lock(&paths),
            Err(DiscoveryError::AlreadyRunning)
        ));
        drop(first);
        acquire_daemon_lock(&paths).unwrap();
    }

    #[test]
    fn stale_record_is_replaced_and_old_instance_cannot_remove_new_one() {
        let directory = tempfile::tempdir().unwrap();
        let paths = prepare_state_paths(directory.path()).unwrap();
        let (_, first_secret) = crate::crypto::generate_secret().unwrap();
        let (_, second_secret) = crate::crypto::generate_secret().unwrap();
        let first_id = Uuid::new_v4().to_string();
        let second_id = Uuid::new_v4().to_string();

        write_daemon_record(&paths, &record(first_id.clone(), first_secret)).unwrap();
        write_daemon_record(&paths, &record(second_id.clone(), second_secret)).unwrap();
        remove_record_if_instance(&paths, &first_id).unwrap();
        assert_eq!(
            read_daemon_record(&paths).unwrap().unwrap().instance_id,
            second_id
        );
        remove_record_if_instance(&paths, &second_id).unwrap();
        assert!(read_daemon_record(&paths).unwrap().is_none());
    }

    #[test]
    fn malformed_and_oversized_records_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let paths = prepare_state_paths(directory.path()).unwrap();
        fs::write(&paths.record, b"{}").unwrap();
        assert!(read_daemon_record(&paths).is_err());
        fs::write(&paths.record, vec![b'x'; MAX_STATE_BYTES as usize + 1]).unwrap();
        assert!(matches!(
            read_daemon_record(&paths),
            Err(DiscoveryError::InvalidRecord)
        ));
    }
}
