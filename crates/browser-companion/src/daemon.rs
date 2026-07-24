//! Browser daemon bootstrap and idle lifetime.

use std::net::{Ipv4Addr, SocketAddrV4};
use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;
use tokio::net::TcpListener;
use uuid::Uuid;

use crate::crypto::{CryptoError, generate_secret};
use crate::discovery::{
    DaemonRecord, DiscoveryError, STATE_VERSION, acquire_daemon_lock, prepare_state_paths,
    remove_record_if_instance, write_daemon_record,
};
use crate::server::{BridgeConfig, BridgeState, router, wait_until_idle};

#[derive(Debug, Clone)]
pub struct DaemonOptions {
    pub state_dir: PathBuf,
    pub idle_timeout: Duration,
    pub fixture_stage_delay: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonExit {
    Idle,
    AlreadyRunning,
}

#[derive(Debug, Error)]
pub enum DaemonError {
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Crypto(#[from] CryptoError),
    #[error("browser daemon listener failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Bind literally to IPv4 loopback and let the OS select an unused port.
pub async fn bind_random_loopback() -> Result<TcpListener, std::io::Error> {
    TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).await
}

pub async fn run_daemon(options: DaemonOptions) -> Result<DaemonExit, DaemonError> {
    let paths = prepare_state_paths(options.state_dir)?;
    let daemon_lock = match acquire_daemon_lock(&paths) {
        Ok(lock) => lock,
        Err(DiscoveryError::AlreadyRunning) => return Ok(DaemonExit::AlreadyRunning),
        Err(error) => return Err(error.into()),
    };

    let listener = bind_random_loopback().await?;
    let address = listener.local_addr()?;
    debug_assert_eq!(address.ip(), std::net::IpAddr::V4(Ipv4Addr::LOCALHOST));
    let port = address.port();
    let (control_secret, encoded_control_secret) = generate_secret()?;
    let instance_id = Uuid::new_v4().to_string();
    let started_at_unix_ms =
        u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or_default();
    let record = DaemonRecord {
        state_version: STATE_VERSION,
        instance_id: instance_id.clone(),
        pid: std::process::id(),
        port,
        control_secret: encoded_control_secret,
        started_at_unix_ms,
    };
    write_daemon_record(&paths, &record)?;

    let mut config = BridgeConfig::for_port(port);
    config.idle_timeout = options.idle_timeout;
    config.fixture_stage_delay = options.fixture_stage_delay;
    let state = BridgeState::new(config, control_secret);
    let service = router(state.clone());
    let serve_result = axum::serve(listener, service)
        .with_graceful_shutdown(wait_until_idle(state))
        .await;

    let cleanup_result = remove_record_if_instance(&paths, &instance_id);
    drop(daemon_lock);
    serve_result?;
    cleanup_result?;
    Ok(DaemonExit::Idle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn listener_is_ipv4_loopback_with_random_port() {
        let listener = bind_random_loopback().await.unwrap();
        let address = listener.local_addr().unwrap();
        assert_eq!(
            address.ip(),
            std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
        );
        assert_ne!(address.port(), 0);
    }
}
