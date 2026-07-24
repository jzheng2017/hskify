#![cfg(windows)]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use browser_companion::discovery::{prepare_state_paths, read_daemon_record};
use browser_companion::launcher::{request_session, spawn_detached_daemon};

const ORIGIN: &str = "moz-extension://00000000-0000-4000-8000-000000000001";

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

#[test]
fn detached_daemon_survives_spawn_scope_and_duplicate_exits() {
    let directory = tempfile::tempdir().expect("temporary state directory");
    let paths = prepare_state_paths(directory.path()).expect("state paths");
    let executable = PathBuf::from(env!("CARGO_BIN_EXE_hsk-manga-browser-daemon"));
    let (first, breakaway_available) = match spawn_detached_daemon(
        &executable,
        directory.path(),
        Some(Duration::from_millis(700)),
    ) {
        Ok(child) => (child, true),
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
            // Some CI/desktop harness jobs intentionally omit
            // JOB_OBJECT_LIMIT_BREAKAWAY_OK. The unit test still asserts that
            // production sets CREATE_BREAKAWAY_FROM_JOB; use an in-job child
            // here to continue exercising discovery, duplicate prevention,
            // control auth, and idle cleanup.
            eprintln!("Windows harness denied breakaway spawn; exercising lifecycle in-job");
            (
                spawn_in_job(&executable, directory.path(), Duration::from_millis(700))
                    .expect("spawn in-job lifecycle daemon"),
                false,
            )
        }
        Err(error) => panic!("spawn detached daemon with CREATE_BREAKAWAY_FROM_JOB: {error}"),
    };
    let mut first = ChildGuard(first);

    let deadline = Instant::now() + Duration::from_secs(5);
    let record = loop {
        if let Some(record) = read_daemon_record(&paths).ok().flatten() {
            break record;
        }
        assert!(
            Instant::now() < deadline,
            "detached daemon did not publish discovery state"
        );
        thread::sleep(Duration::from_millis(25));
    };
    assert_ne!(record.pid, std::process::id());
    let ready = request_session(&record, ORIGIN).expect("control-secret session issuance");
    assert_eq!(ready.port, record.port);

    let mut duplicate = if breakaway_available {
        spawn_detached_daemon(
            &executable,
            directory.path(),
            Some(Duration::from_millis(700)),
        )
        .expect("spawn duplicate contender")
    } else {
        spawn_in_job(&executable, directory.path(), Duration::from_millis(700))
            .expect("spawn in-job duplicate contender")
    };
    let deadline = Instant::now() + Duration::from_secs(3);
    let status = loop {
        if let Some(status) = duplicate.try_wait().expect("poll duplicate") {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "duplicate daemon did not exit after failing the lock"
        );
        thread::sleep(Duration::from_millis(25));
    };
    assert!(status.success());
    assert_eq!(
        read_daemon_record(&paths)
            .expect("read winner state")
            .expect("winner record")
            .instance_id,
        record.instance_id
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = first.0.try_wait().expect("poll idle daemon") {
            assert!(status.success());
            break;
        }
        assert!(
            Instant::now() < deadline,
            "detached daemon did not honor idle shutdown"
        );
        thread::sleep(Duration::from_millis(25));
    }
    assert!(
        read_daemon_record(&paths)
            .expect("read cleaned state")
            .is_none(),
        "idle daemon should remove only its own discovery record"
    );
}

fn spawn_in_job(
    executable: &PathBuf,
    state_dir: &std::path::Path,
    idle_timeout: Duration,
) -> std::io::Result<Child> {
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::System::Threading::{CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW};

    Command::new(executable)
        .arg("--state-dir")
        .arg(state_dir)
        .arg("--idle-milliseconds")
        .arg(idle_timeout.as_millis().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
        .spawn()
}
