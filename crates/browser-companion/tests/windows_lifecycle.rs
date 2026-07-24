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
fn in_job_daemon_lifecycle_covers_discovery_lock_and_idle_cleanup() {
    exercise_lifecycle(LaunchMode::InCurrentJob);
}

#[test]
#[ignore = "requires a Windows parent job that grants JOB_OBJECT_LIMIT_BREAKAWAY_OK; this is a production-flag launch probe, not a Firefox packaging smoke test"]
fn production_breakaway_launch_probe_covers_detached_lifecycle() {
    exercise_lifecycle(LaunchMode::ProductionBreakaway);
}

#[derive(Clone, Copy, Debug)]
enum LaunchMode {
    InCurrentJob,
    ProductionBreakaway,
}

impl LaunchMode {
    fn spawn(
        self,
        executable: &PathBuf,
        state_dir: &std::path::Path,
        idle_timeout: Duration,
    ) -> std::io::Result<Child> {
        match self {
            Self::InCurrentJob => spawn_in_job(executable, state_dir, idle_timeout),
            Self::ProductionBreakaway => {
                spawn_detached_daemon(executable, state_dir, Some(idle_timeout))
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::InCurrentJob => "in-job",
            Self::ProductionBreakaway => "production breakaway",
        }
    }
}

fn exercise_lifecycle(mode: LaunchMode) {
    let directory = tempfile::tempdir().expect("temporary state directory");
    let paths = prepare_state_paths(directory.path()).expect("state paths");
    let executable = PathBuf::from(env!("CARGO_BIN_EXE_hsk-manga-browser-daemon"));
    let first = mode
        .spawn(
            &executable,
            directory.path(),
            Duration::from_millis(700),
        )
        .unwrap_or_else(|error| {
            panic!(
                "{} daemon launch failed: {error}; the production probe requires the parent job to grant JOB_OBJECT_LIMIT_BREAKAWAY_OK",
                mode.label()
            )
        });
    let mut first = ChildGuard(first);

    let deadline = Instant::now() + Duration::from_secs(5);
    let record = loop {
        if let Some(record) = read_daemon_record(&paths).ok().flatten() {
            break record;
        }
        assert!(
            Instant::now() < deadline,
            "{} daemon did not publish discovery state",
            mode.label()
        );
        thread::sleep(Duration::from_millis(25));
    };
    assert_ne!(record.pid, std::process::id());
    let ready = request_session(&record, ORIGIN).expect("control-secret session issuance");
    assert_eq!(ready.port, record.port);

    let mut duplicate = mode
        .spawn(&executable, directory.path(), Duration::from_millis(700))
        .expect("spawn duplicate contender");
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
            "{} daemon did not honor idle shutdown",
            mode.label()
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
