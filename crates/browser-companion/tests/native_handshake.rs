use std::io::{Cursor, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use browser_companion::contracts::{NativeReadyResponse, Validate};
use browser_companion::discovery::{prepare_state_paths, read_daemon_record};
use browser_companion::launcher::request_session;
use browser_companion::native_framing::{read_frame, write_frame};
use browser_companion::{FIREFOX_EXTENSION_ID, NATIVE_HOST_NAME, PROTOCOL_HEADER};
use serde_json::json;

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
fn native_binary_frames_fresh_session_from_existing_daemon() {
    let directory = tempfile::tempdir().expect("temporary daemon state");
    let paths = prepare_state_paths(directory.path()).expect("state paths");
    let daemon = PathBuf::from(env!("CARGO_BIN_EXE_hsk-manga-browser-daemon"));
    let native_host = PathBuf::from(env!("CARGO_BIN_EXE_hsk-manga-native-host"));

    let daemon_child = Command::new(&daemon)
        .arg("--state-dir")
        .arg(directory.path())
        .arg("--idle-milliseconds")
        .arg("1200")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start lifecycle daemon");
    let mut daemon_child = ChildGuard(daemon_child);

    let deadline = Instant::now() + Duration::from_secs(5);
    let record = loop {
        if let Some(record) = read_daemon_record(&paths).ok().flatten()
            && request_session(&record, ORIGIN).is_ok()
        {
            break record;
        }
        if let Some(status) = daemon_child.0.try_wait().expect("poll daemon") {
            panic!("daemon exited before handshake: {status}");
        }
        assert!(Instant::now() < deadline, "daemon did not become healthy");
        thread::sleep(Duration::from_millis(25));
    };

    let manifest_path = directory.path().join("local.mangalations.hsk_manga.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&json!({
            "name": NATIVE_HOST_NAME,
            "description": "integration test",
            "path": native_host.canonicalize().expect("canonical native host"),
            "type": "stdio",
            "allowed_extensions": [FIREFOX_EXTENSION_ID]
        }))
        .expect("serialize manifest"),
    )
    .expect("write manifest");

    let mut framed_request = Vec::new();
    write_frame(
        &mut framed_request,
        &json!({
            "type": "start-or-discover-daemon",
            "protocolVersion": 1,
            "extensionVersion": "0.1.0",
            "extensionOrigin": ORIGIN
        }),
    )
    .expect("encode native request");
    let mut child = Command::new(&native_host)
        .arg(&manifest_path)
        .arg(FIREFOX_EXTENSION_ID)
        .env("HSK_MANGA_STATE_DIR", directory.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start native host");
    child
        .stdin
        .take()
        .expect("native stdin")
        .write_all(&framed_request)
        .expect("write native request");
    let output = child.wait_with_output().expect("wait native host");
    assert!(
        output.status.success(),
        "native host failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut cursor = Cursor::new(output.stdout);
    let ready: NativeReadyResponse = read_frame(&mut cursor).expect("decode native response");
    ready.validate().expect("valid native ready response");
    assert_eq!(ready.port, record.port);
    assert_eq!(cursor.position() as usize, cursor.get_ref().len());

    let mut stream = TcpStream::connect(SocketAddrV4::new(Ipv4Addr::LOCALHOST, ready.port))
        .expect("connect with native session");
    write!(
        stream,
        "GET /browser/v1/health HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nOrigin: {}\r\n{}: 1\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",
        ready.port, ORIGIN, PROTOCOL_HEADER, ready.token
    )
    .expect("write health request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("read health response");
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
    assert!(
        response
            .to_ascii_lowercase()
            .contains(&format!("access-control-allow-origin: {}", ORIGIN))
    );
}
