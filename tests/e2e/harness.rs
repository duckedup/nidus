//! Spawn a real `nidus serve` and talk to it over HTTP.
//!
//! The awkward parts of driving a child server, solved once here so the suites read as
//! assertions rather than process plumbing:
//!
//! * **Which binary?** `env!("CARGO_BIN_EXE_nidus")` — cargo builds the `nidus` target
//!   before the test runs and hands us its path, so there is no guessing at
//!   `target/debug` and no chance of testing a stale build.
//! * **Which port?** `--addr 127.0.0.1:0` lets the kernel pick, and the startup line
//!   reports the bound address, so concurrent tests never collide on a fixed port.
//! * **Is it up yet?** [`Server::start`] polls `/health` and only returns once the
//!   server answers, so no suite needs a hopeful `sleep`.
//! * **Diagnosing a red run.** Every line the child writes to stderr is captured; a
//!   failed startup panics with that transcript attached instead of an opaque timeout.
//! * **Leaks.** [`Drop`] kills and reaps the child, so a failing assertion (which
//!   unwinds past any explicit cleanup) can't leave an orphan holding the writer lock.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use ureq::Agent;

/// How long to wait for the startup line, and then for `/health` to answer. Generous:
/// a debug-build cold start on a loaded CI runner is slow, and a timeout here is a
/// confusing failure, so we would rather wait than flake.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Build a `nidus serve` invocation. `dim` and the store location are always needed;
/// everything else is extra flags.
pub struct Server {
    dir: std::path::PathBuf,
    dim: usize,
    args: Vec<String>,
    token: Option<String>,
}

/// A running `nidus serve` child process.
pub struct RunningServer {
    child: Child,
    base: String,
    token: Option<String>,
    agent: Agent,
    /// Everything the child has written to stderr, for failure messages.
    stderr: Arc<Mutex<Vec<String>>>,
}

impl Server {
    /// A server over `dir` with the given embedding dimension.
    pub fn new(dir: impl Into<std::path::PathBuf>, dim: usize) -> Self {
        Server {
            dir: dir.into(),
            dim,
            args: Vec::new(),
            token: None,
        }
    }

    /// Extra `nidus serve` flags (`--quantization int8`, `--cluster`, …).
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.args
            .extend(args.into_iter().map(|a| a.as_ref().to_string()));
        self
    }

    /// Require `Authorization: Bearer <token>`, and send it on this client's requests.
    pub fn token(mut self, token: &str) -> Self {
        self.token = Some(token.to_string());
        self
    }

    /// Spawn the server and wait until it answers `/health`.
    ///
    /// Panics — rather than returning an error — because every caller is a test for
    /// which a server that won't start is a failure, and the panic carries the child's
    /// stderr.
    pub fn start(self) -> RunningServer {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_nidus"));
        cmd.arg("serve")
            .arg("--dir")
            .arg(&self.dir)
            .arg("--dim")
            .arg(self.dim.to_string())
            // Port 0: the kernel assigns a free port, which the startup line reports.
            .arg("--addr")
            .arg("127.0.0.1:0")
            .args(&self.args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            // Inherited NIDUS_* vars would silently override the flags under test.
            .env_clear()
            .envs(std::env::vars().filter(|(k, _)| !k.starts_with("NIDUS_")));
        if let Some(token) = &self.token {
            cmd.arg("--token").arg(token);
        }

        let mut child = cmd.spawn().expect("spawn nidus serve");
        let pipe = child.stderr.take().expect("stderr piped");

        // Drain stderr on a thread: it must be read continuously or a chatty child would
        // block on a full pipe, and the first line carries the bound address.
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = channel();
        let log = Arc::clone(&stderr);
        std::thread::spawn(move || {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                if let Some(addr) = bound_addr(&line) {
                    // A closed receiver just means startup already resolved.
                    let _ = tx.send(addr);
                }
                log.lock().expect("stderr log").push(line);
            }
        });

        let base = match await_addr(&rx, &mut child, &stderr) {
            Ok(addr) => format!("http://{addr}"),
            Err(msg) => panic!("{msg}"),
        };

        let server = RunningServer {
            child,
            base,
            token: self.token,
            // 4xx/5xx must arrive as responses, not errors — the suites assert on 401
            // and 413 (mirrors `backend::cloud::Http`).
            agent: Agent::new_with_config(
                ureq::config::Config::builder()
                    .http_status_as_error(false)
                    .timeout_global(Some(Duration::from_secs(30)))
                    .build(),
            ),
            stderr,
        };
        server.await_health();
        server
    }
}

impl RunningServer {
    /// `http://127.0.0.1:<port>` — the base for a raw client.
    pub fn base_url(&self) -> &str {
        &self.base
    }

    /// Everything the child has written to stderr so far.
    pub fn stderr(&self) -> String {
        self.stderr.lock().expect("stderr log").join("\n")
    }

    /// `GET path`, as `(status, body)`. The body is `Value::Null` when not JSON (an
    /// error page, say), so status-only assertions don't have to care.
    pub fn get(&self, path: &str) -> (u16, Value) {
        let res = self
            .auth(self.agent.get(self.url(path)))
            .call()
            .unwrap_or_else(|e| {
                panic!("GET {path}: {e}\n--- server stderr ---\n{}", self.stderr())
            });
        read(res, path, &self.stderr())
    }

    /// `POST path` with a JSON body, as `(status, body)`.
    ///
    /// Serialises here and hands the bytes to [`post_bytes`](Self::post_bytes) rather
    /// than using ureq's `send_json`, which needs a ureq feature the library does not
    /// otherwise want — one request path, and no dependency widened for a test.
    pub fn post(&self, path: &str, body: &Value) -> (u16, Value) {
        let bytes = serde_json::to_vec(body).expect("serialise request body");
        self.post_bytes(path, &bytes)
    }

    /// `POST path` with a raw body and an explicit content type — also the escape hatch
    /// for payloads that must bypass serialisation (e.g. one deliberately larger than
    /// `--max-body-bytes`).
    pub fn post_bytes(&self, path: &str, body: &[u8]) -> (u16, Value) {
        let res = self
            .auth(self.agent.post(self.url(path)))
            .header("content-type", "application/json")
            .send(body)
            .unwrap_or_else(|e| {
                panic!("POST {path}: {e}\n--- server stderr ---\n{}", self.stderr())
            });
        read(res, path, &self.stderr())
    }

    /// Ask the server to shut down the way a supervisor would (SIGTERM), and wait for it
    /// to exit — the path that flushes and releases the writer lock. Returns whether it
    /// exited successfully.
    ///
    /// Unix-only: there is no portable SIGTERM in `std` (`Child::kill` is SIGKILL), and
    /// shelling out to `kill` keeps the harness dependency-free.
    #[cfg(unix)]
    pub fn shutdown(mut self) -> bool {
        let pid = self.child.id().to_string();
        let signalled = Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(signalled, "failed to SIGTERM pid {pid}");
        // `Drop` still runs when this returns, but its kill/wait on an already-reaped
        // child fail harmlessly (both errors are ignored) — no need to leak `self` to
        // suppress it.
        self.child.wait().expect("wait for graceful exit").success()
    }

    /// Kill the server outright (SIGKILL) — the crash path, which flushes nothing and
    /// leaves the on-disk writer lock behind.
    pub fn kill(mut self) {
        self.child.kill().expect("kill server");
        self.child.wait().expect("reap killed server");
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn auth<B>(&self, req: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        match &self.token {
            Some(t) => req.header("authorization", &format!("Bearer {t}")),
            None => req,
        }
    }

    /// Poll `/health` until it answers or [`STARTUP_TIMEOUT`] elapses. The bound port
    /// proves the listener exists; this proves the store opened and the router is live.
    fn await_health(&self) {
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        let url = self.url("/health");
        loop {
            if let Ok(res) = self.agent.get(&url).call()
                && res.status().as_u16() == 200
            {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "server bound {} but /health never answered\n--- server stderr ---\n{}",
                self.base,
                self.stderr()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        // Best-effort: a panicking test unwinds past any explicit shutdown, and an
        // orphaned child would hold the writer lock and fail every later test.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Pull the bound address out of the startup line
/// (`nidus serving on http://127.0.0.1:51513 (Ctrl-C …)`).
fn bound_addr(line: &str) -> Option<String> {
    let rest = line.split_once("http://")?.1;
    let addr = rest.split_whitespace().next()?;
    (!addr.is_empty()).then(|| addr.to_string())
}

/// Wait for the startup line. Distinguishes the two ways this goes wrong — the child
/// died (bad flags, port in use, store already locked) versus it is merely slow — so the
/// failure names the actual cause.
fn await_addr(
    rx: &Receiver<String>,
    child: &mut Child,
    stderr: &Arc<Mutex<Vec<String>>>,
) -> Result<String, String> {
    let transcript = || stderr.lock().expect("stderr log").join("\n");
    match rx.recv_timeout(STARTUP_TIMEOUT) {
        Ok(addr) => Ok(addr),
        Err(RecvTimeoutError::Disconnected) => {
            let status = child.wait().map(|s| s.to_string()).unwrap_or_default();
            Err(format!(
                "nidus serve exited before reporting an address ({status})\
                 \n--- server stderr ---\n{}",
                transcript()
            ))
        }
        Err(RecvTimeoutError::Timeout) => {
            let _ = child.kill();
            Err(format!(
                "nidus serve printed no address within {STARTUP_TIMEOUT:?}\
                 \n--- server stderr ---\n{}",
                transcript()
            ))
        }
    }
}

/// Read a response into `(status, json)`, tolerating a non-JSON body.
fn read(res: ureq::http::Response<ureq::Body>, path: &str, stderr: &str) -> (u16, Value) {
    let status = res.status().as_u16();
    let body = res
        .into_body()
        .read_to_vec()
        .unwrap_or_else(|e| panic!("read body of {path}: {e}\n--- server stderr ---\n{stderr}"));
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}
