//! `nidus code ingest` / `nidus code search` through the real binary (epic nidus-3gm). The
//! no-provider (BM25) path only, mirroring `tests/e2e/ingest_fts.rs`'s template: the claim
//! that matters is what the shipped binary actually indexes and returns, not what an
//! in-process call to `crate::code` would.

#![cfg(all(feature = "memory", feature = "code"))]

use std::path::Path;
use std::process::{Command, Output, Stdio};

use serde_json::Value;

/// The binary with a cleared environment, same rationale as `ingest_fts.rs`: a leaked
/// `NIDUS_EMBED_PROVIDER` from a developer's shell would silently switch the code path
/// this file exists to prove (no provider, no network).
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .envs(std::env::vars().filter(|(k, _)| !k.starts_with("NIDUS_")))
        .output()
        .unwrap_or_else(|e| panic!("spawn nidus {args:?}: {e}"))
}

fn ok_json(args: &[&str]) -> Value {
    let out = run(args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "nidus {args:?} exited {:?}\n--- stderr ---\n{stderr}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("nidus {args:?} stdout is not JSON ({e}):\n{stdout}"))
}

fn code_ingest(store: &Path, root: &Path) -> Value {
    let (store, root) = (store.to_string_lossy(), root.to_string_lossy());
    ok_json(&["code", "ingest", &root, "--dir", &store])
}

fn code_search(store: &Path, query: &str, extra: &[&str]) -> Value {
    let store = store.to_string_lossy();
    let mut args: Vec<&str> = vec!["code", "search", query, "--dir", &store];
    args.extend_from_slice(extra);
    ok_json(&args)
}

fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(path, body).expect("write");
}

/// One Rust source file (two functions, one with a doc comment carrying a word that never
/// appears in its body), one markdown doc, and one file under a dot-directory: enough for
/// dispatch, dot-reach and doc-comment coverage in one corpus.
fn corpus(root: &Path) {
    write(
        &root.join("src/lib.rs"),
        "/// Adds one to x, mentioning zorbatronic nowhere else in this file.\n\
         pub fn alpha(x: i32) -> i32 {\n    x + 1\n}\n\n\
         pub fn beta(x: i32) -> i32 {\n    x - 1\n}\n",
    );
    write(
        &root.join("README.md"),
        "# Widget\n\n## Usage\n\nCall alpha to add one.\n\n## License\n\nMIT.\n",
    );
    write(
        &root.join(".claude/rules/hidden.md"),
        "# Hidden Rule\n\nThis rule mentions quetzalcoatl and lives under a dot-directory.\n",
    );
}

/// The three load-bearing dispatch claims in one ingest: a Rust function becomes a
/// symbol-tagged chunk, a markdown file is chunked by heading rather than AST (ONE CORPUS,
/// two strategies), and a file under a dot-directory is still reached.
#[test]
fn code_ingest_dispatches_each_file_and_reaches_dot_directories() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, root) = (tmp.path().join("store"), tmp.path().join("repo"));
    corpus(&root);

    let report = code_ingest(&store, &root);
    assert_eq!(
        report["matched"], 3,
        "all three files, dot-dir included: {report}"
    );
    assert_eq!(report["ingested"], 3, "{report}");

    let rust_hits = code_search(&store, "zorbatronic", &[]);
    let rust_hits = rust_hits.as_array().expect("array");
    assert_eq!(rust_hits.len(), 1, "one file matched: {rust_hits:?}");
    assert_eq!(rust_hits[0]["path"], "src/lib.rs");
    let symbols = rust_hits[0]["symbols"].as_array().expect("symbols array");
    assert_eq!(
        symbols[0]["symbol"], "alpha",
        "the doc-comment word must resolve to the symbol it documents: {symbols:?}"
    );
    assert_eq!(symbols[0]["kind"], "function");
    assert!(
        symbols[0]["start_line"].is_number() && symbols[0]["end_line"].is_number(),
        "a code chunk must carry a real line span: {symbols:?}"
    );

    let doc_hits = code_search(&store, "quetzalcoatl", &[]);
    let doc_hits = doc_hits.as_array().expect("array");
    assert_eq!(doc_hits.len(), 1, "the hidden markdown file: {doc_hits:?}");
    assert_eq!(doc_hits[0]["path"], ".claude/rules/hidden.md");
    let doc_symbols = doc_hits[0]["symbols"].as_array().expect("symbols array");
    assert!(
        doc_symbols[0]["symbol"].is_null(),
        "markdown is chunked by heading, not AST, so it carries no symbol: {doc_symbols:?}"
    );
}

/// nidus-lvo.2's core claim, carried over to `code ingest`: a re-run over an unchanged tree
/// writes nothing.
#[test]
fn a_second_code_ingest_of_an_unchanged_tree_writes_nothing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, root) = (tmp.path().join("store"), tmp.path().join("repo"));
    corpus(&root);

    code_ingest(&store, &root);
    let second = code_ingest(&store, &root);
    assert_eq!(second["ingested"], 0, "{second}");
    assert_eq!(second["unchanged"], 3, "{second}");
}

/// No source body ever reaches the CLI's own output, in a case that would fail if `search`
/// forgot to route through the presentation layer: path, symbol, kind and line span only.
#[test]
fn code_search_never_prints_source() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, root) = (tmp.path().join("store"), tmp.path().join("repo"));
    corpus(&root);
    code_ingest(&store, &root);

    let out = run(&[
        "code",
        "search",
        "zorbatronic",
        "--dir",
        &store.to_string_lossy(),
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("x + 1"),
        "the function body must never appear in `code search` output: {stdout}"
    );
}

/// `--vector` with no embedder configured is refused up front, naming the flag it needs,
/// instead of silently answering from BM25.
#[test]
fn code_search_vector_with_no_embedder_is_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, root) = (tmp.path().join("store"), tmp.path().join("repo"));
    corpus(&root);
    code_ingest(&store, &root);

    let out = run(&[
        "code",
        "search",
        "zorbatronic",
        "--dir",
        &store.to_string_lossy(),
        "--vector",
    ]);
    assert!(
        !out.status.success(),
        "a forced vector query with no embedder must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--vector") && stderr.contains("embedder"),
        "the error must name the flag and the missing embedder: {stderr}"
    );
}

/// Two mock providers on loopback (the `ingest.rs::Recorder` pattern): an embedder and a
/// summarizer, both recording their request bodies. That recording is the only proof that
/// matters for summarize-then-embed: WHICH text reached the embedder.
mod summarize_mocks {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    pub struct Mock {
        pub url: String,
        pub bodies: Arc<Mutex<Vec<String>>>,
    }

    fn read_body(stream: &mut std::net::TcpStream) -> String {
        let mut reader = BufReader::new(stream.try_clone().expect("clone"));
        let mut len = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                len = v.trim().parse().unwrap_or(0);
            }
            if line == "\r\n" || line == "\n" {
                break;
            }
        }
        let mut body = vec![0u8; len];
        reader.read_exact(&mut body).ok();
        String::from_utf8_lossy(&body).to_string()
    }

    fn respond(mut stream: std::net::TcpStream, json: &str) {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            json.len()
        );
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(json.as_bytes());
        let _ = stream.flush();
    }

    /// `response` is a closure so the embedder can echo a deterministic vector and the
    /// summarizer can answer with prose.
    pub fn start(response: impl Fn(&str) -> String + Send + 'static) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let addr = listener.local_addr().expect("addr");
        let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&bodies);
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let body = read_body(&mut stream);
                sink.lock().expect("sink").push(body.clone());
                respond(stream, &response(&body));
            }
        });
        Mock {
            url: format!("http://{addr}"),
            bodies,
        }
    }

    impl Mock {
        pub fn bodies(&self) -> Vec<String> {
            self.bodies.lock().expect("sink").clone()
        }
    }
}

/// `--summarize` embeds the SUMMARY, not the body. The load-bearing assertion is what the
/// embedder was asked to embed: a test checking only "exit 0" passes with the summarize step
/// deleted, which is exactly how this shipped unwired the first time.
#[test]
fn summarize_embeds_the_summary_and_stores_it_beside_the_body() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, root) = (tmp.path().join("store"), tmp.path().join("repo"));
    write(
        &root.join("src/lib.rs"),
        "/// Adds one.\npub fn alpha(x: i32) -> i32 {\n    x + 1\n}\n",
    );

    const SUMMARY: &str = "This function increments an integer by one.";
    let summarizer = summarize_mocks::start(|_| {
        serde_json::json!({
            "choices": [{ "message": { "role": "assistant", "content": SUMMARY } }]
        })
        .to_string()
    });
    let embedder = summarize_mocks::start(|_| {
        serde_json::json!({ "embeddings": [[0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] })
            .to_string()
    });

    let out = run(&[
        "code",
        "ingest",
        root.to_str().expect("root"),
        "--dir",
        store.to_str().expect("store"),
        "--dim",
        "8",
        "--embed-provider",
        "ollama",
        "--embed-base-url",
        &embedder.url,
        "--summarize",
        "--summarize-provider",
        "openai",
        "--summarize-api-key",
        "test-key",
        "--summarize-base-url",
        &summarizer.url,
    ]);
    assert!(
        out.status.success(),
        "ingest failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Value = serde_json::from_slice(&out.stdout).expect("ingest report is json");
    assert_eq!(report["summarize"]["files_summarized"], 1, "{report}");
    // One file summary plus one symbol summary: wdpkr's symbol prompt takes the file
    // summary as context, so the file call is not optional.
    assert_eq!(report["summarize"]["model_calls"], 2, "{report}");

    // wdpkr's code-summarizer prompt reached the provider, not nidus's generic default.
    let prompts = summarizer.bodies();
    assert!(
        prompts
            .iter()
            .any(|b| b.contains("code summarizer for a semantic search index")),
        "wdpkr's system prompt must reach the wire: {prompts:?}"
    );

    // THE claim: the embedder saw the summary, never the function body.
    let embedded = embedder.bodies();
    assert!(
        embedded.iter().any(|b| b.contains("increments an integer")),
        "the summary must be what gets embedded: {embedded:?}"
    );
    assert!(
        !embedded.iter().any(|b| b.contains("x + 1")),
        "the body must NOT be what gets embedded under --summarize: {embedded:?}"
    );

    // The body is still stored and still BM25-searchable; the summary sits beside it.
    let listed = ok_json(&[
        "list",
        "--dir",
        store.to_str().expect("store"),
        "--limit",
        "5",
    ]);
    let text = listed.to_string();
    assert!(text.contains("nidus.summary"), "{listed}");
    assert!(text.contains("x + 1"), "the body is still stored: {listed}");
}

/// The spend guard: a budget too small for a file's `1 + symbols` calls embeds that file raw
/// and says so, rather than half-summarizing it and leaving two kinds of vector in one
/// corpus with nothing recording which is which.
#[test]
fn a_file_the_summarize_budget_cannot_cover_is_embedded_raw_and_reported() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, root) = (tmp.path().join("store"), tmp.path().join("repo"));
    write(
        &root.join("src/lib.rs"),
        "pub fn a() -> i32 {\n    1\n}\n\npub fn b() -> i32 {\n    2\n}\n",
    );

    let summarizer = summarize_mocks::start(|_| {
        serde_json::json!({
            "choices": [{ "message": { "role": "assistant", "content": "summary" } }]
        })
        .to_string()
    });
    let embedder = summarize_mocks::start(|_| {
        serde_json::json!({ "embeddings": [[0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] })
            .to_string()
    });

    let out = run(&[
        "code",
        "ingest",
        root.to_str().expect("root"),
        "--dir",
        store.to_str().expect("store"),
        "--dim",
        "8",
        "--embed-provider",
        "ollama",
        "--embed-base-url",
        &embedder.url,
        "--summarize",
        "--summarize-provider",
        "openai",
        "--summarize-api-key",
        "test-key",
        "--summarize-base-url",
        &summarizer.url,
        // The file needs three calls (one file plus two symbols); two is not enough.
        "--summarize-budget",
        "2",
    ]);
    assert!(
        out.status.success(),
        "ingest failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Value = serde_json::from_slice(&out.stdout).expect("ingest report is json");
    assert_eq!(report["summarize"]["files_over_budget"], 1, "{report}");
    assert_eq!(report["summarize"]["files_summarized"], 0, "{report}");
    assert_eq!(
        report["summarize"]["model_calls"], 0,
        "not one call may be spent on a file it cannot finish: {report}"
    );
    assert!(
        summarizer.bodies().is_empty(),
        "the provider must not be called at all for a file over budget"
    );
    // The bodies were embedded instead, so the file is still searchable.
    assert!(
        embedder.bodies().iter().any(|b| b.contains("1")),
        "the raw bodies must still be embedded: {:?}",
        embedder.bodies()
    );
}
