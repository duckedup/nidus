//! `nidus ingest` through the real binary (nidus-lvo.2 + nidus-lvo.3). The load-bearing
//! claim — a re-run over an unchanged tree makes no embedding calls and no writes — is only
//! provable by counting what the provider actually received, which no in-process test can
//! see.

#![cfg(feature = "embed-ollama")]

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::harness::{ok_json, read_request_body, run_no_stdin};

const DIM: usize = 8;
/// The ollama adapter embeds this literal once per process to learn its own dimension
/// (`src/embed/ollama.rs:28-36`). It is never corpus content, so it is excluded from every
/// count here — the assertions are about which *chunks* were sent.
const PROBE: &str = "dimension probe";

/// A mock embedder that records every text it was asked for. The count alone is not enough:
/// the cache tests assert on *which* chunks were sent.
struct Recorder {
    url: String,
    texts: Arc<Mutex<Vec<String>>>,
}

impl Recorder {
    fn start() -> Recorder {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock embedder");
        let addr = listener.local_addr().expect("mock embedder addr");
        let texts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&texts);
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let body = read_request_body(&mut stream);
                let text = requested_text(&body);
                sink.lock().expect("mock sink").push(text.clone());
                let vector: Vec<f64> = vector_for(&text).into_iter().map(f64::from).collect();
                write_json_response(stream, &json!({ "embeddings": [vector] }).to_string());
            }
        });
        Recorder {
            url: format!("http://{addr}"),
            texts,
        }
    }

    /// Corpus texts the provider was asked to embed, probe excluded.
    fn documents(&self) -> Vec<String> {
        self.texts
            .lock()
            .expect("mock sink")
            .iter()
            .filter(|t| *t != PROBE)
            .cloned()
            .collect()
    }

    fn reset(&self) {
        self.texts.lock().expect("mock sink").clear();
    }
}

/// A per-text deterministic vector, so a wrong-vector-for-a-chunk bug is visible downstream.
fn vector_for(text: &str) -> Vec<f32> {
    let mut v = vec![0.1f32; DIM];
    for (i, b) in text.bytes().enumerate() {
        v[i % DIM] += (b as f32) + 1.0;
    }
    v
}

fn write_json_response(mut stream: TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
}

/// The text Ollama's `/api/embed` request carries in `input` (`src/embed/ollama.rs`).
fn requested_text(body: &[u8]) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v["input"].as_str().map(str::to_string))
        .unwrap_or_default()
}

/// One ingest invocation over `corpus` into `store`, plus whatever extra flags a test needs.
fn ingest(store: &Path, corpus: &Path, url: &str, extra: &[&str]) -> Value {
    let (store, corpus) = (store.to_string_lossy(), corpus.to_string_lossy());
    let mut args: Vec<&str> = vec![
        "ingest",
        &corpus,
        "--collection",
        "docs",
        "--glob",
        "**/*.md",
        "--dir",
        &store,
        "--dim",
        "8",
        "--embed-provider",
        "ollama",
        "--embed-base-url",
        url,
        "--max-chars",
        "60",
        "--overlap-chars",
        "0",
    ];
    args.extend_from_slice(extra);
    ok_json(&args)
}

/// The record ids currently in the collection, sorted.
fn ids(store: &Path) -> Vec<String> {
    let store = store.to_string_lossy();
    let hits = ok_json(&["list", "docs", "--dir", &store, "--limit", "500"]);
    let mut out: Vec<String> = hits
        .as_array()
        .expect("list returns an array")
        .iter()
        .map(|h| h["id"].as_str().expect("id").to_string())
        .collect();
    out.sort();
    out
}

fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(path, body).expect("write");
}

/// A file long enough to chunk into several pieces at `--max-chars 60`, one paragraph per
/// chunk so an edit can be localised to exactly one of them.
fn paragraphs(n: usize, last: &str) -> String {
    let mut out: Vec<String> = (0..n - 1)
        .map(|i| format!("paragraph {i} with some distinct filler words"))
        .collect();
    out.push(last.to_string());
    out.join("\n\n")
}

/// nidus-lvo.2's central criterion: a re-run over an unchanged tree does zero embedding work
/// and zero writes. Both halves are asserted separately, because they have different causes.
#[test]
fn a_reingest_of_an_unchanged_tree_embeds_nothing_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("a.md"), "alpha one two three");
    write(&corpus.join("sub/b.md"), "beta four five six");
    write(&corpus.join("c.md"), "gamma seven eight nine");
    let mock = Recorder::start();

    let first = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(first["ingested"], 3, "first run must ingest: {first}");
    assert_eq!(first["unchanged"], 0);
    assert!(first["chunks"].as_u64().unwrap() >= 3, "chunks: {first}");
    assert_eq!(mock.documents().len(), 3, "one chunk per file was embedded");
    let before = ids(&store);

    mock.reset();
    let second = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(
        mock.documents(),
        Vec::<String>::new(),
        "a re-run over an unchanged tree must embed nothing"
    );
    assert_eq!(second["ingested"], 0, "and write nothing: {second}");
    assert_eq!(second["unchanged"], 3);
    assert_eq!(second["chunks"], 0);
    assert_eq!(ids(&store), before, "the record set must be untouched");
}

/// The over-correction the test above cannot see: a skip check that skips everything.
#[test]
fn a_changed_file_re_embeds_only_itself() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("a.md"), "alpha one two three");
    write(&corpus.join("b.md"), "beta four five six");
    write(&corpus.join("c.md"), "gamma seven eight nine");
    let mock = Recorder::start();
    ingest(&store, &corpus, &mock.url, &[]);

    mock.reset();
    write(
        &corpus.join("b.md"),
        "beta rewritten entirely different now",
    );
    let out = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(out["ingested"], 1, "only the edited file: {out}");
    assert_eq!(out["unchanged"], 2);
    assert_eq!(
        mock.documents(),
        vec!["beta rewritten entirely different now".to_string()],
        "and only its text reached the provider"
    );
}

/// nidus-lvo.3: the cache is what makes a file that *did* change cost only its changed
/// chunks. `--no-cache` is the counterfactual proving the cache is what did it.
#[test]
fn an_edited_file_re_embeds_only_its_changed_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    let long = corpus.join("long.md");
    write(&long, &paragraphs(6, "paragraph five original ending text"));
    let mock = Recorder::start();

    let first = ingest(&store, &corpus, &mock.url, &[]);
    let chunks = first["chunks"].as_u64().expect("chunks") as usize;
    assert!(
        chunks >= 5,
        "the fixture must produce several chunks: {first}"
    );
    assert_eq!(mock.documents().len(), chunks, "all of them embedded once");

    mock.reset();
    write(
        &long,
        &paragraphs(6, "paragraph five REWRITTEN ending text here"),
    );
    let cached = ingest(&store, &corpus, &mock.url, &[]);
    let sent = mock.documents().len();
    assert!(
        sent < chunks,
        "the cache must spare the untouched chunks: sent {sent} of {chunks}"
    );
    assert!(
        cached["cache"]["hits"].as_u64().unwrap() > 0,
        "and report the hits it served: {cached}"
    );

    // Same shape of edit, cache disabled: the whole file goes to the provider again.
    mock.reset();
    write(
        &long,
        &paragraphs(6, "paragraph five REWRITTEN a third time now"),
    );
    let uncached = ingest(&store, &corpus, &mock.url, &["--no-cache"]);
    assert_eq!(
        mock.documents().len(),
        uncached["chunks"].as_u64().unwrap() as usize,
        "--no-cache must re-embed every chunk: {uncached}"
    );
    assert_eq!(uncached["cache"]["hits"], 0);
}

/// `--prune` is opt-in because pointing ingest at a partial tree must not empty the
/// collection, and it must never reach a record this command did not write.
#[test]
fn prune_is_opt_in_and_spares_hand_written_memories() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("a.md"), "alpha one two three");
    write(&corpus.join("gone.md"), "this file will be deleted");
    let mock = Recorder::start();
    ingest(&store, &corpus, &mock.url, &[]);

    let store_s = store.to_string_lossy().to_string();
    ok_json(&[
        "remember",
        "docs",
        "a fact nobody ingested",
        "--id",
        "handmade",
        "--dir",
        &store_s,
        "--embed-provider",
        "ollama",
        "--embed-base-url",
        &mock.url,
    ]);
    std::fs::remove_file(corpus.join("gone.md")).unwrap();

    let without = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(without["pruned"], 0, "prune must not happen unasked");
    assert!(
        ids(&store).contains(&"gone.md#0".to_string()),
        "the record survives an unpruned run: {:?}",
        ids(&store)
    );

    let with = ingest(&store, &corpus, &mock.url, &["--prune"]);
    assert_eq!(
        with["pruned"], 1,
        "the gone file's record is removed: {with}"
    );
    let after = ids(&store);
    assert!(
        !after.iter().any(|id| id.starts_with("gone.md#")),
        "{after:?}"
    );
    assert!(
        after.contains(&"a.md#0".to_string()),
        "the live file stays: {after:?}"
    );
    assert!(
        after.contains(&"handmade".to_string()),
        "a record without nidus.source_hash is not ingest's to delete: {after:?}"
    );
}

/// Chunk 0 carries the digest and is written *first* (nidus-lvo.5: `remember_chunked` is not
/// atomic), so a digest-only check reads a torn document as complete forever. Deleting the
/// tail is exactly the state a crash mid-write leaves behind.
#[test]
fn a_torn_document_re_ingests_rather_than_looking_complete() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(
        &corpus.join("long.md"),
        &paragraphs(6, "paragraph five original ending text"),
    );
    let mock = Recorder::start();
    let first = ingest(&store, &corpus, &mock.url, &[]);
    let chunks = first["chunks"].as_u64().unwrap() as usize;
    assert!(chunks >= 3, "the fixture must chunk: {first}");

    // Tear the document: drop every chunk above 0, leaving the digest-bearing one behind.
    let store_s = store.to_string_lossy().to_string();
    let tail: Vec<String> = (1..chunks).map(|i| format!("long.md#{i}")).collect();
    let mut args = vec!["delete", "docs", "--dir", &store_s];
    args.extend(tail.iter().map(|s| s.as_str()));
    ok_json(&args);
    assert_eq!(
        ids(&store),
        vec!["long.md#0".to_string()],
        "torn as intended"
    );

    mock.reset();
    let out = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(
        out["unchanged"], 0,
        "a half-written document must not read as unchanged: {out}"
    );
    assert_eq!(out["ingested"], 1, "{out}");
    assert_eq!(ids(&store).len(), chunks, "and the whole document is back");
}

/// An empty or whitespace-only file writes nothing, so it has no chunk 0 to carry a digest.
/// It must be reported rather than counted as ingested, and must never embed.
#[test]
fn an_empty_file_is_reported_and_never_embedded() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("blank.md"), "   \n\t\n  ");
    write(&corpus.join("real.md"), "actual content here");
    let mock = Recorder::start();

    let out = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(out["matched"], 2, "{out}");
    assert_eq!(out["skipped_empty"], 1, "{out}");
    assert_eq!(out["ingested"], 1, "{out}");
    assert_eq!(mock.documents().len(), 1, "only the real file was embedded");
    assert_eq!(ids(&store), vec!["real.md#0".to_string()]);
}

/// A file that shrank must not leave its old high-index chunks searchable.
#[test]
fn a_shrunk_file_drops_its_stale_tail() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    let long = corpus.join("long.md");
    write(&long, &paragraphs(6, "paragraph five original ending text"));
    let mock = Recorder::start();
    let first = ingest(&store, &corpus, &mock.url, &[]);
    let chunks = first["chunks"].as_u64().unwrap() as usize;
    assert_eq!(ids(&store).len(), chunks);

    write(&long, "now tiny");
    let out = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(out["chunks"], 1, "one chunk now: {out}");
    assert_eq!(
        out["stale_tail_pruned"].as_u64().unwrap() as usize,
        chunks - 1,
        "every higher index removed: {out}"
    );
    assert_eq!(ids(&store), vec!["long.md#0".to_string()]);
}

/// A binary file in the tree is counted and skipped, never fatal.
#[test]
fn a_non_utf8_file_is_skipped_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("good.md"), "good readable text here");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::write(corpus.join("bad.md"), [0xff, 0xfe, 0x00, 0x01, b'x']).unwrap();
    let mock = Recorder::start();

    let out = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(out["matched"], 2, "both files matched the glob: {out}");
    assert_eq!(out["skipped_non_utf8"], 1, "{out}");
    assert_eq!(out["ingested"], 1, "the good one still landed: {out}");
    assert_eq!(ids(&store), vec!["good.md#0".to_string()]);
}

/// `--dry-run` is what you reach for before a `--prune`, so it must touch nothing at all.
#[test]
fn dry_run_embeds_nothing_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("a.md"), "alpha one two three");
    write(&corpus.join("b.md"), "beta four five six");
    let mock = Recorder::start();

    let out = ingest(&store, &corpus, &mock.url, &["--dry-run"]);
    assert_eq!(out["would_ingest"], 2, "{out}");
    assert_eq!(out["ingested"], 0);
    assert_eq!(out["dry_run"], true);
    assert_eq!(
        mock.documents(),
        Vec::<String>::new(),
        "a dry run must not embed"
    );
    assert!(ids(&store).is_empty(), "nor write: {:?}", ids(&store));
}

/// By default, dot-entries are skipped so `nidus ingest .` does not walk `.git`; symlinks are
/// skipped so a cycle cannot hang the walk. Both are behaviours of the real walk over a real
/// tree, unchanged since before nidus-0fw.
#[test]
fn the_walk_skips_dot_entries_and_symlinks_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("a.md"), "alpha one two three");
    write(&corpus.join(".git/HEAD.md"), "ref: refs/heads/main");
    write(&corpus.join(".hidden.md"), "hidden");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&corpus, corpus.join("loop")).unwrap();
    let mock = Recorder::start();

    let out = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(out["matched"], 1, "only the visible file: {out}");
    assert_eq!(ids(&store), vec!["a.md#0".to_string()]);
}

/// With `--include-hidden`, a dot-directory is walked, but `.git` stays skipped at any
/// depth and the symlink is still skipped (nidus-0fw). The case that fails if the flag
/// were implemented as "drop the dot-check" instead of naming `.git` specifically.
#[test]
fn the_walk_with_include_hidden_reaches_dot_directories_but_never_git() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("a.md"), "alpha one two three");
    write(&corpus.join(".git/HEAD.md"), "ref: refs/heads/main");
    write(&corpus.join(".claude/rules/x.md"), "a hidden rule file");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&corpus, corpus.join("loop")).unwrap();
    let mock = Recorder::start();

    let out = ingest(&store, &corpus, &mock.url, &["--include-hidden"]);
    assert_eq!(
        out["matched"], 2,
        "the visible file and the hidden one: {out}"
    );
    assert_eq!(
        ids(&store),
        vec![".claude/rules/x.md#0".to_string(), "a.md#0".to_string()],
        ".git must never be walked, flag or no flag"
    );
}

/// A model swap into a collection that already has vectors is refused outright — the store
/// pins `nidus.embedder` per collection, and vectors from two models are not comparable.
/// Better than re-embedding: the mistake is caught instead of silently ranked.
#[test]
fn a_model_swap_into_an_existing_collection_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(&corpus.join("a.md"), "alpha one two three");
    let mock = Recorder::start();
    ingest(&store, &corpus, &mock.url, &[]);
    let before = ids(&store);

    let (store_s, corpus_s) = (store.to_string_lossy(), corpus.to_string_lossy());
    let out = run_no_stdin(&[
        "ingest",
        &corpus_s,
        "--collection",
        "docs",
        "--glob",
        "**/*.md",
        "--dir",
        &store_s,
        "--dim",
        "8",
        "--embed-provider",
        "ollama",
        "--embed-base-url",
        &mock.url,
        "--max-chars",
        "60",
        "--overlap-chars",
        "0",
        "--embed-model",
        "another-model",
    ]);
    assert!(
        !out.status.success(),
        "a model swap must not silently proceed"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not comparable"),
        "the error must name why, not just fail: {stderr}"
    );
    assert_eq!(
        ids(&store),
        before,
        "and must leave the collection untouched"
    );
}

/// A `--max-chars` change re-chunks, so the stored records no longer describe the file: the
/// digest covers the chunk options for exactly this reason, and the run must not read as
/// "unchanged".
#[test]
fn changing_the_chunk_options_re_ingests() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    write(
        &corpus.join("long.md"),
        &paragraphs(6, "paragraph five original ending text"),
    );
    let mock = Recorder::start();
    let first = ingest(&store, &corpus, &mock.url, &[]);
    assert_eq!(first["unchanged"], 0);

    mock.reset();
    let (store_s, corpus_s) = (store.to_string_lossy(), corpus.to_string_lossy());
    let wider: Value = ok_json(&[
        "ingest",
        &corpus_s,
        "--collection",
        "docs",
        "--glob",
        "**/*.md",
        "--dir",
        &store_s,
        "--dim",
        "8",
        "--embed-provider",
        "ollama",
        "--embed-base-url",
        &mock.url,
        "--max-chars",
        "500",
        "--overlap-chars",
        "0",
    ]);
    assert_eq!(
        wider["unchanged"], 0,
        "a re-chunk is not 'unchanged': {wider}"
    );
    assert_eq!(wider["ingested"], 1, "{wider}");
    assert_ne!(
        wider["chunks"], first["chunks"],
        "and the chunking actually differs: {wider}"
    );
}

/// One `nidus recall` over the real binary, with whatever read flags the test needs.
fn recall(store: &Path, url: &str, query: &str, extra: &[&str]) -> Value {
    let store = store.to_string_lossy();
    let mut args: Vec<&str> = vec![
        "recall",
        "docs",
        query,
        "--dir",
        &store,
        "--dim",
        "8",
        "--embed-provider",
        "ollama",
        "--embed-base-url",
        url,
    ];
    args.extend_from_slice(extra);
    ok_json(&args)
}

/// The epic's own arc, over the real binary: `ingest` a tree, then read it back as
/// documents. Without `--rollup` the corpus answers in chunks, which is the shape every RAG
/// application then has to collapse by hand.
#[test]
fn ingest_then_recall_with_rollup_returns_one_hit_per_document() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    let mock = Recorder::start();
    for doc in ["a", "b"] {
        write(&corpus.join(format!("{doc}.md")), &paragraphs(4, "tail"));
    }
    ingest(&store, &corpus, &mock.url, &[]);

    let raw = recall(&store, &mock.url, "paragraph 1", &["-k", "20"]);
    let hits = raw.as_array().expect("recall returns an array");
    assert!(hits.len() > 2, "chunk hits, not document hits: {raw}");

    let rolled = recall(
        &store,
        &mock.url,
        "paragraph 1",
        &["-k", "20", "--rollup", "1"],
    );
    let rolled = rolled.as_array().expect("recall returns an array");
    let mut parents: Vec<&str> = rolled
        .iter()
        .map(|h| {
            h["attrs"]["nidus.parent_id"]["Str"]
                .as_str()
                .expect("parent")
        })
        .collect();
    parents.sort();
    parents.dedup();
    assert_eq!(
        rolled.len(),
        parents.len(),
        "one hit per document: {rolled:?}"
    );
    assert_eq!(parents.len(), 2, "both documents present: {parents:?}");
}

/// The ticket's provable-ordering criterion at the real wire: expansion changes the payload
/// and nothing else. Compares the `(id, score)` sequence, not just the count.
#[test]
fn neighbour_expansion_widens_a_hit_without_reordering_it() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    let mock = Recorder::start();
    write(&corpus.join("a.md"), &paragraphs(5, "tail"));
    ingest(&store, &corpus, &mock.url, &[]);

    let ranking = |v: &Value| -> Vec<(String, f64)> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|h| {
                (
                    h["id"].as_str().expect("id").to_string(),
                    h["score"].as_f64().expect("score"),
                )
            })
            .collect()
    };
    let plain = recall(&store, &mock.url, "paragraph 2", &["-k", "10"]);
    let widened = recall(
        &store,
        &mock.url,
        "paragraph 2",
        &["-k", "10", "--rollup", "10", "--neighbours", "1"],
    );

    assert_eq!(ranking(&plain), ranking(&widened), "payload only");
    for hit in plain.as_array().expect("array") {
        assert!(hit.get("context").is_none(), "no context asked for: {hit}");
    }
    for hit in widened.as_array().expect("array") {
        let own = hit["attrs"]["nidus.text"]["Str"].as_str().expect("text");
        let context = hit["context"].as_str().expect("context");
        assert!(context.contains(own), "the winning chunk is in its window");
    }
    // The middle chunk's window reaches its neighbours, which its own text does not.
    let widened = widened.as_array().expect("array");
    let middle = widened
        .iter()
        .find(|h| h["id"] == json!("a.md#2"))
        .expect("chunk 2 ranked");
    let context = middle["context"].as_str().expect("context");
    assert!(context.contains("paragraph 1"), "{context}");
    assert!(context.contains("paragraph 3"), "{context}");
}

/// Expansion is keyed on `(parent_id, chunk_index)`, so two documents whose indices overlap
/// must not bleed into each other — a bug a single-document corpus cannot see.
#[test]
fn expansion_does_not_cross_a_document_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    let mock = Recorder::start();
    write(&corpus.join("a.md"), &paragraphs(4, "alpha tail marker"));
    write(
        &corpus.join("b.md"),
        &paragraphs(4, "bravo tail CONTAMINANT"),
    );
    ingest(&store, &corpus, &mock.url, &[]);

    let hits = recall(
        &store,
        &mock.url,
        "alpha tail marker",
        &["-k", "20", "--rollup", "10", "--neighbours", "5"],
    );
    for hit in hits.as_array().expect("array") {
        let id = hit["id"].as_str().expect("id");
        let context = hit["context"].as_str().expect("context");
        if id.starts_with("a.md#") {
            assert!(
                !context.contains("CONTAMINANT"),
                "a.md#'s window pulled b.md's text: {context}"
            );
        }
    }
}

/// The de-overlap: with `--overlap-chars` set, a stitched window must be the source once,
/// not the source with every seam repeated. A length check would not catch a wrong trim.
#[test]
fn an_expanded_window_does_not_repeat_the_chunk_overlap() {
    let dir = tempfile::tempdir().unwrap();
    let (corpus, store) = (dir.path().join("corpus"), dir.path().join("store"));
    let mock = Recorder::start();
    let body = paragraphs(4, "tail");
    write(&corpus.join("a.md"), &body);
    // Overlapping chunks: the naive concatenation would repeat 20 chars at each seam. Spelled
    // out rather than via `ingest`, whose baked-in `--overlap-chars 0` cannot be overridden.
    let (store_s, corpus_s) = (store.to_string_lossy(), corpus.to_string_lossy());
    ok_json(&[
        "ingest",
        &corpus_s,
        "--collection",
        "docs",
        "--glob",
        "**/*.md",
        "--dir",
        &store_s,
        "--dim",
        "8",
        "--embed-provider",
        "ollama",
        "--embed-base-url",
        &mock.url,
        "--max-chars",
        "60",
        "--overlap-chars",
        "20",
    ]);

    let hits = recall(
        &store,
        &mock.url,
        "paragraph 1",
        &["-k", "20", "--rollup", "10", "--neighbours", "10"],
    );
    let hits = hits.as_array().expect("array");
    let widest = hits
        .iter()
        .map(|h| h["context"].as_str().expect("context"))
        .max_by_key(|c| c.len())
        .expect("at least one hit");
    // The whole document is one contiguous slice of the source, seams and all.
    assert_eq!(
        widest,
        body.as_str(),
        "the window must be the source once, not once per seam"
    );
}
