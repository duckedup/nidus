//! `nidus ingest --fts-only` through the real binary (nidus-gmy.6). The claim these tests
//! exist for is a *negative* one — that the whole path runs with no embedding provider, no
//! API key and no network call — and the only way to prove it is to run the shipped binary
//! with every `NIDUS_*` variable stripped and no `--embed-*` flag, which is what `run` does.

#![cfg(feature = "memory")]

use std::path::Path;
use std::process::{Command, Output, Stdio};

use serde_json::Value;

/// The binary with a cleared environment: no `NIDUS_EMBED_PROVIDER`, no key, nothing a
/// developer's shell might be exporting. A test that inherited those would pass here and
/// fail in CI, proving nothing about the no-provider path.
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

/// One `--fts-only` ingest over `corpus` into `store`. No `--dim` and no `--embed-*`: the
/// store is created by this call, and the flags it is *not* given are the point.
fn ingest_fts(store: &Path, corpus: &Path, fields: &[&str], extra: &[&str]) -> Value {
    ingest_fts_into(store, corpus, "docs", fields, extra)
}

fn ingest_fts_into(
    store: &Path,
    corpus: &Path,
    collection: &str,
    fields: &[&str],
    extra: &[&str],
) -> Value {
    let (store, corpus) = (store.to_string_lossy(), corpus.to_string_lossy());
    let mut args: Vec<&str> = vec![
        "ingest",
        &corpus,
        "--collection",
        collection,
        "--glob",
        "**/*.md",
        "--dir",
        &store,
        "--strategy",
        "markdown",
        "--max-chars",
        "200",
        "--overlap-chars",
        "0",
    ];
    for field in fields {
        args.extend_from_slice(&["--fts-only", field]);
    }
    args.extend_from_slice(extra);
    ok_json(&args)
}

fn text_search(store: &Path, query: &str) -> Vec<String> {
    text_search_field(store, "nidus.text", query)
}

/// BM25 over an arbitrary declared field, so a test can ask whether a *widened* schema
/// actually reached the index rather than trusting the ingest report that says it did.
fn text_search_field(store: &Path, field: &str, query: &str) -> Vec<String> {
    let store = store.to_string_lossy();
    let hits = ok_json(&["text-search", "--dir", &store, field, query, "--top-k", "5"]);
    hits.as_array()
        .expect("text-search returns an array")
        .iter()
        .map(|h| h["id"].as_str().expect("id").to_string())
        .collect()
}

fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(path, body).expect("write");
}

/// Two files with disjoint vocabulary, so a hit on one is attributable to the query and not
/// to "the only document in the store".
fn corpus(root: &Path) {
    write(
        &root.join("mmap.md"),
        "# Memory mapping\n\nWhen the corpus outgrows RAM, nidus maps the vector segments \
         instead of loading them.\n",
    );
    write(
        &root.join("durability.md"),
        "# Durability\n\nPer-batch fsync and an append-only log mean a crash loses at most \
         the in-flight batch.\n",
    );
}

/// nidus-gmy.6's central claim: a corpus becomes keyword-searchable with no embedder at all.
/// Asserting the *ranking* rather than "the command exited 0" — a run that ingested nothing
/// would also exit 0, and did while this was being written.
#[test]
fn an_fts_only_ingest_needs_no_provider_and_answers_bm25() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, docs) = (tmp.path().join("store"), tmp.path().join("docs"));
    corpus(&docs);

    let report = ingest_fts(&store, &docs, &["nidus.text"], &[]);
    assert_eq!(report["ingested"], 2, "both files ingested: {report}");
    assert_eq!(report["chunks"], 2, "one chunk each: {report}");
    assert_eq!(
        report["embedder"], "fts-only:nidus.text",
        "the digest identity names the declared field set: {report}"
    );

    assert_eq!(
        text_search(&store, "corpus outgrows RAM"),
        vec!["mmap.md#0".to_string()],
        "BM25 must find the mmap doc and only it"
    );
    assert_eq!(
        text_search(&store, "crash in-flight batch"),
        vec!["durability.md#0".to_string()],
        "and the durability doc for its own vocabulary"
    );
}

/// Every chunk is a text-only record, so the store holds documents and *no* vector rows.
/// `doc_count` alone would pass if each chunk carried a filler vector — which is the failure
/// this path exists to avoid, since filler vectors silently poison search and hybrid-search.
#[test]
fn fts_only_records_occupy_no_vector_rows() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, docs) = (tmp.path().join("store"), tmp.path().join("docs"));
    corpus(&docs);
    ingest_fts(&store, &docs, &["nidus.text"], &[]);

    let stats = ok_json(&["stats", "--dir", &store.to_string_lossy()]);
    assert_eq!(stats["footprint"]["doc_count"], 2, "both chunks stored");
    assert_eq!(
        stats["footprint"]["rows"], 0,
        "a text-only record takes no data row: {stats}"
    );
    assert_eq!(
        stats["footprint"]["vector_bytes"], 0,
        "and no vector bytes: {stats}"
    );
    assert_eq!(
        stats["dimension"], 0,
        "a fresh fts-only store declares no embedding space: {stats}"
    );
}

/// A vector query must say *why* it cannot be answered. Before the guard, a zero-length query
/// matched the zero dimension and came back as an empty ranking — indistinguishable from
/// "your query matched nothing", which is the wrong thing for a caller to conclude.
#[test]
fn a_vector_query_against_an_fts_only_store_is_refused_with_the_reason() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, docs) = (tmp.path().join("store"), tmp.path().join("docs"));
    corpus(&docs);
    ingest_fts(&store, &docs, &["nidus.text"], &[]);

    // Through a query *file*: `search` reads stdin otherwise, and a closed stdin fails at
    // JSON parsing before the store is ever consulted — passing the test for the wrong reason.
    let query = tmp.path().join("query.json");
    write(&query, "[0.1, 0.2, 0.3]");
    let out = run(&[
        "search",
        "--dir",
        &store.to_string_lossy(),
        "--query-file",
        &query.to_string_lossy(),
        "--top-k",
        "3",
    ]);
    assert!(
        !out.status.success(),
        "a vector query must not succeed here"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("dimension 0") && stderr.contains("fts-only"),
        "the error must name the cause and the flag that produced it, got: {stderr}"
    );
}

/// The re-ingest skip has to survive having no embedder to hash: `source_hash` folds the
/// embedder identity, so an unstable stand-in would rewrite the whole corpus on every run.
#[test]
fn an_fts_only_reingest_of_an_unchanged_tree_writes_nothing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, docs) = (tmp.path().join("store"), tmp.path().join("docs"));
    corpus(&docs);
    ingest_fts(&store, &docs, &["nidus.text"], &[]);

    let second = ingest_fts(&store, &docs, &["nidus.text"], &[]);
    assert_eq!(second["matched"], 2, "the walk still sees both files");
    assert_eq!(second["ingested"], 0, "and rewrites neither: {second}");
    assert_eq!(second["unchanged"], 2, "both skipped: {second}");
}

/// Changing which attrs are full-text indexed changes what the index can answer, so it must
/// re-ingest. Without the field set in the digest the chunks stay indexed under the old
/// schema and the new field silently returns nothing.
#[test]
fn changing_the_fts_field_set_reingests() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, docs) = (tmp.path().join("store"), tmp.path().join("docs"));
    corpus(&docs);
    ingest_fts(&store, &docs, &["nidus.text"], &[]);

    let widened = ingest_fts(&store, &docs, &["nidus.text", "nidus.source_path"], &[]);
    assert_eq!(
        widened["ingested"], 2,
        "a changed field set must re-ingest, not skip: {widened}"
    );
    assert_eq!(widened["unchanged"], 0, "nothing may be skipped: {widened}");

    // The counts above pass while the widened field is silently absent from the index.
    // `.md` is a shared token, so both match; the claim is the field is indexed at all and
    // ranks the right doc first. Before the fix this came back empty.
    let by_path = text_search_field(&store, "nidus.source_path", "mmap.md");
    assert_eq!(
        by_path.first().map(String::as_str),
        Some("mmap.md#0"),
        "the newly declared field must actually be searchable, not merely re-ingested: \
         {by_path:?}"
    );
    assert_eq!(
        text_search(&store, "corpus outgrows RAM"),
        vec!["mmap.md#0".to_string()],
        "and the original field must keep working"
    );
}

/// `set_fts_schema` refuses an alias outright (nidus-klh), so the declare must resolve one
/// first. Without that, `--fts-only` into any aliased collection fails on every single run —
/// which is exactly the blue/green target the docs-index use case points at.
#[test]
fn an_fts_only_ingest_works_through_a_collection_alias() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, docs) = (tmp.path().join("store"), tmp.path().join("docs"));
    corpus(&docs);
    let dir = store.to_string_lossy().to_string();

    ingest_fts_into(&store, &docs, "docs_v1", &["nidus.text"], &[]);
    ok_json(&["set-alias", "docs", "docs_v1", "--dir", &dir]);

    write(
        &docs.join("later.md"),
        "# Compaction\n\nRewriting the base segment reclaims the rows tombstoned by \
         deletes.\n",
    );
    // Through the ALIAS. Before the fix this exited 1 on every run with `set_fts_schema`
    // requires a concrete collection, so the assertion is that it lands at all.
    let report = ingest_fts_into(&store, &docs, "docs", &["nidus.text"], &[]);
    assert_eq!(
        report["ingested"], 1,
        "only the new file is written through the alias: {report}"
    );
    assert_eq!(
        report["unchanged"], 2,
        "the first two are skipped: {report}"
    );
    assert_eq!(
        text_search(&store, "reclaims tombstoned rows"),
        vec!["later.md#0".to_string()],
        "the alias-routed chunk must be full-text searchable"
    );
}

/// Field order is not meaning: `--fts-only a --fts-only b` and the reverse declare the same
/// schema, so the digest must not treat them as different and re-ingest the whole corpus.
#[test]
fn the_fts_field_digest_ignores_the_order_the_fields_were_given_in() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, docs) = (tmp.path().join("store"), tmp.path().join("docs"));
    corpus(&docs);
    ingest_fts(&store, &docs, &["nidus.text", "nidus.source_path"], &[]);

    let reordered = ingest_fts(&store, &docs, &["nidus.source_path", "nidus.text"], &[]);
    assert_eq!(
        reordered["unchanged"], 2,
        "reordering the same fields must skip, not rewrite: {reordered}"
    );
}

/// `--fts-only` and `--embed-provider` describe incompatible runs, and clap must say so
/// before any walking happens rather than letting one silently win.
#[test]
fn fts_only_and_an_embed_provider_conflict_at_parse_time() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (store, docs) = (tmp.path().join("store"), tmp.path().join("docs"));
    corpus(&docs);

    let out = run(&[
        "ingest",
        &docs.to_string_lossy(),
        "--collection",
        "docs",
        "--dir",
        &store.to_string_lossy(),
        "--fts-only",
        "nidus.text",
        "--embed-provider",
        "ollama",
    ]);
    assert!(!out.status.success(), "the two flags must not be accepted");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot be used with"),
        "clap must report the conflict, got: {stderr}"
    );
    assert!(!store.exists(), "the run must fail before creating a store");
}
