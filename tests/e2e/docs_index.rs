//! `scripts/docs-index.sh` and the ranked tier it feeds under `.claude/skills/nidus/bin/spec`
//! (nidus-3gm unit 11): the collapse from three staged `--fts-only` ingests into ONE
//! `code ingest --include-hidden` over the repo root. Runs the real script and the real
//! `spec` tool as subprocesses against this actual checkout: an ingest whose whole point is
//! "the repo root" cannot be sandboxed away from the repo it is testing.
//!
//! A run-the-script-and-check-exit-code test proves nothing (the old three-ingest script
//! exited 0 too), so every assertion here is a counterfactual: content that must be
//! findable, a digest that must change, `spec find` that must answer differently depending
//! on whether the index exists and whether it is fresh.

#![cfg(all(feature = "memory", feature = "code"))]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn run_script(args: &[&str]) -> Output {
    Command::new("bash")
        .arg("scripts/docs-index.sh")
        .args(args)
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("spawn docs-index.sh {args:?}: {e}"))
}

/// `--digest` is pure `git ls-files` + `shasum`, no cargo and no walk, so it is cheap and
/// safe to call on its own to probe the sentinel math without paying for a full build.
fn digest() -> String {
    let out = run_script(&["--digest"]);
    assert!(
        out.status.success(),
        "docs-index.sh --digest failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn nidus(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(args)
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .envs(std::env::vars().filter(|(k, _)| !k.starts_with("NIDUS_")))
        .output()
        .unwrap_or_else(|e| panic!("spawn nidus {args:?}: {e}"))
}

fn text_search(query: &str) -> Vec<Value> {
    let out = nidus(&[
        "text-search",
        "--dir",
        "target/docs-index",
        "nidus.text",
        query,
        "--top-k",
        "10",
    ]);
    assert!(
        out.status.success(),
        "text-search {query:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("text-search stdout is JSON")
}

/// The digest the store's `meta` sentinel record carries, mirroring what
/// `.claude/skills/nidus/lib/spec.mjs`'s `recordedDigest()` reads back.
fn recorded_digest() -> Option<String> {
    let out = nidus(&["get", "meta", "--dir", "target/docs-index"]);
    if !out.status.success() {
        return None;
    }
    let rows: Value = serde_json::from_slice(&out.stdout).ok()?;
    rows.as_array()?
        .iter()
        .find(|r| r["id"] == "docs-index.digest")?
        .get("attrs")?
        .get("digest")?
        .get("Str")?
        .as_str()
        .map(str::to_string)
}

fn spec_find(query: &str) -> Output {
    Command::new(repo_root().join(".claude/skills/nidus/bin/spec"))
        .args(["find", query])
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("spawn bin/spec find {query:?}: {e}"))
}

/// Stages a scratch file (adds it to what `git ls-files` names) without touching any
/// existing file's content, and always unstages and removes it again on drop -- a panic
/// midway through the test must not leave the real checkout's git state dirty.
struct StagedScratch {
    repo: PathBuf,
    rel: &'static str,
}

impl StagedScratch {
    fn new(repo: &Path, rel: &'static str) -> Self {
        std::fs::write(repo.join(rel), "# docs-index e2e scratch\n").expect("write scratch");
        let out = Command::new("git")
            .args(["add", "--", rel])
            .current_dir(repo)
            .output()
            .expect("git add scratch");
        assert!(out.status.success(), "git add {rel} failed");
        Self {
            repo: repo.to_path_buf(),
            rel,
        }
    }
}

impl Drop for StagedScratch {
    fn drop(&mut self) {
        let _ = Command::new("git")
            .args(["reset", "--", self.rel])
            .current_dir(&self.repo)
            .output();
        let _ = std::fs::remove_file(self.repo.join(self.rel));
    }
}

/// The whole unit-11 claim in one pass, in this order, so the one expensive real build
/// (`docs-index.sh` with no `--digest`, which walks the actual repo root) is paid once and
/// every assertion rides the same store rather than each rebuilding it.
#[test]
fn docs_index_sh_builds_one_ranked_store_spanning_docs_and_code() {
    let repo = repo_root();
    let store = repo.join("target/docs-index");

    let digest_before = digest();

    // The D0013 floor: with no store at all, `spec find` must still answer, and say why
    // it fell back rather than silently degrading.
    let _ = std::fs::remove_dir_all(&store);
    let floor = spec_find("miri discipline");
    let floor_stderr = String::from_utf8_lossy(&floor.stderr);
    assert!(
        floor_stderr.contains("no docs index yet"),
        "the floor must name the reason it fell back: {floor_stderr}"
    );

    // Build it for real, over this real checkout -- exactly what `just docs-index` runs.
    let build = run_script(&[]);
    assert!(
        build.status.success(),
        "docs-index.sh failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        store.join("manifest").exists(),
        "a manifest must exist after a build"
    );

    // ONE CORPUS: a markdown heading chunk from SPEC.md and a symbol chunk from a src/
    // file must both be findable, in the same store.
    let md_hits = text_search("brute-force cosine");
    assert!(
        md_hits
            .iter()
            .any(|h| h["attrs"]["code.path"]["Str"] == "SPEC.md"),
        "a SPEC.md chunk must be findable: {md_hits:?}"
    );
    let code_hits = text_search("dot-entries are always walked");
    assert!(
        code_hits.iter().any(|h| h["attrs"]["code.path"]["Str"]
            .as_str()
            .is_some_and(|p| p.ends_with(".rs"))),
        "a src/ symbol chunk must be findable: {code_hits:?}"
    );

    // A fresh build's sentinel matches the tree it was just built from.
    let recorded = recorded_digest();
    assert_eq!(
        recorded.as_deref(),
        Some(digest_before.as_str()),
        "a fresh build's recorded digest must match the tree it covers"
    );

    // Staleness: stage a new tracked file (no existing content touched) and the digest
    // the script computes now must differ from what the store still has recorded --
    // exactly the comparison `spec find`'s freshness check makes.
    let scratch = StagedScratch::new(&repo, "zz-docs-index-e2e-scratch.md");
    let digest_after = digest();
    assert_ne!(
        digest_after, digest_before,
        "staging a new tracked file must change the digest"
    );
    assert_ne!(
        Some(digest_after),
        recorded,
        "the now-current digest must not match the store's stale recording"
    );

    let stale = spec_find("miri discipline");
    let stale_stderr = String::from_utf8_lossy(&stale.stderr);
    assert!(
        stale_stderr.contains("stale"),
        "spec find must report the index stale rather than serve silently wrong ranks: \
         {stale_stderr}"
    );

    drop(scratch); // restores the tree before the ranked check below

    // Ranked tier: with the index present and fresh again, `spec find` must answer from
    // it rather than falling back to the floor.
    let ranked = spec_find("miri discipline");
    let ranked_stderr = String::from_utf8_lossy(&ranked.stderr);
    assert!(
        !ranked_stderr.contains("using text search"),
        "a fresh index must not fall back to the floor: {ranked_stderr}"
    );
    let ranked_stdout = String::from_utf8_lossy(&ranked.stdout);
    assert!(
        ranked_stdout.contains("fetch: spec"),
        "ranked output must carry a fetchable ref: {ranked_stdout}"
    );
}
