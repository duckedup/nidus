//! End-to-end smoke test of the public API — the quickest way to see nidus work.
//!
//! Run it:
//!     cargo run --example demo        (or: just demo)
//!
//! It opens a file-backed store in a temp dir, indexes a handful of toy 4-dim
//! vectors across two collections, then demonstrates: single-collection search,
//! whole-store (`Scope::All`) search, a metadata filter + `min_score`, and
//! durability by reopening the store from disk.

use std::collections::BTreeMap;

use nidus::{Config, Nidus, Predicate, Record, Scope, SearchOpts, Value};

/// Tiny helper to build a record with a couple of string attrs.
fn rec(id: &str, vector: Vec<f32>, path: &str, kind: &str) -> Record {
    let mut attrs = BTreeMap::new();
    attrs.insert("path".to_string(), Value::Str(path.to_string()));
    attrs.insert("kind".to_string(), Value::Str(kind.to_string()));
    Record::new(id, vector, attrs)
}

fn print_hits(label: &str, hits: &[nidus::Hit]) {
    println!("\n{label}");
    for h in hits {
        let path = match h.attrs.get("path") {
            Some(Value::Str(s)) => s.as_str(),
            _ => "?",
        };
        println!(
            "  {:>6.3}  [{}] {}  ({})",
            h.score, h.collection, h.id, path
        );
    }
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join("nidus-demo");
    let _ = std::fs::remove_dir_all(&dir); // start clean

    // ── Open a file-backed store (dimension 4) ───────────────────────────
    let mut db = Nidus::open(Config::new(&dir, 4))?;
    println!("opened store at {} (dim {})", dir.display(), db.dimension());

    db.create_collection("code")?;
    db.create_collection("docs")?;

    // ── Index some toy vectors ───────────────────────────────────────────
    db.upsert(
        "code",
        &[
            rec("a", vec![1.0, 0.0, 0.0, 0.0], "src/auth/login.rs", "file"),
            rec("b", vec![0.9, 0.1, 0.0, 0.0], "src/auth/token.rs", "file"),
            rec("c", vec![0.0, 1.0, 0.0, 0.0], "src/render/mod.rs", "file"),
        ],
    )?;
    db.upsert(
        "docs",
        &[
            rec("d", vec![0.8, 0.2, 0.0, 0.0], "docs/auth.md", "section"),
            rec("e", vec![0.0, 0.0, 1.0, 0.0], "docs/render.md", "section"),
        ],
    )?;
    println!(
        "indexed {} code + {} docs records",
        db.get_all("code").len(),
        db.get_all("docs").len()
    );

    let query = [1.0, 0.0, 0.0, 0.0]; // "something about auth"
    let opts = SearchOpts {
        top_k: 5,
        ..Default::default()
    };

    // ── 1. Search one collection ─────────────────────────────────────────
    print_hits(
        "search(\"code\", auth-ish):",
        &db.search("code", &query, &opts)?,
    );

    // ── 2. Search the whole store (one merged ranking) ───────────────────
    print_hits(
        "search(Scope::All, auth-ish):",
        &db.search(Scope::All, &query, &opts)?,
    );

    // ── 3. Filter (path glob) + min_score ────────────────────────────────
    let filtered = SearchOpts {
        top_k: 5,
        filter: nidus::Filter(vec![Predicate::Glob("path".into(), "src/auth/*".into())]),
        min_score: Some(0.5),
    };
    print_hits(
        "search(Scope::All) WHERE path GLOB 'src/auth/*' AND score>=0.5:",
        &db.search(Scope::All, &query, &filtered)?,
    );

    // ── 4. Durability: drop, reopen from disk, search again ──────────────
    drop(db);
    let db2 = Nidus::open(Config::new(&dir, 4))?;
    let reopened = db2.search("code", &query, &opts)?;
    print_hits("after reopen — search(\"code\"):", &reopened);
    assert_eq!(reopened.first().map(|h| h.id.as_str()), Some("a"));

    println!("\nOK — durability + ranking verified. Cleaning up.");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
