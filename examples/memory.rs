//! The all-in-one memory: text in (`remember`), relevant text out (`recall`).
//!
//! This walks the AI ingest layer end to end — pick an embedder (and,
//! optionally, a summarizer), wrap a store in a [`Memory`], `remember` a few
//! notes, then `recall` the nearest ones to a natural-language query. It also
//! shows the **bring-your-own-vector escape hatch**: the raw [`Nidus`] API is
//! always there underneath, with zero async and zero provider deps.
//!
//! Run it (the AI features are all off by default, so opt them in):
//!
//! ```bash
//! cargo run --example memory --features memory,embed-all,summarize-all
//! ```
//!
//! **Offline-safe.** With no provider configured it still runs: the BYO-vector
//! section needs no network, and the provider-backed section prints a clear,
//! actionable message instead of a stack trace when there is no API key or
//! reachable backend. Point it at a provider with environment variables:
//!
//! ```bash
//! # OpenAI (the default embed provider):
//! export NIDUS_EMBED_API_KEY=sk-...
//! # Or a fully local, keyless setup via Ollama:
//! export NIDUS_EMBED_PROVIDER=ollama            # keyless; talks to localhost:11434
//! export NIDUS_EMBED_BASE_URL=http://localhost:11434   # override the host if needed
//! # Optionally summarize-then-embed (needs a chat provider):
//! export NIDUS_SUMMARIZE_PROVIDER=anthropic
//! export NIDUS_SUMMARIZE_API_KEY=sk-ant-...
//! ```
//!
//! The `[[example]]` stanza this file needs (in `Cargo.toml`):
//!
//! ```toml
//! [[example]]
//! name = "memory"
//! required-features = ["memory", "embed-all", "summarize-all"]
//! ```

use std::collections::BTreeMap;

use nidus::embed::{AnyEmbedder, EmbedConfig, EmbedProvider, Embedder};
use nidus::summarize::{AnySummarizer, SummarizeConfig, SummarizeProvider};
use nidus::{Config, Memory, Nidus, RecallOpts, Record, RememberMode, SearchOpts, Value};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── 1. The escape hatch — raw vectors, no network, no async ──────────────
    // `Memory` is a strictly additive convenience layer. If you already have
    // your own embeddings (or want a model nidus ships no adapter for), keep
    // using `Nidus` directly: it takes a caller-supplied `Vec<f32>` and is
    // completely untouched by the AI layer. This part always runs.
    byo_vector_demo()?;

    // ── 2. Build an embedder from the environment (or bail out cleanly) ──────
    // Every provider is selected at runtime through the closed `AnyEmbedder`
    // enum — no `Box<dyn>`. The model is optional: leaving it empty uses the
    // provider's default (e.g. `text-embedding-3-small` for OpenAI).
    let embedder = match build_embedder().await {
        Ok(embedder) => embedder,
        Err(msg) => {
            println!("\n── provider-backed memory: skipped ──");
            println!("{msg}");
            println!(
                "Set NIDUS_EMBED_API_KEY (and optionally NIDUS_EMBED_PROVIDER / \
                 NIDUS_EMBED_BASE_URL) to run the remember/recall demo."
            );
            return Ok(());
        }
    };

    println!(
        "\n── provider-backed memory ──\nembedder: {}/{}  (dimension {})",
        embedder.provider_name(),
        embedder.model_name(),
        embedder.dimension(),
    );

    // ── 3. Open a store whose dimension matches the embedder ─────────────────
    // The embedding dimension is pinned into the store at creation. Asking the
    // embedder for its dimension and opening the store to match keeps the two
    // in lockstep (a mismatch is a hard error on the first `remember`).
    let dir = std::env::temp_dir().join("nidus-memory-example");
    let _ = std::fs::remove_dir_all(&dir); // start clean
    let dim = embedder.dimension();
    let db = Nidus::open(Config::new(&dir, dim))?;

    // ── 4. Optionally attach a summarizer for RememberMode::Summarize ────────
    let mut memory = Memory::new(db, embedder);
    let mut can_summarize = false;
    if let Some(summarizer) = build_summarizer().await {
        println!(
            "summarizer: {} (Summarize mode enabled)",
            summarizer_label(&summarizer)
        );
        memory = memory.with_summarizer(summarizer);
        can_summarize = true;
    } else {
        println!("summarizer: none (Raw mode only)");
    }

    // ── 5. remember() — text in ──────────────────────────────────────────────
    // Raw mode embeds the text as-is. The first write into a collection pins
    // the embedder's "provider/model" identity + dimension into the collection
    // metadata; a later write with a *different* embedder is refused, so you
    // can't accidentally mix incomparable vector spaces in one collection.
    let notes = [
        (
            "login",
            "Users authenticate with a bearer token issued at login.",
        ),
        (
            "cache",
            "The warm search index can be shared across workers via Redis.",
        ),
        (
            "durability",
            "Every batch is fsynced; a crash loses at most the in-flight write.",
        ),
    ];
    for (id, text) in notes {
        match memory
            .remember("notes", id, text, attrs(text), RememberMode::Raw)
            .await
        {
            Ok(()) => println!("  remembered [{id}]"),
            Err(e) => {
                // e.g. no reachable backend — report it and stop the demo cleanly.
                println!("  could not reach the embedding backend: {e:#}");
                let _ = std::fs::remove_dir_all(&dir);
                return Ok(());
            }
        }
    }

    // Summarize mode: summarize the text first, embed the *summary*, and store
    // both the summary and the original source so a hit stays explainable.
    if can_summarize {
        let long = "In 2019 the team migrated the auth service off session cookies \
                    and onto short-lived bearer tokens, cutting a class of CSRF bugs \
                    and simplifying horizontal scaling.";
        match memory
            .remember(
                "notes",
                "auth-history",
                long,
                BTreeMap::new(),
                RememberMode::Summarize,
            )
            .await
        {
            Ok(()) => println!("  remembered [auth-history] (summarized then embedded)"),
            Err(e) => println!("  summarize path failed: {e:#}"),
        }
    }

    // ── 6. recall() — relevant text out ──────────────────────────────────────
    // The query is embedded (via the query side of the embedder) and matched by
    // cosine similarity. `RecallOpts` maps onto the store's SearchOpts — top_k,
    // a score floor, and an optional metadata filter.
    let opts = RecallOpts {
        top_k: 3,
        min_score: 0.0,
        filter: None,
    };
    let hits = memory
        .recall("notes", "how do users sign in?", &opts)
        .await?;
    println!("\nrecall(\"how do users sign in?\"):");
    for h in &hits {
        let text = match h.attrs.get("text") {
            Some(Value::Str(s)) => s.as_str(),
            _ => "",
        };
        println!("  {:>6.3}  [{}]  {}", h.score, h.id, text);
    }

    // The bare store is always reachable underneath — drop back to it any time.
    let _db: &Nidus = memory.db();

    println!("\nOK. Cleaning up.");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// The bring-your-own-vector path: no provider, no async, no network. A host
/// that already has embeddings uses `Nidus` directly and never touches the AI
/// layer at all.
fn byo_vector_demo() -> anyhow::Result<()> {
    println!("── bring-your-own-vector (raw Nidus, no network) ──");
    let mut db = Nidus::open_in_memory(4)?;
    db.create_collection("vectors")?;
    db.upsert(
        "vectors",
        &[
            Record::new("a", vec![1.0, 0.0, 0.0, 0.0], BTreeMap::new()),
            Record::new("b", vec![0.0, 1.0, 0.0, 0.0], BTreeMap::new()),
        ],
    )?;
    let hits = db.search(
        "vectors",
        &[0.9, 0.1, 0.0, 0.0],
        &SearchOpts {
            top_k: 1,
            ..Default::default()
        },
    )?;
    println!(
        "  nearest to [0.9, 0.1, 0, 0] is [{}] ({:.3})",
        hits[0].id, hits[0].score
    );
    Ok(())
}

/// Build an [`AnyEmbedder`] from the environment. Returns a human-readable
/// message (not an error type) when there is nothing to build — the caller
/// prints it and moves on, keeping the example offline-safe.
async fn build_embedder() -> Result<AnyEmbedder, String> {
    let name = std::env::var("NIDUS_EMBED_PROVIDER").unwrap_or_else(|_| "openai".to_string());
    let provider = EmbedProvider::from_name(&name)
        .ok_or_else(|| format!("unknown embed provider '{name}'"))?;

    // Leaving `model` empty uses the provider default (openai-compat has none).
    let mut config = EmbedConfig::new(std::env::var("NIDUS_EMBED_MODEL").unwrap_or_default());
    if let Ok(key) = std::env::var("NIDUS_EMBED_API_KEY") {
        config = config.api_key(key);
    }
    // Base-URL override: an OpenAI-compatible gateway, or the Ollama host.
    if let Ok(base) = std::env::var("NIDUS_EMBED_BASE_URL") {
        config = config.base_url(base);
    }

    // `build` is async because keyless local providers (Ollama, openai-compat)
    // probe their embedding dimension with a live call during construction.
    AnyEmbedder::build(provider, config)
        .await
        .map_err(|e| format!("could not build embedder: {e}"))
}

/// Build an optional [`AnySummarizer`]. `None` when unconfigured — Summarize
/// mode is simply skipped in that case.
async fn build_summarizer() -> Option<AnySummarizer> {
    let name = std::env::var("NIDUS_SUMMARIZE_PROVIDER").ok()?;
    let provider = SummarizeProvider::from_name(&name)?;
    let mut config =
        SummarizeConfig::new(std::env::var("NIDUS_SUMMARIZE_MODEL").unwrap_or_default());
    if config.model.is_empty() {
        config = SummarizeConfig::new(provider.default_model());
    }
    if let Ok(key) = std::env::var("NIDUS_SUMMARIZE_API_KEY") {
        config = config.api_key(key);
    }
    if let Ok(base) = std::env::var("NIDUS_SUMMARIZE_BASE_URL") {
        config = config.base_url(base);
    }
    AnySummarizer::build(provider, config).await.ok()
}

/// A single `text` attr so recall hits can print what was remembered.
fn attrs(text: &str) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert("text".to_string(), Value::Str(text.to_string()));
    m
}

/// `provider/model` for a built summarizer (the trait exposes both parts).
fn summarizer_label(s: &AnySummarizer) -> String {
    use nidus::summarize::Summarizer;
    format!("{}/{}", s.provider_name(), s.model_name())
}
