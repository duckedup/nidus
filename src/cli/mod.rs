//! The `nidus` command line: store operations over a directory, plus `nidus
//! serve`. Everything here is synchronous (matching the library); only `serve`
//! spins up a Tokio runtime, so the common, fast subcommands pay no async cost.

use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;

use crate::server::dto::{AnnDto, FootprintDto, HitDto};
use crate::{
    AggregateOpts, AnnConfig, Config, Distance, Filter, Fsync, FtsClause, FtsCombine, FtsField,
    FtsQuery, HighlightOpts, HybridOpts, LeaseWait, LimitPer, ListOpts, Nidus, OpenMode, OrderBy,
    Projection, Quantization, Record, Scope, SearchOpts,
};

// AI-ingest (memory) wiring for `serve`: only under the `memory` feature (pulled
// by the `serve` umbrella). A plain `cli` build has no `--embed-provider` flags.
#[cfg(feature = "memory")]
use std::sync::Arc;

#[cfg(feature = "memory")]
use crate::embed::{AnyEmbedder, EmbedConfig, EmbedProvider};
#[cfg(all(feature = "memory", feature = "summarize"))]
use crate::summarize::{AnySummarizer, SummarizeConfig, SummarizeProvider};

mod backup;
#[cfg(feature = "memory")]
mod memory;

#[derive(Parser, Debug)]
#[command(
    name = "nidus",
    version,
    about = "A small, pure-Rust vector store — CLI and HTTP server"
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Store location, shared by every subcommand. For an existing store the dimension and distance
/// are read from the on-disk header, so `--dim`/`--distance` are only needed when creating one, or
/// to double-check an existing one — a mismatch is then a hard error.
#[derive(Args, Debug, Default)]
struct StoreArgs {
    /// Store directory (created on first write). Unused — but still required — when
    /// `--persistence` names an object store, where the durable bytes live remotely.
    #[arg(long, short = 'd', env = "NIDUS_DIR")]
    dir: PathBuf,
    /// Embedding dimension. Inferred from an existing store; required to create one.
    #[arg(long, env = "NIDUS_DIM")]
    dim: Option<usize>,
    /// Distance metric: cosine, euclidean, or dot. Inferred from an existing
    /// store; defaults to cosine when creating one.
    #[arg(long, env = "NIDUS_DISTANCE")]
    distance: Option<DistanceArg>,
    /// Open without taking the writer lock (rejects mutations).
    #[arg(long, env = "NIDUS_READ_ONLY")]
    read_only: bool,
    /// Opt into an approximate-nearest-neighbour index: `hnsw` or `ivf`. Omit for exact brute-force
    /// search (the default). Unlike `--dim`/`--distance`, the ANN choice is *not* stored in the
    /// header — pass it on every open (including `serve`) where you want the index built/consulted.
    #[arg(long, env = "NIDUS_ANN")]
    ann: Option<AnnKindArg>,
    /// HNSW: max neighbours per node above layer 0. Ignored without `--ann hnsw`.
    #[arg(long, env = "NIDUS_ANN_M")]
    ann_m: Option<usize>,
    /// HNSW: build-time beam width. Ignored without `--ann hnsw`.
    #[arg(long, env = "NIDUS_ANN_EF_CONSTRUCTION")]
    ann_ef_construction: Option<usize>,
    /// HNSW: search-time beam width. Ignored without `--ann hnsw`.
    #[arg(long, env = "NIDUS_ANN_EF_SEARCH")]
    ann_ef_search: Option<usize>,
    /// IVF: number of k-means lists (`0` = auto `~sqrt(n)`). Ignored without `--ann ivf`.
    #[arg(long, env = "NIDUS_ANN_N_LISTS")]
    ann_n_lists: Option<usize>,
    /// IVF: lists probed per query. Ignored without `--ann ivf`.
    #[arg(long, env = "NIDUS_ANN_N_PROBE")]
    ann_n_probe: Option<usize>,
    /// Candidate over-fetch multiple (`top_k * overscan`) before post-filter + rerank.
    /// Applies to both ANN kinds. Ignored without `--ann`.
    #[arg(long, env = "NIDUS_ANN_OVERSCAN")]
    ann_overscan: Option<usize>,
    /// Build PRNG seed (deterministic index). Applies to both ANN kinds. Ignored without `--ann`.
    #[arg(long, env = "NIDUS_ANN_SEED")]
    ann_seed: Option<u64>,
    /// Where the durable bytes live (SPEC §13.2). Omit for local files under `--dir`; `s3://…` or
    /// `gs://…` for a live object-backed store. Pass `--dim` with an object store, since the remote
    /// header is not peeked; credentials come from the standard environment.
    #[arg(long, env = "NIDUS_PERSISTENCE")]
    persistence: Option<String>,
    /// Share the in-RAM working set across processes (SPEC §13.3): a `redis://…` (or
    /// `valkey://…`, `keydb://…`, `dragonfly://…`) URL. Omit (or `local`) to keep it
    /// process-local. The working set is published on flush and adopted on open.
    #[arg(long, env = "NIDUS_MEMORY")]
    memory: Option<String>,
    /// Run as one of several cooperating instances over a *shared* store (SPEC §14.6):
    /// requires an object-store `--persistence` **and** a Redis-family `--memory` tier.
    #[arg(long, env = "NIDUS_CLUSTER")]
    cluster: bool,
    /// Memory-map immutable segments instead of holding them in RAM — lets a store
    /// exceed RAM on one node. Local filesystem + little-endian only; other segments
    /// fall back to a RAM load.
    #[arg(long, env = "NIDUS_MMAP")]
    mmap: bool,
    /// Quantize the search first pass for speed, then rerank the candidates in exact
    /// f32: `int8` (4× less memory traffic) or `binary` (32×, cosine only). Omit for
    /// exact-only search. Like `--ann`, not stored in the header — pass it on every open.
    #[arg(long, env = "NIDUS_QUANTIZATION")]
    quantization: Option<QuantArg>,
    /// Candidate over-fetch multiple for the quantized first pass (`top_k * rescore`
    /// reranked in f32). Defaults per kind: 4 for int8, 16 for binary. Ignored without
    /// `--quantization`.
    #[arg(long, env = "NIDUS_QUANT_RESCORE")]
    quant_rescore: Option<usize>,
    /// Worker threads for a single exact search (`1` = serial, the default). Splits one
    /// query's scan across threads; unrelated to serving concurrent requests.
    #[arg(long, env = "NIDUS_QUERY_THREADS")]
    query_threads: Option<usize>,
    /// Seal the active segment once it reaches this many rows (omit = never seal, one
    /// growing segment). Sealed segments are immutable — the unit of mmap and per-segment
    /// indexing.
    #[arg(long, env = "NIDUS_SEGMENT_MAX_ROWS")]
    segment_max_rows: Option<u64>,
    /// Minimum rows for a sealed segment to get its own IVF index (omit = never index,
    /// exact brute-force). Needs `--segment-max-rows` to have any effect.
    #[arg(long, env = "NIDUS_SEGMENT_INDEX_MIN_ROWS")]
    segment_index_min_rows: Option<u64>,
    /// fsync policy: `per-batch` (durable per call, the default) or `on-flush` (faster,
    /// weaker — a crash can lose acknowledged writes).
    #[arg(long, env = "NIDUS_FSYNC")]
    fsync: Option<FsyncArg>,
    /// Rewrite the data matrix when this fraction of rows is dead (default `0.5`).
    #[arg(long, env = "NIDUS_AUTO_COMPACT", conflicts_with = "no_auto_compact")]
    auto_compact: Option<f32>,
    /// Never auto-compact; reclaim dead rows only on an explicit `compact`.
    #[arg(long, env = "NIDUS_NO_AUTO_COMPACT")]
    no_auto_compact: bool,
    /// Seconds before another process may reclaim a stale writer lock (default `60`).
    /// In `--cluster` mode this is also the writer-lease window.
    #[arg(long, value_name = "SECONDS", env = "NIDUS_LOCK_TTL")]
    lock_ttl: Option<u64>,
    /// Wait for the writer handle instead of exiting when another instance holds it: this becomes a
    /// standby, promoted within roughly `--lock-ttl` of the holder dying. Without it a second writer
    /// exits immediately and failover takes as long as the supervisor's backoff.
    #[arg(
        long,
        value_name = "SECONDS",
        num_args = 0..=1,
        default_missing_value = "forever",
        env = "NIDUS_WAIT_FOR_LEASE"
    )]
    wait_for_lease: Option<String>,
    /// Fail the readiness probe once a `--read-only` instance has gone this many seconds
    /// without verifying it is current. Omit for no bound (the default).
    #[arg(long, value_name = "SECONDS", env = "NIDUS_MAX_STALENESS")]
    max_staleness: Option<u64>,
    /// Refuse to open a store whose vector matrix would exceed this many bytes — the
    /// overcommit guard (SPEC §6.6). Omit for no ceiling.
    #[arg(long, env = "NIDUS_MAX_VECTOR_BYTES")]
    max_vector_bytes: Option<u64>,
}

impl StoreArgs {
    /// Resolve the `(dimension, distance)` to open with. An explicit flag wins and is verified
    /// against the header on open; otherwise the value comes from an existing store's header. With
    /// neither — no store and no `--dim` — creation cannot proceed.
    fn resolve(&self) -> Result<(usize, Distance)> {
        // The local-file header peek only applies to a local store; an object-store
        // location (`s3://`/`gs://`) has no peekable local `data`, so `--dim` is required.
        let peeked = if self.is_object_store() {
            None
        } else {
            crate::data::peek_header(&self.dir.join("data"))?
        };
        let dimension = match (self.dim, peeked) {
            (Some(d), _) => d,
            (None, Some((d, _))) => d,
            (None, None) => bail!(
                "no store at {} yet — pass --dim to create one",
                self.dir.display()
            ),
        };
        let distance = match (self.distance, peeked) {
            (Some(d), _) => d.into(),
            (None, Some((_, dist))) => dist,
            (None, None) => Distance::default(),
        };
        Ok((dimension, distance))
    }

    /// Whether `--persistence` names a (non-local) object store.
    fn is_object_store(&self) -> bool {
        self.persistence.as_deref().is_some_and(|p| {
            let p = p.to_ascii_lowercase();
            p.starts_with("s3://") || p.starts_with("gs://") || p.starts_with("gcs://")
        })
    }

    /// Whether `--memory` names a (non-local) shared Redis-family tier — the same
    /// RESP schemes [`crate::open_memory_tier`] routes to a `RedisTier`.
    fn is_shared_memory(&self) -> bool {
        self.memory.as_deref().is_some_and(|m| {
            let m = m.to_ascii_lowercase();
            crate::backend::REDIS_SCHEMES
                .iter()
                .any(|s| m.starts_with(&format!("{s}://")))
        })
    }

    /// Build the open [`Config`] from these args — the single place the store flags
    /// (`--dim`/`--distance`/`--ann*`/`--persistence`/`--memory`/mode) are assembled, so
    /// the read and serve paths can't drift.
    fn config(&self, mode: OpenMode) -> Result<Config> {
        let (dim, distance) = self.resolve()?;
        let mut cfg = Config::new(self.dir.clone(), dim)
            .distance(distance)
            .ann(self.ann_config())
            .quantization(self.quant_config())
            .persistence(self.persistence.clone().unwrap_or_default())
            .memory(self.memory.clone().unwrap_or_default())
            .cluster(self.cluster)
            .mmap(self.mmap)
            .segment_max_rows(self.segment_max_rows)
            .segment_index_min_rows(self.segment_index_min_rows)
            .max_vector_bytes(self.max_vector_bytes)
            .open_mode(mode);
        // The remaining knobs have non-`Option` defaults in `Config`, so only an
        // explicitly-supplied flag may overwrite them.
        if let Some(n) = self.query_threads {
            cfg = cfg.query_threads(n);
        }
        if let Some(f) = self.fsync {
            cfg = cfg.fsync(f.into());
        }
        if self.no_auto_compact {
            cfg = cfg.auto_compact(None);
        } else if let Some(ratio) = self.auto_compact {
            cfg = cfg.auto_compact(Some(ratio));
        }
        if let Some(secs) = self.lock_ttl {
            cfg = cfg.lock_ttl(std::time::Duration::from_secs(secs));
        }
        cfg = cfg.lease_wait(self.lease_wait()?);
        cfg = cfg.max_staleness(self.max_staleness.map(std::time::Duration::from_secs));
        Ok(cfg)
    }

    /// Parse `--wait-for-lease` into a [`LeaseWait`]. Absent → `Fail` (unchanged
    /// behaviour); bare flag → `Forever`; a number → that many seconds.
    fn lease_wait(&self) -> Result<LeaseWait> {
        let Some(raw) = self.wait_for_lease.as_deref() else {
            return Ok(LeaseWait::Fail);
        };
        if raw.eq_ignore_ascii_case("forever") {
            return Ok(LeaseWait::Forever);
        }
        let secs: u64 = raw.parse().with_context(|| {
            format!(
                "--wait-for-lease expects a number of seconds (or no value at all, \
                     to wait indefinitely), got {raw:?}"
            )
        })?;
        Ok(LeaseWait::Timeout(std::time::Duration::from_secs(secs)))
    }

    /// Build the `Option<Quantization>` from the `--quantization`/`--quant-rescore`
    /// flags — `None` (no `--quantization`) keeps the exact-only search path.
    fn quant_config(&self) -> Option<Quantization> {
        let base = match self.quantization? {
            QuantArg::Int8 => Quantization::int8(),
            QuantArg::Binary => Quantization::binary(),
        };
        Some(match self.quant_rescore {
            Some(n) => base.rescore(n),
            None => base,
        })
    }

    /// Build the `Option<AnnConfig>` from the `--ann*` flags. No `--ann` keeps exact brute force;
    /// otherwise start from the kind's defaults and override only what was supplied. Param flags for
    /// the other kind are accepted but inert, matching `AnnConfig`'s own semantics.
    fn ann_config(&self) -> Option<AnnConfig> {
        let base = match self.ann? {
            AnnKindArg::Hnsw => AnnConfig::hnsw(),
            AnnKindArg::Ivf => AnnConfig::ivf(),
        };
        let mut cfg = base;
        if let Some(v) = self.ann_m {
            cfg = cfg.m(v);
        }
        if let Some(v) = self.ann_ef_construction {
            cfg = cfg.ef_construction(v);
        }
        if let Some(v) = self.ann_ef_search {
            cfg = cfg.ef_search(v);
        }
        if let Some(v) = self.ann_n_lists {
            cfg = cfg.n_lists(v);
        }
        if let Some(v) = self.ann_n_probe {
            cfg = cfg.n_probe(v);
        }
        if let Some(v) = self.ann_overscan {
            cfg = cfg.overscan(v);
        }
        if let Some(v) = self.ann_seed {
            cfg = cfg.seed(v);
        }
        Some(cfg)
    }
}

/// Embedder/summarizer configuration for `nidus serve`'s text-native memory routes, flattened into
/// the `Serve` subcommand only under the `memory` feature. With no `--embed-provider` the server
/// still starts, serving the raw vector endpoints while the memory routes answer `400`.
#[cfg(feature = "memory")]
#[derive(Args, Debug, Default)]
struct IngestArgs {
    /// Embedding provider for `/remember` and `/recall`: voyage, openai, ollama,
    /// cohere, gemini, mistral, jina, or openai-compat. Omit to serve only the
    /// raw vector endpoints (the memory routes then answer 400).
    #[arg(long, env = "NIDUS_EMBED_PROVIDER")]
    embed_provider: Option<String>,
    /// Embedding model. Defaults to the provider's default when omitted
    /// (openai-compat has none — pass one).
    #[arg(long, env = "NIDUS_EMBED_MODEL")]
    embed_model: Option<String>,
    /// API key for the embedding provider (some, e.g. Ollama, need none).
    #[arg(long, env = "NIDUS_EMBED_API_KEY")]
    embed_api_key: Option<String>,
    /// Base-URL override for the embedding provider (required for openai-compat
    /// and self-hosted gateways).
    #[arg(long, env = "NIDUS_EMBED_BASE_URL")]
    embed_base_url: Option<String>,

    /// Summarizer provider enabling `mode: "summarize"` on `/remember`:
    /// anthropic or openai. Omit for raw-embed only.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    #[arg(long, env = "NIDUS_SUMMARIZE_PROVIDER")]
    summarize_provider: Option<String>,
    /// Summarizer model. Defaults to the provider's default when omitted.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    #[arg(long, env = "NIDUS_SUMMARIZE_MODEL")]
    summarize_model: Option<String>,
    /// API key for the summarizer provider.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    #[arg(long, env = "NIDUS_SUMMARIZE_API_KEY")]
    summarize_api_key: Option<String>,
    /// Base-URL override for the summarizer provider.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    #[arg(long, env = "NIDUS_SUMMARIZE_BASE_URL")]
    summarize_base_url: Option<String>,
}

#[cfg(feature = "memory")]
impl IngestArgs {
    /// Build the embedder from `--embed-provider …`, or `None` when the flag was
    /// omitted (the server then serves only the raw endpoints). Async because
    /// some adapters probe their dimension with a live call on construction.
    async fn embedder(&self) -> Result<Option<Arc<AnyEmbedder>>> {
        Ok(self.build_embedder().await?.map(Arc::new))
    }

    /// The embedder itself, unwrapped. `serve`/`mcp` share one behind an `Arc`; the
    /// `remember`/`recall` subcommands hand ownership to a `Memory` instead.
    pub(super) async fn build_embedder(&self) -> Result<Option<AnyEmbedder>> {
        let Some(name) = self.embed_provider.as_deref() else {
            return Ok(None);
        };
        let provider = EmbedProvider::from_name(name)
            .ok_or_else(|| anyhow::anyhow!("unknown embed provider '{name}'"))?;
        // An empty model lets `AnyEmbedder::build` fill the provider default.
        let mut config = EmbedConfig::new(self.embed_model.clone().unwrap_or_default());
        if let Some(k) = &self.embed_api_key {
            config = config.api_key(k);
        }
        if let Some(u) = &self.embed_base_url {
            config = config.base_url(u);
        }
        let embedder = AnyEmbedder::build(provider, config)
            .await
            .map_err(|e| anyhow::anyhow!("building embedder '{name}': {e}"))?;
        Ok(Some(embedder))
    }

    /// Build the summarizer from `--summarize-provider …`, or `None` when omitted.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    async fn summarizer(&self) -> Result<Option<Arc<AnySummarizer>>> {
        Ok(self.build_summarizer().await?.map(Arc::new))
    }

    /// The summarizer itself, unwrapped — `Memory::with_summarizer` takes ownership.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    pub(super) async fn build_summarizer(&self) -> Result<Option<AnySummarizer>> {
        let Some(name) = self.summarize_provider.as_deref() else {
            return Ok(None);
        };
        let provider = SummarizeProvider::from_name(name)
            .ok_or_else(|| anyhow::anyhow!("unknown summarize provider '{name}'"))?;
        let model = self
            .summarize_model
            .clone()
            .unwrap_or_else(|| provider.default_model().to_string());
        let mut config = SummarizeConfig::new(model);
        if let Some(k) = &self.summarize_api_key {
            config = config.api_key(k);
        }
        if let Some(u) = &self.summarize_base_url {
            config = config.base_url(u);
        }
        let summarizer = AnySummarizer::build(provider, config)
            .await
            .map_err(|e| anyhow::anyhow!("building summarizer '{name}': {e}"))?;
        Ok(Some(summarizer))
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum DistanceArg {
    Cosine,
    Euclidean,
    Dot,
}

impl From<DistanceArg> for Distance {
    fn from(d: DistanceArg) -> Self {
        match d {
            DistanceArg::Cosine => Distance::Cosine,
            DistanceArg::Euclidean => Distance::Euclidean,
            DistanceArg::Dot => Distance::DotProduct,
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum AnnKindArg {
    Hnsw,
    Ivf,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum QuantArg {
    Int8,
    Binary,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum FsyncArg {
    PerBatch,
    OnFlush,
}

impl From<FsyncArg> for Fsync {
    fn from(f: FsyncArg) -> Self {
        match f {
            FsyncArg::PerBatch => Fsync::PerBatch,
            FsyncArg::OnFlush => Fsync::OnFlush,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum CombineArg {
    #[default]
    Sum,
    Max,
}

impl From<CombineArg> for FtsCombine {
    fn from(c: CombineArg) -> Self {
        match c {
            CombineArg::Sum => FtsCombine::Sum,
            CombineArg::Max => FtsCombine::Max,
        }
    }
}

/// The multi-clause / annotation half of a text or hybrid query, shared by both subcommands
/// so the two cannot drift (nidus-m50.10, nidus-m50.5).
#[derive(Args, Debug, Default)]
struct TextQueryArgs {
    /// An extra query clause, `field=text` (repeatable). Use instead of the positional
    /// field + query pair, never alongside it.
    #[arg(long = "clause")]
    clauses: Vec<String>,
    /// How several clauses fold into one score.
    #[arg(long, value_enum, default_value_t = CombineArg::Sum)]
    combine: CombineArg,
    /// Annotate each hit with its per-clause (and, for hybrid, per-leg) scores.
    #[arg(long)]
    explain: bool,
    /// Return highlighted fragments of the matched text.
    #[arg(long)]
    highlight: bool,
    /// Fragments per field when highlighting.
    #[arg(long, default_value_t = HighlightOpts::default().max_fragments)]
    max_fragments: usize,
    /// Characters per fragment when highlighting.
    #[arg(long, default_value_t = HighlightOpts::default().fragment_chars)]
    fragment_chars: usize,
}

impl TextQueryArgs {
    /// Build the query, taking the clauses from the positional `field`/`text` pair or from
    /// repeated `--clause field=text` — the same either/or the HTTP body enforces.
    fn query(&self, field: Option<String>, text: Option<String>) -> anyhow::Result<FtsQuery> {
        let clauses = match (field, text, self.clauses.is_empty()) {
            (Some(f), Some(t), true) => vec![FtsClause::new(f, t)],
            (None, None, false) => self
                .clauses
                .iter()
                .map(|c| {
                    c.split_once('=')
                        .map(|(f, t)| FtsClause::new(f, t))
                        .ok_or_else(|| anyhow::anyhow!("--clause must be field=text, got '{c}'"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
            (None, None, true) => {
                anyhow::bail!("give a field and its query text, or one or more --clause field=text")
            }
            _ => {
                anyhow::bail!("the positional field/text pair and --clause are mutually exclusive")
            }
        };
        let mut q = FtsQuery::multi(clauses).combine(self.combine.into());
        if self.highlight {
            q = q.highlight(
                HighlightOpts::default()
                    .max_fragments(self.max_fragments)
                    .fragment_chars(self.fragment_chars),
            );
        }
        Ok(q)
    }
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the HTTP server.
    Serve {
        #[command(flatten)]
        store: StoreArgs,
        /// Address to bind. Bind `0.0.0.0:7700` to serve outside localhost (e.g. in
        /// a container); pair it with `--token`.
        #[arg(long, default_value = "127.0.0.1:7700", env = "NIDUS_ADDR")]
        addr: String,
        /// Require `Authorization: Bearer <token>` on every request except
        /// `/health`. Strongly advised when binding anything other than localhost.
        #[arg(long, env = "NIDUS_TOKEN")]
        token: Option<String>,
        /// Maximum request body size in bytes (also the largest single upsert).
        /// Default 256 MiB.
        #[arg(long, default_value_t = 256 * 1024 * 1024, env = "NIDUS_MAX_BODY_BYTES")]
        max_body_bytes: usize,
        /// Cap on store-touching requests in flight. Past it, requests are **shed** with a
        /// retryable `503` + `Retry-After` rather than queued — the store's working set is
        /// in RAM, so an unbounded queue of in-flight bodies competes with the data itself.
        #[arg(long, default_value_t = 0, env = "NIDUS_MAX_CONCURRENT_REQUESTS")]
        max_concurrent_requests: usize,
        /// Deadline in seconds for a read request (search, list, stats). Default 30.
        /// `0` disables it.
        #[arg(
            long,
            value_name = "SECONDS",
            default_value_t = 30,
            env = "NIDUS_READ_TIMEOUT"
        )]
        read_timeout: u64,
        /// Deadline in seconds for a mutating request (upsert, delete, compact, flush).
        /// Default 600. `0` disables it.
        #[arg(
            long,
            value_name = "SECONDS",
            default_value_t = 600,
            env = "NIDUS_WRITE_TIMEOUT"
        )]
        write_timeout: u64,
        /// Abandon a request body that goes this many seconds without delivering data.
        /// Default 15. `0` disables it.
        #[arg(
            long,
            value_name = "SECONDS",
            default_value_t = 15,
            env = "NIDUS_BODY_IDLE_TIMEOUT"
        )]
        body_idle_timeout: u64,
        /// Refresh this instance every N seconds so a `--read-only` reader stays current
        /// without a sidecar or cron calling `POST /refresh`. Omit to leave refreshing
        /// entirely to the caller (the default).
        #[arg(long, value_name = "SECONDS", env = "NIDUS_REFRESH_INTERVAL")]
        refresh_interval: Option<u64>,
        /// Refuse to start unless the store is on shared, non-local backends: an object-store
        /// `--persistence` *and* a Redis-family `--memory` tier. The contract the published Docker
        /// image runs under, since a container has no durable disk and would lose data on restart.
        #[arg(long, env = "NIDUS_REQUIRE_REMOTE")]
        require_remote: bool,
        /// Embedder / summarizer flags for the text-native `/remember` + `/recall`
        /// routes. Present only when built with the `memory` feature (the `serve`
        /// umbrella).
        #[cfg(feature = "memory")]
        #[command(flatten)]
        ingest: IngestArgs,
    },
    /// Speak MCP over stdio, for `claude mcp add nidus -- nidus mcp --dir ~/.nidus`.
    #[cfg(feature = "mcp")]
    Mcp {
        #[command(flatten)]
        store: StoreArgs,
        #[cfg(feature = "memory")]
        #[command(flatten)]
        ingest: IngestArgs,
    },
    /// List collections.
    Collections {
        #[command(flatten)]
        store: StoreArgs,
    },
    /// Create a collection.
    Create {
        #[command(flatten)]
        store: StoreArgs,
        name: String,
    },
    /// Drop a collection and its records.
    Drop {
        #[command(flatten)]
        store: StoreArgs,
        name: String,
    },
    /// Upsert records (JSON array of records) from a file or stdin.
    Upsert {
        #[command(flatten)]
        store: StoreArgs,
        collection: String,
        /// Read records from this file instead of stdin.
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Nearest-neighbour search. Query vector is a JSON array of floats.
    Search {
        #[command(flatten)]
        store: StoreArgs,
        /// Collections to search; omit to search every collection.
        collections: Vec<String>,
        /// Read the query vector from this file instead of stdin.
        #[arg(long)]
        query_file: Option<PathBuf>,
        #[arg(long, short = 'k', default_value_t = 10)]
        top_k: usize,
        /// Skip this many top-ranked hits before returning (pagination).
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// Drop hits scoring below this cosine similarity.
        #[arg(long)]
        min_score: Option<f32>,
        /// AND-filter as JSON. Leaves: Eq, Ne, Glob, IGlob, In, NotIn, Lt, Le, Gt, Ge, Contains, NotContains, ContainsAny. Groups: All, Any, Not.
        /// E.g. '[{"Ge":["ts",{"Int":1700000000}]},{"Ne":["status",{"Str":"archived"}]}]'.
        #[arg(long = "where")]
        filter: Option<String>,
        /// Force the exact scan, bypassing any ANN index and the quantized first pass.
        #[arg(long)]
        exact: bool,
        /// Return only this attr (repeatable). Mutually exclusive with --exclude-attr.
        #[arg(long = "include-attr")]
        include_attributes: Vec<String>,
        /// Return every attr but this one (repeatable). Mutually exclusive with --include-attr.
        #[arg(long = "exclude-attr")]
        exclude_attributes: Vec<String>,
        /// Ranking expression as JSON, e.g. '{"Decay":{"field":"ts","origin":1700000000000,"scale":604800000,"lambda":0.2}}'.
        #[arg(long = "rank-by")]
        rank_by: Option<String>,
        /// Cap hits per distinct value of this attribute (needs --limit-per-max).
        #[arg(long = "limit-per", requires = "limit_per_max")]
        limit_per: Option<String>,
        /// Maximum hits kept per distinct --limit-per value.
        #[arg(long = "limit-per-max", requires = "limit_per")]
        limit_per_max: Option<usize>,
    },
    /// Count records matching a filter, and sum numeric attributes, without listing them.
    Aggregate {
        #[command(flatten)]
        store: StoreArgs,
        /// Collections to aggregate over; omit for every collection.
        collections: Vec<String>,
        /// AND-filter as JSON (same form as `search --where`).
        #[arg(long = "where")]
        filter: Option<String>,
        /// Attribute to sum (repeatable). Missing and non-numeric values are skipped.
        #[arg(long = "sum")]
        sum: Vec<String>,
        /// Report one row per distinct value of this attribute, alongside the totals.
        #[arg(long = "group-by")]
        group_by: Option<String>,
    },
    /// List records by metadata filter (no vector query).
    List {
        #[command(flatten)]
        store: StoreArgs,
        /// Collections to list from; omit to list from every collection.
        collections: Vec<String>,
        /// Skip this many matches before returning (pagination).
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// Maximum number of results.
        #[arg(long, short = 'n', default_value_t = 100)]
        limit: usize,
        /// AND-filter as JSON. Leaves: Eq, Ne, Glob, IGlob, In, NotIn, Lt, Le, Gt, Ge, Contains, NotContains, ContainsAny. Groups: All, Any, Not.
        /// E.g. '[{"Ge":["ts",{"Int":1700000000}]},{"Ne":["status",{"Str":"archived"}]}]'.
        #[arg(long = "where")]
        filter: Option<String>,
        /// Return only this attr (repeatable). Mutually exclusive with --exclude-attr.
        #[arg(long = "include-attr")]
        include_attributes: Vec<String>,
        /// Return every attr but this one (repeatable). Mutually exclusive with --include-attr.
        #[arg(long = "exclude-attr")]
        exclude_attributes: Vec<String>,
        /// Sort by this attribute instead of storage order.
        #[arg(long = "order-by")]
        order_by: Option<String>,
        /// Sort --order-by descending.
        #[arg(long, requires = "order_by")]
        desc: bool,
    },
    /// Declare a collection's full-text-indexed fields (BM25). The tuning flags below apply to
    /// every `--field`; use `--field-spec` to tune one field on its own. Re-running rebuilds
    /// the affected field indexes.
    SetFtsSchema {
        #[command(flatten)]
        store: StoreArgs,
        collection: String,
        /// Attribute field to full-text index, taking the tuning flags below (repeatable).
        #[arg(long = "field")]
        fields: Vec<String>,
        /// One field with its own tuning, e.g. `--field-spec 'body:k1=1.5,b=0.3'`. Keys:
        /// k1, b, ascii_folding, max_token_len. Unnamed keys keep the flag defaults.
        #[arg(long = "field-spec")]
        field_specs: Vec<String>,
        /// BM25 term-frequency saturation (default 1.2).
        #[arg(long)]
        k1: Option<f32>,
        /// BM25 length normalization, 0..=1 (default 0.75).
        #[arg(long)]
        b: Option<f32>,
        /// Fold Latin diacritics to ASCII, so "café" and "cafe" share a term.
        #[arg(long)]
        ascii_folding: bool,
        /// Drop tokens longer than this many characters (default: no limit).
        #[arg(long)]
        max_token_len: Option<usize>,
    },
    /// Full-text (BM25) search of fields declared via `set-fts-schema`.
    TextSearch {
        #[command(flatten)]
        store: StoreArgs,
        /// The full-text-indexed field to search. Omit when using --clause.
        field: Option<String>,
        /// Query text (analyzed the same way documents were indexed).
        query: Option<String>,
        #[command(flatten)]
        text: TextQueryArgs,
        /// Collections to search; omit to search every collection.
        #[arg(long = "in")]
        collections: Vec<String>,
        #[arg(long, short = 'k', default_value_t = 10)]
        top_k: usize,
        /// Skip this many top-ranked hits before returning (pagination).
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// Drop hits scoring below this raw BM25 score.
        #[arg(long)]
        min_score: Option<f32>,
        /// AND-filter as JSON (same form as `search --where`).
        #[arg(long = "where")]
        filter: Option<String>,
    },
    /// Hybrid search: fuse a vector query and a BM25 text query with RRF.
    HybridSearch {
        #[command(flatten)]
        store: StoreArgs,
        /// The full-text-indexed field for the BM25 leg. Omit when using --clause.
        field: Option<String>,
        /// Query text for the BM25 leg.
        text: Option<String>,
        #[command(flatten)]
        query: TextQueryArgs,
        /// Read the query vector (JSON array) from this file instead of stdin.
        #[arg(long)]
        query_file: Option<PathBuf>,
        /// Collections to search; omit to search every collection.
        #[arg(long = "in")]
        collections: Vec<String>,
        #[arg(long, short = 'k', default_value_t = 10)]
        top_k: usize,
        /// Skip this many top-ranked fused hits before returning (pagination).
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// AND-filter as JSON, applied to both legs.
        #[arg(long = "where")]
        filter: Option<String>,
        /// RRF rank-bias constant.
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f32,
        /// Candidates pulled per leg before fusing.
        #[arg(long, default_value_t = 100)]
        candidates: usize,
        /// Weight on the vector leg's fused contribution.
        #[arg(long, default_value_t = 1.0)]
        vector_weight: f32,
        /// Weight on the BM25 leg's fused contribution.
        #[arg(long, default_value_t = 1.0)]
        text_weight: f32,
    },
    /// Print every record in a collection (JSON).
    Get {
        #[command(flatten)]
        store: StoreArgs,
        collection: String,
    },
    /// Delete records by id, or by `--where` filter.
    Delete {
        #[command(flatten)]
        store: StoreArgs,
        collection: String,
        /// Ids to delete.
        ids: Vec<String>,
        /// Delete by filter (JSON) instead of ids.
        #[arg(long = "where", conflicts_with = "ids")]
        filter: Option<String>,
    },
    /// Reclaim dead rows and superseded log records.
    Compact {
        #[command(flatten)]
        store: StoreArgs,
    },
    /// Snapshot a store into a single compressed archive (`.tar.gz`).
    Backup {
        /// Store directory to back up (the source when `--persistence` is omitted).
        #[arg(long, short = 'd')]
        dir: PathBuf,
        /// Read the source store from this persistence location instead of `--dir` —
        /// e.g. `s3://bucket/store` or `gs://bucket/store` for an object-backed store.
        #[arg(long)]
        persistence: Option<String>,
        /// Output archive location — a local path, `file://…`, `s3://…`, or `gs://…`.
        /// Defaults to `<dir-name>-<unix-secs>.tar.gz` in the current directory.
        #[arg(long, short = 'o')]
        out: Option<String>,
    },
    /// Restore a store from a `nidus backup` archive (`.tar.gz`).
    Restore {
        /// Backup archive location to restore from (a local path, `file://…`, `s3://…`).
        #[arg(long = "in", short = 'i')]
        input: String,
        /// Target store directory (created if absent; the target when `--persistence`
        /// is omitted).
        #[arg(long, short = 'd')]
        dir: PathBuf,
        /// Restore into this persistence location instead of `--dir` — e.g.
        /// `s3://bucket/store` for an object-backed store.
        #[arg(long)]
        persistence: Option<String>,
        /// Overwrite an existing store without prompting (for cron / scripts).
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Print store footprint and collections (JSON).
    Stats {
        #[command(flatten)]
        store: StoreArgs,
    },
    /// Remember a fact: embed `text` (optionally summarizing first) and store it.
    /// Needs an embedder — the same `--embed-*` flags (and `NIDUS_EMBED_*` envs) `serve` takes.
    #[cfg(feature = "memory")]
    Remember {
        #[command(flatten)]
        store: StoreArgs,
        #[command(flatten)]
        ingest: IngestArgs,
        collection: String,
        /// The text to remember.
        text: String,
        /// Id to store under. Omit to derive a stable one from the text, which makes
        /// re-remembering the same fact idempotent instead of accumulating duplicates.
        #[arg(long)]
        id: Option<String>,
        /// Extra attrs as a JSON object of typed values, e.g. '{"tag":{"Str":"ops"}}'.
        #[arg(long)]
        attrs: Option<String>,
        /// Summarize the text first and embed the summary, storing both.
        #[cfg(feature = "summarize")]
        #[arg(long)]
        summarize: bool,
    },
    /// Recall the nearest remembered text to `query`. Opens read-only, so it runs
    /// alongside a `nidus serve` holding the writer lock.
    #[cfg(feature = "memory")]
    Recall {
        #[command(flatten)]
        store: StoreArgs,
        #[command(flatten)]
        ingest: IngestArgs,
        collection: String,
        /// The query text, embedded the same way the stored text was.
        query: String,
        #[arg(long, short = 'k', default_value_t = 10)]
        top_k: usize,
        /// Drop hits scoring at or below this cosine similarity.
        #[arg(long)]
        min_score: Option<f32>,
        /// AND-filter as JSON (same form as `search --where`).
        #[arg(long = "where")]
        filter: Option<String>,
    },
}

/// Parse-and-dispatch entry point used by `main`.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Serve {
            store,
            addr,
            token,
            max_body_bytes,
            max_concurrent_requests,
            read_timeout,
            write_timeout,
            body_idle_timeout,
            refresh_interval,
            require_remote,
            #[cfg(feature = "memory")]
            ingest,
        } => serve(
            ServeArgs {
                addr,
                token,
                max_body_bytes,
                max_concurrent_requests,
                read_timeout,
                write_timeout,
                body_idle_timeout,
                refresh_interval,
                require_remote,
            },
            store,
            #[cfg(feature = "memory")]
            ingest,
        ),
        #[cfg(feature = "mcp")]
        Command::Mcp {
            store,
            #[cfg(feature = "memory")]
            ingest,
        } => mcp(
            store,
            #[cfg(feature = "memory")]
            ingest,
        ),
        Command::Collections { store } => {
            let db = open(&store, false)?;
            print_json(&db.collections())
        }
        Command::Create { store, name } => {
            let mut db = open(&store, true)?;
            db.create_collection(&name)?;
            print_json(&serde_json::json!({ "created": name }))
        }
        Command::Drop { store, name } => {
            let mut db = open(&store, true)?;
            db.drop_collection(&name)?;
            print_json(&serde_json::json!({ "dropped": name }))
        }
        Command::Upsert {
            store,
            collection,
            file,
        } => {
            let mut db = open(&store, true)?;
            let records: Vec<Record> = serde_json::from_str(&read_input(file.as_ref())?)?;
            let n = db.upsert(&collection, &records)?;
            print_json(&serde_json::json!({ "upserted": n }))
        }
        Command::Search {
            store,
            collections,
            query_file,
            top_k,
            offset,
            min_score,
            filter,
            exact,
            include_attributes,
            exclude_attributes,
            rank_by,
            limit_per,
            limit_per_max,
        } => {
            let db = open(&store, false)?;
            let query: Vec<f32> = serde_json::from_str(&read_input(query_file.as_ref())?)?;
            let filter = match filter {
                Some(s) => serde_json::from_str(&s)?,
                None => Filter::default(),
            };
            let opts = SearchOpts {
                top_k,
                offset,
                min_score,
                filter,
                exact,
                projection: projection(include_attributes, exclude_attributes)?,
                rank_by: rank_by.map(|s| serde_json::from_str(&s)).transpose()?,
                // clap's `requires` pairing means either both flags are present or neither is.
                limit_per: limit_per
                    .zip(limit_per_max)
                    .map(|(f, m)| LimitPer::new(f, m)),
                ..Default::default()
            };
            let refs: Vec<&str> = collections.iter().map(String::as_str).collect();
            let hits = if refs.is_empty() {
                db.search(Scope::All, &query, &opts)?
            } else {
                db.search(Scope::Collections(&refs), &query, &opts)?
            };
            let out: Vec<HitDto> = hits.into_iter().map(HitDto::from).collect();
            print_json(&out)
        }
        Command::List {
            store,
            collections,
            offset,
            limit,
            filter,
            include_attributes,
            exclude_attributes,
            order_by,
            desc,
        } => {
            let db = open(&store, false)?;
            let filter = match filter {
                Some(s) => serde_json::from_str(&s)?,
                None => Filter::default(),
            };
            let opts = ListOpts {
                offset,
                limit,
                filter,
                projection: projection(include_attributes, exclude_attributes)?,
                order_by: order_by.map(|f| OrderBy {
                    field: f,
                    descending: desc,
                }),
            };
            let refs: Vec<&str> = collections.iter().map(String::as_str).collect();
            let hits = if refs.is_empty() {
                db.list(Scope::All, &opts)?
            } else {
                db.list(Scope::Collections(&refs), &opts)?
            };
            let out: Vec<HitDto> = hits.into_iter().map(HitDto::from).collect();
            print_json(&out)
        }
        Command::Aggregate {
            store,
            collections,
            filter,
            sum,
            group_by,
        } => {
            let db = open(&store, false)?;
            let filter = match filter {
                Some(s) => serde_json::from_str(&s)?,
                None => Filter::default(),
            };
            let opts = AggregateOpts {
                filter,
                sum,
                group_by,
            };
            let refs: Vec<&str> = collections.iter().map(String::as_str).collect();
            let out = if refs.is_empty() {
                db.aggregate(Scope::All, &opts)?
            } else {
                db.aggregate(Scope::Collections(&refs), &opts)?
            };
            print_json(&out)
        }
        Command::SetFtsSchema {
            store,
            collection,
            fields,
            field_specs,
            k1,
            b,
            ascii_folding,
            max_token_len,
        } => {
            let defaults = FieldDefaults {
                k1,
                b,
                ascii_folding,
                max_token_len,
            };
            let mut decl: Vec<FtsField> = fields
                .iter()
                .map(|name| defaults.apply(FtsField::new(name)))
                .collect();
            for spec in &field_specs {
                decl.push(parse_field_spec(spec, &defaults)?);
            }
            if decl.is_empty() {
                bail!("set-fts-schema needs at least one --field or --field-spec");
            }
            if let Some(dupe) = first_duplicate(&decl) {
                bail!("field '{dupe}' is declared twice; give it one --field or --field-spec");
            }
            let mut db = open(&store, true)?;
            db.set_fts_schema(&collection, &decl)?;
            // Per field, not one summary: the whole point of --field-spec is that they differ.
            let reported: Vec<serde_json::Value> = decl
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "field": f.field,
                        "k1": f.k1,
                        "b": f.b,
                        "ascii_folding": f.analyzer.ascii_folding,
                        "max_token_len": f.analyzer.max_token_len,
                    })
                })
                .collect();
            print_json(&serde_json::json!({
                "collection": collection,
                "fts_fields": reported,
            }))
        }
        Command::TextSearch {
            store,
            field,
            query,
            text,
            collections,
            top_k,
            offset,
            min_score,
            filter,
        } => {
            let q = text.query(field, query)?;
            let db = open(&store, false)?;
            let filter = match filter {
                Some(s) => serde_json::from_str(&s)?,
                None => Filter::default(),
            };
            let opts = SearchOpts {
                top_k,
                offset,
                min_score,
                filter,
                explain: text.explain,
                ..Default::default()
            };
            let refs: Vec<&str> = collections.iter().map(String::as_str).collect();
            let hits = if refs.is_empty() {
                db.text_search(Scope::All, &q, &opts)?
            } else {
                db.text_search(Scope::Collections(&refs), &q, &opts)?
            };
            let out: Vec<HitDto> = hits.into_iter().map(HitDto::from).collect();
            print_json(&out)
        }
        Command::HybridSearch {
            store,
            field,
            text,
            query,
            query_file,
            collections,
            top_k,
            offset,
            filter,
            rrf_k,
            candidates,
            vector_weight,
            text_weight,
        } => {
            let q = query.query(field, text)?;
            let db = open(&store, false)?;
            let vector: Vec<f32> = serde_json::from_str(&read_input(query_file.as_ref())?)?;
            let filter = match filter {
                Some(s) => serde_json::from_str(&s)?,
                None => Filter::default(),
            };
            let opts = HybridOpts {
                top_k,
                offset,
                filter,
                rrf_k,
                candidates,
                explain: query.explain,
                vector_weight,
                text_weight,
            };
            let refs: Vec<&str> = collections.iter().map(String::as_str).collect();
            let hits = if refs.is_empty() {
                db.hybrid_search(Scope::All, &vector, &q, &opts)?
            } else {
                db.hybrid_search(Scope::Collections(&refs), &vector, &q, &opts)?
            };
            let out: Vec<HitDto> = hits.into_iter().map(HitDto::from).collect();
            print_json(&out)
        }
        Command::Get { store, collection } => {
            let db = open(&store, false)?;
            print_json(&db.get_all(&collection))
        }
        Command::Delete {
            store,
            collection,
            ids,
            filter,
        } => {
            let mut db = open(&store, true)?;
            let n = match filter {
                Some(s) => {
                    let f: Filter = serde_json::from_str(&s)?;
                    db.delete_where(&collection, &f)?
                }
                None => {
                    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
                    db.delete(&collection, &refs)?
                }
            };
            print_json(&serde_json::json!({ "deleted": n }))
        }
        Command::Compact { store } => {
            let mut db = open(&store, true)?;
            db.compact()?;
            print_json(&serde_json::json!({ "ok": true }))
        }
        Command::Backup {
            dir,
            persistence,
            out,
        } => {
            let out = out.unwrap_or_else(|| backup::default_out_name(&dir));
            let source = persistence.unwrap_or_else(|| dir.to_string_lossy().into_owned());
            print_json(&backup::backup(&source, &out)?)
        }
        Command::Restore {
            input,
            dir,
            persistence,
            yes,
        } => {
            let target = persistence.unwrap_or_else(|| dir.to_string_lossy().into_owned());
            print_json(&backup::restore(&input, &target, yes)?)
        }
        Command::Stats { store } => {
            let db = open(&store, false)?;
            print_json(&serde_json::json!({
                "dimension": db.dimension(),
                "distance": format!("{:?}", db.config().distance),
                "ann": db.config().ann.map(AnnDto::from),
                "collections": db.collections(),
                "footprint": FootprintDto::from(db.footprint()),
            }))
        }
        #[cfg(feature = "memory")]
        Command::Remember {
            store,
            ingest,
            collection,
            text,
            id,
            attrs,
            #[cfg(feature = "summarize")]
            summarize,
        } => memory::remember(
            store,
            ingest,
            collection,
            text,
            id,
            attrs,
            #[cfg(feature = "summarize")]
            summarize,
        ),
        #[cfg(feature = "memory")]
        Command::Recall {
            store,
            ingest,
            collection,
            query,
            top_k,
            min_score,
            filter,
        } => memory::recall(store, ingest, collection, query, top_k, min_score, filter),
    }
}

/// The invocation-wide `set-fts-schema` tuning flags, which every declared field starts from
/// and a `--field-spec` may then override key by key.
struct FieldDefaults {
    k1: Option<f32>,
    b: Option<f32>,
    ascii_folding: bool,
    max_token_len: Option<usize>,
}

impl FieldDefaults {
    fn apply(&self, mut f: FtsField) -> FtsField {
        f.k1 = self.k1.unwrap_or(f.k1);
        f.b = self.b.unwrap_or(f.b);
        f.analyzer.ascii_folding = self.ascii_folding;
        f.analyzer.max_token_len = self.max_token_len;
        f
    }
}

/// Parse `body:k1=1.5,b=0.3,ascii_folding=true,max_token_len=40` (nidus-9jp). An unknown or
/// unparseable key is refused rather than ignored — a silently dropped knob leaves the field
/// indexed differently than the caller asked, and nothing downstream would say so.
fn parse_field_spec(spec: &str, defaults: &FieldDefaults) -> Result<FtsField> {
    let (name, rest) = spec.split_once(':').unwrap_or((spec, ""));
    if name.is_empty() {
        bail!("--field-spec '{spec}' has no field name; want 'field:key=value,…'");
    }
    let mut f = defaults.apply(FtsField::new(name));
    for pair in rest.split(',').filter(|p| !p.trim().is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, "true"));
        let (key, value) = (key.trim(), value.trim());
        let bad =
            |what: &str| anyhow::anyhow!("--field-spec '{spec}': {key}={value} is not {what}");
        match key {
            "k1" => f.k1 = value.parse().map_err(|_| bad("a number"))?,
            "b" => f.b = value.parse().map_err(|_| bad("a number"))?,
            "ascii_folding" => {
                f.analyzer.ascii_folding = value.parse().map_err(|_| bad("a bool"))?
            }
            "max_token_len" => {
                f.analyzer.max_token_len = Some(value.parse().map_err(|_| bad("a whole number"))?);
            }
            _ => bail!(
                "--field-spec '{spec}': unknown key '{key}'; want k1, b, ascii_folding, or max_token_len"
            ),
        }
    }
    Ok(f)
}

/// The first field name declared twice, whether across `--field` or `--field-spec`. Two
/// declarations of one field would leave which tuning won up to ordering.
fn first_duplicate(decl: &[FtsField]) -> Option<&str> {
    let mut seen = std::collections::HashSet::new();
    decl.iter()
        .find(|f| !seen.insert(f.field.as_str()))
        .map(|f| f.field.as_str())
}

/// Open the store. `mutating` commands take the writer lock; read commands open
/// read-only so they never contend with a running `nidus serve` writer.
fn open(store: &StoreArgs, mutating: bool) -> Result<Nidus> {
    if mutating && store.read_only {
        bail!("--read-only was set, but this command mutates the store");
    }
    let mode = if mutating {
        OpenMode::ReadWrite
    } else {
        OpenMode::ReadOnly
    };
    Nidus::open(store.config(mode)?)
}

/// The `serve` flags, as one struct.
struct ServeArgs {
    addr: String,
    token: Option<String>,
    max_body_bytes: usize,
    max_concurrent_requests: usize,
    read_timeout: u64,
    write_timeout: u64,
    body_idle_timeout: u64,
    refresh_interval: Option<u64>,
    require_remote: bool,
}

/// Seconds from a flag to a deadline, where `0` means "no deadline".
fn timeout_secs(secs: u64) -> Option<std::time::Duration> {
    (secs > 0).then(|| std::time::Duration::from_secs(secs))
}

fn serve(
    args: ServeArgs,
    store: StoreArgs,
    #[cfg(feature = "memory")] ingest: IngestArgs,
) -> Result<()> {
    let ServeArgs {
        addr,
        token,
        max_body_bytes,
        max_concurrent_requests,
        read_timeout,
        write_timeout,
        body_idle_timeout,
        refresh_interval,
        require_remote,
    } = args;
    // The container contract: no durable local disk, so refuse anything that would
    // keep its state process-local (a local-file store or process-RAM working set).
    if require_remote {
        if !store.is_object_store() {
            bail!(
                "--require-remote: --persistence must be an object store (s3://… or gs://…), got {:?}",
                store.persistence.as_deref().unwrap_or("<local files>")
            );
        }
        if !store.is_shared_memory() {
            bail!(
                "--require-remote: --memory must be a shared Redis-family tier (redis://…), got {:?}",
                store.memory.as_deref().unwrap_or("<process RAM>")
            );
        }
    }
    let mode = if store.read_only {
        OpenMode::ReadOnly
    } else {
        OpenMode::ReadWrite
    };
    // Resolve the config here so a bad flag fails before anything binds, but defer the
    // OPEN itself to the server: with `--wait-for-lease` it can block indefinitely, and
    // the listener must already be answering liveness probes while a standby waits.
    let open_config = store.config(mode)?;
    // An empty --token / NIDUS_TOKEN (clap reads the env var) means no auth.
    let token = token.filter(|t| !t.is_empty());
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Build the embedder/summarizer (async — some adapters probe on construction)
    // on the same runtime that will drive the server. `None` when the flags were
    // omitted: the server then serves only the raw endpoints.
    #[cfg(feature = "memory")]
    let embedder = rt.block_on(ingest.embedder())?;
    #[cfg(all(feature = "memory", feature = "summarize"))]
    let summarizer = rt.block_on(ingest.summarizer())?;

    let cfg = crate::server::ServeConfig {
        addr,
        token,
        max_body_bytes,
        max_concurrent_requests,
        read_timeout: timeout_secs(read_timeout),
        write_timeout: timeout_secs(write_timeout),
        body_idle_timeout: timeout_secs(body_idle_timeout),
        max_staleness: open_config.max_staleness,
        // A third of the lease TTL: frequent enough that a long batch cannot let the lease
        // lapse, infrequent enough to be a rounding error in object-store cost.
        lease_renew_interval: open_config.lock_ttl / 3,
        refresh_interval: refresh_interval.map(std::time::Duration::from_secs),
        #[cfg(feature = "memory")]
        embedder,
        #[cfg(all(feature = "memory", feature = "summarize"))]
        summarizer,
    };
    rt.block_on(crate::server::serve(move || Nidus::open(open_config), cfg))
}

/// `nidus mcp`: speak MCP over stdio. Always opens read-write — there is exactly one
/// client and no reason to run a memory server it cannot write to.
#[cfg(feature = "mcp")]
fn mcp(store: StoreArgs, #[cfg(feature = "memory")] ingest: IngestArgs) -> Result<()> {
    // Honour `--read-only` as `serve` does: a reader that never takes the writer lock is a
    // legitimate way to run this alongside a writer. `remember`/`forget` then fail honestly.
    let mode = if store.read_only {
        OpenMode::ReadOnly
    } else {
        OpenMode::ReadWrite
    };
    let open_config = store.config(mode)?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    #[cfg(feature = "memory")]
    let embedder = rt.block_on(ingest.embedder())?;
    #[cfg(all(feature = "memory", feature = "summarize"))]
    let summarizer = rt.block_on(ingest.summarizer())?;

    let cfg = crate::server::StdioConfig {
        #[cfg(feature = "memory")]
        embedder,
        #[cfg(all(feature = "memory", feature = "summarize"))]
        summarizer,
        // A third of the lease TTL, as in `serve`.
        lease_renew_interval: open_config.lock_ttl / 3,
    };
    rt.block_on(crate::server::serve_stdio(
        move || Nidus::open(open_config),
        cfg,
    ))
}

/// Resolve `--include-attr`/`--exclude-attr` into a [`Projection`]. Both given is an error,
/// not a precedence rule — the same refusal the HTTP surface makes (nidus-m50.15).
fn projection(include: Vec<String>, exclude: Vec<String>) -> Result<Projection> {
    match (include.is_empty(), exclude.is_empty()) {
        (true, true) => Ok(Projection::All),
        (false, true) => Ok(Projection::Include(include)),
        (true, false) => Ok(Projection::Exclude(exclude)),
        (false, false) => anyhow::bail!("--include-attr and --exclude-attr are mutually exclusive"),
    }
}

/// Read JSON from `file`, or from stdin when absent.
fn read_input(file: Option<&PathBuf>) -> Result<String> {
    match file {
        Some(p) => Ok(std::fs::read_to_string(p)?),
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            Ok(s)
        }
    }
}

/// Pretty-print a value as JSON to stdout (still valid JSON for piping).
fn print_json<T: Serialize>(v: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnnKind, QuantKind};

    #[test]
    fn no_subcommand_errors() {
        assert!(Cli::try_parse_from(["nidus"]).is_err());
    }

    #[test]
    fn serve_defaults_addr() {
        let cli = Cli::try_parse_from(["nidus", "serve", "--dir", "/tmp/s", "--dim", "8"]).unwrap();
        match cli.command {
            Command::Serve {
                addr, store, token, ..
            } => {
                assert_eq!(addr, "127.0.0.1:7700");
                assert_eq!(store.dim, Some(8));
                assert!(!store.read_only);
                assert_eq!(token, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    /// The `serve` memory flags parse into `IngestArgs` (present only under the
    /// `memory` feature, which the `serve` umbrella pulls).
    #[cfg(feature = "memory")]
    #[test]
    fn serve_parses_ingest_flags() {
        let cli = Cli::try_parse_from([
            "nidus",
            "serve",
            "--dir",
            "/tmp/s",
            "--dim",
            "3",
            "--embed-provider",
            "openai",
            "--embed-model",
            "text-embedding-3-small",
            "--embed-api-key",
            "sk-test",
        ])
        .unwrap();
        match cli.command {
            Command::Serve { ingest, .. } => {
                assert_eq!(ingest.embed_provider.as_deref(), Some("openai"));
                assert_eq!(
                    ingest.embed_model.as_deref(),
                    Some("text-embedding-3-small")
                );
                assert_eq!(ingest.embed_api_key.as_deref(), Some("sk-test"));
                assert_eq!(ingest.embed_base_url, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    /// With no ingest flags, `IngestArgs` is all-`None` (memory routes then 400).
    #[cfg(feature = "memory")]
    #[test]
    fn serve_without_ingest_flags_is_none() {
        let cli = Cli::try_parse_from(["nidus", "serve", "--dir", "/tmp/s", "--dim", "3"]).unwrap();
        match cli.command {
            Command::Serve { ingest, .. } => {
                assert_eq!(ingest.embed_provider, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    /// `remember` takes the collection and text positionally and reuses `serve`'s ingest
    /// flags, so a caller configures the embedder exactly one way across the whole CLI.
    #[cfg(feature = "memory")]
    #[test]
    fn remember_parses_positionals_and_ingest_flags() {
        let cli = Cli::try_parse_from([
            "nidus",
            "remember",
            "--dir",
            "/tmp/s",
            "--embed-provider",
            "ollama",
            "--embed-base-url",
            "http://127.0.0.1:11434",
            "--id",
            "manual",
            "--attrs",
            r#"{"tag":{"Str":"ops"}}"#,
            "notes",
            "deploys run at noon",
        ])
        .unwrap();
        match cli.command {
            Command::Remember {
                store,
                ingest,
                collection,
                text,
                id,
                attrs,
                ..
            } => {
                assert_eq!(store.dir, PathBuf::from("/tmp/s"));
                assert_eq!(store.dim, None, "dimension comes from the embedder");
                assert_eq!(ingest.embed_provider.as_deref(), Some("ollama"));
                assert_eq!(
                    ingest.embed_base_url.as_deref(),
                    Some("http://127.0.0.1:11434")
                );
                assert_eq!(collection, "notes");
                assert_eq!(text, "deploys run at noon");
                assert_eq!(id.as_deref(), Some("manual"));
                assert_eq!(attrs.as_deref(), Some(r#"{"tag":{"Str":"ops"}}"#));
            }
            _ => panic!("expected Remember"),
        }
    }

    /// Without `--id` the id is left for the handler to derive from the text.
    #[cfg(feature = "memory")]
    #[test]
    fn remember_without_an_id_leaves_it_unset() {
        let cli = Cli::try_parse_from([
            "nidus",
            "remember",
            "--dir",
            "/tmp/s",
            "--embed-provider",
            "ollama",
            "notes",
            "a fact",
        ])
        .unwrap();
        match cli.command {
            Command::Remember { id, attrs, .. } => {
                assert_eq!(id, None);
                assert_eq!(attrs, None);
            }
            _ => panic!("expected Remember"),
        }
    }

    /// `--summarize` is a plain flag, present only when the summarizer is compiled in.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    #[test]
    fn remember_parses_the_summarize_flag() {
        let cli = Cli::try_parse_from([
            "nidus",
            "remember",
            "--dir",
            "/tmp/s",
            "--embed-provider",
            "ollama",
            "--summarize-provider",
            "anthropic",
            "--summarize",
            "notes",
            "a long wall of text",
        ])
        .unwrap();
        match cli.command {
            Command::Remember {
                summarize, ingest, ..
            } => {
                assert!(summarize);
                assert_eq!(ingest.summarize_provider.as_deref(), Some("anthropic"));
            }
            _ => panic!("expected Remember"),
        }
    }

    /// `recall` mirrors `search`'s query knobs (`-k`, `--min-score`, `--where`).
    #[cfg(feature = "memory")]
    #[test]
    fn recall_parses_query_knobs() {
        let cli = Cli::try_parse_from([
            "nidus",
            "recall",
            "--dir",
            "/tmp/s",
            "--embed-provider",
            "ollama",
            "-k",
            "3",
            "--min-score",
            "0.4",
            "--where",
            r#"[{"Eq":["tag",{"Str":"ops"}]}]"#,
            "notes",
            "when do deploys run",
        ])
        .unwrap();
        match cli.command {
            Command::Recall {
                collection,
                query,
                top_k,
                min_score,
                filter,
                ..
            } => {
                assert_eq!(collection, "notes");
                assert_eq!(query, "when do deploys run");
                assert_eq!(top_k, 3);
                assert_eq!(min_score, Some(0.4));
                assert_eq!(filter.as_deref(), Some(r#"[{"Eq":["tag",{"Str":"ops"}]}]"#));
            }
            _ => panic!("expected Recall"),
        }
    }

    /// `recall`'s `top_k` defaults to the same 10 `search` uses.
    #[cfg(feature = "memory")]
    #[test]
    fn recall_defaults_top_k() {
        let cli = Cli::try_parse_from([
            "nidus",
            "recall",
            "--dir",
            "/tmp/s",
            "--embed-provider",
            "ollama",
            "notes",
            "q",
        ])
        .unwrap();
        match cli.command {
            Command::Recall {
                top_k,
                min_score,
                filter,
                ..
            } => {
                assert_eq!(top_k, 10);
                assert_eq!(min_score, None);
                assert_eq!(filter, None);
            }
            _ => panic!("expected Recall"),
        }
    }

    /// Both subcommands need their two positionals: a collection with no text is a parse
    /// error, not a remember of the empty string.
    #[cfg(feature = "memory")]
    #[test]
    fn memory_subcommands_require_both_positionals() {
        assert!(Cli::try_parse_from(["nidus", "remember", "--dir", "/tmp/s", "notes"]).is_err());
        assert!(Cli::try_parse_from(["nidus", "recall", "--dir", "/tmp/s", "notes"]).is_err());
    }

    #[test]
    fn search_parses_collections_and_flags() {
        let cli = Cli::try_parse_from([
            "nidus",
            "search",
            "--dir",
            "/tmp/s",
            "--dim",
            "3",
            "docs",
            "notes",
            "-k",
            "5",
            "--min-score",
            "0.2",
        ])
        .unwrap();
        match cli.command {
            Command::Search {
                collections,
                top_k,
                min_score,
                ..
            } => {
                assert_eq!(collections, vec!["docs", "notes"]);
                assert_eq!(top_k, 5);
                assert_eq!(min_score, Some(0.2));
            }
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn search_parses_the_ranking_knobs() {
        let cli = Cli::try_parse_from([
            "nidus",
            "search",
            "--dir",
            "/tmp/s",
            "docs",
            "--rank-by",
            r#"{"Decay":{"field":"ts","origin":0}}"#,
            "--limit-per",
            "file",
            "--limit-per-max",
            "2",
        ])
        .unwrap();
        match cli.command {
            Command::Search {
                rank_by,
                limit_per,
                limit_per_max,
                ..
            } => {
                assert!(rank_by.is_some());
                assert_eq!(limit_per.as_deref(), Some("file"));
                assert_eq!(limit_per_max, Some(2));
            }
            _ => panic!("expected Search"),
        }
        // The cap's two halves are useless apart, so clap requires them together.
        assert!(
            Cli::try_parse_from([
                "nidus",
                "search",
                "--dir",
                "/tmp/s",
                "docs",
                "--limit-per",
                "file"
            ])
            .is_err()
        );
    }

    /// A `--field-spec` tunes ONE field, starting from the invocation-wide flags and
    /// overriding only the keys it names (nidus-9jp).
    #[test]
    fn a_field_spec_overrides_the_invocation_wide_flags_per_field() {
        let defaults = FieldDefaults {
            k1: Some(1.1),
            b: None,
            ascii_folding: false,
            max_token_len: Some(20),
        };
        let f = parse_field_spec("body:k1=1.5,ascii_folding,max_token_len=40", &defaults).unwrap();
        assert_eq!(f.field, "body");
        assert_eq!(f.k1, 1.5, "the spec wins over --k1");
        assert_eq!(
            f.b,
            FtsField::new("x").b,
            "an unnamed key keeps its default"
        );
        assert!(f.analyzer.ascii_folding, "a bare key means true");
        assert_eq!(f.analyzer.max_token_len, Some(40));

        // A spec naming no keys is just the field at the invocation-wide defaults.
        let bare = parse_field_spec("title", &defaults).unwrap();
        assert_eq!(bare.field, "title");
        assert_eq!(bare.k1, 1.1);
        assert_eq!(bare.analyzer.max_token_len, Some(20));
    }

    /// Each of these would otherwise index a field differently than asked, with nothing
    /// downstream to say so — so each is refused rather than ignored.
    #[test]
    fn a_malformed_field_spec_is_refused_not_ignored() {
        let defaults = FieldDefaults {
            k1: None,
            b: None,
            ascii_folding: false,
            max_token_len: None,
        };
        for spec in [
            "body:nope=1",            // unknown key
            "body:k1=fast",           // unparseable float
            "body:max_token_len=-1",  // unparseable usize
            "body:ascii_folding=yes", // unparseable bool
            ":k1=1.5",                // no field name
        ] {
            assert!(
                parse_field_spec(spec, &defaults).is_err(),
                "accepted '{spec}'"
            );
        }
    }

    /// Two declarations of one field would leave the winning tuning up to argument order.
    #[test]
    fn a_field_declared_twice_is_refused() {
        let decl = [
            FtsField::new("title"),
            FtsField::new("body"),
            FtsField::new("title"),
        ];
        assert_eq!(first_duplicate(&decl), Some("title"));
        assert_eq!(first_duplicate(&decl[..2]), None);
    }

    #[test]
    fn set_fts_schema_takes_field_specs_alongside_plain_fields() {
        let cli = Cli::try_parse_from([
            "nidus",
            "set-fts-schema",
            "--dir",
            "/tmp/s",
            "--dim",
            "3",
            "docs",
            "--field",
            "title",
            "--field-spec",
            "body:k1=1.5",
        ])
        .unwrap();
        match cli.command {
            Command::SetFtsSchema {
                fields,
                field_specs,
                ..
            } => {
                assert_eq!(fields, ["title"]);
                assert_eq!(field_specs, ["body:k1=1.5"]);
            }
            _ => panic!("expected SetFtsSchema"),
        }
    }

    #[test]
    fn list_parses_order_by_and_aggregate_parses_sums() {
        let cli = Cli::try_parse_from([
            "nidus",
            "list",
            "--dir",
            "/tmp/s",
            "--order-by",
            "ts",
            "--desc",
        ])
        .unwrap();
        match cli.command {
            Command::List { order_by, desc, .. } => {
                assert_eq!(order_by.as_deref(), Some("ts"));
                assert!(desc);
            }
            _ => panic!("expected List"),
        }
        // `--desc` alone has nothing to reverse.
        assert!(Cli::try_parse_from(["nidus", "list", "--dir", "/tmp/s", "--desc"]).is_err());

        let cli = Cli::try_parse_from([
            "nidus",
            "aggregate",
            "--dir",
            "/tmp/s",
            "docs",
            "--sum",
            "bytes",
        ])
        .unwrap();
        match cli.command {
            Command::Aggregate {
                collections, sum, ..
            } => {
                assert_eq!(collections, vec!["docs"]);
                assert_eq!(sum, vec!["bytes"]);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn hybrid_search_leg_weights_default_to_one() {
        let cli = Cli::try_parse_from([
            "nidus",
            "hybrid-search",
            "--dir",
            "/tmp/s",
            "body",
            "quantum",
        ])
        .unwrap();
        match cli.command {
            Command::HybridSearch {
                vector_weight,
                text_weight,
                ..
            } => assert_eq!((vector_weight, text_weight), (1.0, 1.0)),
            _ => panic!("expected HybridSearch"),
        }
    }

    #[test]
    fn delete_ids_and_filter_conflict() {
        // --where conflicts with positional ids.
        assert!(
            Cli::try_parse_from([
                "nidus", "delete", "--dir", "/tmp/s", "--dim", "3", "docs", "a", "--where", "[]",
            ])
            .is_err()
        );
    }

    #[test]
    fn store_args_require_dir_but_not_dim() {
        // --dir is always required.
        assert!(Cli::try_parse_from(["nidus", "collections"]).is_err());
        // --dim is now optional (inferred from an existing store's header).
        let cli = Cli::try_parse_from(["nidus", "collections", "--dir", "/tmp/s"]).unwrap();
        match cli.command {
            Command::Collections { store } => assert_eq!(store.dim, None),
            _ => panic!("expected Collections"),
        }
    }

    #[test]
    fn resolve_infers_dim_and_distance_from_existing_store() {
        let dir = tempfile::tempdir().unwrap();
        // Create a euclidean store, then drop it.
        {
            let cfg = Config::new(dir.path().to_path_buf(), 5).distance(Distance::Euclidean);
            Nidus::open(cfg).unwrap();
        }
        // No --dim / --distance: both come from the header.
        let args = StoreArgs {
            dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        assert_eq!(args.resolve().unwrap(), (5, Distance::Euclidean));
    }

    #[test]
    fn backup_parses_dir_and_optional_out() {
        let cli =
            Cli::try_parse_from(["nidus", "backup", "--dir", "/tmp/s", "-o", "/tmp/s.tar.gz"])
                .unwrap();
        match cli.command {
            Command::Backup { dir, out, .. } => {
                assert_eq!(dir, PathBuf::from("/tmp/s"));
                assert_eq!(out.as_deref(), Some("/tmp/s.tar.gz"));
            }
            _ => panic!("expected Backup"),
        }
        // --out is optional (a timestamped default is synthesized).
        let cli = Cli::try_parse_from(["nidus", "backup", "-d", "/tmp/s"]).unwrap();
        match cli.command {
            Command::Backup { out, .. } => assert_eq!(out, None),
            _ => panic!("expected Backup"),
        }
    }

    #[test]
    fn restore_parses_in_dir_and_yes() {
        let cli = Cli::try_parse_from([
            "nidus",
            "restore",
            "--in",
            "/tmp/s.tar.gz",
            "--dir",
            "/tmp/s2",
            "-y",
        ])
        .unwrap();
        match cli.command {
            Command::Restore {
                input, dir, yes, ..
            } => {
                assert_eq!(input, "/tmp/s.tar.gz");
                assert_eq!(dir, PathBuf::from("/tmp/s2"));
                assert!(yes);
            }
            _ => panic!("expected Restore"),
        }
        // -y defaults off.
        let cli = Cli::try_parse_from(["nidus", "restore", "-i", "/tmp/s.tar.gz", "-d", "/tmp/s2"])
            .unwrap();
        match cli.command {
            Command::Restore { yes, .. } => assert!(!yes),
            _ => panic!("expected Restore"),
        }
    }

    #[test]
    fn ann_defaults_off() {
        // No --ann: exact brute-force (Config::ann stays None).
        let cli =
            Cli::try_parse_from(["nidus", "search", "--dir", "/tmp/s", "--dim", "3"]).unwrap();
        match cli.command {
            Command::Search { store, .. } => assert!(store.ann_config().is_none()),
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn ann_hnsw_with_param_overrides() {
        let cli = Cli::try_parse_from([
            "nidus",
            "serve",
            "--dir",
            "/tmp/s",
            "--dim",
            "3",
            "--ann",
            "hnsw",
            "--ann-m",
            "32",
            "--ann-ef-search",
            "128",
            "--ann-overscan",
            "8",
        ])
        .unwrap();
        match cli.command {
            Command::Serve { store, .. } => {
                let ann = store.ann_config().expect("ann enabled");
                assert_eq!(ann.kind, AnnKind::Hnsw);
                assert_eq!(ann.m, 32); // overridden
                assert_eq!(ann.ef_search, 128); // overridden
                assert_eq!(ann.overscan, 8); // overridden
                assert_eq!(ann.ef_construction, AnnConfig::hnsw().ef_construction); // default kept
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn ann_ivf_uses_ivf_defaults() {
        let cli = Cli::try_parse_from([
            "nidus",
            "search",
            "--dir",
            "/tmp/s",
            "--dim",
            "3",
            "--ann",
            "ivf",
            "--ann-n-probe",
            "16",
        ])
        .unwrap();
        match cli.command {
            Command::Search { store, .. } => {
                let ann = store.ann_config().expect("ann enabled");
                assert_eq!(ann.kind, AnnKind::Ivf);
                assert_eq!(ann.n_probe, 16); // overridden
                assert_eq!(ann.n_lists, AnnConfig::ivf().n_lists); // default kept
            }
            _ => panic!("expected Search"),
        }
    }

    /// Parse a `serve` command line and hand back the `StoreArgs` it produced.
    fn serve_store(args: &[&str]) -> StoreArgs {
        let mut argv = vec!["nidus", "serve", "--dir", "/tmp/s", "--dim", "3"];
        argv.extend_from_slice(args);
        match Cli::try_parse_from(argv).expect("parses").command {
            Command::Serve { store, .. } => store,
            _ => panic!("expected Serve"),
        }
    }

    /// The whole point of nidus-1e6.1: every one of these `Config` knobs used to be
    /// library-only, so `nidus serve` could ONLY run the default exact, all-RAM,
    /// single-writer store. Assert each flag actually reaches `Config`.
    #[test]
    fn store_flags_reach_config() {
        let cfg = serve_store(&[
            "--cluster",
            "--mmap",
            "--query-threads",
            "4",
            "--segment-max-rows",
            "100000",
            "--segment-index-min-rows",
            "50000",
            "--fsync",
            "on-flush",
            "--no-auto-compact",
            "--lock-ttl",
            "15",
            "--max-vector-bytes",
            "4096",
        ])
        .config(OpenMode::ReadWrite)
        .expect("config builds");

        assert!(cfg.cluster);
        assert!(cfg.mmap);
        assert_eq!(cfg.query_threads, 4);
        assert_eq!(cfg.segment_max_rows, Some(100_000));
        assert_eq!(cfg.segment_index_min_rows, Some(50_000));
        assert_eq!(cfg.fsync, Fsync::OnFlush);
        assert_eq!(cfg.auto_compact, None);
        assert_eq!(cfg.lock_ttl, std::time::Duration::from_secs(15));
        assert_eq!(cfg.max_vector_bytes, Some(4096));
    }

    /// With none of the new flags passed, `Config`'s own defaults must survive — a
    /// flag defaulting to `Some(..)` would silently change behaviour for everyone.
    #[test]
    fn store_flags_omitted_keep_config_defaults() {
        let cfg = serve_store(&[])
            .config(OpenMode::ReadWrite)
            .expect("config");
        let default = Config::new("/tmp/s", 3);

        assert!(!cfg.cluster);
        assert!(!cfg.mmap);
        assert_eq!(cfg.quantization, None);
        assert_eq!(cfg.query_threads, default.query_threads);
        assert_eq!(cfg.segment_max_rows, default.segment_max_rows);
        assert_eq!(cfg.segment_index_min_rows, default.segment_index_min_rows);
        assert_eq!(cfg.fsync, default.fsync);
        assert_eq!(cfg.auto_compact, default.auto_compact);
        assert_eq!(cfg.lock_ttl, default.lock_ttl);
        assert_eq!(cfg.max_vector_bytes, default.max_vector_bytes);
    }

    #[test]
    fn quantization_kinds_and_rescore_override() {
        // int8 with its default rescore.
        let q = serve_store(&["--quantization", "int8"])
            .quant_config()
            .expect("quant enabled");
        assert_eq!(q.kind, QuantKind::Int8);
        assert_eq!(q.rescore, Quantization::int8().rescore);

        // binary keeps its own (coarser → higher) default rescore, not int8's.
        let q = serve_store(&["--quantization", "binary"])
            .quant_config()
            .expect("quant enabled");
        assert_eq!(q.kind, QuantKind::Binary);
        assert_eq!(q.rescore, Quantization::binary().rescore);

        // --quant-rescore overrides the per-kind default.
        let q = serve_store(&["--quantization", "binary", "--quant-rescore", "3"])
            .quant_config()
            .expect("quant enabled");
        assert_eq!(q.kind, QuantKind::Binary);
        assert_eq!(q.rescore, 3);

        // No --quantization: exact-only search.
        assert!(serve_store(&[]).quant_config().is_none());
    }

    /// `--wait-for-lease` is what turns a losing writer into a standby, so the three
    /// forms must map exactly: absent keeps the historical fail-fast behaviour, bare waits
    /// forever, and a number bounds the wait.
    #[test]
    fn wait_for_lease_forms() {
        assert_eq!(
            serve_store(&[])
                .config(OpenMode::ReadWrite)
                .unwrap()
                .lease_wait,
            LeaseWait::Fail,
            "absent must not change behaviour for existing users"
        );
        assert_eq!(
            serve_store(&["--wait-for-lease"])
                .config(OpenMode::ReadWrite)
                .unwrap()
                .lease_wait,
            LeaseWait::Forever,
        );
        assert_eq!(
            serve_store(&["--wait-for-lease", "45"])
                .config(OpenMode::ReadWrite)
                .unwrap()
                .lease_wait,
            LeaseWait::Timeout(std::time::Duration::from_secs(45)),
        );

        // A non-numeric value is a clear error, not a silent fall-back to waiting forever.
        let err = serve_store(&["--wait-for-lease", "soon"])
            .config(OpenMode::ReadWrite)
            .expect_err("non-numeric value must be rejected")
            .to_string();
        assert!(err.contains("--wait-for-lease"), "unhelpful error: {err}");
    }

    #[test]
    fn auto_compact_ratio_and_disable_conflict() {
        let cfg = serve_store(&["--auto-compact", "0.25"])
            .config(OpenMode::ReadWrite)
            .expect("config");
        assert_eq!(cfg.auto_compact, Some(0.25));

        // Setting a ratio and disabling at once is contradictory, so clap refuses it.
        assert!(
            Cli::try_parse_from([
                "nidus",
                "serve",
                "--dir",
                "/tmp/s",
                "--dim",
                "3",
                "--auto-compact",
                "0.25",
                "--no-auto-compact",
            ])
            .is_err()
        );
    }

    /// A `StoreArgs` with the given persistence/memory and everything else defaulted —
    /// keeps the backend-predicate tests below readable.
    fn store_args(persistence: Option<&str>, memory: Option<&str>) -> StoreArgs {
        StoreArgs {
            dir: PathBuf::from("/tmp/s"),
            dim: Some(8),
            persistence: persistence.map(str::to_string),
            memory: memory.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn object_store_and_shared_memory_predicates() {
        // Object-store persistence: the three accepted schemes (case-insensitive), and
        // not a local path / file:// URL.
        assert!(store_args(Some("s3://bucket/store"), None).is_object_store());
        assert!(store_args(Some("gs://bucket/store"), None).is_object_store());
        assert!(store_args(Some("GCS://Bucket/Store"), None).is_object_store());
        assert!(!store_args(Some("file:///data"), None).is_object_store());
        assert!(!store_args(Some("/data"), None).is_object_store());
        assert!(!store_args(None, None).is_object_store());

        // Shared memory: the Redis family, and not local / process RAM.
        assert!(store_args(None, Some("redis://cache:6379")).is_shared_memory());
        assert!(store_args(None, Some("rediss://cache:6379")).is_shared_memory());
        assert!(store_args(None, Some("valkey://cache:6379")).is_shared_memory());
        assert!(store_args(None, Some("dragonfly://cache:6379")).is_shared_memory());
        assert!(!store_args(None, Some("local")).is_shared_memory());
        assert!(!store_args(None, None).is_shared_memory());
    }

    /// `--require-remote` args that never get as far as binding: the check under test
    /// fails first. Everything but `require_remote` is therefore a placeholder.
    fn require_remote_args() -> ServeArgs {
        ServeArgs {
            addr: "x".into(),
            token: None,
            max_body_bytes: 1,
            max_concurrent_requests: 0,
            read_timeout: 30,
            write_timeout: 600,
            body_idle_timeout: 15,
            refresh_interval: None,
            require_remote: true,
        }
    }

    #[test]
    fn serve_require_remote_rejects_local_backends() {
        // Local-file persistence (the default) is refused under --require-remote.
        let err = serve(
            require_remote_args(),
            store_args(None, Some("redis://c")),
            #[cfg(feature = "memory")]
            IngestArgs::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--persistence must be an object store"),
            "{err}"
        );

        // Object store but process-RAM memory is refused too.
        let err = serve(
            require_remote_args(),
            store_args(Some("s3://b/s"), None),
            #[cfg(feature = "memory")]
            IngestArgs::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("--memory must be a shared"), "{err}");
    }

    /// `0` is the documented "no deadline" escape hatch on both timeout flags.
    #[test]
    fn zero_means_no_deadline() {
        assert_eq!(timeout_secs(0), None);
        assert_eq!(timeout_secs(30), Some(std::time::Duration::from_secs(30)));
    }

    #[test]
    fn resolve_requires_dim_when_no_store_yet() {
        let dir = tempfile::tempdir().unwrap();
        let args = StoreArgs {
            dir: dir.path().join("does-not-exist-yet"),
            ..Default::default()
        };
        let err = args.resolve().unwrap_err().to_string();
        assert!(err.contains("--dim"), "unexpected error: {err}");
    }
}
