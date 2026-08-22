---
title: Remember & recall
description: Store natural-language text and get the relevant pieces back. nidus embeds the text for you (optionally summarizing first) with the provider of your choice, then answers queries by cosine similarity.
---

nidus is a vector store, so it works in vectors: you hand it a `Vec<f32>` and it
answers nearest-neighbour queries. The **memory** layer adds the step before and
after that: **text in, relevant text out**. You `remember` a piece of text and
nidus embeds it for you (optionally summarizing it first) with the provider you
choose; you `recall` with a natural-language query and get the closest pieces
back, ranked by cosine similarity.

It sits on top of the same store: a thin, async convenience layer over the
synchronous core. The raw `Vec<f32>` API underneath never changes, so if you
already have your own embeddings you can skip this entirely (see [the escape
hatch](#the-escape-hatch-bring-your-own-vector) below).

## What you have to provide

**nidus does not embed text itself, and ships no built-in model.** It stores vectors
and answers queries over them; turning text into a vector is delegated to a provider
you choose. So before `remember`/`recall` will work, you need exactly one of:

- **A hosted provider and its API key**: Voyage, OpenAI, Cohere, Gemini, Mistral,
  Jina, or any OpenAI-compatible endpoint. Enable its `embed-<name>` feature and set
  the key. See [the provider table](#embedding-providers-and-their-default-models).
- **A local daemon**: [Ollama](#a-fully-local-keyless-setup-with-ollama). No API key
  and nothing leaves your machine, at the cost of running a daemon.
- **Your own vectors**: skip the memory layer and use the raw `Vec<f32>` API. See
  [the escape hatch](#the-escape-hatch-bring-your-own-vector).

There is deliberately no bundled local embedder. A model table worth using is 8–32 MB,
which every `cargo add nidus` would carry whether or not it embeds anything, and static
embeddings score meaningfully below a real model, so it would be both the heaviest and
the weakest option on the list. Ollama already covers the fully-local case. The
reasoning is recorded in `SPEC.md` §9.1.

## Turn it on

The memory layer and its provider adapters are **off by default**: the plain
`cargo add nidus` stays a pure, dependency-lean sync vector store. Opt in with
Cargo features: `memory` for the `remember`/`recall` surface, one `embed-<name>`
feature per embedding provider you want, and (optionally) a `summarize-<name>`
feature for the summarize-then-embed mode.

```toml
# Cargo.toml: the all-in-one memory with OpenAI embeddings and
# Anthropic summarization:
[dependencies]
nidus = { version = "0.55", features = ["memory", "embed-openai", "summarize-anthropic"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Enable only the providers you use: each `embed-<name>` compiles just that one
adapter. The umbrella features `embed-all` and `summarize-all` pull in every
shipped adapter at once.

:::note
Enabling `embed`/`summarize` adds `reqwest` (with rustls TLS, reusing the `ring`
already present) plus `tokio` and `serde_json`. There is no new C toolchain and
no bundled OpenSSL, so the build stays fast. The memory API is async; the store
underneath is still synchronous.
:::

## Pick a provider

Both the embedder and the summarizer are chosen **at runtime** through a closed
enum: no `Box<dyn>`, no dynamic dispatch cost. Build one from a provider and a
config:

```rust
use nidus::embed::{AnyEmbedder, EmbedConfig, EmbedProvider};

// Leaving the model empty uses the provider default (below).
let embedder = AnyEmbedder::build(
    EmbedProvider::OpenAi,
    EmbedConfig::new("").api_key(std::env::var("OPENAI_API_KEY")?),
).await?;
# anyhow::Ok(())
```

### Embedding providers and their default models

| Provider | Feature | Enum | Default model |
|---|---|---|---|
| Voyage | `embed-voyage` | `EmbedProvider::Voyage` | `voyage-4` |
| OpenAI | `embed-openai` | `EmbedProvider::OpenAi` | `text-embedding-3-small` |
| Ollama | `embed-ollama` | `EmbedProvider::Ollama` | `nomic-embed-text` |
| Cohere | `embed-cohere` | `EmbedProvider::Cohere` | `embed-english-v3.0` |
| Gemini | `embed-gemini` | `EmbedProvider::Gemini` | `text-embedding-004` |
| Mistral | `embed-mistral` | `EmbedProvider::Mistral` | `mistral-embed` |
| Jina | `embed-jina` | `EmbedProvider::Jina` | `jina-embeddings-v3` |
| OpenAI-compatible | `embed-openai-compat` | `EmbedProvider::OpenAiCompat` | *(none: set a model)* |

Voyage's current generation is the **Voyage 4** family (`voyage-4-large`,
`voyage-4`, `voyage-4-lite`, `voyage-code-4`, and the open-weight
`voyage-4-nano`): 1024 dimensions natively, a 32K-token context, and Matryoshka
truncation to 256, 512, or 2048 via `EmbedConfig::output_dimension` (the
`--embed-dimension` flag). Since the store pins its dimension at creation,
choose the width before the first upsert.

:::caution[The Voyage default moved to `voyage-4` in 0.77.0]
It was `voyage-3` through 0.76.x. Both are 1024 dimensions natively, so no store
needs re-creating and nothing is re-embedded behind your back. What does change:
a collection written through the memory API carries a pinned `voyage/voyage-3`
identity, and a recall under the new default refuses with an embedder-mismatch
error rather than returning cross-space results. Pass `--embed-model voyage-3`
(or `EmbedConfig::new("voyage-3")`) to stay exactly where you were.
:::

```rust
let cfg = EmbedConfig::new("voyage-4-large")
    .api_key(std::env::var("VOYAGE_API_KEY")?)
    .output_dimension(256);
```

**OpenAI** honours `output_dimension` too, on `text-embedding-3-small` and
`text-embedding-3-large`. Those take any width from 1 up to the model's native
size (1536 and 3072 respectively) rather than a fixed set, so
`output_dimension(768)` is valid against `text-embedding-3-large` where the
Voyage equivalent is not.

Asking for a width a model cannot honour is an error, not a silent fallback:
fixed-width Voyage models, `text-embedding-ada-002`, and every provider that
does not advertise the capability reject `output_dimension` at construction
rather than pinning the store to a dimension the API will not fill.

The **OpenAI-compatible** adapter is the catch-all: point its `base_url` at any
service that speaks the standard `/v1/embeddings` shape: Azure OpenAI, Together,
Fireworks, vLLM, LiteLLM, DeepInfra, and so on. It has no default model, so pass
one explicitly.

### Summarization providers and their default models

| Provider | Feature | Enum | Default model |
|---|---|---|---|
| Anthropic | `summarize-anthropic` | `SummarizeProvider::Anthropic` | `claude-haiku-4-5-20251001` |
| OpenAI | `summarize-openai` | `SummarizeProvider::OpenAi` | `gpt-4o-mini` |

The OpenAI summarizer speaks the chat-completions shape, so its `base_url` also
reaches Azure, LiteLLM, vLLM, and Ollama's `/v1` endpoint.

## Configure the connection

Both `EmbedConfig` and `SummarizeConfig` are fluent builders over the same knobs:

```rust
use nidus::embed::EmbedConfig;

let config = EmbedConfig::new("text-embedding-3-large")   // model (empty = default)
    .api_key("sk-...")                                    // bearer token
    .base_url("https://my-gateway.example.com")           // route via a gateway
    .header("x-org-id", "acme");                          // extra header per request
# let _ = config;
```

- **`api_key`**: the bearer token. Keyless providers (Ollama, and some
  OpenAI-compatible gateways) leave it empty.
- **`base_url`**: override the provider's default endpoint. This is how you
  route through a self-hosted proxy or gateway, reach an OpenAI-compatible
  service, or point at a mock in tests.
- **`header(name, value)`**: extra headers applied to every request, for
  gateway auth or tenant routing. Chain it more than once.

### A fully local, keyless setup with Ollama

Ollama needs no API key. Leave `api_key` empty and set `base_url` to your Ollama
host (it defaults to `http://localhost:11434`):

```rust
use nidus::embed::{AnyEmbedder, EmbedConfig, EmbedProvider};

let embedder = AnyEmbedder::build(
    EmbedProvider::Ollama,
    EmbedConfig::new("nomic-embed-text")
        .base_url("http://localhost:11434"),
).await?;
# anyhow::Ok(())
```

Ollama (and the OpenAI-compatible adapter) probe their embedding dimension with
a live call while building, so `build` is `async` and will surface a clear error
if the host is unreachable.

## Remember and recall

Wrap a store and an embedder in a `Memory`, then `remember` text and `recall` it:

```rust
use std::collections::BTreeMap;
use nidus::{Config, Memory, Nidus, RecallOpts, RememberMode};
use nidus::embed::{AnyEmbedder, EmbedConfig, EmbedProvider, Embedder};

# async fn run() -> anyhow::Result<()> {
let embedder = AnyEmbedder::build(
    EmbedProvider::OpenAi,
    EmbedConfig::new("").api_key(std::env::var("OPENAI_API_KEY")?),
).await?;

// Open the store to match the embedder's dimension (see pinning, below).
let db = Nidus::open(Config::new("./store", embedder.dimension()))?;
let mut memory = Memory::new(db, embedder);

// Text in.
memory.remember(
    "notes", "login",
    "Users authenticate with a bearer token issued at login.",
    BTreeMap::new(),
    RememberMode::Raw,
).await?;

// Relevant text out.
let hits = memory.recall("notes", "how do users sign in?", &RecallOpts {
    top_k: 3,
    ..Default::default()
}).await?;
for h in &hits {
    println!("{:.3}  [{}] {}", h.score, h.collection, h.id);
}
# anyhow::Ok(())
# }
```

`remember` creates the collection if it does not exist, embeds the text, and
upserts a record under your `id` with your `attrs`. `recall` embeds the query
(using the provider's query side, where it distinguishes document from query
vectors) and runs a vector search. `RecallOpts` maps straight onto the store's
search options: `top_k`, a `min_score` floor, and an optional metadata
[`Filter`](/guides/search/). Two fields are sentinels: `top_k: 0` (the default)
means 10, and `min_score: 0.0` means no floor.

`recall` also filters expiry automatically: an entry whose `nidus.expires_at` is
in the past is invisible to it (and to the server's memory reads), while the raw
`search`/`list` store API still returns it. Only the HTTP and MCP write paths set
that attr today; see the parity note below.

:::note
The in-process `Memory::remember` is leaner than the server's write path: it does
not take `ttl_seconds`/`dedupe_threshold` options and it does not stamp the
`nidus.text` attr the auto-provisioned full-text schema indexes. If you need TTL,
dedupe, or BM25 over remembered text from Rust today, write through a
`nidus serve` instance, or track
[#131](https://github.com/duckedup/nidus/issues/131) and the parity work.
:::

### Raw vs. Summarize

`RememberMode` chooses what actually gets embedded:

- **`Raw`** embeds the text exactly as given. Best when the text is already the
  right size and shape for retrieval.
- **`Summarize`** first runs the text through a summarizer, embeds the
  **summary**, and stores *both* the summary and the original text alongside
  the record (under the `nidus.summary` and `nidus.text` attrs) so a hit stays
  explainable back to what you ingested. Use it for long or noisy inputs where a
  dense summary is a better embedding target than the raw text.

Summarize mode needs a summarizer attached:

```rust
use nidus::summarize::{AnySummarizer, SummarizeConfig, SummarizeProvider};
use nidus::{Memory, RememberMode};
# use std::collections::BTreeMap;

# async fn run(mut memory: Memory) -> anyhow::Result<()> {
let summarizer = AnySummarizer::build(
    SummarizeProvider::Anthropic,
    SummarizeConfig::new("").api_key(std::env::var("ANTHROPIC_API_KEY")?),
).await?;

let mut memory = memory.with_summarizer(summarizer);
memory.remember(
    "notes", "auth-history",
    "In 2019 the team migrated the auth service off session cookies onto \
     short-lived bearer tokens, cutting a class of CSRF bugs.",
    BTreeMap::new(),
    RememberMode::Summarize,
).await?;
# anyhow::Ok(())
# }
```

Requesting `Summarize` without a summarizer attached is an error: the message
tells you to add one with `with_summarizer`.

## Dimension and embedder-identity pinning

Vectors from different models live in incomparable spaces, so mixing them in one
collection would make cosine ranking meaningless. The memory layer guards against
that on two axes:

- **Dimension.** The embedding dimension is pinned into the store at creation.
  If the embedder's dimension does not match the store's, the first `remember`
  fails with an error naming both. Opening the store to
  `embedder.dimension()` (as above) keeps them in lockstep.
- **Embedder identity.** On the first write into a collection, nidus records the
  embedder's `"provider/model"` identity in the collection metadata (under
  `nidus.embedder`). Every later write re-checks it and **refuses** if a
  different embedder is now in play, catching an accidental cross-model write
  before it corrupts a collection's ranking. To switch models, use a separate
  collection.

A collection written straight through `upsert`, by nidus or by any other tool,
carries no `nidus.embedder` at all, so neither check has anything to compare: a
recall with a mismatched embedder returns plausible-looking scores from two
different spaces. That case logs a warning (once per collection and embedder) and
otherwise proceeds, since refusing it would break every store built on raw
upserts. Set `Config::strict_embedder_identity` (`--strict-embedder-identity`,
`NIDUS_STRICT_EMBEDDER_IDENTITY`) to refuse instead: an unpinned collection then
errors on recall, and a `remember` into one that already holds rows errors rather
than stamping nidus's own identity onto vectors it did not produce.

## The escape hatch: bring your own vector

`Memory` is strictly additive. The underlying `Nidus` store (with its raw,
synchronous, dependency-free `Vec<f32>` API) is always right there:

```rust
use nidus::{Memory, Nidus};
# fn f(memory: Memory) {
let db: &Nidus = memory.db();          // borrow it
# }
# fn g(mut memory: Memory) {
let db: &mut Nidus = memory.db_mut();  // mutably borrow it
# }
# fn h(memory: Memory) {
let db: Nidus = memory.into_inner();   // unwrap back to the bare store
# }
```

So if you already produce your own embeddings, or want a model nidus ships no
adapter for, keep using `Nidus` directly: upsert your own vectors and search
with your own query vectors, with zero async and zero provider dependencies.
Embedding is a property of *this handle*, not of the on-disk store: one process
can wrap a store with an OpenAI embedder while another opens the same directory
raw.

## Offline and failure behaviour

Nothing about this layer assumes a provider is reachable. Building an embedder
that requires a key without one, or pointing at a host that is down, returns a
**typed, descriptive error** (`EmbedError` / `SummarizeError`, each with
`Config`, `Backend`, `Api { status, body }`, and `Decode` variants), not a panic.
Transient failures (HTTP 429 and 5xx) are retried with backoff before the error
surfaces. Match on the variant to decide whether to fall back, retry, or fail.

## Parity across the surfaces

`remember`/`recall` exist on four surfaces: the Rust `Memory` API you have been reading
about, the HTTP `/remember` + `/recall` routes, the MCP memory tools, and the `nidus
remember`/`nidus recall` CLI subcommands. They agree on almost everything, but not
everything, and the differences are exactly the places callers get burned. Verify
against source rather than assuming one surface behaves like another.

| | Rust `Memory` | HTTP | MCP | CLI |
|---|---|---|---|---|
| `ttl_seconds` | `RememberOpts.ttl_seconds` | `RememberRequest.ttl_seconds` | `remember` arg `ttl_seconds` | `--ttl-seconds` |
| `dedupe_threshold` | `RememberOpts.dedupe_threshold` | `RememberRequest.dedupe_threshold` | `remember` arg `dedupe_threshold` | `--dedupe-threshold` |
| `nidus.text` | stamped on every write | stamped on every write | stamped on every write | stamped on every write |
| `nidus.source` | never stamped (legacy) | never stamped (legacy) | never stamped (legacy) | never stamped (legacy) |
| Derived ids | no, `id` is required | no, `id` is required | yes, from content when omitted | yes, from content when omitted |
| TTL-on-read | only `Memory::recall` | only `/recall` | `recall`, `get`, `browse`, `text_search`, `hybrid_search` | only `recall` |

A couple of things the table cannot show:

- `ttl_seconds` counts from the moment of the write, not from whenever a caller
  happens to read the entry back.
- `dedupe_threshold` is a cosine floor: a write that lands within it of an existing
  entry updates that entry in place instead of inserting a competitor, so the id you
  get back may not be the id you sent.

### Derived ids are not universal

`Memory::remember` and the HTTP `RememberRequest` both take `id` as a required
string; there is no server-side derivation, so an omitted id is a caller error on
either surface. The MCP `remember` tool and the `nidus remember` CLI subcommand
both derive a stable id from the text when the caller omits one, using the same
`DefaultHasher`-based scheme, so the same fact written from either entry point lands
on the same record instead of accumulating duplicates. This is why the CLI gets its
own column above rather than being folded into "Rust": it behaves like MCP here, not
like the library it is built on.

### `nidus.text` is always stamped; `nidus.source` never is

Every surface stamps `nidus.text` with the raw remembered text on every write,
regardless of `Raw` or `Summarize` mode. `nidus.source` is not written by any
surface. It is a legacy attr, kept only so records written before the fix in
nidus-133 remain readable; do not expect a current write to produce it.

### TTL-on-read is not store-wide

The not-expired filter is only AND-ed into the memory-shaped read paths: `/recall`
over HTTP, `Memory::recall` in Rust and in the `nidus recall` CLI subcommand built
on it, and the `recall`, `get`, `browse`, `text_search`, and `hybrid_search` tools
over MCP. It is never applied to the generic vector-search, list, full-text, or
hybrid-search routes, over either HTTP (`/search`, `/list`, `/text-search`,
`/hybrid-search`) or the CLI (`search`, `list`, `text-search`, `hybrid-search`). A
record with a past `nidus.expires_at` that has not yet been swept is invisible to
`recall` but still returned by a plain search against the same collection.

## Run the example

The repository ships a runnable end-to-end example. It is offline-safe: with no
provider configured it still runs the bring-your-own-vector section and prints a
clear message for the provider-backed part.

```bash
cargo run --example memory --features memory,embed-all,summarize-all
```

Point it at a provider with environment variables. See the comments at the top
of `examples/memory.rs`.

## Where to next

- [MCP (agent memory)](/guides/mcp/): expose this layer to an agent over the
  Model Context Protocol.
- [Search & filters](/guides/search/): what `recall` runs underneath.
- [Embedding in a host app](/guides/integrating/): mapping your document type
  onto a `Record`.
- [API reference](/reference/api/): the full surface.
