---
title: Remember & recall
description: Store natural-language text and get the relevant pieces back — nidus embeds the text for you (optionally summarizing first) with the provider of your choice, then answers queries by cosine similarity.
---

nidus is a vector store, so it works in vectors: you hand it a `Vec<f32>` and it
answers nearest-neighbour queries. The **memory** layer adds the step before and
after that — **text in, relevant text out**. You `remember` a piece of text and
nidus embeds it for you (optionally summarizing it first) with the provider you
choose; you `recall` with a natural-language query and get the closest pieces
back, ranked by cosine similarity.

It sits on top of the same store — a thin, async convenience layer over the
synchronous core. The raw `Vec<f32>` API underneath never changes, so if you
already have your own embeddings you can skip this entirely (see [the escape
hatch](#the-escape-hatch-bring-your-own-vector) below).

## Turn it on

The memory layer and its provider adapters are **off by default** — the plain
`cargo add nidus` stays a pure, dependency-lean sync vector store. Opt in with
Cargo features: `memory` for the `remember`/`recall` surface, one `embed-<name>`
feature per embedding provider you want, and (optionally) a `summarize-<name>`
feature for the summarize-then-embed mode.

```toml
# Cargo.toml — the all-in-one memory with OpenAI embeddings and
# Anthropic summarization:
[dependencies]
nidus = { version = "0.28", features = ["memory", "embed-openai", "summarize-anthropic"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Enable only the providers you use — each `embed-<name>` compiles just that one
adapter. The umbrella features `embed-all` and `summarize-all` pull in every
shipped adapter at once.

:::note
Enabling `embed`/`summarize` adds `reqwest` (with rustls TLS, reusing the `ring`
already present) plus `tokio` and `serde_json`. There is no new C toolchain and
no bundled OpenSSL — the build stays fast. The memory API is async; the store
underneath is still synchronous.
:::

## Pick a provider

Both the embedder and the summarizer are chosen **at runtime** through a closed
enum — no `Box<dyn>`, no dynamic dispatch cost. Build one from a provider and a
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
| Voyage | `embed-voyage` | `EmbedProvider::Voyage` | `voyage-3` |
| OpenAI | `embed-openai` | `EmbedProvider::OpenAi` | `text-embedding-3-small` |
| Ollama | `embed-ollama` | `EmbedProvider::Ollama` | `nomic-embed-text` |
| Cohere | `embed-cohere` | `EmbedProvider::Cohere` | `embed-english-v3.0` |
| Gemini | `embed-gemini` | `EmbedProvider::Gemini` | `text-embedding-004` |
| Mistral | `embed-mistral` | `EmbedProvider::Mistral` | `mistral-embed` |
| Jina | `embed-jina` | `EmbedProvider::Jina` | `jina-embeddings-v3` |
| OpenAI-compatible | `embed-openai-compat` | `EmbedProvider::OpenAiCompat` | *(none — set a model)* |

The **OpenAI-compatible** adapter is the catch-all: point its `base_url` at any
service that speaks the standard `/v1/embeddings` shape — Azure OpenAI, Together,
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
    .base_url("https://my-gateway.example.com")           // route through a gateway / proxy
    .header("x-org-id", "acme");                           // extra header on every request
# let _ = config;
```

- **`api_key`** — the bearer token. Keyless providers (Ollama, and some
  OpenAI-compatible gateways) leave it empty.
- **`base_url`** — override the provider's default endpoint. This is how you
  route through a self-hosted proxy or gateway, reach an OpenAI-compatible
  service, or point at a mock in tests.
- **`header(name, value)`** — extra headers applied to every request, for
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
search options — `top_k`, a `min_score` floor, and an optional metadata
[`Filter`](/guides/search/).

### Raw vs. Summarize

`RememberMode` chooses what actually gets embedded:

- **`Raw`** embeds the text exactly as given. Best when the text is already the
  right size and shape for retrieval.
- **`Summarize`** first runs the text through a summarizer, embeds the
  **summary**, and stores *both* the summary and the original source alongside
  the record (under the `nidus.summary` and `nidus.source` attrs) so a hit stays
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

Requesting `Summarize` without a summarizer attached is an error — the message
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
  different embedder is now in play — catching an accidental cross-model write
  before it corrupts a collection's ranking. To switch models, use a separate
  collection.

## The escape hatch: bring your own vector

`Memory` is strictly additive. The underlying `Nidus` store — with its raw,
synchronous, dependency-free `Vec<f32>` API — is always right there:

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
adapter for, keep using `Nidus` directly — upsert your own vectors and search
with your own query vectors, with zero async and zero provider dependencies.
Embedding is a property of *this handle*, not of the on-disk store: one process
can wrap a store with an OpenAI embedder while another opens the same directory
raw.

## Offline and failure behaviour

Nothing about this layer assumes a provider is reachable. Building an embedder
that requires a key without one, or pointing at a host that is down, returns a
**typed, descriptive error** — `EmbedError` / `SummarizeError`, each with
`Config`, `Backend`, `Api { status, body }`, and `Decode` variants — not a panic.
Transient failures (HTTP 429 and 5xx) are retried with backoff before the error
surfaces. Match on the variant to decide whether to fall back, retry, or fail.

## Run the example

The repository ships a runnable end-to-end example. It is offline-safe: with no
provider configured it still runs the bring-your-own-vector section and prints a
clear message for the provider-backed part.

```bash
cargo run --example memory --features memory,embed-all,summarize-all
```

Point it at a provider with environment variables — see the comments at the top
of `examples/memory.rs`.

## Where to next

- [Search & filters](/guides/search/) — what `recall` runs underneath.
- [Embedding in a host app](/guides/integrating/) — mapping your document type
  onto a `Record`.
- [API reference](/reference/api/) — the full surface.
