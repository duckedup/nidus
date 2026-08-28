"""nidus memory providers for the Agent Memory Benchmark (AMB).

Three variants, each a separate entry in AMB's provider registry, because they
exercise different halves of nidus:

  nidus-fts     dimension-0 store, BM25 `text_search` only. No API key, no network.
                The honest floor, directly comparable to AMB's `bm25` baseline.
  nidus         dense + BM25 hybrid via `hybrid_search` RRF fusion. Vectors are
                embedded client-side in batches and written with the batch `upsert`
                route, because nidus has no batch text-ingest route yet (nidus-amb-2).
  nidus-memory  the all-in-one memory layer end to end: `POST /remember` per document,
                `POST /recall` per query, embedding server-side. One HTTP call, no glue
                code, and the slowest ingest of the three by two orders of magnitude.

Every variant runs a real `nidus serve` subprocess and drives it over the published
Python SDK, so a number here is a number the shipped binary produced.

Configuration is environment-only (AMB providers take no constructor args):

  NIDUS_BIN               path to the nidus binary (default: `nidus` on PATH)
  NIDUS_EMBED_PROVIDER    voyage | openai | gemini | ollama | cohere | ... (default: voyage)
  NIDUS_EMBED_MODEL       provider default when unset
  NIDUS_EMBED_API_KEY     falls back to VOYAGE_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
  NIDUS_EMBED_DIMENSION   narrower embedding (Voyage Matryoshka / OpenAI v3)
  NIDUS_RERANK_PROVIDER   voyage | cohere — enables the cross-encoder stage
  NIDUS_RERANK_MODEL, NIDUS_RERANK_API_KEY
  NIDUS_ANN               hnsw | ivf — opt into approximate search
  NIDUS_TOP_K             retrieval depth (default 50, matching the qdrant baseline)
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

from nidus import NidusClient, f

from ..models import Document
from ..utils import chunk_text
from .base import MemoryProvider

_COLLECTION = "bench"
_TEXT = "nidus.text"
_PARENT = "nidus.parent_id"
_INDEX = "nidus.chunk_index"

# Native widths, so the store can be created before the first embed call.
_NATIVE_DIMS = {
    ("voyage", "voyage-3.5"): 1024,
    ("voyage", "voyage-3.5-lite"): 1024,
    ("voyage", "voyage-3-large"): 1024,
    ("openai", "text-embedding-3-small"): 1536,
    ("openai", "text-embedding-3-large"): 3072,
    ("gemini", "gemini-embedding-001"): 3072,
    ("cohere", "embed-v4.0"): 1536,
    ("jina", "jina-embeddings-v3"): 1024,
    ("mistral", "mistral-embed"): 1024,
}
_DEFAULT_MODEL = {
    "voyage": "voyage-3.5",
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "cohere": "embed-v4.0",
    "jina": "jina-embeddings-v3",
    "mistral": "mistral-embed",
    "ollama": "nomic-embed-text",
}
_KEY_ENV = {
    "voyage": "VOYAGE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "jina": "JINA_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _embed_config() -> tuple[str, str, str | None, int | None]:
    provider = os.getenv("NIDUS_EMBED_PROVIDER", "voyage")
    model = os.getenv("NIDUS_EMBED_MODEL") or _DEFAULT_MODEL.get(provider, "")
    key = os.getenv("NIDUS_EMBED_API_KEY") or os.getenv(_KEY_ENV.get(provider, ""), "") or None
    dim_env = os.getenv("NIDUS_EMBED_DIMENSION")
    dim = int(dim_env) if dim_env else _NATIVE_DIMS.get((provider, model))
    return provider, model, key, dim


class _NidusBase(MemoryProvider):
    """Server lifecycle and chunking shared by all three variants."""

    kind = "local"
    provider = "nidus"
    link = "https://nidus.duckedup.org"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=nidus.duckedup.org"
    concurrency = 8

    #: dimension-0 store (BM25 only) unless the subclass wants vectors.
    needs_embedder = True

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._db: NidusClient | None = None
        self._port: int | None = None
        self._log = None
        self.top_k = int(os.getenv("NIDUS_TOP_K", "50"))

    # ── lifecycle ────────────────────────────────────────────────────────────

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        store = store_dir / "nidus"
        if reset and store.exists():
            import shutil

            shutil.rmtree(store)
        store.mkdir(parents=True, exist_ok=True)

        provider, model, key, dim = _embed_config()
        if self.needs_embedder and dim is None:
            raise RuntimeError(
                f"unknown native dimension for {provider}/{model}; set NIDUS_EMBED_DIMENSION"
            )

        self._port = _free_port()
        cmd = [
            os.getenv("NIDUS_BIN", "nidus"),
            "serve",
            "--dir", str(store),
            "--dim", str(dim if self.needs_embedder else 0),
            "--addr", f"127.0.0.1:{self._port}",
            "--fsync", "on-flush",       # benchmark ingest; crash safety is tested elsewhere
            "--read-timeout", "120",
            "--write-timeout", "1200",
        ]
        if self.needs_embedder:
            cmd += ["--embed-provider", provider]
            if model:
                cmd += ["--embed-model", model]
            if key:
                cmd += ["--embed-api-key", key]
            if os.getenv("NIDUS_EMBED_DIMENSION"):
                cmd += ["--embed-dimension", os.environ["NIDUS_EMBED_DIMENSION"]]
        if os.getenv("NIDUS_RERANK_PROVIDER"):
            cmd += ["--rerank-provider", os.environ["NIDUS_RERANK_PROVIDER"]]
            if os.getenv("NIDUS_RERANK_MODEL"):
                cmd += ["--rerank-model", os.environ["NIDUS_RERANK_MODEL"]]
            rk = os.getenv("NIDUS_RERANK_API_KEY") or key
            if rk:
                cmd += ["--rerank-api-key", rk]
        if os.getenv("NIDUS_ANN"):
            cmd += ["--ann", os.environ["NIDUS_ANN"]]

        self._log = open(store_dir / "nidus-serve.log", "w")
        self._proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self._db = NidusClient(f"http://127.0.0.1:{self._port}", timeout=1200.0)
        self._await_health()

        if _COLLECTION not in self._db.collections():
            self._db.create_collection(_COLLECTION)
        self._db.set_fts_schema(_COLLECTION, [_TEXT])
        self._db.set_filter_index(_COLLECTION, ["user_id"])

    def _await_health(self, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(f"nidus serve exited with {self._proc.returncode}")
            try:
                # `/health` answers ok as soon as the socket is up, *before* the store is
                # open — poll `/ready`, which is the gate that actually means "will serve
                # a request". Getting this wrong yields "store is not open yet: this
                # instance is waiting for the writer handle" on the first upsert.
                if getattr(self._db.ready(), "ready", False):
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise RuntimeError("nidus serve did not become ready in 60s")

    def cleanup(self) -> None:
        if self._db is not None:
            try:
                self._db.flush()
            except Exception:
                pass
            self._db.close()
            self._db = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None

    # ── shared helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _chunks(documents: list[Document]):
        """(chunk_id, text, doc_id, chunk_index, user_id) for every 512-token window.

        Uses AMB's own `chunk_text`, so the chunk boundaries match the `bm25` and
        `qdrant` baselines exactly and the comparison is about retrieval, not chunking.
        """
        for doc in documents:
            for i, chunk in enumerate(chunk_text(doc.content)):
                yield f"{doc.id}#{i}", chunk, doc.id, i, doc.user_id

    @staticmethod
    def _to_documents(hits) -> tuple[list[Document], dict]:
        docs, raw = [], []
        for h in hits:
            attrs = h.attrs or {}
            text = h.context or attrs.get(_TEXT, "")
            docs.append(
                Document(
                    id=attrs.get(_PARENT, h.id),
                    content=text,
                    user_id=attrs.get("user_id"),
                )
            )
            raw.append({"id": h.id, "score": h.score, "parent": attrs.get(_PARENT)})
        return docs, {"results": raw}

    def _user_filter(self, user_id: str | None):
        return [f.eq("user_id", user_id)] if user_id is not None else None


class NidusFtsProvider(_NidusBase):
    """BM25 only, dimension-0 store. Zero API keys, zero network, zero embedding cost."""

    name = "nidus-fts"
    variant = "bm25"
    description = (
        "nidus BM25 full-text search over a dimension-0 store (no vectors at all). "
        "Documents chunked into 512-token windows, indexed via `set_fts_schema`, "
        "queried via `POST /text-search`. Runs entirely offline — the honest floor, "
        "directly comparable to AMB's `bm25` baseline."
    )
    needs_embedder = False

    def ingest(self, documents: list[Document]) -> None:
        records = [
            {
                "id": cid,
                "vector": [],
                "attrs": {_TEXT: text, _PARENT: pid, _INDEX: idx, "user_id": uid or ""},
            }
            for cid, text, pid, idx, uid in self._chunks(documents)
        ]
        for i in range(0, len(records), 2000):
            self._db.upsert(_COLLECTION, records[i : i + 2000])

    def retrieve(self, query, k=None, user_id=None, query_timestamp=None):
        hits = self._db.text_search(
            field=_TEXT,
            query=query,
            top_k=k or self.top_k,
            filter=self._user_filter(user_id),
        )
        return self._to_documents(hits)


class NidusProvider(_NidusBase):
    """Dense + BM25 hybrid RRF. Vectors embedded client-side, in batches."""

    name = "nidus"
    variant = "hybrid"
    description = (
        "nidus hybrid search: a dense vector leg and a BM25 leg fused with RRF via "
        "`POST /hybrid-search`. Documents chunked into 512-token windows. Vectors are "
        "embedded client-side in batches and written with the batch `upsert` route — "
        "nidus has no batch text-ingest route, which is the gap `nidus-memory` exposes."
    )

    def __init__(self) -> None:
        super().__init__()
        self._embedder = None

    def _embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if self._embedder is None:
            from .nidus_embed import make_embedder

            self._embedder = make_embedder(*_embed_config())
        return self._embedder(texts, is_query)

    def ingest(self, documents: list[Document]) -> None:
        rows = list(self._chunks(documents))
        for i in range(0, len(rows), 128):
            batch = rows[i : i + 128]
            vectors = self._embed([r[1] for r in batch])
            self._db.upsert(
                _COLLECTION,
                [
                    {
                        "id": cid,
                        "vector": vec,
                        "attrs": {_TEXT: text, _PARENT: pid, _INDEX: idx, "user_id": uid or ""},
                    }
                    for (cid, text, pid, idx, uid), vec in zip(batch, vectors)
                ],
            )

    def retrieve(self, query, k=None, user_id=None, query_timestamp=None):
        k = k or self.top_k
        qvec = self._embed([query], is_query=True)[0]
        kwargs = {}
        if os.getenv("NIDUS_RERANK_PROVIDER"):
            kwargs["rerank"] = {"query": query}
        hits = self._db.hybrid_search(
            vector=qvec,
            field=_TEXT,
            text=query,
            top_k=k,
            candidates=k * 4,
            filter=self._user_filter(user_id),
            **kwargs,
        )
        return self._to_documents(hits)


class NidusServerMemoryProvider(_NidusBase):
    """The all-in-one memory layer: /remember and /recall, embedding server-side."""

    name = "nidus-memory"
    variant = "memory"
    description = (
        "nidus's all-in-one memory layer end to end: `POST /remember` per document "
        "(server-side embed) and `POST /recall` per query, with `rollup` reading the "
        "chunked corpus back as documents. No client-side embedding and no glue code — "
        "and one HTTP request plus one single-text embed call per document, which is "
        "why its ingest is the slowest of the three."
    )
    concurrency = 4

    def ingest(self, documents: list[Document]) -> None:
        for cid, text, pid, idx, uid in self._chunks(documents):
            self._db.remember(
                _COLLECTION,
                cid,
                text,
                attrs={_PARENT: pid, _INDEX: idx, "user_id": uid or ""},
            )

    def retrieve(self, query, k=None, user_id=None, query_timestamp=None):
        hits = self._db.recall(
            _COLLECTION,
            query,
            top_k=k or self.top_k,
            filter=self._user_filter(user_id),
            rollup={"per_parent": 1, "neighbours": 1},
        )
        return self._to_documents(hits)
