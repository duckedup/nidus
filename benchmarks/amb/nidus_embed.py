"""Client-side batch embedding for the `nidus` AMB provider.

This file exists only because nidus's text-ingest route (`POST /remember`) takes one
document per request and embeds one text per provider call. A 24k-document dataset would
be 24k HTTP round-trips and 24k single-text embed calls, which measures the missing batch
route rather than the retrieval quality. So the `nidus` variant embeds in batches here
and writes through the batch `upsert` route instead. Delete this file the day nidus grows
a batch remember (nidus-amb-2) — `nidus-memory` is already the shape it should have.

Deliberately urllib-only: the AMB venv is heavy enough, and this mirrors the Python SDK's
own zero-dependency posture.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable

_RETRY_STATUS = {429, 500, 502, 503, 529}


def _post(url: str, body: dict, headers: dict, attempts: int = 5) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            if err.code in _RETRY_STATUS and attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{url} -> {err.code}: {err.read()[:400].decode(errors='replace')}") from err
        except urllib.error.URLError:
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def make_embedder(provider: str, model: str, key: str | None, dim: int | None) -> Callable[[list[str], bool], list[list[float]]]:
    """Return `f(texts, is_query) -> vectors` for one provider."""
    if provider == "voyage":
        def embed(texts, is_query=False):
            body = {
                "input": texts,
                "model": model,
                "input_type": "query" if is_query else "document",
            }
            if dim:
                body["output_dimension"] = dim
            out = _post("https://api.voyageai.com/v1/embeddings", body, {"authorization": f"Bearer {key}"})
            return [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]
        return embed

    if provider == "openai":
        def embed(texts, is_query=False):
            body = {"input": texts, "model": model}
            if dim:
                body["dimensions"] = dim
            out = _post("https://api.openai.com/v1/embeddings", body, {"authorization": f"Bearer {key}"})
            return [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]
        return embed

    if provider == "gemini":
        def embed(texts, is_query=False):
            task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
            body = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task,
                        **({"outputDimensionality": dim} if dim else {}),
                    }
                    for t in texts
                ]
            }
            out = _post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
                body,
                {"x-goog-api-key": key or ""},
            )
            return [e["values"] for e in out["embeddings"]]
        return embed

    if provider == "ollama":
        import os

        base = os.getenv("NIDUS_EMBED_BASE_URL", "http://127.0.0.1:11434")

        def embed(texts, is_query=False):
            out = _post(f"{base}/api/embed", {"model": model, "input": texts}, {})
            return out["embeddings"]
        return embed

    raise RuntimeError(
        f"no client-side batch embedder for '{provider}'. "
        "Use the nidus-memory variant (server-side embed), or add one here."
    )
