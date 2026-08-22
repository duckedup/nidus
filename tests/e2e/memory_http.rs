//! The HTTP memory routes (`/remember`, `/recall`) through the real binary (#107).
//! These handlers duplicate the MCP tool logic in a separate code path
//! (`src/server/mod.rs` vs `src/server/mcp/remember.rs`), so the dedupe, recency,
//! TTL, and provisioning behaviours the `mcp/*` suites pin need pinning here too.

#![cfg(feature = "embed-ollama")]

use serde_json::{Value, json};

use crate::harness::RunningServer;
use crate::mcp::support::{DIM, mock_embedder_per_text, per_text_embedder_server, vector_for};

/// `POST /collections/notes/remember`, asserting success and returning the response.
fn remember(server: &RunningServer, args: Value) -> Value {
    let (status, body) = server.post("/collections/notes/remember", &args);
    assert_eq!(status, 200, "remember failed: {body}");
    body
}

/// `POST /collections/notes/recall`, returning the ranked hits.
fn recall(server: &RunningServer, query: &str) -> Vec<Value> {
    let (status, body) = server.post(
        "/collections/notes/recall",
        &json!({"query": query, "top_k": 10}),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    body.as_array().expect("recall returns an array").clone()
}

/// `POST /collections/notes/recall` with a full request body — for cases needing fields
/// beyond `query`/`top_k` (`reinforce`, `extend_ttl_seconds`, a stamping filter, …), so
/// `recall` above stays untouched for the cases that don't.
fn recall_with(server: &RunningServer, body: Value) -> Vec<Value> {
    let (status, body) = server.post("/collections/notes/recall", &body);
    assert_eq!(status, 200, "recall failed: {body}");
    body.as_array().expect("recall returns an array").clone()
}

fn ids(hits: &[Value]) -> Vec<&str> {
    hits.iter().map(|h| h["id"].as_str().unwrap()).collect()
}

/// The epoch-ms behind a `{"DateTime": ms}` attr on a recall hit.
fn stamp(hit: &Value, key: &str) -> i64 {
    hit["attrs"][key]["DateTime"]
        .as_i64()
        .unwrap_or_else(|| panic!("{key} missing or not DateTime: {hit}"))
}

/// The raw attrs of one entry, read back through `/list` (unguarded by the recall-time
/// TTL filter, and the only route that shows a reinforcement stamp regardless of ranking).
fn listed_attrs(server: &RunningServer, id: &str) -> Value {
    let (status, listed) = server.post("/list", &json!({"limit": 1000}));
    assert_eq!(status, 200, "list failed: {listed}");
    listed
        .as_array()
        .expect("list returns an array")
        .iter()
        .find(|h| h["id"] == id)
        .unwrap_or_else(|| panic!("id '{id}' not found in list: {listed}"))["attrs"]
        .clone()
}

/// `nidus.access_count`, as a plain `i64`, or `None` when the key is absent.
fn access_count(attrs: &Value) -> Option<i64> {
    attrs.get("nidus.access_count").map(|v| {
        v["Int"]
            .as_i64()
            .unwrap_or_else(|| panic!("access_count not an Int: {v}"))
    })
}

fn epoch_ms_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before epoch")
        .as_millis() as i64
}

/// remember → recall round-trips with zero setup: the collection and its FTS schema
/// are provisioned on first write, and both recency stamps land as `DateTime`.
#[test]
fn remember_then_recall_round_trips_and_provisions() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"id": "bug", "text": "the ranking bug is in the upsert path"}),
    );
    remember(
        &server,
        json!({"id": "groceries", "text": "a completely unrelated grocery list"}),
    );

    let hits = recall(&server, "the ranking bug is in the upsert path");
    assert_eq!(
        hits[0]["id"], "bug",
        "matching text must rank first: {hits:?}"
    );
    assert!(stamp(&hits[0], "nidus.created_at") > 0);
    assert!(stamp(&hits[0], "nidus.updated_at") > 0);

    // First-write provisioning declared the default FTS schema over `nidus.text`.
    let (status, hits) = server.post(
        "/text-search",
        &json!({"field": "nidus.text", "query": "ranking", "top_k": 5}),
    );
    assert_eq!(
        status, 200,
        "text-search over the provisioned schema: {hits}"
    );
    assert_eq!(hits[0]["id"], "bug", "{hits}");
}

/// Re-remembering an id keeps its birth date — the handler reads the prior value
/// back before the wholesale-attr upsert, same as the MCP tool.
#[test]
fn re_remember_preserves_created_at_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(&server, json!({"id": "same", "text": "first version"}));
    let created = stamp(&recall(&server, "first version")[0], "nidus.created_at");

    remember(&server, json!({"id": "same", "text": "second version"}));
    let hit = &recall(&server, "second version")[0];
    assert_eq!(hit["id"], "same");
    assert_eq!(
        created,
        stamp(hit, "nidus.created_at"),
        "created_at must carry forward across a re-remember: {hit}"
    );
}

/// A near-duplicate write with `dedupe_threshold` updates the matched entry in place:
/// the response says so, redirects to the survivor's id, and merges attrs (supplied
/// keys win, omitted keys survive) instead of replacing them.
#[test]
fn dedupe_updates_in_place_and_merges_attrs_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    let text = "the deploy runbook lives in the ops repo";

    remember(
        &server,
        json!({"id": "first", "text": text,
               "attrs": {"keep": {"Str": "x"}, "k": {"Str": "v1"}}}),
    );
    let body = remember(
        &server,
        json!({"id": "second", "text": text, "dedupe_threshold": 0.95,
               "attrs": {"k": {"Str": "v2"}}}),
    );
    assert_eq!(body["deduped"], true, "{body}");
    assert_eq!(
        body["id"], "first",
        "write must redirect to the match: {body}"
    );

    let (status, listed) = server.post("/list", &json!({"limit": 100}));
    assert_eq!(status, 200);
    assert_eq!(
        listed.as_array().unwrap().len(),
        1,
        "one entry, not two: {listed}"
    );
    let attrs = &listed[0]["attrs"];
    assert_eq!(
        attrs["k"]["Str"], "v2",
        "supplied key wins the collision: {attrs}"
    );
    assert_eq!(
        attrs["keep"]["Str"], "x",
        "omitted key survives the merge: {attrs}"
    );
}

/// An expired entry is not a dedupe candidate — matching one would inherit its past
/// `expires_at` and land a write that reports success but is never visible.
#[test]
fn dedupe_does_not_match_an_expired_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    let text = "an ephemeral note about the flaky test";

    remember(
        &server,
        json!({"id": "old", "text": text, "ttl_seconds": 0}),
    );
    let body = remember(
        &server,
        json!({"id": "new", "text": text, "dedupe_threshold": 0.9}),
    );
    assert_eq!(body["deduped"], false, "{body}");
    assert_eq!(body["id"], "new", "{body}");

    let hits = recall(&server, text);
    assert!(
        ids(&hits).contains(&"new"),
        "the fresh entry must be live: {hits:?}"
    );
}

/// **The #106 regression.** `/recall` hides an expired entry and still surfaces one
/// that never got a TTL (D5) — while the raw `/list` route deliberately sees the
/// expired row: TTL is read-time memory semantics, not deletion.
#[test]
fn recall_hides_expired_entries_but_raw_list_still_sees_them() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"id": "gone", "text": "ephemeral scratch note", "ttl_seconds": 0}),
    );
    remember(&server, json!({"id": "kept", "text": "durable note"}));

    let hits = recall(&server, "ephemeral scratch note");
    assert!(
        !ids(&hits).contains(&"gone"),
        "expired entry leaked: {hits:?}"
    );
    let hits = recall(&server, "durable note");
    assert!(
        ids(&hits).contains(&"kept"),
        "no-TTL entry must surface: {hits:?}"
    );

    let (status, listed) = server.post("/list", &json!({"limit": 100}));
    assert_eq!(status, 200);
    let listed: Vec<&str> = listed
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_str().unwrap())
        .collect();
    assert!(
        listed.contains(&"gone") && listed.contains(&"kept"),
        "raw list is unguarded by design and sees both: {listed:?}"
    );
}

/// A plain recall (`reinforce` omitted) must stay a pure read: asserting "recall returned
/// hits" would pass whether or not reinforcement exists at all, so this checks the raw attrs.
#[test]
fn a_default_recall_stamps_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    remember(&server, json!({"id": "note", "text": "a plain memory"}));

    let hits = recall(&server, "a plain memory");
    assert!(ids(&hits).contains(&"note"), "{hits:?}");

    let attrs = listed_attrs(&server, "note");
    assert!(
        access_count(&attrs).is_none(),
        "a default recall must not stamp: {attrs}"
    );
}

/// `"reinforce": true` stamps the returned entry, and a second reinforced recall increments
/// it rather than resetting it — proving the counter, not just its presence.
#[test]
fn a_reinforced_recall_stamps_the_returned_entries() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    remember(&server, json!({"id": "note", "text": "a durable memory"}));

    let before = epoch_ms_now();
    let hits = recall_with(
        &server,
        json!({"query": "a durable memory", "top_k": 10, "reinforce": true}),
    );
    assert!(ids(&hits).contains(&"note"), "{hits:?}");

    let attrs = listed_attrs(&server, "note");
    assert_eq!(access_count(&attrs), Some(1), "{attrs}");
    let last_accessed = attrs["nidus.last_accessed"]["DateTime"]
        .as_i64()
        .unwrap_or_else(|| panic!("last_accessed missing or not DateTime: {attrs}"));
    assert!(
        last_accessed >= before,
        "last_accessed must be a plausible epoch ms: {attrs}"
    );

    recall_with(
        &server,
        json!({"query": "a durable memory", "top_k": 10, "reinforce": true}),
    );
    let attrs = listed_attrs(&server, "note");
    assert_eq!(
        access_count(&attrs),
        Some(2),
        "the second recall must increment: {attrs}"
    );
}

/// A reinforced recall stamps only the entries it actually returned — a `top_k` of 1 must
/// leave the entry that lost the ranking untouched.
#[test]
fn a_reinforced_recall_does_not_stamp_an_entry_it_did_not_return() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    remember(
        &server,
        json!({"id": "winner", "text": "the target memory text"}),
    );
    remember(
        &server,
        json!({"id": "loser", "text": "a totally unrelated memory"}),
    );

    let hits = recall_with(
        &server,
        json!({"query": "the target memory text", "top_k": 1, "reinforce": true}),
    );
    assert_eq!(ids(&hits), ["winner"], "{hits:?}");

    assert_eq!(access_count(&listed_attrs(&server, "winner")), Some(1));
    assert_eq!(access_count(&listed_attrs(&server, "loser")), None);
}

/// `extend_ttl_seconds` moves an existing expiry forward and never fabricates one on an
/// entry that had none — the same semantics `crate::memory::reinforce_hits` documents,
/// pinned here through the actual HTTP surface.
#[test]
fn extend_ttl_seconds_moves_an_existing_expiry_and_creates_none() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    remember(
        &server,
        json!({"id": "mortal", "text": "a memory with a ttl", "ttl_seconds": 60}),
    );
    remember(
        &server,
        json!({"id": "eternal", "text": "a memory with no ttl"}),
    );

    let original_expiry = listed_attrs(&server, "mortal")["nidus.expires_at"]["DateTime"]
        .as_i64()
        .expect("mortal must start with an expiry");

    recall_with(
        &server,
        json!({
            "query": "a memory",
            "top_k": 10,
            "reinforce": true,
            "extend_ttl_seconds": 3600,
        }),
    );

    let mortal_attrs = listed_attrs(&server, "mortal");
    let new_expiry = mortal_attrs["nidus.expires_at"]["DateTime"]
        .as_i64()
        .expect("mortal must still have an expiry");
    assert!(new_expiry > original_expiry, "{mortal_attrs}");

    let eternal_attrs = listed_attrs(&server, "eternal");
    assert!(
        eternal_attrs.get("nidus.expires_at").is_none(),
        "extend_ttl_seconds must not create an expiry: {eternal_attrs}"
    );
}

/// The durable half of the point: the stamp is a real log append, not an in-RAM tweak, so
/// it must survive the process that wrote it going away.
#[test]
fn a_reinforced_recall_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    remember(&server, json!({"id": "note", "text": "a durable memory"}));
    recall_with(
        &server,
        json!({"query": "a durable memory", "top_k": 10, "reinforce": true}),
    );
    assert_eq!(access_count(&listed_attrs(&server, "note")), Some(1));

    assert!(server.shutdown(), "clean shutdown should exit successfully");

    let restarted = per_text_embedder_server(dir.path(), DIM);
    let attrs = listed_attrs(&restarted, "note");
    assert_eq!(
        access_count(&attrs),
        Some(1),
        "the reinforcement stamp must survive a restart: {attrs}"
    );
}

/// A caller that asked for the write over the wire is refused on a `--read-only` server, not
/// answered as though the stamp happened. The in-process `Memory::recall` degrades instead: a
/// library caller may not own `open_mode`, where a request naming `reinforce` does.
#[test]
fn a_reinforced_recall_on_a_read_only_server_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    {
        let server = per_text_embedder_server(dir.path(), DIM);
        remember(&server, json!({"id": "note", "text": "a durable memory"}));
        assert!(server.shutdown(), "clean shutdown should exit successfully");
    }

    let embed_url = mock_embedder_per_text(DIM);
    let server = crate::harness::Server::new(dir.path(), DIM)
        .args([
            "--embed-provider",
            "ollama",
            "--embed-base-url",
            &embed_url,
            "--read-only",
        ])
        .start();

    let (status, body) = server.post(
        "/collections/notes/recall",
        &json!({"query": "a durable memory", "top_k": 10, "reinforce": true}),
    );
    assert_ne!(
        status, 200,
        "a reinforced recall must not report success: {body}"
    );
    assert!(
        body.to_string().contains("read-only"),
        "the refusal must name the cause: {body}"
    );
    assert_eq!(
        access_count(&listed_attrs(&server, "note")),
        None,
        "and nothing may have been written"
    );

    // The same server still answers an ordinary recall: only the write was refused.
    let hits = recall_with(&server, json!({"query": "a durable memory", "top_k": 10}));
    assert!(ids(&hits).contains(&"note"), "{hits:?}");
}

/// The criterion that matters end to end: a reinforced entry's durable count must be able to
/// flip a ranking. Reinforced via `/recall` isolated by `filter`, then ranked through both
/// `/search` (exact vectors, so the cosine gap to overturn is known) and `/recall`.
#[test]
fn a_reinforced_recall_ranks_by_count() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    assert_eq!(server.post("/collections/notes", &json!({})).0, 200);

    // hot at 0 degrees, cold at 10 degrees, the tie query at 6 degrees: cold's raw cosine
    // edges out hot's by ~0.003, small enough for the count term (~0.33 at count 5) to
    // overturn once hot alone is reinforced.
    let angle =
        |deg: f32| -> Vec<f32> { vec![deg.to_radians().cos(), deg.to_radians().sin(), 0.0] };
    let hot = angle(0.0);
    let cold = angle(10.0);
    let tie = angle(6.0);
    let rec =
        |id: &str, v: Vec<f32>| json!({"id": id, "vector": v, "attrs": {"which": {"Str": id}}});
    let (status, body) = server.post(
        "/collections/notes/upsert",
        &json!({"records": [rec("hot", hot), rec("cold", cold)]}),
    );
    assert_eq!(status, 200, "upsert failed: {body}");

    for _ in 0..5 {
        recall_with(
            &server,
            json!({
                "query": "anything",
                "top_k": 10,
                "reinforce": true,
                "filter": [{"Eq": ["which", {"Str": "hot"}]}],
            }),
        );
    }
    assert_eq!(access_count(&listed_attrs(&server, "hot")), Some(5));
    assert_eq!(access_count(&listed_attrs(&server, "cold")), None);

    let ranked = |rank_by: Option<Value>| -> Vec<String> {
        let mut req = json!({"query": tie.clone(), "top_k": 2, "scope": ["notes"]});
        if let Some(rb) = rank_by {
            req["rank_by"] = rb;
        }
        let (status, body) = server.post("/search", &req);
        assert_eq!(status, 200, "search failed: {body}");
        body.as_array()
            .expect("search returns an array")
            .iter()
            .map(|h| h["id"].as_str().expect("id").to_string())
            .collect()
    };
    assert_eq!(
        ranked(None),
        vec!["cold".to_string(), "hot".to_string()],
        "cold's raw cosine must edge out hot's before any count term applies"
    );
    assert_eq!(
        ranked(Some(json!({"Decay": {
            "field": "", "origin": 0, "count_field": "nidus.access_count"
        }}))),
        vec!["hot".to_string(), "cold".to_string()],
        "the reinforced entry must out-rank the fresher-scoring one once counted"
    );

    // `count_lambda` of 10 dwarfs any cosine gap (which cannot exceed 2), so "hot" must lead;
    // pointed at a key nothing carries, every hit pays the same penalty and the order falls
    // back to the plain one. The pair proves the term is read, not merely accepted.
    let recalled = |rank_by: Option<Value>| -> Vec<String> {
        let mut req = json!({"query": "anything", "top_k": 2});
        if let Some(rb) = rank_by {
            req["rank_by"] = rb;
        }
        ids(&recall_with(&server, req))
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    let term = |field: &str| {
        json!({"Decay": {
            "field": "", "origin": 0, "count_field": field, "count_lambda": 10.0
        }})
    };
    assert_eq!(
        recalled(Some(term("nidus.access_count")))[0],
        "hot",
        "a count term on /recall must promote the reinforced entry"
    );
    assert_eq!(
        recalled(Some(term("nidus.no_such_key"))),
        recalled(None),
        "a count term nothing carries must leave the plain recall order alone"
    );
}

/// Recall is the surface where near-duplicate crowding hurts most, so `diversity` has to
/// reach it through the real binary and not just the library. The corpus is built in the
/// query's own embedding space, so which hits are redundant is not a guess.
#[test]
fn recall_diversity_spreads_a_crowded_window_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    assert_eq!(server.post("/collections/notes", &json!({})).0, 200);

    // The mock embeds by hashing the text, so the query's direction is computable in-test.
    let u = unit(vector_for("query", DIM));
    let w = orthogonal_unit(&u);
    let mix = |a: f32, b: f32| -> Vec<f32> { (0..DIM).map(|i| a * u[i] + b * w[i]).collect() };
    let rec = |id: &str, v: Vec<f32>, text: &str| json!({"id": id, "vector": v, "attrs": {"nidus.text": {"Str": text}}});
    let (status, body) = server.post(
        "/collections/notes/upsert",
        &json!({"records": [
            rec("dup0", mix(1.0, 0.0), "alpha"),
            rec("dup1", mix(0.9999, 0.0141), "alpha again"),
            rec("novel", mix(0.6, 0.8), "something else")
        ]}),
    );
    assert_eq!(status, 200, "upsert failed: {body}");

    let recalled = |diversity: Option<f32>| -> Vec<String> {
        let mut req = json!({"query": "query", "top_k": 2});
        if let Some(d) = diversity {
            req["diversity"] = json!(d);
        }
        let (status, body) = server.post("/collections/notes/recall", &req);
        assert_eq!(status, 200, "recall failed: {body}");
        body.as_array()
            .expect("recall returns an array")
            .iter()
            .map(|h| h["id"].as_str().expect("id").to_string())
            .collect()
    };
    assert_eq!(recalled(None), ["dup0", "dup1"]);
    assert_eq!(
        recalled(Some(0.3)),
        ["dup0", "novel"],
        "diversity changed nothing"
    );
    // `0.0` is a real lambda, not "unset" — the trap `RecallOpts`' zero sentinel sets.
    assert_eq!(recalled(Some(0.0)), ["dup0", "novel"]);
}

/// `v` scaled to unit length. A zero vector is returned unchanged, as the store does.
fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// A unit vector orthogonal to `u`, for building a corpus at a known cosine.
fn orthogonal_unit(u: &[f32]) -> Vec<f32> {
    let pick = if u[0].abs() < 0.9 { 0 } else { 1 };
    let mut e = vec![0.0f32; u.len()];
    e[pick] = 1.0;
    let dot: f32 = u.iter().zip(&e).map(|(a, b)| a * b).sum();
    for (i, x) in e.iter_mut().enumerate() {
        *x -= dot * u[i];
    }
    unit(e)
}
