//! [`RedisTier`]: a shared [`MemoryTier`] over the Redis wire protocol (SPEC §13.3).
//!
//! A single `redis-rs` *blocking* client speaks RESP, so one backend covers the whole
//! RESP-compatible family — **Redis, Valkey, KeyDB, and DragonflyDB** — selected by URL
//! scheme:
//!
//! - `redis://…`  / `valkey://…` / `keydb://…` / `dragonfly://…` → plain TCP
//! - `rediss://…` / `valkeys://…`                               → TLS (via `tls-rustls`)
//!
//! The non-`redis` schemes are pure aliases: they are rewritten to `redis://`/`rediss://`
//! before being handed to the client, since the servers are protocol-identical.
//!
//! A `?cluster=true` query opens a **Redis/Valkey Cluster** client instead (via the `cluster`
//! feature): the host is a seed node — or several, comma-separated, to tolerate one being
//! down at startup (`redis://a,b,c?cluster=true`) — the rest of the topology is discovered,
//! and slot routing + `MOVED`/`ASK` redirection are handled by the client. Single-node and
//! cluster connections share one code path — both are driven as `&mut dyn ConnectionLike`
//! through the low-level [`Cmd`](redis::Cmd) API.
//!
//! As a [`MemoryTier`] this is **model (a)** (SPEC §13.3): a *shared, rebuildable* cache
//! of the serialized working set, not a source of truth. `store` is `SET` (with `EX`
//! when a ttl is given), `load` is `GET`; an evicted or absent key is `Ok(None)`, never
//! fatal — the persistence tier remains authoritative.
//!
//! Sync by design: `default-features = false` keeps `redis-rs` on its blocking
//! `Connection`, so nothing async (no tokio) enters the tree. A connection is cached
//! behind a `Mutex` and transparently reopened if a command fails (a dropped/expired
//! TCP connection), so callers never juggle reconnection.

use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use redis::{Client, ConnectionLike, RedisResult, cluster::ClusterClient};

use super::MemoryTier;

/// A live blocking connection — a single-node [`redis::Connection`] or a
/// [`ClusterConnection`](redis::cluster::ClusterConnection) — behind one object-safe trait
/// so the rest of the tier is topology-agnostic. Commands are issued through the low-level
/// [`Cmd`](redis::Cmd) API, which takes `&mut dyn ConnectionLike`.
type Conn = Box<dyn ConnectionLike + Send>;

/// A shared memory tier backed by a Redis-protocol server (Redis/Valkey/KeyDB/Dragonfly),
/// either a single node or a **cluster** (slot-routed, with MOVED/ASK redirection handled by
/// the client).
pub struct RedisTier {
    /// Opens a fresh connection. One boxed closure regardless of single-node vs cluster, so
    /// the reconnect path below is shared.
    open: Box<dyn Fn() -> RedisResult<Conn> + Send + Sync>,
    /// A cached blocking connection, lazily opened and reopened on failure.
    conn: Mutex<Option<Conn>>,
    /// Optional key namespace (`<prefix>:<key>`), from `?prefix=…` in the URL.
    prefix: String,
}

impl RedisTier {
    /// Build from a memory-tier location: a `redis://`/`rediss://`/`valkey://`/
    /// `valkeys://`/`keydb://`/`dragonfly://` URL. An optional `?prefix=<ns>` query
    /// namespaces every key (`<ns>:<key>`), and `?cluster=true` opens a Redis/Valkey
    /// **Cluster** client over one or more comma-separated seed hosts
    /// (`redis://a,b,c?cluster=true`) — the rest of the topology is discovered. Both query
    /// keys are stripped before the URL reaches the client. The connection is opened lazily
    /// on first use, so construction never blocks on the network.
    pub(crate) fn from_url(location: &str) -> Result<RedisTier> {
        let (nodes, prefix, cluster) = normalize_url(location)?;
        let open: Box<dyn Fn() -> RedisResult<Conn> + Send + Sync> = if cluster {
            // Every comma-separated host is a seed; the client bootstraps cluster discovery
            // (CLUSTER SLOTS) from whichever is reachable, so several tolerate one being down.
            let client = ClusterClient::new(nodes.clone())
                .with_context(|| format!("invalid Redis cluster seed nodes {nodes:?}"))?;
            Box::new(move || client.get_connection().map(|c| Box::new(c) as Conn))
        } else {
            if nodes.len() > 1 {
                bail!("multiple comma-separated seed nodes require ?cluster=true: {nodes:?}");
            }
            let url = nodes
                .into_iter()
                .next()
                .expect("normalize_url yields ≥1 node");
            let client = Client::open(url.as_str())
                .with_context(|| format!("invalid Redis memory-tier URL {url:?}"))?;
            Box::new(move || client.get_connection().map(|c| Box::new(c) as Conn))
        };
        Ok(RedisTier {
            open,
            conn: Mutex::new(None),
            prefix,
        })
    }

    /// Namespace a key with the configured prefix (`<prefix>:<key>`), or pass it
    /// through unchanged when no prefix is set.
    fn namespaced(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}:{}", self.prefix, key)
        }
    }

    /// Run `f` against a live connection, opening one if needed and retrying **once**
    /// on a connection-level failure (a server-side timeout or restart can silently
    /// drop the cached socket; the retry transparently reconnects).
    fn run<T>(&self, f: impl Fn(&mut dyn ConnectionLike) -> RedisResult<T>) -> Result<T> {
        let mut guard = self
            .conn
            .lock()
            .map_err(|_| anyhow!("Redis memory-tier connection lock poisoned"))?;
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some((self.open)().context("failed to connect to the Redis memory tier")?);
            }
            let conn = guard.as_mut().expect("connection just ensured");
            match f(conn.as_mut()) {
                Ok(v) => return Ok(v),
                // Drop the (possibly-dead) connection and let the next attempt reopen.
                Err(e) if attempt == 0 && e.is_connection_dropped() => *guard = None,
                Err(e) => return Err(anyhow!("Redis memory-tier command failed: {e}")),
            }
        }
        unreachable!("the loop returns on success or on the final attempt's error")
    }
}

impl MemoryTier for RedisTier {
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let k = self.namespaced(key);
        // Low-level `Cmd::query` (vs the `Commands` helpers) because it takes
        // `&mut dyn ConnectionLike`, so the same call drives a single-node or cluster
        // connection without the trait being `Sized`.
        self.run(|conn| redis::cmd("GET").arg(&k).query::<Option<Vec<u8>>>(conn))
    }

    fn store(&self, key: &str, bytes: &[u8], ttl: Option<Duration>) -> Result<()> {
        let k = self.namespaced(key);
        let mut command = redis::cmd("SET");
        command.arg(&k).arg(bytes);
        if let Some(d) = ttl {
            // `EX <seconds>` — clamp any non-`None` ttl up to ≥ 1s: Redis rejects `EX 0` as
            // an invalid expire time, and a sub-second ttl truncates to 0, so a caller asking
            // to "store this briefly" must still get a valid expiry.
            command.arg("EX").arg(d.as_secs().max(1));
        }
        self.run(|conn| command.query::<()>(conn))
    }
}

/// Rewrite a memory-tier location into a `redis-rs`-acceptable URL plus the optional key
/// prefix and a cluster flag. Maps the alias schemes onto Redis's own (`valkey`/`keydb`/
/// `dragonfly` → `redis`, `valkeys` → `rediss`, since the servers are protocol-identical)
/// and strips the `?prefix=<ns>` / `?cluster=<bool>` query keys. Pure string logic —
/// unit-tested directly.
fn normalize_url(location: &str) -> Result<(Vec<String>, String, bool)> {
    let (scheme, rest) = location
        .split_once("://")
        .ok_or_else(|| anyhow!("Redis memory-tier location {location:?} is missing a scheme"))?;
    let canon = match scheme.to_ascii_lowercase().as_str() {
        "redis" | "valkey" | "keydb" | "dragonfly" => "redis",
        "rediss" | "valkeys" => "rediss",
        other => bail!("{other:?} is not a Redis-family scheme"),
    };
    // Split off the query; we honour `prefix=<ns>` and `cluster=<bool>`.
    let (body, prefix, cluster) = match rest.split_once('?') {
        Some((body, query)) => (
            body,
            query_value(query, "prefix"),
            query_flag(query, "cluster"),
        ),
        None => (rest, String::new(), false),
    };
    if body.is_empty() {
        bail!("Redis memory-tier location {location:?} is missing a host");
    }
    Ok((expand_nodes(canon, body), prefix, cluster))
}

/// Expand `[userinfo@]host[,host…][/db]` into one full `scheme://…` URL per host, so a
/// comma-separated cluster seed list becomes individual node URLs that each carry the shared
/// credentials and database. A single host yields a one-element vec (the common case).
fn expand_nodes(scheme: &str, body: &str) -> Vec<String> {
    // Userinfo is everything before the last `@` (hosts never contain `@`, so a password
    // containing one still splits correctly); the `/db` path, if any, follows the host list.
    let (userinfo, hostpath) = match body.rsplit_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, body),
    };
    let (hosts, path) = match hostpath.split_once('/') {
        Some((h, p)) => (h, Some(p)),
        None => (hostpath, None),
    };
    hosts
        .split(',')
        .map(|host| {
            let mut url = format!("{scheme}://");
            if let Some(u) = userinfo {
                url.push_str(u);
                url.push('@');
            }
            url.push_str(host);
            if let Some(p) = path {
                url.push('/');
                url.push_str(p);
            }
            url
        })
        .collect()
}

/// Value of `<key>=…` in a URL query string (`""` if absent; the rest is ignored).
fn query_value(query: &str, key: &str) -> String {
    let needle = format!("{key}=");
    query
        .split('&')
        .find_map(|kv| kv.strip_prefix(&needle))
        .unwrap_or("")
        .to_string()
}

/// Whether a boolean query flag is truthy (`<key>=true` / `=1` / `=yes`).
fn query_flag(query: &str, key: &str) -> bool {
    matches!(
        query_value(query, key).to_ascii_lowercase().as_str(),
        "true" | "1" | "yes"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_maps_aliases_to_redis_schemes() {
        for (loc, want) in [
            ("redis://h:6379", "redis://h:6379"),
            ("valkey://h:6379", "redis://h:6379"),
            ("keydb://h", "redis://h"),
            ("dragonfly://h/0", "redis://h/0"),
            ("rediss://h:6380", "rediss://h:6380"),
            ("valkeys://h", "rediss://h"),
            ("REDIS://H", "redis://H"), // scheme is case-insensitive
        ] {
            let (nodes, prefix, cluster) = normalize_url(loc).unwrap();
            assert_eq!(nodes, vec![want.to_string()], "{loc}");
            assert!(prefix.is_empty(), "{loc}");
            assert!(!cluster, "{loc}");
        }
    }

    #[test]
    fn normalize_extracts_and_strips_prefix() {
        let (nodes, prefix, _) = normalize_url("valkey://h:6379/0?prefix=nidus").unwrap();
        assert_eq!(nodes, vec!["redis://h:6379/0"]);
        assert_eq!(prefix, "nidus");
        // A non-prefix query is simply dropped.
        let (nodes, prefix, _) = normalize_url("redis://h?foo=bar").unwrap();
        assert_eq!(nodes, vec!["redis://h"]);
        assert!(prefix.is_empty());
    }

    #[test]
    fn normalize_detects_cluster_flag_with_prefix() {
        // Cluster opt-in via `?cluster=true`, composable with `?prefix=`.
        let (nodes, prefix, cluster) =
            normalize_url("valkey://seed:6379?cluster=true&prefix=ns").unwrap();
        assert_eq!(nodes, vec!["redis://seed:6379"]);
        assert_eq!(prefix, "ns");
        assert!(cluster);
        // Truthy variants, and default-off.
        assert!(normalize_url("redis://h?cluster=1").unwrap().2);
        assert!(normalize_url("rediss://h?cluster=yes").unwrap().2);
        assert!(!normalize_url("redis://h?cluster=false").unwrap().2);
        assert!(!normalize_url("redis://h").unwrap().2);
    }

    #[test]
    fn normalize_expands_comma_separated_seed_list() {
        // Each host becomes its own node URL, all sharing the scheme.
        let (nodes, _, cluster) =
            normalize_url("valkey://a:6379,b:6379,c:6379?cluster=true").unwrap();
        assert_eq!(
            nodes,
            vec!["redis://a:6379", "redis://b:6379", "redis://c:6379"]
        );
        assert!(cluster);
        // Shared userinfo and db are distributed to every node.
        let (nodes, _, _) = normalize_url("rediss://u:p@a:1,b:2/0?cluster=true").unwrap();
        assert_eq!(nodes, vec!["rediss://u:p@a:1/0", "rediss://u:p@b:2/0"]);
        // A password containing '@' still splits at the userinfo boundary.
        let (nodes, _, _) = normalize_url("redis://u:p@ss@a:1,b:2?cluster=true").unwrap();
        assert_eq!(nodes, vec!["redis://u:p@ss@a:1", "redis://u:p@ss@b:2"]);
    }

    #[test]
    fn from_url_rejects_seed_list_without_cluster() {
        // A comma-separated list only makes sense for a cluster; demand the opt-in.
        // (`.err()` avoids needing `Debug` on the closure-holding `RedisTier`.)
        let err = RedisTier::from_url("redis://a:6379,b:6379")
            .err()
            .expect("multiple seeds without ?cluster should error")
            .to_string();
        assert!(err.contains("require ?cluster=true"), "{err}");
    }

    #[test]
    fn normalize_rejects_bad_locations() {
        assert!(normalize_url("redis-no-scheme").is_err());
        assert!(normalize_url("kafka://h").is_err());
        assert!(normalize_url("redis://").is_err()); // no host
    }

    #[test]
    fn namespaced_prepends_prefix() {
        // `from_url` only parses (no connection until first use), so this stays Miri-clean.
        let tier = RedisTier::from_url("redis://localhost?prefix=nidus").unwrap();
        assert_eq!(tier.namespaced("warm"), "nidus:warm");
        let flat = RedisTier::from_url("redis://localhost").unwrap();
        assert_eq!(flat.namespaced("warm"), "warm");
    }

    // ── End-to-end against an in-process RESP server (Miri-ignored: real TCP) ────────

    /// `load`/`store` round-trip the actual `redis-rs` blocking client against a tiny
    /// RESP mock — proving the wire path (handshake, SET/SETEX/GET framing, binary values)
    /// without a real server. Also covers the alias scheme and `?prefix=` namespacing.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn redis_tier_round_trips_through_resp_server() {
        let server = mock::spawn();
        // `valkey://` proves the alias rewrite reaches the same client; `?prefix=` proves
        // namespacing is applied to the wire keys.
        let loc = format!("valkey://{}?prefix=nidus", server.addr);
        let tier = RedisTier::from_url(&loc).unwrap();

        assert!(
            tier.load("workingset").unwrap().is_none(),
            "absent key → None"
        );

        let blob = vec![0u8, 1, 2, 250, 255, 42]; // binary-safe
        tier.store("workingset", &blob, None).unwrap();
        assert_eq!(
            tier.load("workingset").unwrap().as_deref(),
            Some(blob.as_slice())
        );

        // ttl path (SETEX) stores the value too.
        tier.store("warm", b"hot", Some(Duration::from_secs(30)))
            .unwrap();
        assert_eq!(
            tier.load("warm").unwrap().as_deref(),
            Some(b"hot".as_slice())
        );

        // A zero/sub-second ttl is clamped to ≥1s (Redis rejects `EX 0`), so this stores
        // rather than erroring.
        tier.store("brief", b"x", Some(Duration::ZERO)).unwrap();
        assert_eq!(
            tier.load("brief").unwrap().as_deref(),
            Some(b"x".as_slice())
        );

        // The prefix was applied on the wire (`nidus:workingset`, not `workingset`).
        assert!(server.has_key("nidus:workingset"));
        assert!(!server.has_key("workingset"));
    }

    /// A minimal RESP server: enough of GET/SET/SETEX (and a permissive `+OK` for the
    /// client's connection-handshake chatter) to round-trip the tier.
    mod mock {
        use std::collections::HashMap;
        use std::io::{BufRead, BufReader, Write};
        use std::net::{TcpListener, TcpStream};
        use std::sync::{Arc, Mutex};
        use std::thread;

        type Store = Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>;

        pub struct MockRedis {
            pub addr: String,
            store: Store,
        }

        impl MockRedis {
            pub fn has_key(&self, key: &str) -> bool {
                self.store.lock().unwrap().contains_key(key.as_bytes())
            }
        }

        pub fn spawn() -> MockRedis {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            let store: Store = Arc::new(Mutex::new(HashMap::new()));
            let srv = store.clone();
            thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { break };
                    let store = srv.clone();
                    thread::spawn(move || serve(stream, store));
                }
            });
            MockRedis { addr, store }
        }

        fn serve(stream: TcpStream, store: Store) {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut out = stream;
            while let Some(cmd) = read_command(&mut reader) {
                if cmd.is_empty() {
                    continue;
                }
                let name = String::from_utf8_lossy(&cmd[0]).to_ascii_uppercase();
                let reply = match name.as_str() {
                    "GET" => match store.lock().unwrap().get(&cmd[1]) {
                        Some(v) => {
                            let mut r = format!("${}\r\n", v.len()).into_bytes();
                            r.extend_from_slice(v);
                            r.extend_from_slice(b"\r\n");
                            r
                        }
                        None => b"$-1\r\n".to_vec(),
                    },
                    // SET key value [EX n …]; SETEX key seconds value.
                    "SET" => {
                        store.lock().unwrap().insert(cmd[1].clone(), cmd[2].clone());
                        b"+OK\r\n".to_vec()
                    }
                    "SETEX" => {
                        store.lock().unwrap().insert(cmd[1].clone(), cmd[3].clone());
                        b"+OK\r\n".to_vec()
                    }
                    "DEL" => {
                        store.lock().unwrap().remove(&cmd[1]);
                        b":1\r\n".to_vec()
                    }
                    // Handshake chatter (CLIENT SETINFO, PING, …) — accept permissively.
                    _ => b"+OK\r\n".to_vec(),
                };
                if out.write_all(&reply).is_err() {
                    return;
                }
            }
        }

        /// Read one RESP array-of-bulk-strings command; `None` on EOF / a malformed frame.
        fn read_command(r: &mut impl BufRead) -> Option<Vec<Vec<u8>>> {
            let mut line = String::new();
            if r.read_line(&mut line).ok()? == 0 {
                return None; // EOF
            }
            let line = line.trim_end();
            let n: usize = line.strip_prefix('*')?.parse().ok()?;
            let mut args = Vec::with_capacity(n);
            for _ in 0..n {
                let mut hdr = String::new();
                if r.read_line(&mut hdr).ok()? == 0 {
                    return None;
                }
                let len: usize = hdr.trim_end().strip_prefix('$')?.parse().ok()?;
                let mut buf = vec![0u8; len + 2]; // payload + trailing CRLF
                r.read_exact(&mut buf).ok()?;
                buf.truncate(len);
                args.push(buf);
            }
            Some(args)
        }
    }
}
