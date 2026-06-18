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
use redis::{Client, Commands};

use super::MemoryTier;

/// A shared memory tier backed by a Redis-protocol server (Redis/Valkey/KeyDB/Dragonfly).
pub struct RedisTier {
    client: Client,
    /// A cached blocking connection, lazily opened and reopened on failure.
    conn: Mutex<Option<redis::Connection>>,
    /// Optional key namespace (`<prefix>:<key>`), from `?prefix=…` in the URL.
    prefix: String,
}

impl RedisTier {
    /// Build from a memory-tier location: a `redis://`/`rediss://`/`valkey://`/
    /// `valkeys://`/`keydb://`/`dragonfly://` URL. An optional `?prefix=<ns>` query
    /// namespaces every key (`<ns>:<key>`); it is stripped before the URL reaches the
    /// client. The connection is opened lazily on first use, so construction never
    /// blocks on the network.
    pub(crate) fn from_url(location: &str) -> Result<RedisTier> {
        let (url, prefix) = normalize_url(location)?;
        let client = Client::open(url.as_str())
            .with_context(|| format!("invalid Redis memory-tier URL {url:?}"))?;
        Ok(RedisTier {
            client,
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
    fn run<T>(&self, f: impl Fn(&mut redis::Connection) -> redis::RedisResult<T>) -> Result<T> {
        let mut guard = self
            .conn
            .lock()
            .map_err(|_| anyhow!("Redis memory-tier connection lock poisoned"))?;
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some(
                    self.client
                        .get_connection()
                        .context("failed to connect to the Redis memory tier")?,
                );
            }
            let conn = guard.as_mut().expect("connection just ensured");
            match f(conn) {
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
        self.run(|conn| conn.get(&k))
    }

    fn store(&self, key: &str, bytes: &[u8], ttl: Option<Duration>) -> Result<()> {
        let k = self.namespaced(key);
        match ttl {
            // SET key value EX <seconds> — round sub-second ttls up to 1s (0 would be
            // an immediate-expiry no-op, surprising for a "store this for a moment").
            Some(d) => {
                let secs = d.as_secs().max(if d.is_zero() { 0 } else { 1 });
                self.run(|conn| conn.set_ex(&k, bytes, secs))
            }
            None => self.run(|conn| conn.set(&k, bytes)),
        }
    }
}

/// Rewrite a memory-tier location into a `redis-rs`-acceptable URL plus an optional key
/// prefix. Maps the alias schemes onto Redis's own (`valkey`/`keydb`/`dragonfly` →
/// `redis`, `valkeys` → `rediss`, since the servers are protocol-identical) and strips a
/// trailing `?prefix=<ns>` query. Pure string logic — unit-tested directly.
fn normalize_url(location: &str) -> Result<(String, String)> {
    let (scheme, rest) = location
        .split_once("://")
        .ok_or_else(|| anyhow!("Redis memory-tier location {location:?} is missing a scheme"))?;
    let canon = match scheme.to_ascii_lowercase().as_str() {
        "redis" | "valkey" | "keydb" | "dragonfly" => "redis",
        "rediss" | "valkeys" => "rediss",
        other => bail!("{other:?} is not a Redis-family scheme"),
    };
    // Split off an optional `?prefix=<ns>` (the only query key we honour).
    let (body, prefix) = match rest.split_once('?') {
        Some((body, query)) => (body, parse_prefix(query)),
        None => (rest, String::new()),
    };
    if body.is_empty() {
        bail!("Redis memory-tier location {location:?} is missing a host");
    }
    Ok((format!("{canon}://{body}"), prefix))
}

/// Extract a `prefix=<ns>` value from a URL query string (the rest is ignored).
fn parse_prefix(query: &str) -> String {
    query
        .split('&')
        .find_map(|kv| kv.strip_prefix("prefix="))
        .unwrap_or("")
        .to_string()
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
            let (url, prefix) = normalize_url(loc).unwrap();
            assert_eq!(url, want, "{loc}");
            assert!(prefix.is_empty(), "{loc}");
        }
    }

    #[test]
    fn normalize_extracts_and_strips_prefix() {
        let (url, prefix) = normalize_url("valkey://h:6379/0?prefix=nidus").unwrap();
        assert_eq!(url, "redis://h:6379/0");
        assert_eq!(prefix, "nidus");
        // A non-prefix query is simply dropped.
        let (url, prefix) = normalize_url("redis://h?foo=bar").unwrap();
        assert_eq!(url, "redis://h");
        assert!(prefix.is_empty());
    }

    #[test]
    fn normalize_rejects_bad_locations() {
        assert!(normalize_url("redis-no-scheme").is_err());
        assert!(normalize_url("kafka://h").is_err());
        assert!(normalize_url("redis://").is_err()); // no host
    }

    #[test]
    fn namespaced_prepends_prefix() {
        let tier = RedisTier {
            client: Client::open("redis://localhost").unwrap(),
            conn: Mutex::new(None),
            prefix: "nidus".to_string(),
        };
        assert_eq!(tier.namespaced("warm"), "nidus:warm");
        let flat = RedisTier {
            client: Client::open("redis://localhost").unwrap(),
            conn: Mutex::new(None),
            prefix: String::new(),
        };
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
