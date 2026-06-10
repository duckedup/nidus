//! The `nidus` command line: store operations over a directory, plus `nidus
//! serve`. Everything here is synchronous (matching the library); only `serve`
//! spins up a Tokio runtime, so the common, fast subcommands pay no async cost.

use std::io::Read;
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;

use crate::server::dto::{FootprintDto, HitDto};
use crate::{Config, Distance, Filter, Nidus, OpenMode, Record, Scope, SearchOpts};

mod backup;

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

/// Store location, shared by every subcommand. For an existing store both the
/// dimension and distance metric are read from the on-disk header, so `--dim`
/// and `--distance` are only needed when creating a store (or to override and
/// double-check an existing one — a mismatch is then a hard error).
#[derive(Args, Debug)]
struct StoreArgs {
    /// Store directory (created on first write).
    #[arg(long, short = 'd')]
    dir: PathBuf,
    /// Embedding dimension. Inferred from an existing store; required to create one.
    #[arg(long)]
    dim: Option<usize>,
    /// Distance metric: cosine, euclidean, or dot. Inferred from an existing
    /// store; defaults to cosine when creating one.
    #[arg(long)]
    distance: Option<DistanceArg>,
    /// Open without taking the writer lock (rejects mutations).
    #[arg(long)]
    read_only: bool,
}

impl StoreArgs {
    /// Resolve the `(dimension, distance)` to open with. An explicit flag always
    /// wins (and is then verified against the header on open); otherwise the
    /// value is read from an existing store's header. When neither is available
    /// — no store yet and no `--dim` — creation cannot proceed, so we ask for it.
    fn resolve(&self) -> Result<(usize, Distance)> {
        let peeked = crate::data::peek_header(&self.dir.join("data"))?;
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

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the HTTP server.
    Serve {
        #[command(flatten)]
        store: StoreArgs,
        /// Address to bind.
        #[arg(long, default_value = "127.0.0.1:7700")]
        addr: String,
        /// Require `Authorization: Bearer <token>` on every request except
        /// `/health`. Falls back to the `NIDUS_TOKEN` env var. Strongly advised
        /// when binding anything other than localhost.
        #[arg(long)]
        token: Option<String>,
        /// Maximum request body size in bytes (also the largest single upsert).
        /// Default 256 MiB.
        #[arg(long, default_value_t = 256 * 1024 * 1024)]
        max_body_bytes: usize,
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
        /// Drop hits scoring below this cosine similarity.
        #[arg(long)]
        min_score: Option<f32>,
        /// AND-filter as JSON. Predicates: Eq, Ne, Glob, In, NotIn, Lt, Le, Gt, Ge.
        /// E.g. '[{"Ge":["ts",{"Int":1700000000}]},{"Ne":["status",{"Str":"archived"}]}]'.
        #[arg(long = "where")]
        filter: Option<String>,
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
        /// AND-filter as JSON. Predicates: Eq, Ne, Glob, In, NotIn, Lt, Le, Gt, Ge.
        /// E.g. '[{"Ge":["ts",{"Int":1700000000}]},{"Ne":["status",{"Str":"archived"}]}]'.
        #[arg(long = "where")]
        filter: Option<String>,
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
    ///
    /// Safe to run alongside a writer (e.g. `nidus serve`): it captures a
    /// consistent, lock-free snapshot without blocking writes. Ideal for a
    /// pre-upgrade backup or a periodic cron snapshot.
    Backup {
        /// Store directory to back up.
        #[arg(long, short = 'd')]
        dir: PathBuf,
        /// Output archive path. Defaults to `<dir-name>-<unix-secs>.tar.gz`.
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
    },
    /// Restore a store from a `nidus backup` archive (`.tar.gz`).
    Restore {
        /// Backup archive to restore from.
        #[arg(long = "in", short = 'i')]
        input: PathBuf,
        /// Target store directory (created if absent).
        #[arg(long, short = 'd')]
        dir: PathBuf,
        /// Overwrite an existing store without prompting (for cron / scripts).
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Print store footprint and collections (JSON).
    Stats {
        #[command(flatten)]
        store: StoreArgs,
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
        } => serve(store, addr, token, max_body_bytes),
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
            min_score,
            filter,
        } => {
            let db = open(&store, false)?;
            let query: Vec<f32> = serde_json::from_str(&read_input(query_file.as_ref())?)?;
            let filter = match filter {
                Some(s) => serde_json::from_str(&s)?,
                None => Filter::default(),
            };
            let opts = SearchOpts {
                top_k,
                min_score,
                filter,
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
        } => {
            let db = open(&store, false)?;
            let filter = match filter {
                Some(s) => serde_json::from_str(&s)?,
                None => Filter::default(),
            };
            let refs: Vec<&str> = collections.iter().map(String::as_str).collect();
            let hits = if refs.is_empty() {
                db.list(Scope::All, &filter, offset, limit)?
            } else {
                db.list(Scope::Collections(&refs), &filter, offset, limit)?
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
        Command::Backup { dir, out } => {
            let out = out.unwrap_or_else(|| PathBuf::from(backup::default_out_name(&dir)));
            print_json(&backup::backup(&dir, &out)?)
        }
        Command::Restore { input, dir, yes } => print_json(&backup::restore(&input, &dir, yes)?),
        Command::Stats { store } => {
            let db = open(&store, false)?;
            print_json(&serde_json::json!({
                "dimension": db.dimension(),
                "distance": format!("{:?}", db.config().distance),
                "collections": db.collections(),
                "footprint": FootprintDto::from(db.footprint()),
            }))
        }
    }
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
    let (dim, distance) = store.resolve()?;
    Nidus::open(
        Config::new(store.dir.clone(), dim)
            .distance(distance)
            .open_mode(mode),
    )
}

fn serve(
    store: StoreArgs,
    addr: String,
    token: Option<String>,
    max_body_bytes: usize,
) -> Result<()> {
    let mode = if store.read_only {
        OpenMode::ReadOnly
    } else {
        OpenMode::ReadWrite
    };
    let (dim, distance) = store.resolve()?;
    let db = Nidus::open(
        Config::new(store.dir.clone(), dim)
            .distance(distance)
            .open_mode(mode),
    )?;
    // An explicit --token wins; otherwise fall back to the NIDUS_TOKEN env var.
    let token = token.or_else(|| std::env::var("NIDUS_TOKEN").ok().filter(|t| !t.is_empty()));
    let cfg = crate::server::ServeConfig {
        addr,
        token,
        max_body_bytes,
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(crate::server::serve(db, cfg))
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
            dim: None,
            distance: None,
            read_only: false,
        };
        assert_eq!(args.resolve().unwrap(), (5, Distance::Euclidean));
    }

    #[test]
    fn backup_parses_dir_and_optional_out() {
        let cli =
            Cli::try_parse_from(["nidus", "backup", "--dir", "/tmp/s", "-o", "/tmp/s.tar.gz"])
                .unwrap();
        match cli.command {
            Command::Backup { dir, out } => {
                assert_eq!(dir, PathBuf::from("/tmp/s"));
                assert_eq!(out, Some(PathBuf::from("/tmp/s.tar.gz")));
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
            Command::Restore { input, dir, yes } => {
                assert_eq!(input, PathBuf::from("/tmp/s.tar.gz"));
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
    fn resolve_requires_dim_when_no_store_yet() {
        let dir = tempfile::tempdir().unwrap();
        let args = StoreArgs {
            dir: dir.path().join("does-not-exist-yet"),
            dim: None,
            distance: None,
            read_only: false,
        };
        let err = args.resolve().unwrap_err().to_string();
        assert!(err.contains("--dim"), "unexpected error: {err}");
    }
}
