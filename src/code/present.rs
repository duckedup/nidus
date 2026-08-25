//! Presentation for code hits: group flat [`Hit`]s by file with their matching symbols
//! attached, for the CLI and MCP surfaces. **This is presentation, never ranking** — prior
//! art on the unmerged `wdpkr-retrieval-eval` branch (commit c55d7e5) measured wdpkr's
//! `group_results_multi` beating a flat cosine ranking for code queries because it
//! hard-partitions file-level chunks from symbol chunks, and concluded that grouping is the
//! client's job, not `store::search`'s. So: no new `store::search` mode here, no
//! `limit_per` change. Point and describe: path, symbol, kind, line span. Never the source
//! body; the agent reads the file for ground truth.

use crate::{Hit, Value};

use super::{META_END_LINE, META_KIND, META_PATH, META_START_LINE, META_SYMBOL};

/// One matched symbol within a file: everything an agent needs to go read the real source,
/// never the source itself.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolHit {
    pub symbol: Option<String>,
    pub kind: Option<String>,
    pub start_line: Option<i64>,
    pub end_line: Option<i64>,
    pub score: f32,
}

/// Every hit that landed in one file, its symbols ordered by descending score.
#[derive(Debug, Clone, PartialEq)]
pub struct FileGroup {
    pub path: String,
    pub symbols: Vec<SymbolHit>,
}

/// Group `hits` by `code.path`: each file's symbols ranked by descending score, and files
/// ranked by their own best-scoring symbol. A hit carrying no `code.path` did not come from
/// `code` ingest and is skipped; this module has nothing to say about it.
pub fn group_by_file(hits: &[Hit]) -> Vec<FileGroup> {
    let mut by_path: Vec<(String, Vec<SymbolHit>)> = Vec::new();
    for hit in hits {
        let Some(path) = str_attr(hit, META_PATH) else {
            continue;
        };
        let symbol = SymbolHit {
            symbol: str_attr(hit, META_SYMBOL).map(str::to_string),
            kind: str_attr(hit, META_KIND).map(str::to_string),
            start_line: int_attr(hit, META_START_LINE),
            end_line: int_attr(hit, META_END_LINE),
            score: hit.score,
        };
        match by_path.iter_mut().find(|(p, _)| p == path) {
            Some((_, syms)) => syms.push(symbol),
            None => by_path.push((path.to_string(), vec![symbol])),
        }
    }

    let mut groups: Vec<FileGroup> = by_path
        .into_iter()
        .map(|(path, mut symbols)| {
            symbols.sort_by(|a, b| b.score.total_cmp(&a.score));
            FileGroup { path, symbols }
        })
        .collect();
    groups.sort_by(|a, b| best_score(b).total_cmp(&best_score(a)));
    groups
}

/// A group's own rank key: its best (first, post-sort) symbol's score.
fn best_score(g: &FileGroup) -> f32 {
    g.symbols.first().map(|s| s.score).unwrap_or(f32::MIN)
}

fn str_attr<'a>(hit: &'a Hit, key: &str) -> Option<&'a str> {
    match hit.attrs.get(key) {
        Some(Value::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn int_attr(hit: &Hit, key: &str) -> Option<i64> {
    match hit.attrs.get(key) {
        Some(Value::Int(i)) => Some(*i),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn hit(path: &str, symbol: &str, kind: &str, start: i64, end: i64, score: f32) -> Hit {
        let mut attrs = BTreeMap::new();
        attrs.insert(META_PATH.to_string(), Value::Str(path.to_string()));
        attrs.insert(META_SYMBOL.to_string(), Value::Str(symbol.to_string()));
        attrs.insert(META_KIND.to_string(), Value::Str(kind.to_string()));
        attrs.insert(META_START_LINE.to_string(), Value::Int(start));
        attrs.insert(META_END_LINE.to_string(), Value::Int(end));
        Hit::new("docs", format!("{path}#{symbol}"), score, attrs)
    }

    #[test]
    fn groups_by_file_and_ranks_symbols_within_a_file() {
        let hits = vec![
            hit("src/a.rs", "low", "function", 1, 2, 0.2),
            hit("src/a.rs", "high", "function", 10, 20, 0.9),
        ];
        let groups = group_by_file(&hits);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].path, "src/a.rs");
        assert_eq!(groups[0].symbols[0].symbol.as_deref(), Some("high"));
        assert_eq!(groups[0].symbols[1].symbol.as_deref(), Some("low"));
    }

    #[test]
    fn a_files_best_symbol_outranks_another_files_best() {
        let hits = vec![
            hit("src/weak.rs", "only", "function", 1, 2, 0.3),
            hit("src/strong.rs", "meh", "function", 1, 2, 0.4),
            hit("src/strong.rs", "best", "function", 10, 20, 0.95),
        ];
        let groups = group_by_file(&hits);
        assert_eq!(groups[0].path, "src/strong.rs");
        assert_eq!(groups[1].path, "src/weak.rs");
    }

    #[test]
    fn hits_without_a_path_attr_are_skipped() {
        let hit_no_path = Hit::new("docs", "x", 0.5, BTreeMap::new());
        let groups = group_by_file(&[hit_no_path]);
        assert!(groups.is_empty());
    }

    #[test]
    fn no_result_carries_a_source_body() {
        let hits = vec![hit("src/a.rs", "f", "function", 1, 2, 0.5)];
        let groups = group_by_file(&hits);
        // `SymbolHit` has no field a source body could occupy — structural, not incidental.
        let s = &groups[0].symbols[0];
        assert_eq!(s.symbol.as_deref(), Some("f"));
        assert_eq!(s.start_line, Some(1));
        assert_eq!(s.end_line, Some(2));
    }
}
