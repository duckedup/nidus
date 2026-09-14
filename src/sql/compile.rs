//! Syntax tree -> `model.rs` values. Every semantic rule lives here: which `Compiled`
//! variant a statement dispatches to, and which `WITH` keys that variant can honour.

use crate::model::{
    AggregateOpts, Decay, Expand, Filter, FtsClause, FtsQuery, HybridOpts, LimitPer, ListOpts,
    OrderBy, Predicate, Projection, RankBy, RerankOpts, SearchOpts, Value,
};

use super::Compiled;
use super::error::{
    SEC_AGG, SEC_ANNOTATE, SEC_EXPAND, SEC_PLAN, SEC_RANK, SEC_SYNTAX, SEC_TEXT, SqlError,
};
use super::parse::{
    AnyLit, ClauseLit, CmpOp, DecayLit, Lit, OptArg, OptValue, PredNode, Ranking, Selection,
    Statement, WithOption,
};

/// `LIMIT` omitted on a ranked query means 10, not `SearchOpts::default()`'s 0: a bare
/// `SELECT ... ORDER BY knn(...)` must not read as "no matches". `HybridOpts` already
/// defaults to 10, so the three ranked arms agree.
const DEFAULT_RANKED_LIMIT: usize = 10;

/// Turn one parsed statement into the `Compiled` value that runs it. The `ORDER BY` head
/// alone picks the dispatch (root blueprint's dispatch table); nothing else affects it.
pub(super) fn compile(stmt: &Statement) -> Result<Compiled, SqlError> {
    match &stmt.order {
        Some(Ranking::Knn { vector, decay, at }) => {
            compile_search(stmt, vector, decay.as_ref(), *at)
        }
        Some(Ranking::Match { clauses, at }) => compile_text_search(stmt, clauses, *at),
        Some(Ranking::Hybrid {
            vector,
            clauses,
            at,
        }) => compile_hybrid(stmt, vector, clauses, *at),
        Some(Ranking::Field { field, desc }) => {
            compile_list_or_aggregate(stmt, Some((field.as_str(), *desc)))
        }
        None => compile_list_or_aggregate(stmt, None),
    }
}

// ── dispatch: Search (`ORDER BY knn(...)`) ──────────────────────────────────────────

fn compile_search(
    stmt: &Statement,
    vector: &[f32],
    decay: Option<&DecayLit>,
    at: usize,
) -> Result<Compiled, SqlError> {
    if stmt.group_by.is_some() {
        return Err(SqlError::new(
            at,
            "ORDER BY knn(...) cannot combine with GROUP BY",
            SEC_AGG,
        ));
    }
    let mut opts = SearchOpts {
        filter: compile_filter(stmt.filter.as_ref())?,
        projection: compile_projection(&stmt.select),
        top_k: DEFAULT_RANKED_LIMIT,
        ..Default::default()
    };
    if let Some(n) = stmt.limit {
        opts.top_k = to_usize(n, stmt.at, SEC_SYNTAX)?;
    }
    if let Some(n) = stmt.offset {
        opts.offset = to_usize(n, stmt.at, SEC_SYNTAX)?;
    }
    if let Some(d) = decay {
        opts.rank_by = Some(RankBy::Decay(compile_decay(d)));
    }
    apply_with_search(&mut opts, &stmt.with, true)?;
    Ok(Compiled::Search {
        collections: stmt.from.clone(),
        vector: vector.to_vec(),
        opts,
    })
}

// ── dispatch: TextSearch (`ORDER BY match(...)`) ────────────────────────────────────

fn compile_text_search(
    stmt: &Statement,
    clauses: &[ClauseLit],
    at: usize,
) -> Result<Compiled, SqlError> {
    if stmt.group_by.is_some() {
        return Err(SqlError::new(
            at,
            "ORDER BY match(...) cannot combine with GROUP BY",
            SEC_AGG,
        ));
    }
    let mut opts = SearchOpts {
        filter: compile_filter(stmt.filter.as_ref())?,
        projection: compile_projection(&stmt.select),
        top_k: DEFAULT_RANKED_LIMIT,
        ..Default::default()
    };
    if let Some(n) = stmt.limit {
        opts.top_k = to_usize(n, stmt.at, SEC_SYNTAX)?;
    }
    if let Some(n) = stmt.offset {
        opts.offset = to_usize(n, stmt.at, SEC_SYNTAX)?;
    }
    // `false`: there is no `Nidus::text_search_with_plan` sibling to honour WITH (plan).
    apply_with_search(&mut opts, &stmt.with, false)?;
    Ok(Compiled::TextSearch {
        collections: stmt.from.clone(),
        query: compile_clauses(clauses),
        opts,
    })
}

// ── dispatch: Hybrid (`ORDER BY knn(...) FUSE match(...)`) ──────────────────────────

fn compile_hybrid(
    stmt: &Statement,
    vector: &[f32],
    clauses: &[ClauseLit],
    at: usize,
) -> Result<Compiled, SqlError> {
    if stmt.group_by.is_some() {
        return Err(SqlError::new(
            at,
            "ORDER BY knn(...) FUSE match(...) cannot combine with GROUP BY",
            SEC_AGG,
        ));
    }
    require_select_all(stmt, "on a FUSE (hybrid) query", SEC_RANK)?;
    let mut opts = HybridOpts {
        filter: compile_filter(stmt.filter.as_ref())?,
        ..Default::default()
    };
    if let Some(n) = stmt.limit {
        opts.top_k = to_usize(n, stmt.at, SEC_SYNTAX)?;
    }
    if let Some(n) = stmt.offset {
        opts.offset = to_usize(n, stmt.at, SEC_SYNTAX)?;
    }
    apply_with_hybrid(&mut opts, &stmt.with)?;
    Ok(Compiled::Hybrid {
        collections: stmt.from.clone(),
        vector: vector.to_vec(),
        text: compile_clauses(clauses),
        opts,
    })
}

// ── dispatch: List / Aggregate (a bare field, or no ORDER BY at all) ────────────────

fn compile_list_or_aggregate(
    stmt: &Statement,
    order_field: Option<(&str, bool)>,
) -> Result<Compiled, SqlError> {
    let collections = stmt.from.clone();
    // Either spelling selects the aggregate arm: `SELECT sum(x)` (whole-scope totals, the
    // `group_by: None` case) or a `GROUP BY` clause. `sums` may come from either place.
    let projected = match &stmt.select {
        Selection::Aggregates { sums, .. } => Some(sums.clone()),
        _ => None,
    };
    if stmt.group_by.is_some() || projected.is_some() {
        if projected.is_none() {
            require_select_all(stmt, "on an aggregate query (GROUP BY)", SEC_AGG)?;
        }
        if stmt.limit.is_some() || stmt.offset.is_some() {
            return Err(SqlError::new(
                stmt.at,
                "LIMIT/OFFSET are not supported on an aggregate query",
                SEC_AGG,
            ));
        }
        if order_field.is_some() {
            return Err(SqlError::new(
                stmt.at,
                "ORDER BY is not supported on an aggregate query",
                SEC_AGG,
            ));
        }
        reject_all_with(&stmt.with, "an aggregate query")?;
        let mut sum = projected.unwrap_or_default();
        sum.extend(stmt.sums.iter().cloned());
        sum.dedup();
        let opts = AggregateOpts {
            filter: compile_filter(stmt.filter.as_ref())?,
            sum,
            group_by: stmt.group_by.clone(),
        };
        return Ok(Compiled::Aggregate { collections, opts });
    }

    reject_all_with(
        &stmt.with,
        "a plain listing query (no ranking, no GROUP BY)",
    )?;
    let mut opts = ListOpts {
        filter: compile_filter(stmt.filter.as_ref())?,
        projection: compile_projection(&stmt.select),
        ..Default::default()
    };
    if let Some(n) = stmt.limit {
        opts.limit = to_usize(n, stmt.at, SEC_SYNTAX)?;
    }
    if let Some(n) = stmt.offset {
        opts.offset = to_usize(n, stmt.at, SEC_SYNTAX)?;
    }
    opts.order_by = order_field.map(|(f, desc)| {
        if desc {
            OrderBy::desc(f)
        } else {
            OrderBy::asc(f)
        }
    });
    Ok(Compiled::List { collections, opts })
}

// ── SELECT projection / filter / literals ───────────────────────────────────────────

fn compile_projection(select: &Selection) -> Projection {
    match select {
        Selection::All => Projection::All,
        Selection::AllExcept(fields) => Projection::Exclude(fields.clone()),
        Selection::Fields(fields) => Projection::Include(fields.clone()),
        // An aggregate projection never reaches a hit-returning arm, so it has no
        // `Projection`; the aggregate dispatch reads its sums directly.
        Selection::Aggregates { .. } => Projection::All,
    }
}

/// `HybridOpts`/`AggregateOpts` carry no projection knob (§7.12: no new execution logic),
/// so a query dispatching there must ask for every column.
fn require_select_all(stmt: &Statement, why: &str, section: &'static str) -> Result<(), SqlError> {
    if stmt.select != Selection::All {
        return Err(SqlError::new(
            stmt.at,
            format!("SELECT columns are not supported {why}"),
            section,
        ));
    }
    Ok(())
}

/// A top-level `AND` unwraps directly into `Filter`'s own AND list rather than nesting one
/// `Predicate::All` inside it, so `WHERE a=1 AND b=2` compiles exactly as a typed caller
/// would write `Filter(vec![Eq(..), Eq(..)])`.
fn compile_filter(pred: Option<&PredNode>) -> Result<Filter, SqlError> {
    let Some(p) = pred else {
        return Ok(Filter::default());
    };
    let preds = match p {
        PredNode::All(items) => items
            .iter()
            .map(compile_predicate)
            .collect::<Result<_, _>>()?,
        other => vec![compile_predicate(other)?],
    };
    Ok(Filter(preds))
}

fn compile_predicate(p: &PredNode) -> Result<Predicate, SqlError> {
    Ok(match p {
        PredNode::Any(items) => Predicate::Any(compile_all(items)?),
        PredNode::All(items) => Predicate::All(compile_all(items)?),
        PredNode::Not(inner) => Predicate::Not(Box::new(compile_predicate(inner)?)),
        PredNode::Cmp { field, op, value } => {
            let v = compile_lit(value);
            match op {
                CmpOp::Eq => Predicate::Eq(field.clone(), v),
                CmpOp::Ne => Predicate::Ne(field.clone(), v),
                CmpOp::Lt => Predicate::Lt(field.clone(), v),
                CmpOp::Le => Predicate::Le(field.clone(), v),
                CmpOp::Gt => Predicate::Gt(field.clone(), v),
                CmpOp::Ge => Predicate::Ge(field.clone(), v),
            }
        }
        PredNode::In {
            field,
            negate,
            values,
        } => {
            let vs = values.iter().map(compile_lit).collect();
            if *negate {
                Predicate::NotIn(field.clone(), vs)
            } else {
                Predicate::In(field.clone(), vs)
            }
        }
        PredNode::Like {
            field,
            negate,
            pattern,
        } => {
            let g = Predicate::Glob(field.clone(), pattern.clone());
            if *negate {
                Predicate::Not(Box::new(g))
            } else {
                g
            }
        }
        PredNode::ILike { field, pattern } => Predicate::IGlob(field.clone(), pattern.clone()),
        PredNode::Regex { field, pattern } => Predicate::Regex(field.clone(), pattern.clone()),
        PredNode::Contains { field, value } => {
            Predicate::Contains(field.clone(), compile_lit(value))
        }
        PredNode::NotContains { field, value } => {
            Predicate::NotContains(field.clone(), compile_lit(value))
        }
        PredNode::ContainsAny { field, values } => {
            Predicate::ContainsAny(field.clone(), values.iter().map(compile_lit).collect())
        }
        PredNode::Fuzzy {
            field,
            needle,
            edits,
            at,
        } => {
            let edits = usize::try_from(*edits).map_err(|_| {
                SqlError::new(*at, "fuzzy edit count must not be negative", SEC_TEXT)
            })?;
            Predicate::Fuzzy(field.clone(), needle.clone(), edits)
        }
        PredNode::MatchAll { field, query } => {
            Predicate::ContainsAllTokens(field.clone(), query.clone())
        }
        PredNode::MatchAny { field, query } => {
            Predicate::ContainsAnyToken(field.clone(), query.clone())
        }
        PredNode::Phrase { field, query } => {
            Predicate::ContainsTokenSequence(field.clone(), query.clone())
        }
    })
}

fn compile_all(items: &[PredNode]) -> Result<Vec<Predicate>, SqlError> {
    items.iter().map(compile_predicate).collect()
}

fn compile_lit(l: &Lit) -> Value {
    match l {
        Lit::Int(n) => Value::Int(*n),
        Lit::DateTime(ms) => Value::DateTime(*ms),
        Lit::Float(f) => Value::Float(*f),
        Lit::Str(s) => Value::Str(s.clone()),
        Lit::Bool(b) => Value::Bool(*b),
        Lit::Null => Value::Null,
    }
}

fn compile_clauses(clauses: &[ClauseLit]) -> FtsQuery {
    FtsQuery::multi(clauses.iter().map(|c| {
        let clause = FtsClause::new(c.field.clone(), c.text.clone());
        if c.prefix { clause.prefix() } else { clause }
    }))
}

fn compile_decay(d: &DecayLit) -> Decay {
    let mut decay = Decay::new(d.field.clone(), d.origin, d.scale);
    if let Some(x) = d.decay {
        decay = decay.decay(x);
    }
    if let Some(x) = d.lambda {
        decay = decay.lambda(x);
    }
    decay
}

fn to_usize(n: i64, at: usize, section: &'static str) -> Result<usize, SqlError> {
    usize::try_from(n).map_err(|_| SqlError::new(at, "expected a non-negative integer", section))
}

// ── WITH (...) — every key, validated against what the dispatch's opts can hold ─────

/// Which `SPEC.md` §7 section a `WITH` key's own rows live under (root blueprint's table).
fn with_key_section(key: &str) -> &'static str {
    match key {
        "annotations" => SEC_ANNOTATE,
        "plan" => SEC_PLAN,
        "context" => SEC_EXPAND,
        "diversity" | "limit_per" => SEC_AGG,
        _ => SEC_RANK, // exact, min_score, rerank, candidates, rrf_k, weights
    }
}

fn contradiction(w: &WithOption, why: &str) -> SqlError {
    SqlError::new(
        w.name_at,
        format!("'{}' {why}", w.name),
        with_key_section(&w.name.to_ascii_lowercase()),
    )
}

fn bad_value(w: &WithOption, expected: &str) -> SqlError {
    SqlError::new(
        w.name_at,
        format!("'{}' requires {expected}", w.name),
        with_key_section(&w.name.to_ascii_lowercase()),
    )
}

fn require_flag(w: &WithOption) -> Result<(), SqlError> {
    match &w.value {
        OptValue::Flag => Ok(()),
        _ => Err(contradiction(w, "takes no value")),
    }
}

fn scalar_f32(w: &WithOption) -> Result<f32, SqlError> {
    match &w.value {
        OptValue::Scalar(AnyLit::Val(Lit::Int(n))) => Ok(*n as f32),
        OptValue::Scalar(AnyLit::Val(Lit::Float(f))) => Ok(*f as f32),
        _ => Err(bad_value(w, "a number")),
    }
}

fn scalar_usize(w: &WithOption) -> Result<usize, SqlError> {
    match &w.value {
        OptValue::Scalar(AnyLit::Val(Lit::Int(n))) if *n >= 0 => Ok(*n as usize),
        _ => Err(bad_value(w, "a non-negative integer")),
    }
}

fn list_args(w: &WithOption) -> Result<&[OptArg], SqlError> {
    match &w.value {
        OptValue::List(args) => Ok(args),
        _ => Err(bad_value(w, "a parenthesized argument list")),
    }
}

fn keyed<'a>(args: &'a [OptArg], key: &str) -> Option<&'a AnyLit> {
    args.iter().find_map(|a| match a {
        OptArg::Keyed(name, v) if name.eq_ignore_ascii_case(key) => Some(v),
        _ => None,
    })
}

fn bare_at(args: &[OptArg], index: usize) -> Option<&AnyLit> {
    args.iter()
        .filter_map(|a| match a {
            OptArg::Bare(v) => Some(v),
            OptArg::Keyed(..) => None,
        })
        .nth(index)
}

fn lit_i64(v: &AnyLit) -> Option<i64> {
    match v {
        AnyLit::Val(Lit::Int(n)) => Some(*n),
        _ => None,
    }
}

fn lit_f32(v: &AnyLit) -> Option<f32> {
    match v {
        AnyLit::Val(Lit::Int(n)) => Some(*n as f32),
        AnyLit::Val(Lit::Float(f)) => Some(*f as f32),
        _ => None,
    }
}

/// A field name, spelled either as a bare/quoted identifier or as a string literal — the
/// `WITH` grammar's `f` placeholder accepts both (root blueprint's key table).
fn lit_field(v: &AnyLit) -> Option<String> {
    match v {
        AnyLit::Ident(s) => Some(s.clone()),
        AnyLit::Val(Lit::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

fn compile_limit_per(w: &WithOption) -> Result<LimitPer, SqlError> {
    let args = list_args(w)?;
    if args.len() != 2 {
        return Err(bad_value(w, "(field, n) with a non-negative n"));
    }
    let field = bare_at(args, 0).and_then(lit_field);
    let max = bare_at(args, 1).and_then(lit_i64);
    match (field, max) {
        (Some(field), Some(max)) if max >= 0 => Ok(LimitPer::new(field, max as usize)),
        _ => Err(bad_value(w, "(field, n) with a non-negative n")),
    }
}

fn compile_context(w: &WithOption) -> Result<Expand, SqlError> {
    let args = list_args(w)?;
    let radius = keyed(args, "radius")
        .and_then(lit_i64)
        .ok_or_else(|| bad_value(w, "at least a radius, e.g. (radius 2)"))?;
    let mut expand =
        Expand::new(usize::try_from(radius).map_err(|_| bad_value(w, "a non-negative radius"))?);
    if let Some(f) = keyed(args, "parent").and_then(lit_field) {
        expand.parent_field = f;
    }
    if let Some(f) = keyed(args, "index").and_then(lit_field) {
        expand.index_field = f;
    }
    if let Some(f) = keyed(args, "text").and_then(lit_field) {
        expand.text_field = f;
    }
    Ok(expand)
}

fn compile_rerank(w: &WithOption) -> Result<RerankOpts, SqlError> {
    let args = list_args(w)?;
    let mut opts = RerankOpts::default();
    if let Some(n) = keyed(args, "overscan").and_then(lit_i64) {
        opts.overscan = usize::try_from(n).map_err(|_| bad_value(w, "a non-negative overscan"))?;
    }
    if let Some(f) = keyed(args, "text").and_then(lit_field) {
        opts.text_attr = f;
    }
    Ok(opts)
}

fn compile_weights(w: &WithOption) -> Result<(f32, f32), SqlError> {
    let args = list_args(w)?;
    if args.len() != 2 {
        return Err(bad_value(w, "(vector_weight, text_weight) as two numbers"));
    }
    let v = bare_at(args, 0).and_then(lit_f32);
    let t = bare_at(args, 1).and_then(lit_f32);
    match (v, t) {
        (Some(v), Some(t)) => Ok((v, t)),
        _ => Err(bad_value(w, "(vector_weight, text_weight) as two numbers")),
    }
}

/// `WITH` keys shared by `Search` and `TextSearch` (both run on `SearchOpts`). `allow_plan`
/// is `false` for `TextSearch`: there is no `Nidus::text_search_with_plan` sibling to honour
/// `WITH (plan)` with, so that combination is a compile-time contradiction, not a silent no-op.
fn apply_with_search(
    opts: &mut SearchOpts,
    with: &[WithOption],
    allow_plan: bool,
) -> Result<(), SqlError> {
    for w in with {
        match w.name.to_ascii_lowercase().as_str() {
            "annotations" => {
                require_flag(w)?;
                opts.explain = true;
            }
            "plan" => {
                require_flag(w)?;
                if !allow_plan {
                    return Err(contradiction(
                        w,
                        "requires ORDER BY knn(...) or knn(...) FUSE match(...), not match(...) alone",
                    ));
                }
                opts.plan = true;
            }
            "exact" => {
                require_flag(w)?;
                opts.exact = true;
            }
            "min_score" => opts.min_score = Some(scalar_f32(w)?),
            "diversity" => opts.diversity = Some(scalar_f32(w)?),
            "limit_per" => opts.limit_per = Some(compile_limit_per(w)?),
            "context" => opts.expand = Some(compile_context(w)?),
            "rerank" => opts.rerank = Some(compile_rerank(w)?),
            "candidates" | "rrf_k" | "weights" => {
                return Err(contradiction(
                    w,
                    "requires ORDER BY knn(...) FUSE match(...)",
                ));
            }
            _ => return Err(unknown_with_key(w)),
        }
    }
    Ok(())
}

fn apply_with_hybrid(opts: &mut HybridOpts, with: &[WithOption]) -> Result<(), SqlError> {
    for w in with {
        match w.name.to_ascii_lowercase().as_str() {
            "annotations" => {
                require_flag(w)?;
                opts.explain = true;
            }
            "plan" => {
                require_flag(w)?;
                opts.plan = true;
            }
            "context" => opts.expand = Some(compile_context(w)?),
            "rerank" => opts.rerank = Some(compile_rerank(w)?),
            "candidates" => opts.candidates = scalar_usize(w)?,
            "rrf_k" => opts.rrf_k = scalar_f32(w)?,
            "weights" => {
                let (v, t) = compile_weights(w)?;
                opts.vector_weight = v;
                opts.text_weight = t;
            }
            "exact" | "min_score" | "diversity" | "limit_per" => {
                return Err(contradiction(
                    w,
                    "requires a vector-only ORDER BY knn(...), not knn(...) FUSE match(...)",
                ));
            }
            _ => return Err(unknown_with_key(w)),
        }
    }
    Ok(())
}

/// `List`/`AggregateOpts` carry none of the `WITH` table's knobs, so *any* key there is a
/// contradiction — named by `context` (why this dispatch has no such knob to set).
fn reject_all_with(with: &[WithOption], context: &str) -> Result<(), SqlError> {
    if let Some(w) = with.first() {
        return Err(contradiction(w, &format!("is not supported on {context}")));
    }
    Ok(())
}

fn unknown_with_key(w: &WithOption) -> SqlError {
    SqlError::new(
        w.name_at,
        format!("unknown WITH option '{}'", w.name),
        SEC_SYNTAX,
    )
}
