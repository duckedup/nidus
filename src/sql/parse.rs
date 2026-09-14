//! Recursive descent: tokens -> a small private syntax tree. One function per grammar
//! production (root blueprint's grammar); `compile.rs` turns the tree into `model.rs` values.

use super::MAX_NEST_DEPTH;
use super::error::{
    SEC_AGG, SEC_ANNOTATE, SEC_ARRAY, SEC_BATCH, SEC_BOOL, SEC_GLOB, SEC_RANK, SEC_REGEX,
    SEC_SYNTAX, SEC_TEXT, SqlError,
};
use super::lex::{Kind, Token, lex, unescape_string};

/// `SELECT` projection, before it becomes a [`crate::model::Projection`].
#[derive(Debug, PartialEq)]
pub(super) enum Selection {
    All,
    AllExcept(Vec<String>),
    Fields(Vec<String>),
    /// `SELECT sum(a), count(*)`: an aggregate projection. Selects the `Aggregate`
    /// dispatch whether or not a `GROUP BY` follows, so whole-scope totals are reachable.
    Aggregates {
        sums: Vec<String>,
        count: bool,
    },
}

/// A comparison-side or predicate-argument literal. Never a bare identifier — `WITH` option
/// values are the only place a bare word can stand for something other than itself, and
/// they use [`AnyLit`] instead.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Lit {
    Int(i64),
    /// `timestamp '...'` or `timestamp <millis>`: an absolute UTC instant (§3).
    DateTime(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// One node of the `WHERE` boolean tree (`predicate` in the grammar).
#[derive(Debug, PartialEq)]
pub(super) enum PredNode {
    Any(Vec<PredNode>),
    All(Vec<PredNode>),
    Not(Box<PredNode>),
    Cmp {
        field: String,
        op: CmpOp,
        value: Lit,
    },
    In {
        field: String,
        negate: bool,
        values: Vec<Lit>,
    },
    Like {
        field: String,
        negate: bool,
        pattern: String,
    },
    ILike {
        field: String,
        pattern: String,
    },
    Regex {
        field: String,
        pattern: String,
    },
    Contains {
        field: String,
        value: Lit,
    },
    NotContains {
        field: String,
        value: Lit,
    },
    ContainsAny {
        field: String,
        values: Vec<Lit>,
    },
    Fuzzy {
        field: String,
        needle: String,
        edits: i64,
        at: usize,
    },
    MatchAll {
        field: String,
        query: String,
    },
    MatchAny {
        field: String,
        query: String,
    },
    Phrase {
        field: String,
        query: String,
    },
}

/// One clause of a `match(...)` ranking or `WITH (context = ...)`'s sibling — see [`ClauseLit`].
#[derive(Debug, PartialEq)]
pub(super) struct ClauseLit {
    pub field: String,
    pub text: String,
    pub prefix: bool,
}

/// The `decay_tail` production: `- decay(field, origin, scale [, decay [, lambda]])`.
#[derive(Debug, PartialEq)]
pub(super) struct DecayLit {
    pub field: String,
    pub origin: i64,
    pub scale: i64,
    pub decay: Option<f32>,
    pub lambda: Option<f32>,
}

/// The `ORDER BY` head — the sole input to dispatch (root blueprint's dispatch table).
#[derive(Debug, PartialEq)]
pub(super) enum Ranking {
    Knn {
        vector: Vec<f32>,
        decay: Option<DecayLit>,
        at: usize,
    },
    Match {
        clauses: Vec<ClauseLit>,
        at: usize,
    },
    Hybrid {
        vector: Vec<f32>,
        clauses: Vec<ClauseLit>,
        at: usize,
    },
    Field {
        field: String,
        desc: bool,
    },
}

/// A `WITH` option value that is not a bare literal — a field name or other bare word,
/// distinguished from [`Lit`] so ordinary predicate/ranking literals can never be one.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum AnyLit {
    Val(Lit),
    Ident(String),
}

/// One argument inside a parenthesized `WITH` option value: a bare value, or a `keyword
/// value` pair (`radius 5`, `parent "pid"`) — see root blueprint's `WITH` key table.
#[derive(Debug, PartialEq)]
pub(super) enum OptArg {
    Bare(AnyLit),
    Keyed(String, AnyLit),
}

#[derive(Debug, PartialEq)]
pub(super) enum OptValue {
    /// No `=` at all — `WITH (annotations, plan, exact)`.
    Flag,
    Scalar(AnyLit),
    List(Vec<OptArg>),
}

#[derive(Debug, PartialEq)]
pub(super) struct WithOption {
    pub name: String,
    pub name_at: usize,
    pub value: OptValue,
}

/// One `SELECT ... FROM ...` statement, fully parsed and ready for `compile.rs`.
#[derive(Debug, PartialEq)]
pub(super) struct Statement {
    /// Byte offset of the leading `SELECT` — the position a whole-statement contradiction
    /// (no single token to blame) reports, e.g. "GROUP BY" columns.
    pub at: usize,
    pub select: Selection,
    /// Concrete collection names; empty means every collection (`FROM *` or an omitted
    /// `FROM` — recommendation in the root blueprint's open questions).
    pub from: Vec<String>,
    pub filter: Option<PredNode>,
    pub group_by: Option<String>,
    pub sums: Vec<String>,
    pub order: Option<Ranking>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
    pub with: Vec<WithOption>,
}

/// Parse a whole script: one or more `;`-separated statements (§7.9), an optional trailing
/// `;`. At least one statement is required — an empty or all-whitespace script is an error.
pub(super) fn parse(src: &str) -> Result<Vec<Statement>, SqlError> {
    let toks = lex(src)?;
    let mut p = Parser {
        src,
        toks,
        pos: 0,
        depth: 0,
    };
    p.parse_script()
}

struct Parser<'a> {
    src: &'a str,
    toks: Vec<Token>,
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn cur(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn kind(&self) -> Kind {
        self.cur().kind
    }

    fn at(&self) -> usize {
        self.cur().at
    }

    fn text(&self, tok: &Token) -> &'a str {
        &self.src[tok.text.clone()]
    }

    fn bump(&mut self) -> Token {
        let tok = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        tok
    }

    fn eat(&mut self, kind: Kind) -> bool {
        if self.kind() == kind {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: Kind, what: &str, section: &'static str) -> Result<Token, SqlError> {
        if self.kind() == kind {
            Ok(self.bump())
        } else {
            Err(self.err(format!("expected {what}"), section))
        }
    }

    fn err(&self, message: impl Into<String>, section: &'static str) -> SqlError {
        SqlError::new(self.at(), message, section)
    }

    /// Lowercased text of the current token, when it is a bare identifier — the disambiguation
    /// point for function names, `WITH` keys, and the `knn`/`match`/`fuse`/`decay` markers that
    /// the lexer deliberately does not tokenize as keywords.
    fn ident_lc(&self) -> Option<String> {
        (self.kind() == Kind::Ident).then(|| self.text(self.cur()).to_ascii_lowercase())
    }

    fn peek2_kind(&self) -> Kind {
        self.toks.get(self.pos + 1).map_or(Kind::Eof, |t| t.kind)
    }

    fn expect_ident_lc(&mut self, word: &str, section: &'static str) -> Result<(), SqlError> {
        if self.ident_lc().as_deref() == Some(word) {
            self.bump();
            Ok(())
        } else {
            Err(self.err(format!("expected '{word}'"), section))
        }
    }

    // ── script / statement ──────────────────────────────────────────────────────

    fn parse_script(&mut self) -> Result<Vec<Statement>, SqlError> {
        let mut out = Vec::new();
        loop {
            if self.kind() == Kind::Eof {
                break;
            }
            out.push(self.parse_statement()?);
            if self.eat(Kind::Semicolon) {
                continue;
            }
            break;
        }
        if self.kind() != Kind::Eof {
            return Err(self.err("expected ';' between statements", SEC_BATCH));
        }
        if out.is_empty() {
            return Err(SqlError::new(0, "empty query", SEC_SYNTAX));
        }
        Ok(out)
    }

    fn parse_statement(&mut self) -> Result<Statement, SqlError> {
        let at = self.at();
        self.expect(Kind::Select, "'SELECT'", SEC_SYNTAX)?;
        let select = self.parse_selection()?;
        self.expect(Kind::From, "'FROM'", SEC_SYNTAX)?;
        let from = self.parse_scope()?;

        let filter = if self.eat(Kind::Where) {
            Some(self.parse_predicate()?)
        } else {
            None
        };

        let mut group_by = None;
        let mut sums = Vec::new();
        if self.eat(Kind::Group) {
            self.expect(Kind::By, "'BY'", SEC_AGG)?;
            group_by = Some(self.parse_field_sec(SEC_AGG)?);
            while self.eat(Kind::Comma) {
                self.expect(Kind::Sum, "'SUM'", SEC_AGG)?;
                self.expect(Kind::LParen, "'('", SEC_AGG)?;
                sums.push(self.parse_field_sec(SEC_AGG)?);
                self.expect(Kind::RParen, "')'", SEC_AGG)?;
            }
        }

        let order = if self.eat(Kind::Order) {
            self.expect(Kind::By, "'BY'", SEC_SYNTAX)?;
            Some(self.parse_ranking()?)
        } else {
            None
        };

        let limit = if self.eat(Kind::Limit) {
            Some(self.parse_int_lit()?)
        } else {
            None
        };
        let offset = if self.eat(Kind::Offset) {
            Some(self.parse_int_lit()?)
        } else {
            None
        };

        let with = if self.eat(Kind::With) {
            self.parse_with_options()?
        } else {
            Vec::new()
        };

        Ok(Statement {
            at,
            select,
            from,
            filter,
            group_by,
            sums,
            order,
            limit,
            offset,
            with,
        })
    }

    fn parse_selection(&mut self) -> Result<Selection, SqlError> {
        if self.eat(Kind::Star) {
            if self.kind() == Kind::Except {
                self.bump();
                self.expect(Kind::LParen, "'('", SEC_SYNTAX)?;
                let fields = self.parse_field_list()?;
                self.expect(Kind::RParen, "')'", SEC_SYNTAX)?;
                return Ok(Selection::AllExcept(fields));
            }
            return Ok(Selection::All);
        }
        if self.kind() == Kind::Sum || self.at_count_call() {
            return self.parse_aggregate_selection();
        }
        Ok(Selection::Fields(self.parse_field_list()?))
    }

    /// `count` is deliberately NOT a keyword: an attribute may be named `count`, and
    /// `SUM(count)` must keep working. Only `count` directly before `(` is the function.
    fn at_count_call(&self) -> bool {
        self.ident_lc().as_deref() == Some("count") && self.peek2_kind() == Kind::LParen
    }

    /// `sum(a), sum(b), count(*)` in any order. `count(*)` is accepted and ignored:
    /// `Aggregation::count` is always returned, so it exists only to read naturally.
    fn parse_aggregate_selection(&mut self) -> Result<Selection, SqlError> {
        let mut sums = Vec::new();
        let mut count = false;
        loop {
            match self.kind() {
                Kind::Sum => {
                    self.bump();
                    self.expect(Kind::LParen, "'('", SEC_AGG)?;
                    sums.push(self.parse_field_sec(SEC_AGG)?);
                    self.expect(Kind::RParen, "')'", SEC_AGG)?;
                }
                _ if self.at_count_call() => {
                    self.bump();
                    self.expect(Kind::LParen, "'('", SEC_AGG)?;
                    self.expect(Kind::Star, "'*'", SEC_AGG)?;
                    self.expect(Kind::RParen, "')'", SEC_AGG)?;
                    count = true;
                }
                _ => return Err(self.err("expected sum(field) or count(*)", SEC_AGG)),
            }
            if !self.eat(Kind::Comma) {
                return Ok(Selection::Aggregates { sums, count });
            }
        }
    }

    fn parse_field_list(&mut self) -> Result<Vec<String>, SqlError> {
        let mut out = vec![self.parse_field()?];
        while self.eat(Kind::Comma) {
            out.push(self.parse_field()?);
        }
        Ok(out)
    }

    /// `scope := '*' | ident (',' ident)*`. `*` returns empty — see [`Statement::from`].
    fn parse_scope(&mut self) -> Result<Vec<String>, SqlError> {
        if self.eat(Kind::Star) {
            return Ok(Vec::new());
        }
        self.parse_field_list()
    }

    fn parse_field(&mut self) -> Result<String, SqlError> {
        self.parse_field_sec(SEC_SYNTAX)
    }

    /// Like [`Self::parse_field`], but a missing field name is tagged the caller's own
    /// section instead of the generic syntax one (e.g. `GROUP BY` -> §7.7).
    fn parse_field_sec(&mut self, section: &'static str) -> Result<String, SqlError> {
        match self.kind() {
            Kind::Ident | Kind::QuotedIdent => {
                let tok = self.bump();
                Ok(self.text(&tok).to_string())
            }
            _ => Err(self.err("expected a field name", section)),
        }
    }

    // ── predicates: predicate := or_expr, precedence OR < AND < NOT < primary ──────

    fn parse_predicate(&mut self) -> Result<PredNode, SqlError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<PredNode, SqlError> {
        let mut items = vec![self.parse_and()?];
        while self.eat(Kind::Or) {
            items.push(self.parse_and()?);
        }
        Ok(if items.len() == 1 {
            items.pop().expect("just pushed one")
        } else {
            PredNode::Any(items)
        })
    }

    fn parse_and(&mut self) -> Result<PredNode, SqlError> {
        let mut items = vec![self.parse_not()?];
        while self.eat(Kind::And) {
            items.push(self.parse_not()?);
        }
        Ok(if items.len() == 1 {
            items.pop().expect("just pushed one")
        } else {
            PredNode::All(items)
        })
    }

    fn parse_not(&mut self) -> Result<PredNode, SqlError> {
        if self.eat(Kind::Not) {
            Ok(PredNode::Not(Box::new(self.parse_primary()?)))
        } else {
            self.parse_primary()
        }
    }

    /// `primary := '(' predicate ')' | comparison | fn_predicate`. The depth cap guards
    /// **only** this recursive `(...)` entry — the real, unbounded-recursion risk on
    /// attacker-supplied text (root blueprint: `serde_json`'s 128-cap has no SQL analogue).
    fn parse_primary(&mut self) -> Result<PredNode, SqlError> {
        if self.kind() == Kind::LParen {
            self.depth += 1;
            if self.depth > MAX_NEST_DEPTH {
                return Err(self.err(
                    format!("predicate nesting exceeds {MAX_NEST_DEPTH} levels"),
                    SEC_BOOL,
                ));
            }
            self.bump();
            let inner = self.parse_or()?;
            self.expect(Kind::RParen, "')'", SEC_BOOL)?;
            self.depth -= 1;
            return Ok(inner);
        }
        if let Some(name) = self.ident_lc()
            && self.peek2_kind() == Kind::LParen
            && matches!(
                name.as_str(),
                "contains"
                    | "not_contains"
                    | "contains_any"
                    | "fuzzy"
                    | "match_all"
                    | "match_any"
                    | "phrase"
            )
        {
            return self.parse_fn_predicate(&name);
        }
        self.parse_comparison()
    }

    fn parse_fn_predicate(&mut self, name: &str) -> Result<PredNode, SqlError> {
        self.bump(); // the function name
        self.bump(); // '('
        let section = match name {
            "contains" | "not_contains" | "contains_any" => SEC_ARRAY,
            _ => SEC_TEXT,
        };
        let field = self.parse_field_sec(section)?;
        self.expect(Kind::Comma, "','", section)?;
        let node = match name {
            "contains" => PredNode::Contains {
                field,
                value: self.parse_literal()?,
            },
            "not_contains" => PredNode::NotContains {
                field,
                value: self.parse_literal()?,
            },
            "contains_any" => {
                let mut values = vec![self.parse_literal()?];
                while self.eat(Kind::Comma) {
                    values.push(self.parse_literal()?);
                }
                PredNode::ContainsAny { field, values }
            }
            "fuzzy" => {
                let needle = self.parse_string_lit_sec(section)?;
                self.expect(Kind::Comma, "','", section)?;
                let at = self.at();
                let edits = self.parse_int_lit()?;
                PredNode::Fuzzy {
                    field,
                    needle,
                    edits,
                    at,
                }
            }
            "match_all" => PredNode::MatchAll {
                field,
                query: self.parse_string_lit_sec(section)?,
            },
            "match_any" => PredNode::MatchAny {
                field,
                query: self.parse_string_lit_sec(section)?,
            },
            "phrase" => PredNode::Phrase {
                field,
                query: self.parse_string_lit_sec(section)?,
            },
            _ => unreachable!("guarded by the caller's match on `name`"),
        };
        self.expect(Kind::RParen, "')'", section)?;
        Ok(node)
    }

    fn parse_comparison(&mut self) -> Result<PredNode, SqlError> {
        let field = self.parse_field()?;
        let op = match self.kind() {
            Kind::Eq => Some(CmpOp::Eq),
            Kind::Ne => Some(CmpOp::Ne),
            Kind::Lt => Some(CmpOp::Lt),
            Kind::Le => Some(CmpOp::Le),
            Kind::Gt => Some(CmpOp::Gt),
            Kind::Ge => Some(CmpOp::Ge),
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let value = self.parse_literal()?;
            return Ok(PredNode::Cmp { field, op, value });
        }
        if self.eat(Kind::In) {
            let values = self.parse_paren_lit_list(SEC_SYNTAX)?;
            return Ok(PredNode::In {
                field,
                negate: false,
                values,
            });
        }
        if self.eat(Kind::Like) {
            let pattern = self.parse_string_lit_sec(SEC_GLOB)?;
            return Ok(PredNode::Like {
                field,
                negate: false,
                pattern,
            });
        }
        if self.eat(Kind::Ilike) {
            let pattern = self.parse_string_lit_sec(SEC_GLOB)?;
            return Ok(PredNode::ILike { field, pattern });
        }
        if self.eat(Kind::Tilde) {
            let pattern = self.parse_string_lit_sec(SEC_REGEX)?;
            return Ok(PredNode::Regex { field, pattern });
        }
        if self.eat(Kind::Not) {
            if self.eat(Kind::In) {
                let values = self.parse_paren_lit_list(SEC_SYNTAX)?;
                return Ok(PredNode::In {
                    field,
                    negate: true,
                    values,
                });
            }
            if self.eat(Kind::Like) {
                let pattern = self.parse_string_lit_sec(SEC_GLOB)?;
                return Ok(PredNode::Like {
                    field,
                    negate: true,
                    pattern,
                });
            }
            return Err(self.err("expected 'IN' or 'LIKE' after 'NOT'", SEC_SYNTAX));
        }
        Err(self.err(
            "expected a comparison operator, 'IN', 'LIKE', 'ILIKE', or '~'",
            SEC_SYNTAX,
        ))
    }

    fn parse_paren_lit_list(&mut self, section: &'static str) -> Result<Vec<Lit>, SqlError> {
        self.expect(Kind::LParen, "'('", section)?;
        let mut out = vec![self.parse_literal()?];
        while self.eat(Kind::Comma) {
            out.push(self.parse_literal()?);
        }
        self.expect(Kind::RParen, "')'", section)?;
        Ok(out)
    }

    // ── ranking: knn(...) [- decay(...)] [FUSE match(...)] | match(...) | field [ASC|DESC] ──

    fn parse_ranking(&mut self) -> Result<Ranking, SqlError> {
        let at = self.at();
        if self.ident_lc().as_deref() == Some("knn") && self.peek2_kind() == Kind::LParen {
            self.bump();
            self.bump();
            let vector = self.parse_vector()?;
            self.expect(Kind::RParen, "')'", SEC_RANK)?;

            let decay = if self.eat(Kind::Minus) {
                Some(self.parse_decay_tail()?)
            } else {
                None
            };

            if self.kind() == Kind::Fuse {
                if decay.is_some() {
                    return Err(self.err(
                        "decay and FUSE cannot combine on one knn(...) ranking",
                        SEC_RANK,
                    ));
                }
                self.bump();
                self.expect_ident_lc("match", SEC_RANK)?;
                self.expect(Kind::LParen, "'('", SEC_RANK)?;
                let clauses = self.parse_clauses()?;
                self.expect(Kind::RParen, "')'", SEC_RANK)?;
                return Ok(Ranking::Hybrid {
                    vector,
                    clauses,
                    at,
                });
            }
            return Ok(Ranking::Knn { vector, decay, at });
        }
        if self.ident_lc().as_deref() == Some("match") && self.peek2_kind() == Kind::LParen {
            self.bump();
            self.bump();
            let clauses = self.parse_clauses()?;
            self.expect(Kind::RParen, "')'", SEC_RANK)?;
            return Ok(Ranking::Match { clauses, at });
        }
        let field = self.parse_field()?;
        let desc = if self.eat(Kind::Desc) {
            true
        } else {
            self.eat(Kind::Asc);
            false
        };
        Ok(Ranking::Field { field, desc })
    }

    fn parse_vector(&mut self) -> Result<Vec<f32>, SqlError> {
        self.expect(Kind::LBracket, "'['", SEC_RANK)?;
        let mut out = Vec::new();
        if self.kind() != Kind::RBracket {
            out.push(self.parse_number_f32()?);
            while self.eat(Kind::Comma) {
                out.push(self.parse_number_f32()?);
            }
        }
        self.expect(Kind::RBracket, "']'", SEC_RANK)?;
        Ok(out)
    }

    fn parse_decay_tail(&mut self) -> Result<DecayLit, SqlError> {
        let section = SEC_RANK;
        self.expect_ident_lc("decay", section)?;
        self.expect(Kind::LParen, "'('", section)?;
        let field = self.parse_field_sec(section)?;
        self.expect(Kind::Comma, "','", section)?;
        let origin = self.parse_int_lit()?;
        self.expect(Kind::Comma, "','", section)?;
        let scale = self.parse_int_lit()?;
        let mut decay = None;
        let mut lambda = None;
        if self.eat(Kind::Comma) {
            decay = Some(self.parse_float_lit()? as f32);
            if self.eat(Kind::Comma) {
                lambda = Some(self.parse_float_lit()? as f32);
            }
        }
        self.expect(Kind::RParen, "')'", section)?;
        Ok(DecayLit {
            field,
            origin,
            scale,
            decay,
            lambda,
        })
    }

    fn parse_clauses(&mut self) -> Result<Vec<ClauseLit>, SqlError> {
        let section = SEC_RANK;
        let mut out = Vec::new();
        loop {
            let field = self.parse_field_sec(section)?;
            self.expect(Kind::Comma, "','", section)?;
            let text = self.parse_string_lit_sec(section)?;
            let prefix = self.eat(Kind::Prefix);
            out.push(ClauseLit {
                field,
                text,
                prefix,
            });
            if !self.eat(Kind::Comma) {
                break;
            }
        }
        Ok(out)
    }

    // ── WITH (...) options ───────────────────────────────────────────────────────

    fn parse_with_options(&mut self) -> Result<Vec<WithOption>, SqlError> {
        let section = SEC_ANNOTATE;
        self.expect(Kind::LParen, "'('", section)?;
        let mut out = Vec::new();
        loop {
            let name_at = self.at();
            let name = match self.kind() {
                Kind::Ident => {
                    let tok = self.bump();
                    self.text(&tok).to_string()
                }
                _ => return Err(self.err("expected a WITH option name", section)),
            };
            let value = if self.eat(Kind::Eq) {
                self.parse_with_value()?
            } else {
                OptValue::Flag
            };
            out.push(WithOption {
                name,
                name_at,
                value,
            });
            if !self.eat(Kind::Comma) {
                break;
            }
        }
        self.expect(Kind::RParen, "')'", section)?;
        Ok(out)
    }

    fn parse_with_value(&mut self) -> Result<OptValue, SqlError> {
        if self.kind() == Kind::LParen {
            self.bump();
            let mut args = Vec::new();
            if self.kind() != Kind::RParen {
                args.push(self.parse_with_arg()?);
                while self.eat(Kind::Comma) {
                    args.push(self.parse_with_arg()?);
                }
            }
            self.expect(Kind::RParen, "')'", SEC_ANNOTATE)?;
            Ok(OptValue::List(args))
        } else {
            Ok(OptValue::Scalar(self.parse_any_lit()?))
        }
    }

    /// One list argument: a bare value, or `keyword value` when the first token is an
    /// identifier immediately followed by another value (not a `,` or `)`) — see
    /// `WithOption`'s doc and the root blueprint's `WITH` key table (`radius n`, `parent f`).
    fn parse_with_arg(&mut self) -> Result<OptArg, SqlError> {
        let v1 = self.parse_any_lit()?;
        if let AnyLit::Ident(name) = &v1
            && !matches!(self.kind(), Kind::Comma | Kind::RParen)
        {
            let v2 = self.parse_any_lit()?;
            return Ok(OptArg::Keyed(name.clone(), v2));
        }
        Ok(OptArg::Bare(v1))
    }

    fn parse_any_lit(&mut self) -> Result<AnyLit, SqlError> {
        match self.kind() {
            Kind::Ident | Kind::QuotedIdent => {
                let tok = self.bump();
                Ok(AnyLit::Ident(self.text(&tok).to_string()))
            }
            _ => Ok(AnyLit::Val(self.parse_literal()?)),
        }
    }

    // ── literals ─────────────────────────────────────────────────────────────────

    fn parse_literal(&mut self) -> Result<Lit, SqlError> {
        match self.kind() {
            Kind::Str => {
                let raw = self.text(self.cur()).to_string();
                self.bump();
                Ok(Lit::Str(unescape_string(&raw)))
            }
            Kind::True => {
                self.bump();
                Ok(Lit::Bool(true))
            }
            Kind::False => {
                self.bump();
                Ok(Lit::Bool(false))
            }
            Kind::Null => {
                self.bump();
                Ok(Lit::Null)
            }
            _ if self.at_timestamp_lit() => self.parse_timestamp_lit(),
            Kind::Minus | Kind::Int | Kind::Float => self.parse_signed_number(),
            _ => Err(self.err("expected a value", SEC_SYNTAX)),
        }
    }

    /// `timestamp` is NOT a keyword: an attribute may be named `timestamp`, and
    /// `WHERE timestamp > 5` must keep working. Only `timestamp` immediately before a
    /// string or number is the literal prefix.
    fn at_timestamp_lit(&self) -> bool {
        self.ident_lc().as_deref() == Some("timestamp")
            && matches!(
                self.peek2_kind(),
                Kind::Str | Kind::Int | Kind::Float | Kind::Minus
            )
    }

    /// `timestamp '2026-09-13T00:00:00Z'` or `timestamp 1700000000000` (epoch millis).
    fn parse_timestamp_lit(&mut self) -> Result<Lit, SqlError> {
        let at = self.at();
        self.bump();
        match self.kind() {
            Kind::Str => {
                let raw = self.text(self.cur()).to_string();
                self.bump();
                let text = unescape_string(&raw);
                super::datetime::iso8601_to_millis(&text)
                    .map(Lit::DateTime)
                    .ok_or_else(|| {
                        SqlError::new(
                            at,
                            "timestamp wants an ISO-8601 UTC instant, e.g. '2026-09-13T00:00:00Z'",
                            SEC_SYNTAX,
                        )
                    })
            }
            Kind::Minus | Kind::Int | Kind::Float => match self.parse_signed_number()? {
                Lit::Int(n) => Ok(Lit::DateTime(n)),
                _ => Err(SqlError::new(
                    at,
                    "timestamp wants whole milliseconds",
                    SEC_SYNTAX,
                )),
            },
            _ => Err(SqlError::new(
                at,
                "timestamp wants a quoted instant or epoch milliseconds",
                SEC_SYNTAX,
            )),
        }
    }

    fn parse_signed_number(&mut self) -> Result<Lit, SqlError> {
        let negative = self.eat(Kind::Minus);
        match self.kind() {
            Kind::Int => {
                let text = self.text(self.cur()).to_string();
                self.bump();
                let n: i64 = text
                    .parse()
                    .map_err(|_| self.err("invalid integer literal", SEC_SYNTAX))?;
                Ok(Lit::Int(if negative { -n } else { n }))
            }
            Kind::Float => {
                let text = self.text(self.cur()).to_string();
                self.bump();
                let f: f64 = text
                    .parse()
                    .map_err(|_| self.err("invalid float literal", SEC_SYNTAX))?;
                Ok(Lit::Float(if negative { -f } else { f }))
            }
            _ => Err(self.err("expected a number", SEC_SYNTAX)),
        }
    }

    /// A string literal, tagging a missing/wrong-shaped value with the caller's own section
    /// (`LIKE` -> §7.1, `~` -> §7.5, a function argument -> its own predicate family).
    fn parse_string_lit_sec(&mut self, section: &'static str) -> Result<String, SqlError> {
        match self.kind() {
            Kind::Str => match self.parse_literal()? {
                Lit::Str(s) => Ok(s),
                _ => unreachable!("Kind::Str always parses to Lit::Str"),
            },
            _ => Err(self.err("expected a string literal", section)),
        }
    }

    fn parse_int_lit(&mut self) -> Result<i64, SqlError> {
        match self.parse_signed_number()? {
            Lit::Int(n) => Ok(n),
            Lit::Float(_) => Err(self.err("expected an integer, found a float", SEC_SYNTAX)),
            _ => unreachable!("parse_signed_number only returns Int/Float"),
        }
    }

    fn parse_float_lit(&mut self) -> Result<f64, SqlError> {
        match self.parse_signed_number()? {
            Lit::Int(n) => Ok(n as f64),
            Lit::Float(f) => Ok(f),
            _ => unreachable!("parse_signed_number only returns Int/Float"),
        }
    }

    fn parse_number_f32(&mut self) -> Result<f32, SqlError> {
        Ok(self.parse_float_lit()? as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(sql: &str) -> Statement {
        let mut stmts = parse(sql).unwrap();
        assert_eq!(stmts.len(), 1, "expected exactly one statement");
        stmts.remove(0)
    }

    // ── projection / scope ───────────────────────────────────────────────────────

    #[test]
    fn select_star_from_star() {
        let s = one("SELECT * FROM *");
        assert_eq!(s.select, Selection::All);
        assert_eq!(s.from, Vec::<String>::new());
    }

    #[test]
    fn select_fields_from_named_collections() {
        let s = one("SELECT a, b.c FROM docs, notes");
        assert_eq!(s.select, Selection::Fields(vec!["a".into(), "b.c".into()]));
        assert_eq!(s.from, vec!["docs".to_string(), "notes".to_string()]);
    }

    #[test]
    fn select_star_except() {
        let s = one("SELECT * EXCEPT (body, embedding) FROM docs");
        assert_eq!(
            s.select,
            Selection::AllExcept(vec!["body".into(), "embedding".into()])
        );
    }

    #[test]
    fn quoted_field_name_with_a_space() {
        let s = one(r#"SELECT "a field" FROM docs"#);
        assert_eq!(s.select, Selection::Fields(vec!["a field".into()]));
    }

    // ── WHERE: every Predicate shape ─────────────────────────────────────────────

    #[test]
    fn where_eq_ne_range_operators() {
        let s = one(
            "SELECT * FROM docs WHERE a = 1 AND b != 2.5 AND c < 3 AND d <= 4 AND e > 5 AND f >= 6",
        );
        let PredNode::All(items) = s.filter.unwrap() else {
            panic!("expected All")
        };
        assert_eq!(items.len(), 6);
        assert_eq!(
            items[0],
            PredNode::Cmp {
                field: "a".into(),
                op: CmpOp::Eq,
                value: Lit::Int(1)
            }
        );
        assert_eq!(
            items[1],
            PredNode::Cmp {
                field: "b".into(),
                op: CmpOp::Ne,
                value: Lit::Float(2.5)
            }
        );
    }

    #[test]
    fn where_in_and_not_in() {
        let s = one("SELECT * FROM docs WHERE lang IN ('rust', 'go') AND status NOT IN (1, 2)");
        let PredNode::All(items) = s.filter.unwrap() else {
            panic!("expected All")
        };
        assert_eq!(
            items[0],
            PredNode::In {
                field: "lang".into(),
                negate: false,
                values: vec![Lit::Str("rust".into()), Lit::Str("go".into())],
            }
        );
        assert_eq!(
            items[1],
            PredNode::In {
                field: "status".into(),
                negate: true,
                values: vec![Lit::Int(1), Lit::Int(2)],
            }
        );
    }

    #[test]
    fn where_like_not_like_ilike_and_regex() {
        let s = one(
            "SELECT * FROM docs WHERE a LIKE 'x*' AND b NOT LIKE 'y*' AND c ILIKE 'Z*' AND d ~ 'v[0-9]+'",
        );
        let PredNode::All(items) = s.filter.unwrap() else {
            panic!("expected All")
        };
        assert_eq!(
            items[0],
            PredNode::Like {
                field: "a".into(),
                negate: false,
                pattern: "x*".into()
            }
        );
        assert_eq!(
            items[1],
            PredNode::Like {
                field: "b".into(),
                negate: true,
                pattern: "y*".into()
            }
        );
        assert_eq!(
            items[2],
            PredNode::ILike {
                field: "c".into(),
                pattern: "Z*".into()
            }
        );
        assert_eq!(
            items[3],
            PredNode::Regex {
                field: "d".into(),
                pattern: "v[0-9]+".into()
            }
        );
    }

    #[test]
    fn where_contains_family() {
        let s = one(
            "SELECT * FROM docs WHERE contains(tags, 'rust') AND not_contains(tags, 'wip') AND contains_any(tags, 'a', 'b')",
        );
        let PredNode::All(items) = s.filter.unwrap() else {
            panic!("expected All")
        };
        assert_eq!(
            items[0],
            PredNode::Contains {
                field: "tags".into(),
                value: Lit::Str("rust".into())
            }
        );
        assert_eq!(
            items[1],
            PredNode::NotContains {
                field: "tags".into(),
                value: Lit::Str("wip".into())
            }
        );
        assert_eq!(
            items[2],
            PredNode::ContainsAny {
                field: "tags".into(),
                values: vec![Lit::Str("a".into()), Lit::Str("b".into())]
            }
        );
    }

    #[test]
    fn where_fuzzy_and_token_predicates() {
        let s = one(
            "SELECT * FROM docs WHERE fuzzy(id, 'nidus', 2) AND match_all(body, 'a b') AND match_any(body, 'c d') AND phrase(body, 'e f')",
        );
        let PredNode::All(items) = s.filter.unwrap() else {
            panic!("expected All")
        };
        assert_eq!(
            items[0],
            PredNode::Fuzzy {
                field: "id".into(),
                needle: "nidus".into(),
                edits: 2,
                at: items[0].fuzzy_at(),
            }
        );
        assert_eq!(
            items[1],
            PredNode::MatchAll {
                field: "body".into(),
                query: "a b".into()
            }
        );
        assert_eq!(
            items[2],
            PredNode::MatchAny {
                field: "body".into(),
                query: "c d".into()
            }
        );
        assert_eq!(
            items[3],
            PredNode::Phrase {
                field: "body".into(),
                query: "e f".into()
            }
        );
    }

    impl PredNode {
        /// Test-only: pull the `at` back out of a `Fuzzy` node so the byte offset need not
        /// be hand-computed in the assertion above.
        fn fuzzy_at(&self) -> usize {
            match self {
                PredNode::Fuzzy { at, .. } => *at,
                _ => panic!("not a Fuzzy node"),
            }
        }
    }

    #[test]
    fn where_and_or_not_precedence_and_grouping() {
        // OR < AND < NOT < primary: "a=1 OR b=2 AND NOT c=3" is Any[Cmp(a), All[Cmp(b), Not(Cmp(c))]]
        let s = one("SELECT * FROM docs WHERE a = 1 OR b = 2 AND NOT c = 3");
        let PredNode::Any(items) = s.filter.unwrap() else {
            panic!("expected Any at the top")
        };
        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], PredNode::Cmp { .. }));
        let PredNode::All(inner) = &items[1] else {
            panic!("expected All")
        };
        assert!(matches!(inner[0], PredNode::Cmp { .. }));
        assert!(matches!(inner[1], PredNode::Not(_)));
    }

    #[test]
    fn parenthesized_group_overrides_precedence() {
        let s = one("SELECT * FROM docs WHERE (a = 1 OR b = 2) AND c = 3");
        let PredNode::All(items) = s.filter.unwrap() else {
            panic!("expected All at the top")
        };
        assert!(matches!(items[0], PredNode::Any(_)));
    }

    // ── the depth cap (§7.3): 200 nested parens must error cleanly, not overflow ────

    #[test]
    fn two_hundred_nested_parens_is_a_clean_error_not_a_crash() {
        let mut sql = "SELECT * FROM docs WHERE ".to_string();
        sql.push_str(&"(".repeat(200));
        sql.push_str("a = 1");
        sql.push_str(&")".repeat(200));
        let err = parse(&sql).unwrap_err();
        assert_eq!(err.section, SEC_BOOL);
        assert!(err.message.contains("nesting"), "{}", err.message);
    }

    #[test]
    fn depth_at_the_cap_still_parses() {
        // 128 levels is the cap itself, still legal.
        let mut sql = "SELECT * FROM docs WHERE ".to_string();
        sql.push_str(&"(".repeat(super::MAX_NEST_DEPTH));
        sql.push_str("a = 1");
        sql.push_str(&")".repeat(super::MAX_NEST_DEPTH));
        assert!(parse(&sql).is_ok());
    }

    // ── GROUP BY / SUM ───────────────────────────────────────────────────────────

    #[test]
    fn group_by_with_sums() {
        let s = one("SELECT * FROM docs GROUP BY project, SUM(bytes), SUM(count)");
        assert_eq!(s.group_by, Some("project".to_string()));
        assert_eq!(s.sums, vec!["bytes".to_string(), "count".to_string()]);
    }

    // ── ORDER BY: every ranking shape ────────────────────────────────────────────

    #[test]
    fn order_by_knn_vector() {
        let s = one("SELECT * FROM docs ORDER BY knn([1.0, 0.0, -0.5])");
        let Some(Ranking::Knn { vector, decay, .. }) = s.order else {
            panic!("expected Knn")
        };
        assert_eq!(vector, vec![1.0, 0.0, -0.5]);
        assert!(decay.is_none());
    }

    #[test]
    fn order_by_knn_with_decay() {
        let s = one(
            "SELECT * FROM docs ORDER BY knn([1.0]) - decay(created_at, 1700000000000, 604800000, 0.5, 2.0)",
        );
        let Some(Ranking::Knn { decay: Some(d), .. }) = s.order else {
            panic!("expected Knn with decay")
        };
        assert_eq!(d.field, "created_at");
        assert_eq!(d.origin, 1_700_000_000_000);
        assert_eq!(d.scale, 604_800_000);
        assert_eq!(d.decay, Some(0.5));
        assert_eq!(d.lambda, Some(2.0));
    }

    #[test]
    fn order_by_match() {
        let s = one("SELECT * FROM docs ORDER BY match(title, 'rust', body, 'async' PREFIX)");
        let Some(Ranking::Match { clauses, .. }) = s.order else {
            panic!("expected Match")
        };
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].field, "title");
        assert_eq!(clauses[0].text, "rust");
        assert!(!clauses[0].prefix);
        assert_eq!(clauses[1].field, "body");
        assert!(clauses[1].prefix);
    }

    #[test]
    fn order_by_knn_fuse_match() {
        let s = one("SELECT * FROM docs ORDER BY knn([1.0, 0.0]) FUSE match(body, 'rust')");
        let Some(Ranking::Hybrid {
            vector, clauses, ..
        }) = s.order
        else {
            panic!("expected Hybrid")
        };
        assert_eq!(vector, vec![1.0, 0.0]);
        assert_eq!(clauses.len(), 1);
    }

    #[test]
    fn order_by_bare_field_asc_desc() {
        let s = one("SELECT * FROM docs ORDER BY score DESC");
        assert_eq!(
            s.order,
            Some(Ranking::Field {
                field: "score".into(),
                desc: true
            })
        );
        let s2 = one("SELECT * FROM docs ORDER BY score ASC");
        assert_eq!(
            s2.order,
            Some(Ranking::Field {
                field: "score".into(),
                desc: false
            })
        );
    }

    // ── LIMIT / OFFSET ───────────────────────────────────────────────────────────

    #[test]
    fn limit_and_offset() {
        let s = one("SELECT * FROM docs LIMIT 10 OFFSET 5");
        assert_eq!(s.limit, Some(10));
        assert_eq!(s.offset, Some(5));
    }

    // ── WITH: every key shape ────────────────────────────────────────────────────

    #[test]
    fn with_flags() {
        let s = one("SELECT * FROM docs ORDER BY knn([1.0]) WITH (annotations, plan, exact)");
        assert_eq!(s.with.len(), 3);
        for w in &s.with {
            assert_eq!(w.value, OptValue::Flag);
        }
        assert_eq!(s.with[0].name, "annotations");
        assert_eq!(s.with[1].name, "plan");
        assert_eq!(s.with[2].name, "exact");
    }

    #[test]
    fn with_scalars() {
        let s = one(
            "SELECT * FROM docs ORDER BY knn([1.0]) WITH (min_score = 0.5, diversity = 0.2, candidates = 100, rrf_k = 60)",
        );
        assert_eq!(
            s.with[0].value,
            OptValue::Scalar(AnyLit::Val(Lit::Float(0.5)))
        );
        assert_eq!(
            s.with[2].value,
            OptValue::Scalar(AnyLit::Val(Lit::Int(100)))
        );
    }

    #[test]
    fn with_limit_per_tuple() {
        let s = one("SELECT * FROM docs ORDER BY knn([1.0]) WITH (limit_per = (file, 2))");
        let OptValue::List(args) = &s.with[0].value else {
            panic!("expected a list")
        };
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], OptArg::Bare(AnyLit::Ident("file".into())));
        assert_eq!(args[1], OptArg::Bare(AnyLit::Val(Lit::Int(2))));
    }

    #[test]
    fn with_context_keyed_args() {
        let s = one(
            r#"SELECT * FROM docs ORDER BY knn([1.0]) WITH (context = (radius 2, parent "pid", index "idx", text "txt"))"#,
        );
        let OptValue::List(args) = &s.with[0].value else {
            panic!("expected a list")
        };
        assert_eq!(
            args[0],
            OptArg::Keyed("radius".into(), AnyLit::Val(Lit::Int(2)))
        );
        assert_eq!(
            args[1],
            OptArg::Keyed("parent".into(), AnyLit::Ident("pid".into()))
        );
    }

    #[test]
    fn with_weights_tuple() {
        let s = one(
            "SELECT * FROM docs ORDER BY knn([1.0]) FUSE match(body, 'x') WITH (weights = (1.0, 2.0))",
        );
        let OptValue::List(args) = &s.with[0].value else {
            panic!("expected a list")
        };
        assert_eq!(args[0], OptArg::Bare(AnyLit::Val(Lit::Float(1.0))));
        assert_eq!(args[1], OptArg::Bare(AnyLit::Val(Lit::Float(2.0))));
    }

    #[test]
    fn with_rerank_optional_args() {
        let s = one(
            "SELECT * FROM docs ORDER BY knn([1.0]) WITH (rerank = (overscan 20, text \"body\"))",
        );
        let OptValue::List(args) = &s.with[0].value else {
            panic!("expected a list")
        };
        assert_eq!(
            args[0],
            OptArg::Keyed("overscan".into(), AnyLit::Val(Lit::Int(20)))
        );
    }

    // ── multi-query batching (§7.9) ──────────────────────────────────────────────

    #[test]
    fn a_script_splits_on_semicolons() {
        let stmts = parse("SELECT * FROM a; SELECT * FROM b;").unwrap();
        assert_eq!(stmts.len(), 2);
        assert_eq!(stmts[0].from, vec!["a".to_string()]);
        assert_eq!(stmts[1].from, vec!["b".to_string()]);
    }

    #[test]
    fn a_trailing_semicolon_is_optional() {
        assert_eq!(parse("SELECT * FROM a").unwrap().len(), 1);
        assert_eq!(parse("SELECT * FROM a;").unwrap().len(), 1);
    }

    // ── error format: byte offset + §7.x tail, one per class ────────────────────

    #[test]
    fn missing_select_keyword() {
        let err = parse("FROM docs").unwrap_err();
        assert_eq!(err.at, 0);
        assert_eq!(err.section, SEC_SYNTAX);
    }

    #[test]
    fn missing_value_after_operator_reports_the_operator_position() {
        // "SELECT * FROM docs WHERE a = " — the '=' is at byte 27, the missing value at 29.
        let sql = "SELECT * FROM docs WHERE a = ";
        let err = parse(sql).unwrap_err();
        assert_eq!(err.at, sql.len());
        assert_eq!(err.section, SEC_SYNTAX);
    }

    #[test]
    fn empty_query_is_an_error() {
        let err = parse("   ").unwrap_err();
        assert_eq!(err.at, 0);
    }

    #[test]
    fn like_missing_pattern_is_tagged_the_glob_section() {
        let err = parse("SELECT * FROM docs WHERE a LIKE").unwrap_err();
        assert_eq!(err.section, SEC_GLOB);
    }

    #[test]
    fn regex_missing_pattern_is_tagged_the_regex_section() {
        let err = parse("SELECT * FROM docs WHERE a ~").unwrap_err();
        assert_eq!(err.section, SEC_REGEX);
    }

    #[test]
    fn unmatched_open_paren_is_a_boolean_composition_error() {
        let err = parse("SELECT * FROM docs WHERE (a = 1").unwrap_err();
        assert_eq!(err.section, SEC_BOOL);
    }

    #[test]
    fn missing_semicolon_between_statements_is_a_batching_error() {
        let err = parse("SELECT * FROM a SELECT * FROM b").unwrap_err();
        assert_eq!(err.section, SEC_BATCH);
    }

    #[test]
    fn group_by_missing_field_is_an_aggregation_error() {
        let err = parse("SELECT * FROM docs GROUP BY").unwrap_err();
        assert_eq!(err.section, SEC_AGG);
    }

    #[test]
    fn with_bad_option_name_is_an_annotation_section_error() {
        let err = parse("SELECT * FROM docs ORDER BY knn([1.0]) WITH (1)").unwrap_err();
        assert_eq!(err.section, SEC_ANNOTATE);
    }
}
