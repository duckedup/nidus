// The wire types: records and hits, store introspection, and the request structs.
//
// These mirror src/server/dto.rs and the serde-derived core types in src/model.rs.
// The SDK adapts to the server's contract, never the reverse, so what the server
// emits is what the response structs say.
//
// # Omit-vs-zero: which knobs are pointers
//
// The server fills unset fields from #[serde(default)]: top_k = 10, limit = 100,
// rrf_k = 60.0, candidates = 100. An unset field must therefore be *absent* from the
// JSON, which is what `omitempty` is for here (the typed equivalent of the JS SDK's
// prune() helper). The question each field then poses is not "is it defaulted" but
// "does the server treat a zero as a real value":
//
//   - TopK, Limit and Offset are plain ints. Asking for zero results is not a thing
//     anyone means, and a zero Offset is the server's own default, so letting 0 stand
//     for "unset" loses nothing — and sending "top_k": 0 would be the silent
//     empty-result bug this whole scheme exists to prevent.
//   - MinScore, RRFK and Candidates are pointers, because for all three the server's
//     zero is meaningful: a score floor of exactly 0; an RRF constant of 0 (the
//     server fuses with 1/(rrf_k + rank + 1), src/store/read.rs, so 0 is the
//     maximally top-heavy weighting rather than nonsense); and candidates = 0, which
//     the server clamps up to top_k — "fuse exactly top_k deep, no over-fetch". A
//     plain float32/int would substitute the server's default for an explicit zero
//     with no error and no way for the caller to tell.
//
// Those defaults live on the server and are deliberately not restated as Go
// constants: one copy that can drift is one too many.
//
// # Tags on the exported structs
//
// The json tags sit directly on the request types rather than on a parallel set of
// unexported wire structs. A wire struct that merely restates its public twin is a
// field you can add to one and forget in the other — which compiles, marshals, and
// silently omits the new knob from every request. rememberWire and recallWire stay,
// because they are not restatements: they fold call *arguments* (id, text, query)
// into the body alongside the options.

package nidus

import (
	"encoding/json"
	"fmt"
)

// A Record is a document: a caller-supplied id, an optional embedding, and typed
// metadata.
//
// A nil Vector is a text-only doc — one indexed and retrieved purely by full-text
// search and metadata, which occupies no row in the vector matrix and never appears
// in a vector search. `omitempty` keeps the key out of the JSON entirely, so such a
// record round-trips as {id, attrs}.
//
// Note what `omitempty` cannot do: it drops a zero-*length* slice too, so a non-nil
// empty Vector would encode byte-identically to an absent one and quietly turn a
// vector-bearing upsert into a text-only document — invisible to every later vector
// search. Rust distinguishes the two (Some(vec![]) is a dimension mismatch, None is
// text-only) and Go cannot, so [Client.Upsert] refuses an empty slice rather than
// encoding it lossily: absent stays absent, and "empty" is an error at the call site.
type Record struct {
	ID     string    `json:"id"`
	Vector []float32 `json:"vector,omitempty"`
	Attrs  Attrs     `json:"attrs"`
}

// A Hit is one search or list result row.
//
// Attrs keeps typed [Value]s rather than plain Go values — see the note on [Attrs]
// for why, and call Attrs.Decode() for the loose map.
//
// Annotations is nil unless the query asked for Explain or Highlight, which is the
// common case: the server omits the key entirely, so an unannotated response decodes
// exactly as it always did.
type Hit struct {
	Collection  string       `json:"collection"`
	ID          string       `json:"id"`
	Score       float32      `json:"score"`
	Attrs       Attrs        `json:"attrs"`
	Annotations *Annotations `json:"annotations,omitempty"`
	// Context is the hit's chunk widened with its neighbours. nil unless the query asked
	// to Expand (or, on recall, to Rollup); the server omits the key entirely.
	Context *string `json:"context,omitempty"`
}

// Annotations is why a hit matched: each fusion leg's own view of it, each BM25
// clause's contribution, and highlighted fragments of the stored text.
//
// Every part is opt-in and independent, so a field being empty means "not asked for or
// not applicable" rather than "zero". Vector and Text are nil outside a hybrid search,
// which is the only query with two legs to compare.
type Annotations struct {
	Vector     *LegScore     `json:"vector,omitempty"`
	Text       *LegScore     `json:"text,omitempty"`
	Clauses    []ClauseScore `json:"clauses,omitempty"`
	Highlights []Highlight   `json:"highlights,omitempty"`
}

// A LegScore is one fusion leg's view of a document: where it ranked within that leg
// (0-based) and what the leg scored it, before fusion flattened the two into one number.
type LegScore struct {
	Rank  int     `json:"rank"`
	Score float32 `json:"score"`
}

// A ClauseScore is one text clause's own BM25 contribution. Only clauses that actually
// matched are reported, so a hit's Clauses may be shorter than the query's.
type ClauseScore struct {
	Field string  `json:"field"`
	Score float32 `json:"score"`
}

// A Highlight is the fragments found in one full-text field.
type Highlight struct {
	Field     string     `json:"field"`
	Fragments []Fragment `json:"fragments"`
}

// A Fragment is an excerpt of a field's stored text plus the ranges within it that a
// query term matched. The ranges cover the surface form, so a stemmed match ("running"
// for the query "run") highlights the word as the document spells it.
type Fragment struct {
	Text  string `json:"text"`
	Spans []Span `json:"spans"`
}

// A Span is one matched range, in bytes from the start of the [Fragment.Text] it came
// from — so frag.Text[s.Start:s.End] is the matched word, no conversion needed. It
// travels as a two-element array, which is why it marshals by hand.
type Span struct {
	Start int
	End   int
}

// MarshalJSON writes the [start, end] pair the server uses.
func (s Span) MarshalJSON() ([]byte, error) { return json.Marshal([2]int{s.Start, s.End}) }

// UnmarshalJSON reads the [start, end] pair. It decodes through a slice and checks the
// length itself, because decoding into a [2]int would silently pad a one-element span
// with a zero and drop anything past the second — both of which highlight the wrong text.
func (s *Span) UnmarshalJSON(b []byte) error {
	var pair []int
	if err := json.Unmarshal(b, &pair); err != nil {
		return fmt.Errorf("nidus: highlight span is not a [start, end] pair: %w", err)
	}
	if len(pair) != 2 {
		return fmt.Errorf("nidus: highlight span has %d elements, want [start, end]", len(pair))
	}
	s.Start, s.End = pair[0], pair[1]
	return nil
}

// A Footprint is the store's on-disk and in-RAM size, mirroring FootprintDto.
// DeadRows counts rows superseded by an overwrite or delete; compaction reclaims
// them.
type Footprint struct {
	Rows        uint64 `json:"rows"`
	DeadRows    uint64 `json:"dead_rows"`
	Dimension   int    `json:"dimension"`
	VectorBytes uint64 `json:"vector_bytes"`
	DocCount    int    `json:"doc_count"`
}

// AnnInfo is the active approximate-index configuration, mirroring AnnDto.
//
// The per-algorithm knobs are pointers because the server omits the inert ones: an
// HNSW index reports M/EfConstruction/EfSearch and no IVF fields, an IVF index the
// reverse. A pointer keeps "this knob does not apply" distinct from "this knob is
// zero", which a plain int could not.
type AnnInfo struct {
	Kind     string `json:"kind"`
	Overscan int    `json:"overscan"`
	Seed     uint64 `json:"seed"`

	// HNSW only.
	M              *int `json:"m,omitempty"`
	EfConstruction *int `json:"ef_construction,omitempty"`
	EfSearch       *int `json:"ef_search,omitempty"`

	// IVF only.
	NLists *int `json:"n_lists,omitempty"`
	NProbe *int `json:"n_probe,omitempty"`
}

// Stats is store-wide introspection, the /stats response.
//
// Ann is nil when the store does exact brute-force search — the server sends null
// there, and that is the common case rather than an error.
type Stats struct {
	Dimension   int       `json:"dimension"`
	Distance    string    `json:"distance"`
	Ann         *AnnInfo  `json:"ann"`
	Collections []string  `json:"collections"`
	Footprint   Footprint `json:"footprint"`
}

// Readiness is the /ready response: whether this instance can serve right now.
//
// Reason is set only by [Client.Ready] itself, from a 503 body — it is never on the
// wire, hence json:"-". Role is the server's own {:?} spelling, passed through
// verbatim, same as [AnnInfo.Kind].
type Readiness struct {
	Ready         bool   `json:"ready"`
	Role          string `json:"role"`
	StalenessSecs uint64 `json:"staleness_secs"`
	Reason        string `json:"-"`
}

// ClusterStatus is the /cluster response: this instance's role and standing within a
// cluster deployment. LeaseOwner and MaxStalenessSecs are nil when the server sends
// null — a single-instance store has no lease owner, and a staleness bound only
// applies to a replica.
type ClusterStatus struct {
	Role              string  `json:"role"`
	Cluster           bool    `json:"cluster"`
	HoldsWriterHandle bool    `json:"holds_writer_handle"`
	Fenced            bool    `json:"fenced"`
	LeaseOwner        *string `json:"lease_owner"`
	CommitVersion     uint64  `json:"commit_version"`
	StalenessSecs     uint64  `json:"staleness_secs"`
	MaxStalenessSecs  *uint64 `json:"max_staleness_secs"`
}

// StoreVersions is the /versions response: the commit-version landscape of a
// store's recorded history. OldestReadable is nil when no history is recorded at
// all; Pinned is nil unless this instance is pinned to a version.
type StoreVersions struct {
	CommitVersion  uint64   `json:"commit_version"`
	OldestReadable *uint64  `json:"oldest_readable"`
	Pinned         *uint64  `json:"pinned"`
	Readable       []uint64 `json:"readable"`
}

// A RankBy is a ranking expression layered over the store's distance metric, and a
// tagged union on the wire with exactly one variant today. Build it with [DecayRank];
// a RankBy naming no variant is an encode error rather than a 400 from the server.
type RankBy struct {
	Decay *Decay `json:"Decay,omitempty"`
}

// DecayRank ranks by [Decay] — the only ranking expression the server has.
func DecayRank(d Decay) *RankBy { return &RankBy{Decay: &d} }

// MarshalJSON refuses an empty RankBy, which would otherwise travel as {} and come back
// as a serde message about an unknown variant.
func (r RankBy) MarshalJSON() ([]byte, error) {
	if r.Decay == nil {
		return nil, fmt.Errorf("nidus: RankBy names no ranking expression; build it with DecayRank")
	}
	return json.Marshal(map[string]*Decay{"Decay": r.Decay})
}

// A Decay is a recency penalty over a timestamp attribute:
//
//	score = base - Lambda * (1 - Decay^(age/Scale))
//
// Age is measured back from Origin, never from the wall clock, so the same query
// against an unchanged store ranks the same way twice. The penalty is subtracted rather
// than multiplied, which keeps it meaningful for the metrics whose scores are negative
// or unbounded (Euclidean, DotProduct, BM25).
//
// Scale and Decay are plain values because the server rejects a zero for either (Scale
// must be positive, Decay must be in (0, 1)), so 0 can safely mean "take the default" —
// a week, and a factor of 0.5, which together make Scale a half-life. Lambda and Missing
// are pointers because their zeros are real requests: Lambda &0 applies no penalty at
// all, and Missing &0 fully penalizes a record whose timestamp is absent or unusable
// (the default, 1.0, penalizes it not at all).
type Decay struct {
	// The timestamp attribute — a DateTime or an Int, epoch milliseconds.
	Field string `json:"field"`
	// "Now", in epoch milliseconds. Required: there is no server-side default.
	Origin int64 `json:"origin"`
	// The age in milliseconds at which the factor equals Decay.
	Scale int64 `json:"scale,omitempty"`
	// The factor reached at exactly Scale old.
	Decay float32 `json:"decay,omitempty"`
	// How much score a fully-decayed hit gives up.
	Lambda *float32 `json:"lambda,omitempty"`
	// The factor for a record whose Field is missing or not a timestamp.
	Missing *float32 `json:"missing,omitempty"`
	// The attribute counting reinforcement (nidus.access_count is the usual one). The
	// count term only applies when this is set.
	CountField string `json:"count_field,omitempty"`
	// The count at which the penalty reduction reaches CountLambda. The server rejects
	// a zero, so 0 safely means "take the default" (10).
	CountScale float32 `json:"count_scale,omitempty"`
	// The penalty an entirely un-reinforced hit pays. A pointer for the same reason Lambda
	// is: the server accepts a zero (no count penalty at all), so a plain value could not
	// tell "disable the term" from "take the default".
	CountLambda *float32 `json:"count_lambda,omitempty"`
}

// A LimitPer caps how many hits may carry any one value of an attribute — "at most two
// hits per file". Records missing the attribute form one shared group, so an absent
// value cannot bypass the cap. Max must be at least 1.
type LimitPer struct {
	Field string `json:"field"`
	Max   int    `json:"max"`
}

// An Expand widens each hit with the neighbouring chunks of its own document, returned in
// [Hit.Context]. Payload only: the ranking is exactly what it was without it. Every field
// but Radius defaults to the reserved attrs nidus ingest stamps, so &Expand{Radius: 1} is
// the whole option a chunked corpus needs.
type Expand struct {
	Radius      int    `json:"radius"`
	ParentField string `json:"parent_field,omitempty"`
	IndexField  string `json:"index_field,omitempty"`
	TextField   string `json:"text_field,omitempty"`
}

// A Rollup reads a chunked corpus as documents rather than fragments: PerParent chunks are
// kept per document (1 when zero, the best-matching chunk), each widened with Neighbours
// chunks either side. The text-native spelling of LimitPer plus [Expand].
type Rollup struct {
	PerParent  int `json:"per_parent,omitempty"`
	Neighbours int `json:"neighbours,omitempty"`
}

// An OrderBy sorts a [Client.List] by an attribute instead of storage order. Values of a
// different type than the first orderable one, unorderable values (Null, List, NaN), and
// records missing the attribute sort into one trailing bucket, either direction.
type OrderBy struct {
	Field      string `json:"field"`
	Descending bool   `json:"descending,omitempty"`
}

// A SearchRequest is a vector (cosine) nearest-neighbour query. An empty Scope
// searches every collection, merged into one ranking — sound because all
// collections share one embedding space.
// Exact forces the exact brute-force scan for this one query, bypassing any ANN index
// and the quantized first pass — a guaranteed-exact answer without giving up the index
// for every other query. IncludeAttributes/ExcludeAttributes project the returned attrs;
// see [Projection]. RankBy and LimitPer reshape the ranking after scoring; see [Decay]
// and [LimitPer].
type SearchRequest struct {
	Query    []float32 `json:"query"`
	Scope    []string  `json:"scope,omitempty"`
	TopK     int       `json:"top_k,omitempty"`     // 0 takes the server's default
	Offset   int       `json:"offset,omitempty"`    // skip this many top-ranked hits
	MinScore *float32  `json:"min_score,omitempty"` // nil is "no floor"; &0 is a floor of zero
	Filter   Filter    `json:"filter,omitempty"`
	Exact    bool      `json:"exact,omitempty"`
	RankBy   *RankBy   `json:"rank_by,omitempty"`
	LimitPer *LimitPer `json:"limit_per,omitempty"`
	// Diversity is a Maximal Marginal Relevance lambda, spreading hits apart in vector
	// space so near-duplicates stop filling a page: 1 is pure relevance, 0 pure variety.
	// A pointer because &0 is a meaningful lambda that omitempty would drop.
	Diversity *float32       `json:"diversity,omitempty"`
	Expand    *Expand        `json:"expand,omitempty"`
	Rerank    *RerankOptions `json:"rerank,omitempty"`
	Projection
}

// A Projection selects which attrs the returned hits carry. Leave both nil for every attr
// — the default, and what every pre-projection request already sends.
//
// Setting both is a 400 from the server rather than a precedence rule, so pick one. Both
// are embedded (not flattened by a tag Go does not have) into SearchRequest and
// ListRequest, which is why the fields land at the top level of the JSON body.
type Projection struct {
	// Return only these attrs. A named attr the record lacks is simply absent.
	IncludeAttributes []string `json:"include_attributes,omitempty"`
	// Return every attr but these.
	ExcludeAttributes []string `json:"exclude_attributes,omitempty"`
}

// A SimilarRequest is a "more like this" query: neighbours of a record already stored
// under Collection/ID, rather than of a caller-supplied query vector. The source record
// is never in the results, but a true duplicate of it is.
//
// It mirrors [SearchRequest] field for field, minus Query, plus Collection and ID. An
// empty Scope searches the source's own collection — the one place this differs from
// [SearchRequest], where an empty Scope searches every collection.
type SimilarRequest struct {
	Collection string    `json:"collection"`
	ID         string    `json:"id"`
	Scope      []string  `json:"scope,omitempty"`
	TopK       int       `json:"top_k,omitempty"`     // 0 takes the server's default
	Offset     int       `json:"offset,omitempty"`    // skip this many top-ranked hits
	MinScore   *float32  `json:"min_score,omitempty"` // nil is "no floor"; &0 is a floor of zero
	Filter     Filter    `json:"filter,omitempty"`
	Exact      bool      `json:"exact,omitempty"`
	RankBy     *RankBy   `json:"rank_by,omitempty"`
	LimitPer   *LimitPer `json:"limit_per,omitempty"`
	// Diversity is a Maximal Marginal Relevance lambda, spreading hits apart in vector
	// space so near-duplicates stop filling a page: 1 is pure relevance, 0 pure variety.
	// A pointer because &0 is a meaningful lambda that omitempty would drop.
	Diversity *float32 `json:"diversity,omitempty"`
	Expand    *Expand  `json:"expand,omitempty"`
	Projection
}

// An FtsField is one entry of a [Client.SetFtsFields] schema: the attribute to
// full-text index, plus the BM25 and analyzer knobs to override for it.
//
// Every knob is a pointer for the omit-vs-zero reason described at the top of this
// file: the server's zero is meaningful for all four. K1 &0 saturates term frequency
// immediately, B &0 disables length normalization, and AsciiFolding &false is the
// default spelled out. Leave one nil to take the server's default (k1 = 1.2,
// b = 0.75, US English, no folding, no token-length cap).
type FtsField struct {
	Field        string   `json:"field"`
	K1           *float32 `json:"k1,omitempty"`
	B            *float32 `json:"b,omitempty"`
	Language     string   `json:"language,omitempty"`
	AsciiFolding *bool    `json:"ascii_folding,omitempty"`
	MaxTokenLen  *int     `json:"max_token_len,omitempty"`
}

// A FilterIndexField is one entry of a [Client.SetFilterIndexFields] declaration: the
// attribute to index for the text predicates (Fuzzy, ContainsAllTokens, ContainsAnyToken,
// ContainsTokenSequence, Regex), plus which structures to build for it.
//
// Both knobs are pointers for the omit-vs-zero reason described at the top of this file:
// the server defaults both to true, so &false is a meaningful override and nil means
// "leave it alone". Tokens drives the token predicates, Trigrams drives Fuzzy and Regex.
//
// Declaring an index changes how fast those predicates run, never what they return.
type FilterIndexField struct {
	Field    string `json:"field"`
	Tokens   *bool  `json:"tokens,omitempty"`
	Trigrams *bool  `json:"trigrams,omitempty"`
}

// An FtsClause is one clause of a multi-field text query: an indexed field and the raw
// query text for it, so title:"rust" plus body:"async runtime" is a single query.
type FtsClause struct {
	Field string `json:"field"`
	Query string `json:"query"`
}

// How several [FtsClause]s fold into one text score, for the Combine field of a text or
// hybrid request. CombineSum is the server's default, so leaving Combine empty is it.
const (
	// Add every matched clause's BM25 score: a doc hit on title *and* body outranks one
	// hit on either alone.
	CombineSum = "Sum"
	// Take the strongest matched clause, so a long body cannot out-accumulate a precise
	// title match.
	CombineMax = "Max"
)

// HighlightOpts asks for excerpts of the matched text and sizes them. A zero value is
// the request for the server's defaults — one fragment of 160 characters — so
// &HighlightOpts{} is "highlight, and don't tell me how".
//
// FragmentChars is a character budget, cut on codepoint boundaries; the spans it comes
// back with are byte offsets (see [Span]). Highlighting reads the stored text, so it
// still works on a field a [Projection] dropped from the returned attrs.
type HighlightOpts struct {
	MaxFragments  int `json:"max_fragments,omitempty"`
	FragmentChars int `json:"fragment_chars,omitempty"`
}

// RerankOptions opts into the hosted cross-encoder stage. The server ranks
// (Offset+TopK)*Overscan deep, scores each candidate's text against Query, and returns the
// caller's page of that. It needs a server started with --rerank-provider; without one the
// request is a 400 rather than a silent pass-through of the un-reranked order.
//
// Query is required on [SearchRequest] and [HybridSearchRequest], which carry no text of
// their own. It defaults to the request's own text on [Client.Recall] and on the
// Field+Query spelling of [TextSearchRequest]; the Clauses spelling has no single text, so
// it must name Query here.
//
// Overscan is a pointer because &0 is a distinguishable (and rejected) request, so nil is
// how you ask for the server's default of 10.
type RerankOptions struct {
	Query    string `json:"query,omitempty"`
	Overscan *int   `json:"overscan,omitempty"`
	TextAttr string `json:"text_attr,omitempty"`
}

// A TextSearchRequest is a BM25 full-text query. MinScore is a raw BM25 floor, not a
// cosine one — BM25 scores are unbounded above and not comparable across queries, so a
// floor that works for one query may drop everything for another.
//
// Name the fields one of two ways and never both: Field plus Query for a single field,
// or Clauses for several, each with its own text, folded by Combine. Sending both, or
// neither, or an empty Clauses list is a 400 rather than an empty result — a client bug
// there would otherwise read as "the corpus has no matches".
//
// Explain reports each matched clause's own BM25 score on every hit, and Highlight
// returns excerpts; both land in [Hit.Annotations].
type TextSearchRequest struct {
	Field     string         `json:"field,omitempty"`
	Query     string         `json:"query,omitempty"`
	Clauses   []FtsClause    `json:"clauses,omitempty"`
	Combine   string         `json:"combine,omitempty"` // CombineSum (default) or CombineMax
	Scope     []string       `json:"scope,omitempty"`
	TopK      int            `json:"top_k,omitempty"`
	Offset    int            `json:"offset,omitempty"`
	MinScore  *float32       `json:"min_score,omitempty"`
	Filter    Filter         `json:"filter,omitempty"`
	Explain   bool           `json:"explain,omitempty"`
	Highlight *HighlightOpts `json:"highlight,omitempty"`
	RankBy    *RankBy        `json:"rank_by,omitempty"`
	LimitPer  *LimitPer      `json:"limit_per,omitempty"`
	// Diversity is a Maximal Marginal Relevance lambda, spreading hits apart in vector
	// space so near-duplicates stop filling a page: 1 is pure relevance, 0 pure variety.
	// A pointer because &0 is a meaningful lambda that omitempty would drop.
	Diversity *float32       `json:"diversity,omitempty"`
	Expand    *Expand        `json:"expand,omitempty"`
	Rerank    *RerankOptions `json:"rerank,omitempty"`
	Projection
}

// A HybridSearchRequest fuses a vector query and a BM25 text query with reciprocal
// rank fusion. The text leg takes the same Field+Text / Clauses choice, and the same
// Explain and Highlight knobs, as a [TextSearchRequest] — note the field is Text here,
// not Query, matching the wire.
//
// RRFK is the RRF constant (higher flattens the weight of the top ranks) and
// Candidates is how deep each leg goes before fusing. VectorWeight and TextWeight scale
// each leg's contribution to the fused score, both defaulting to 1.0 — which reproduces
// the unweighted fusion exactly.
//
// All four are pointers because a zero is a legal, meaningful request for each: RRFK &0
// is the maximally top-heavy fusion, Candidates &0 fuses exactly TopK deep, and a weight
// of &0 drops that leg's contribution entirely. nil, not zero, is how you ask for the
// server's default (60.0, 100, 1.0 and 1.0).
type HybridSearchRequest struct {
	Vector       []float32      `json:"vector"`
	Field        string         `json:"field,omitempty"`
	Text         string         `json:"text,omitempty"`
	Clauses      []FtsClause    `json:"clauses,omitempty"`
	Combine      string         `json:"combine,omitempty"`
	Scope        []string       `json:"scope,omitempty"`
	TopK         int            `json:"top_k,omitempty"`
	Offset       int            `json:"offset,omitempty"`
	Filter       Filter         `json:"filter,omitempty"`
	RRFK         *float32       `json:"rrf_k,omitempty"`
	Candidates   *int           `json:"candidates,omitempty"`
	Explain      bool           `json:"explain,omitempty"`
	Highlight    *HighlightOpts `json:"highlight,omitempty"`
	VectorWeight *float32       `json:"vector_weight,omitempty"`
	TextWeight   *float32       `json:"text_weight,omitempty"`
	Expand       *Expand        `json:"expand,omitempty"`
	Rerank       *RerankOptions `json:"rerank,omitempty"`
}

// A ListRequest is a metadata-only query: no vector, paginated, filter-driven.
//
// Note the asymmetry with the search requests. Offset's server default is 0, so
// omitting a zero Offset changes nothing — but Limit's default is 100, so a zero
// Limit must be omitted rather than sent, exactly like TopK.
type ListRequest struct {
	Scope   []string `json:"scope,omitempty"`
	Offset  int      `json:"offset,omitempty"`
	Limit   int      `json:"limit,omitempty"`
	Filter  Filter   `json:"filter,omitempty"`
	OrderBy *OrderBy `json:"order_by,omitempty"` // nil keeps storage order
	Projection
}

// An AggregateRequest counts the records a filter matches and sums the named
// attributes. An empty Scope aggregates over every collection, and an empty Filter
// matches every record — so the zero value is "how many records are there".
type AggregateRequest struct {
	Scope  []string `json:"scope,omitempty"`
	Filter Filter   `json:"filter,omitempty"`
	Sum    []string `json:"sum,omitempty"`
	// GroupBy splits the answer into one [Group] per distinct value of this attribute,
	// alongside the whole-scope totals. Empty reports the totals alone.
	GroupBy string `json:"group_by,omitempty"`
}

// An Aggregation is the answer to an [AggregateRequest].
//
// Each sum stays a tagged [Value] rather than becoming a float: it is an Int while
// every addend was an Int, and a Float otherwise, so a byte count does not come back
// having quietly lost precision to float64. Sums has one entry per requested field
// either way: a field with no numeric value anywhere sums to Int(0), not to nothing.
type Aggregation struct {
	Count uint64 `json:"count"`
	Sums  Attrs  `json:"sums"`
	// Groups is one row per distinct AggregateRequest.GroupBy value, ordered by Count
	// descending. Nil when no grouping was asked for.
	Groups []Group `json:"groups,omitempty"`
	// GroupsTruncated reports that distinct values outran the server's group cap and later
	// ones were dropped, so a partial answer is never mistaken for a complete one.
	GroupsTruncated bool `json:"groups_truncated,omitempty"`
}

// A Group is one distinct GroupBy value with the aggregates over just its records.
//
// Value is nil for the records missing the attribute entirely — which is a different group
// from those holding a present null, exactly as the filter predicates treat absent vs Null.
type Group struct {
	Value *Value `json:"value"`
	Count uint64 `json:"count"`
	Sums  Attrs  `json:"sums"`
}

// A BatchSearchRequest answers several vector queries in one round-trip, saving a network
// hop per query when one question is fanned into several phrasings.
//
// Each entry is an ordinary [SearchRequest] with its own scope, filter and top_k. Set Fuse
// to merge the per-query rankings into a single list instead of getting them side by side;
// the server caps a batch at 16 queries.
type BatchSearchRequest struct {
	Queries []SearchRequest `json:"queries"`
	Fuse    *BatchFuse      `json:"fuse,omitempty"`
}

// A BatchFuse merges a batch's per-query rankings with Reciprocal Rank Fusion — the same
// fusion [Client.HybridSearch] runs, over N query legs rather than a vector and a text leg.
//
// Weights must be either empty (every leg neutral) or exactly as long as Queries: the
// server refuses a short list rather than silently re-weighting the wrong leg.
type BatchFuse struct {
	RRFK    float32   `json:"rrf_k,omitempty"`
	Weights []float32 `json:"weights,omitempty"`
	TopK    int       `json:"top_k,omitempty"`
}

// RememberOptions tunes a text-native ingest: the server embeds the text and upserts
// it, so the client only ever sends strings.
//
// Mode is "raw" (embed the text as given, the default) or "summarize" (summarize
// first, embed the summary, and stamp a nidus.summary attr — which needs
// a server started with a summarizer). The raw text is always stored under nidus.text. Attrs is metadata stamped on the record.
//
// TTLSeconds and DedupeThreshold are pointers for the usual reason (see the package
// comment): zero is meaningful in both — a TTL of 0 expires the entry immediately, and a
// dedupe floor of 0 matches any entry at all rather than disabling dedupe.
type RememberOptions struct {
	Mode  string
	Attrs Attrs

	// TTLSeconds is how long this memory lives, counted from the write. Nil never expires.
	TTLSeconds *int64

	// DedupeThreshold is a cosine floor above which this write updates the nearest existing
	// entry instead of inserting a competing near-duplicate; nil disables dedupe. It is a
	// vector search server-side, so it needs the embedder Remember does, and an expired
	// entry is never a candidate — a lapsed TTL cannot be revived by a near-duplicate.
	DedupeThreshold *float32
}

type rememberWire struct {
	ID              string   `json:"id"`
	Text            string   `json:"text"`
	Mode            string   `json:"mode,omitempty"`
	Attrs           Attrs    `json:"attrs,omitempty"`
	TTLSeconds      *int64   `json:"ttl_seconds,omitempty"`
	DedupeThreshold *float32 `json:"dedupe_threshold,omitempty"`
}

// wire takes id and text because they are arguments of the Remember call rather
// than options — the server wants all of them in one body.
func (o RememberOptions) wire(id, text string) rememberWire {
	return rememberWire{
		ID:              id,
		Text:            text,
		Mode:            o.Mode,
		Attrs:           o.Attrs,
		TTLSeconds:      o.TTLSeconds,
		DedupeThreshold: o.DedupeThreshold,
	}
}

// A RememberResult reports what a [Client.Remember] actually wrote.
//
// ID is the record that changed, which is not the requested id when Deduped: a
// DedupeThreshold match redirects the write onto the entry it matched, so this is the
// only way to learn which memory now holds the text.
type RememberResult struct {
	ID       string `json:"id"`
	Upserted int    `json:"upserted"`
	Deduped  bool   `json:"deduped"`
}

// RecallOptions tunes a recall: the server embeds the query text and vector-searches
// the collection. It mirrors [SearchRequest] minus the vector, which the server
// produces from the text.
type RecallOptions struct {
	TopK     int
	MinScore *float32 // a cosine-similarity floor; hits below it are dropped
	Filter   Filter
	// Diversity spreads the recalled window apart in vector space (MMR lambda), so one
	// verbose document's near-identical chunks stop filling it. &0 is meaningful.
	Diversity *float32
	// Rollup reads the collection as a chunked corpus: one hit per document, widened.
	Rollup *Rollup
	Rerank *RerankOptions
	// Reinforce records that these entries proved useful, stamping nidus.access_count and
	// nidus.last_accessed. It makes the recall a write: it takes the server's writer lock
	// and is refused on a read-only server.
	Reinforce bool
	// ExtendTTLSeconds pushes an existing nidus.expires_at out to this many seconds from
	// now. Only applies with Reinforce, and never gives an expiry to an entry without one.
	ExtendTTLSeconds *int64
	// RankBy layers a ranking expression over cosine: decay over nidus.last_accessed, a
	// reinforcement term over nidus.access_count, or both. Build it with [DecayRank].
	RankBy *RankBy
}

type recallWire struct {
	Query            string         `json:"query"`
	TopK             int            `json:"top_k,omitempty"`
	MinScore         *float32       `json:"min_score,omitempty"`
	Filter           Filter         `json:"filter,omitempty"`
	Diversity        *float32       `json:"diversity,omitempty"`
	Rollup           *Rollup        `json:"rollup,omitempty"`
	Rerank           *RerankOptions `json:"rerank,omitempty"`
	Reinforce        bool           `json:"reinforce,omitempty"`
	ExtendTTLSeconds *int64         `json:"extend_ttl_seconds,omitempty"`
	RankBy           *RankBy        `json:"rank_by,omitempty"`
}

func (o RecallOptions) wire(query string) recallWire {
	return recallWire{
		Query:            query,
		TopK:             o.TopK,
		MinScore:         o.MinScore,
		Filter:           o.Filter,
		Diversity:        o.Diversity,
		Rollup:           o.Rollup,
		Rerank:           o.Rerank,
		Reinforce:        o.Reinforce,
		ExtendTTLSeconds: o.ExtendTTLSeconds,
		RankBy:           o.RankBy,
	}
}
