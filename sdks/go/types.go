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
//     anyone means, so letting 0 stand for "unset" loses nothing — and sending
//     "top_k": 0 would be the silent empty-result bug this whole scheme exists to
//     prevent.
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
type Hit struct {
	Collection string  `json:"collection"`
	ID         string  `json:"id"`
	Score      float32 `json:"score"`
	Attrs      Attrs   `json:"attrs"`
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

// A SearchRequest is a vector (cosine) nearest-neighbour query. An empty Scope
// searches every collection, merged into one ranking — sound because all
// collections share one embedding space.
type SearchRequest struct {
	Query    []float32 `json:"query"`
	Scope    []string  `json:"scope,omitempty"`
	TopK     int       `json:"top_k,omitempty"`     // 0 takes the server's default
	MinScore *float32  `json:"min_score,omitempty"` // nil is "no floor"; &0 is a floor of zero
	Filter   Filter    `json:"filter,omitempty"`
}

// A TextSearchRequest is a BM25 full-text query over one indexed field. MinScore is
// a raw BM25 floor, not a cosine one — BM25 scores are unbounded above and not
// comparable across queries, so a floor that works for one query may drop
// everything for another.
type TextSearchRequest struct {
	Field    string   `json:"field"`
	Query    string   `json:"query"`
	Scope    []string `json:"scope,omitempty"`
	TopK     int      `json:"top_k,omitempty"`
	MinScore *float32 `json:"min_score,omitempty"`
	Filter   Filter   `json:"filter,omitempty"`
}

// A HybridSearchRequest fuses a vector query and a BM25 text query with reciprocal
// rank fusion.
//
// RRFK is the RRF constant (higher flattens the weight of the top ranks) and
// Candidates is how deep each leg goes before fusing. Both are pointers because a
// zero is a legal, meaningful request for each — RRFK &0 is the maximally top-heavy
// fusion, Candidates &0 fuses exactly TopK deep — so nil, not zero, is how you ask
// for the server's default (60.0 and 100).
type HybridSearchRequest struct {
	Vector     []float32 `json:"vector"`
	Field      string    `json:"field"`
	Text       string    `json:"text"`
	Scope      []string  `json:"scope,omitempty"`
	TopK       int       `json:"top_k,omitempty"`
	Filter     Filter    `json:"filter,omitempty"`
	RRFK       *float32  `json:"rrf_k,omitempty"`
	Candidates *int      `json:"candidates,omitempty"`
}

// A ListRequest is a metadata-only query: no vector, paginated, filter-driven.
//
// Note the asymmetry with the search requests. Offset's server default is 0, so
// omitting a zero Offset changes nothing — but Limit's default is 100, so a zero
// Limit must be omitted rather than sent, exactly like TopK.
type ListRequest struct {
	Scope  []string `json:"scope,omitempty"`
	Offset int      `json:"offset,omitempty"`
	Limit  int      `json:"limit,omitempty"`
	Filter Filter   `json:"filter,omitempty"`
}

// RememberOptions tunes a text-native ingest: the server embeds the text and upserts
// it, so the client only ever sends strings.
//
// Mode is "raw" (embed the text as given, the default) or "summarize" (summarize
// first, embed the summary, and stamp nidus.summary/nidus.source attrs — which needs
// a server started with a summarizer). Attrs is metadata stamped on the record.
type RememberOptions struct {
	Mode  string
	Attrs Attrs
}

type rememberWire struct {
	ID    string `json:"id"`
	Text  string `json:"text"`
	Mode  string `json:"mode,omitempty"`
	Attrs Attrs  `json:"attrs,omitempty"`
}

// wire takes id and text because they are arguments of the Remember call rather
// than options — the server wants all four in one body.
func (o RememberOptions) wire(id, text string) rememberWire {
	return rememberWire{ID: id, Text: text, Mode: o.Mode, Attrs: o.Attrs}
}

// RecallOptions tunes a recall: the server embeds the query text and vector-searches
// the collection. It mirrors [SearchRequest] minus the vector, which the server
// produces from the text.
type RecallOptions struct {
	TopK     int
	MinScore *float32 // a cosine-similarity floor; hits below it are dropped
	Filter   Filter
}

type recallWire struct {
	Query    string   `json:"query"`
	TopK     int      `json:"top_k,omitempty"`
	MinScore *float32 `json:"min_score,omitempty"`
	Filter   Filter   `json:"filter,omitempty"`
}

func (o RecallOptions) wire(query string) recallWire {
	return recallWire{Query: query, TopK: o.TopK, MinScore: o.MinScore, Filter: o.Filter}
}
