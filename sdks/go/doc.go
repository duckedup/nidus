// Package nidus is the Go client for nidus — a small, fast vector store.
//
// It drives a running `nidus serve` over its HTTP API: one method per endpoint,
// every call taking a context.Context. "Local vs remote" is just the base URL —
// point the client at a `nidus serve` on your laptop or at any reachable host and
// nothing else about the code changes.
//
// # Connecting
//
//	db, err := nidus.NewClient("http://127.0.0.1:7700")
//	if err != nil {
//		log.Fatal(err)
//	}
//	hits, err := db.Search(context.Background(), nidus.SearchRequest{
//		Query:  []float32{0.1, 0.2, 0.3},
//		TopK:   5,
//		Filter: nidus.And(nidus.Eq("lang", "rust"), nidus.Ge("year", 2020)),
//	})
//
// The whole create → upsert → search flow is in the Example below — and it is there
// rather than written out here on purpose: an Example is compiled by `go test`, so it
// cannot drift out of step with the API the way a snippet in a comment can.
//
// When the server was started with `nidus serve --token`, pass the same token with
// [WithToken]; bring your own transport, retries, or instrumentation with
// [WithHTTPClient].
//
// # Zero dependencies
//
// The SDK is standard library only — net/http, encoding/json, context, and friends.
// There is no go.sum, nothing to audit, and nothing that can pull a transitive
// surprise into a consuming binary. That is the same posture as the crate itself and
// as the JavaScript SDK, and it is a constraint on future changes, not an accident of
// the current feature set.
//
// # Versioning
//
// The SDK ships at the crate's version: the Rust crate, this module, and the other
// client SDKs all move together off one source of truth, so module version vX.Y.Z is
// the client for nidus X.Y.Z. Matching the two is how you know the wire contract
// lines up — the SDK adapts to the server's contract, never the reverse.
//
// Version numbers here are deliberately written as placeholders rather than as the
// current release: a concrete example in a doc comment is a copy that goes stale on
// the next bump, and one that is wrong-by-one teaches a reader to distrust the rest.
//
// # A note on the module path
//
// The module is github.com/duckedup/nidus/sdks/go, but the package is nidus, so
// imports read as you would want:
//
//	import "github.com/duckedup/nidus/sdks/go"
//
//	db, err := nidus.NewClient("http://127.0.0.1:7700")
//
// A final path element that differs from the package name is legal Go and needs no
// import alias; some editors and linters will nonetheless suggest one. Because the
// module lives in a repo subdirectory, its release tags carry that directory prefix:
// `sdks/go/vX.Y.Z`, not `vX.Y.Z` — which is what `go get …/sdks/go@vX.Y.Z` resolves
// against. The README has the copy-pasteable form at the current version.
//
// While nidus is 0.x the path needs no major-version suffix. When the crate reaches
// 2.0.0 the module path must become .../sdks/go/v2 (Go's import compatibility rule),
// which is a breaking change to every import line — flagged here so it is not a
// surprise later.
package nidus

// This file holds nothing but the package documentation. Keeping it that way means
// the doc that greets a reader on pkg.go.dev is edited deliberately rather than
// drifting as a side effect of a change to some type that happened to live here.
