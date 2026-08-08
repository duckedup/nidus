// Compiled examples — the package's documentation that cannot rot.
//
// Everything else in this SDK is pinned by a test; the public snippets were not, so a
// renamed field or a changed signature could have drifted out of doc.go and README.md
// with a green build. An Example function is Go's one mechanism for a snippet the
// compiler checks and pkg.go.dev renders, so the documented flow lives here and the
// prose points at it.
//
// None of these has an "// Output:" comment, which is what keeps the unit tier
// hermetic: `go test` compiles an Example without an output comment but does not run
// it. They may therefore reference a server that is not there — that is the point.
// Behaviour against a real server is the integration tier's job (integration_test.go).
//
// package nidus_test rather than nidus, so the examples read exactly as a caller writes
// them: qualified with the package name, using only the exported surface.

package nidus_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	nidus "github.com/duckedup/nidus/sdks/go"
)

// ExampleNewClient — connect to a local server, then to a remote one with the bearer
// token it was started with.
func ExampleNewClient() {
	// Local: "local vs remote" is only ever the base URL.
	db, err := nidus.NewClient("http://127.0.0.1:7700")
	if err != nil {
		log.Fatal(err)
	}

	// Remote, authenticated, with a per-request deadline that composes with the
	// caller's own context.
	remote, err := nidus.NewClient(
		"https://nidus.internal.example.com",
		nidus.WithToken(os.Getenv("NIDUS_TOKEN")),
		nidus.WithTimeout(5*time.Second),
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	fmt.Println(db.Health(ctx), remote.Health(ctx))
}

// Example walks the whole flow doc.go and README.md describe: create, upsert, search.
func Example() {
	db, err := nidus.NewClient("http://127.0.0.1:7700")
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	if err := db.CreateCollection(ctx, "docs"); err != nil {
		log.Fatal(err)
	}

	// Attributes are typed, and the constructor is the type: Int and Float are distinct
	// on the server, so nidus.Int(2024) and nidus.Float(2024) never compare equal.
	if _, err := db.Upsert(ctx, "docs", []nidus.Record{
		{
			ID:     "a",
			Vector: []float32{0.1, 0.2, 0.3},
			Attrs: nidus.Attrs{
				"lang": nidus.Str("rust"), "year": nidus.Int(2024), "score": nidus.Float(0.75),
			},
		},
		// No vector: a text-only document, findable by text search and metadata.
		{ID: "b", Attrs: nidus.Attrs{"body": nidus.Str("vector stores are neat")}},
	}); err != nil {
		log.Fatal(err)
	}

	hits, err := db.Search(ctx, nidus.SearchRequest{
		Query:  []float32{0.1, 0.2, 0.3},
		TopK:   5, // leave at 0 to take the server's default rather than asking for none
		Filter: nidus.And(nidus.Eq("lang", "rust"), nidus.Ge("year", 2020)),
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, hit := range hits {
		lang, _ := hit.Attrs["lang"].Str()
		fmt.Println(hit.ID, hit.Score, lang)
	}
}

// ExampleClient_HybridSearch — the knobs whose zero the server treats as a real value
// are pointers, so nil is how the default is requested.
func ExampleClient_HybridSearch() {
	db, err := nidus.NewClient("http://127.0.0.1:7700")
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	rrfK := float32(0) // maximally top-heavy fusion; nil would mean the default, 60
	hits, err := db.HybridSearch(ctx, nidus.HybridSearchRequest{
		Vector: []float32{0.1, 0.2, 0.3},
		Field:  "body",
		Text:   "vector store",
		TopK:   10,
		RRFK:   &rrfK,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(hits))
}

// ExampleError — a failed call carries the status the server chose, which is the part a
// caller can act on.
func ExampleError() {
	db, err := nidus.NewClient("http://127.0.0.1:7700")
	if err != nil {
		log.Fatal(err)
	}

	_, err = db.Upsert(context.Background(), "docs", []nidus.Record{
		{ID: "a", Vector: []float32{0.1, 0.2, 0.3}},
	})

	var nerr *nidus.Error
	if errors.As(err, &nerr) {
		switch {
		case nerr.IsTransport():
			// status 0: no answer at all, and the write may still have been applied
		case nerr.IsBadRequest():
			// 400/422: the request itself is wrong — retrying cannot help
		case nerr.IsLocked(), nerr.IsUnavailable():
			// 409/503: transient — retry with backoff
		}
		fmt.Println(nerr.Status, nerr.Message)
	}
}
