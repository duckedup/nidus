// The Go SDK is its own module, nested in the nidus repo under sdks/go, so that
// `go get` never drags the crate's tooling along and the SDK can be tagged
// independently of the Rust workspace.
//
// Go resolves a module in a repo subdirectory by a tag of the form
// `<subdir>/v<semver>`, so releases are tagged `sdks/go/vX.Y.Z` — the module path's
// directory prefix is load-bearing, not cosmetic. README.md is the one place that
// spells the mechanism out at a concrete version; no copy of it here to go stale.
//
// There are deliberately no `require` directives: the SDK is standard library only,
// which is why no go.sum exists beside this file.
module github.com/duckedup/nidus/sdks/go

go 1.23
