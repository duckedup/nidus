# `glob` module — spec

Implement `pub fn glob_match(pattern: &str, text: &str) -> bool` in `mod.rs`.
**Do not change the signature.** Root design: `../../SPEC.md` §7.1.

## Behavior
A minimal GLOB matcher over Unicode scalar values (`char`s):
- `*` — matches any run of characters, including empty (with backtracking).
- `?` — matches exactly one character.
- `[...]` — character class: matches one char in the set. Supports ranges (`a-z`),
  and negation if the class begins with `!` or `^` (`[!0-9]`, `[^abc]`).
- Any other char matches itself literally.
- The whole pattern must match the whole `text` (anchored both ends).

Edge cases to get right: `*` at start/end/middle; consecutive `**` (same as `*`);
empty pattern matches only empty text; `?` does not match empty; an unterminated
`[` (no closing `]`) is treated as a literal `[`; empty text vs patterns like `*`.

## Constraints
- Pure safe Rust. The crate root is `#![forbid(unsafe_code)]`.
- A small hand-rolled recursive/iterative matcher is expected (no new dependency
  needed). Do not pull in a glob crate.
- No `unwrap`/`expect` on attacker-controllable input paths (there is no IO here).

## Tests (inline `#[cfg(test)] mod tests`)
Cover, at minimum: literal match/non-match; `*` empty + greedy + multiple; `?`;
ranges and negation; unterminated class as literal; anchoring (`"ab"` must not match
pattern `"a"`); realistic path patterns used by callers, e.g.
`glob_match("src/*", "src/main.rs") == true`, `glob_match("src/*.rs", "src/a.rs")`,
`glob_match("src/x/*", "src/y/a.rs") == false`. These are pure-logic tests and MUST
run under Miri (no `#[cfg_attr(miri, ignore)]`).
