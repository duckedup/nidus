//! reference/cli.md claims to list every flag the binary accepts; nothing enforced that,
//! and four docs-drift gaps shipped before this gate existed (nidus-3zv).

use clap::{Command, CommandFactory};

use super::Cli;

const DOC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/docs/src/content/docs/reference/cli.md"
));

/// Subcommands whose section says they take another's flags *wholesale*. Only `similar`
/// qualifies; `hybrid-search` cross-references six flags by name, which grants nothing.
const INHERITS: [(&str, &str); 1] = [("similar", "search")];

/// Everything before the per-subcommand sections: the shared Store/Ingest/Rerank tables.
fn preamble() -> &'static str {
    DOC.split("## Subcommands")
        .next()
        .expect("cli.md has a '## Subcommands' heading")
}

/// The body of each subcommand's `###` section, keyed by name. A heading with no
/// backticked name ("### Turning a flag back off") is prose, not a subcommand.
fn sections() -> Vec<(&'static str, &'static str)> {
    DOC.split("\n### ")
        .skip(1)
        .filter_map(|s| {
            let rest = s.strip_prefix('`')?;
            let end = rest.find('`')?;
            Some((&rest[..end], s))
        })
        .collect()
}

fn section<'a>(secs: &[(&'a str, &'a str)], name: &str) -> Option<&'a str> {
    secs.iter().find(|(n, _)| *n == name).map(|(_, body)| *body)
}

/// Whether `hay` documents `flag` itself, rather than a longer flag it prefixes:
/// `--rerank` must not be satisfied by a row for `--rerank-query`.
fn mentions(hay: &str, flag: &str) -> bool {
    hay.match_indices(flag).any(|(i, _)| {
        hay[i + flag.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '-')
    })
}

/// Long flags a caller can actually type, so a hidden arg (absent from `--help`) is
/// exempt, as are clap's auto-generated `--help`/`--version`.
fn long_flags(cmd: &Command) -> Vec<&str> {
    cmd.get_arguments()
        .filter(|a| !a.is_hide_set())
        .filter_map(|a| a.get_long())
        .filter(|l| *l != "help" && *l != "version")
        .collect()
}

/// Every leaf subcommand this build exposes, minus clap's built-in `help`, keyed by its
/// full space-joined name (`"code ingest"`, not `"code"`). A group like `code` — routing
/// to nested subcommands rather than taking flags of its own — contributes no entry itself.
fn subcommands(cli: &Command) -> Vec<(String, &Command)> {
    fn walk<'a>(cmd: &'a Command, prefix: &str, out: &mut Vec<(String, &'a Command)>) {
        for sub in cmd.get_subcommands().filter(|c| c.get_name() != "help") {
            let name = match prefix {
                "" => sub.get_name().to_string(),
                p => format!("{p} {}", sub.get_name()),
            };
            if sub.get_subcommands().next().is_some() {
                walk(sub, &name, out);
            } else {
                out.push((name, sub));
            }
        }
    }
    let mut out = Vec::new();
    walk(cli, "", &mut out);
    out
}

#[test]
fn every_long_flag_has_a_docs_row() {
    let (secs, pre) = (sections(), preamble());
    let cli = Cli::command();
    let mut missing = Vec::new();

    for (name, sub) in subcommands(&cli) {
        let Some(own) = section(&secs, &name) else {
            missing.push(format!(
                "{name}: cli.md has no \"### {name}\" section at all"
            ));
            continue;
        };
        let inherited = INHERITS
            .iter()
            .find(|(from, _)| *from == name.as_str())
            .and_then(|(_, to)| section(&secs, to))
            .unwrap_or("");

        for flag in long_flags(sub) {
            let long = format!("--{flag}");
            if !mentions(own, &long) && !mentions(inherited, &long) && !mentions(pre, &long) {
                missing.push(format!("{name}: --{flag}"));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "docs/src/content/docs/reference/cli.md claims to be exhaustive but is missing \
         {} flag row(s); add each under its subcommand's own section:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

#[test]
fn subcommand_list_is_current() {
    let pre = preamble();
    let start = pre
        .find("The binary has **")
        .expect("cli.md states the subcommand count");
    let sentence = &pre[start..][..pre[start..]
        .find("`.")
        .expect("the subcommand sentence ends with a backticked name")
        + 2];

    let stated: usize = sentence
        .split("**")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .expect("the count is written as **N subcommands**");
    let listed: Vec<&str> = sentence.split('`').skip(1).step_by(2).collect();

    assert_eq!(
        stated,
        listed.len(),
        "cli.md says {stated} subcommands but names {}: {listed:?}",
        listed.len()
    );

    let cli = Cli::command();
    let names: Vec<String> = subcommands(&cli).into_iter().map(|(n, _)| n).collect();
    let absent: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| !listed.contains(n))
        .collect();
    assert!(
        absent.is_empty(),
        "cli.md's subcommand list omits {absent:?}; update the count and the names"
    );
}
