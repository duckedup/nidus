//! The `nidus` binary: a thin entry point that parses arguments and dispatches.
//! All logic lives in the feature-gated `nidus::cli` / `nidus::server` modules.

use nidus::cli::Cli;

fn main() {
    let cli = Cli::parse_checked();
    if let Err(e) = nidus::cli::run(cli) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
