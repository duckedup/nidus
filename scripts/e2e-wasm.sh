#!/usr/bin/env bash
# Browser bring-up for the wasm_opfs E2E suite (nidus-y67 U6): the single
# source of truth for driver detection and the pinned wasm-bindgen-cli
# install, called by BOTH `just test-wasm-e2e` and integration.yml — the same
# reason `scripts/e2e-services.sh` exists, so CI and a local run cannot drift.
#
# Driver choice is OS-dependent because chromedriver is only usable on the CI
# runner: brew's cask fails the Gatekeeper check, is SIGKILLed on launch, and
# still hangs after `xattr -d com.apple.quarantine`. On macOS this prefers
# Safari (`safaridriver`, already on the box) then Firefox (`geckodriver`),
# and fails fast naming the remediation rather than hanging on chromedriver.
set -euo pipefail

WASM_BINDGEN_CLI_VERSION=0.2.122
STATE_DIR="${TMPDIR:-/tmp}/nidus-e2e-wasm"
SAFARIDRIVER_DEFAULT=/System/Cryptexes/App/usr/bin/safaridriver

# Prints "$ENV_VAR:$path" for the driver to use, "UNUSABLE_CHROMEDRIVER" if
# chromedriver is the only thing present on Darwin, or "" if nothing usable
# was found.
detect_driver() {
    if [ "$(uname -s)" = "Darwin" ]; then
        if command -v safaridriver >/dev/null 2>&1; then
            echo "SAFARIDRIVER:$(command -v safaridriver)"
        elif [ -x "$SAFARIDRIVER_DEFAULT" ]; then
            echo "SAFARIDRIVER:$SAFARIDRIVER_DEFAULT"
        elif command -v geckodriver >/dev/null 2>&1; then
            echo "GECKODRIVER:$(command -v geckodriver)"
        elif command -v chromedriver >/dev/null 2>&1; then
            echo "UNUSABLE_CHROMEDRIVER"
        else
            echo ""
        fi
    else
        if command -v chromedriver >/dev/null 2>&1; then
            echo "CHROMEDRIVER:$(command -v chromedriver)"
        elif command -v geckodriver >/dev/null 2>&1; then
            echo "GECKODRIVER:$(command -v geckodriver)"
        else
            echo ""
        fi
    fi
}

no_driver_message() {
    echo "::error::no usable browser driver found for the wasm_opfs suite." >&2
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "On macOS, use Safari or Firefox — NOT chromedriver (brew's cask fails" >&2
        echo "the Gatekeeper check and hangs on launch even after clearing quarantine):" >&2
        echo "  A. Safari: run 'sudo safaridriver --enable', then in Safari turn on" >&2
        echo "     Develop > Allow Remote Automation. safaridriver ships at" >&2
        echo "     $SAFARIDRIVER_DEFAULT, no install needed." >&2
        echo "  B. Firefox: 'brew install geckodriver' (and Firefox itself if missing)." >&2
    else
        echo "Install Chrome + chromedriver, or Firefox + geckodriver, and put the" >&2
        echo "driver binary on PATH." >&2
    fi
    exit 1
}

ensure_wasm_target() {
    rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
}

# The runner refuses the compiled module if its version does not match
# Cargo.lock's wasm-bindgen, so pin exactly and never install `latest`.
ensure_wasm_bindgen_cli() {
    if cargo install --list 2>/dev/null | grep -q "^wasm-bindgen-cli v${WASM_BINDGEN_CLI_VERSION}:"; then
        return
    fi
    echo "installing wasm-bindgen-cli ${WASM_BINDGEN_CLI_VERSION} (must match Cargo.lock)…"
    cargo install wasm-bindgen-cli --version "${WASM_BINDGEN_CLI_VERSION}" --locked
}

up() {
    ensure_wasm_target
    ensure_wasm_bindgen_cli

    driver="$(detect_driver)"
    if [ "$driver" = "UNUSABLE_CHROMEDRIVER" ]; then
        echo "::error::found chromedriver, but it is not usable on macOS (Gatekeeper" >&2
        echo "kills it on launch, even after clearing quarantine)." >&2
        no_driver_message
    fi
    if [ -z "$driver" ]; then
        no_driver_message
    fi
    mkdir -p "$STATE_DIR"
    echo "$driver" >"$STATE_DIR/driver"
    echo "wasm-opfs e2e ready: driver=${driver%%:*} ($(echo "$driver" | cut -d: -f2-))"
}

test_cmd() {
    up
    driver="$(cat "$STATE_DIR/driver")"
    var="${driver%%:*}"
    path="${driver#*:}"
    export "${var}=${path}"

    log="$STATE_DIR/test.log"
    set +e
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
        cargo test --target wasm32-unknown-unknown --test wasm_opfs 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    set -e

    if [ "$status" -ne 0 ]; then
        echo "::group::wasm_opfs failure — browser/driver output" >&2
        cat "$log" >&2
        echo "::endgroup::" >&2
        if [ "$var" = "SAFARIDRIVER" ] && grep -qE '(http status: 500|session not created|Unexpected server response)' "$log"; then
            echo "::error::safaridriver returned an error typical of remote automation being" >&2
            echo "disabled. Run 'sudo safaridriver --enable', then in Safari turn on" >&2
            echo "Develop > Allow Remote Automation, and retry." >&2
        fi
        exit "$status"
    fi
}

down() {
    rm -rf "$STATE_DIR"
}

# Builds the wasm binding + docs site, serves docs/dist over loopback HTTP (OPFS
# needs a secure context — file:// does not qualify, 127.0.0.1 does) on a free
# port, and drives docs/e2e/terminal.mjs against it in the detected browser.
docs_cmd() {
    up
    driver="$(cat "$STATE_DIR/driver")"
    var="${driver%%:*}"
    path="${driver#*:}"
    export "${var}=${path}"

    just build-wasm-binding
    just docs-build

    port="$(python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()')"

    python3 -m http.server "$port" --bind 127.0.0.1 --directory docs/dist \
        >"$STATE_DIR/docs-http.log" 2>&1 &
    server_pid=$!
    trap 'kill "$server_pid" >/dev/null 2>&1 || true' EXIT

    for _ in $(seq 1 50); do
        curl -sf "http://127.0.0.1:${port}/" >/dev/null 2>&1 && break
        sleep 0.1
    done

    bun docs/e2e/terminal.mjs "http://127.0.0.1:${port}"
}

case "${1:-}" in
up) up ;;
test) test_cmd ;;
docs) docs_cmd ;;
down) down ;;
*)
    echo "usage: $0 {up|test|docs|down}" >&2
    exit 2
    ;;
esac
