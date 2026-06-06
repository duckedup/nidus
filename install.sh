#!/bin/sh
# nidus installer — fetch a prebuilt `nidus` binary from GitHub Releases and drop
# it on your PATH. No Rust toolchain required.
#
#   curl -fsSL https://raw.githubusercontent.com/duckedup/nidus/main/install.sh | sh
#
# Environment overrides:
#   NIDUS_VERSION   release tag to install (default: latest)
#   NIDUS_BIN_DIR   install directory (default: $HOME/.local/bin)
#
# Pure POSIX sh; needs curl (or wget), tar, and uname — present on a stock
# macOS/Linux box. Windows users: download the .zip from the releases page.

set -eu

REPO="duckedup/nidus"
BIN_DIR="${NIDUS_BIN_DIR:-$HOME/.local/bin}"

say() { printf 'nidus: %s\n' "$1"; }
err() { printf 'nidus: error: %s\n' "$1" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1; }

# A downloader that writes a URL to a file: curl or wget, whichever exists.
download() {
    url="$1"; dest="$2"
    if need curl; then
        curl -fsSL "$url" -o "$dest"
    elif need wget; then
        wget -qO "$dest" "$url"
    else
        err "need curl or wget to download"
    fi
}

# Verify a file against an expected SHA-256 (the first field of a `shasum`-style
# sum file). Uses sha256sum or shasum, whichever exists; warns and skips if
# neither is present rather than failing the install on a checksum-less box.
verify_sha256() {
    file="$1"; sumfile="$2"
    expected="$(awk '{print $1}' "$sumfile" | head -1)"
    [ -n "$expected" ] || err "checksum file was empty: $sumfile"
    if need sha256sum; then
        actual="$(sha256sum "$file" | awk '{print $1}')"
    elif need shasum; then
        actual="$(shasum -a 256 "$file" | awk '{print $1}')"
    else
        say "warning: no sha256 tool found — skipping checksum verification"
        return 0
    fi
    [ "$expected" = "$actual" ] \
        || err "checksum mismatch for $file (expected $expected, got $actual)"
}

# Map uname output to a Rust target triple matching the release asset names.
detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os" in
        Linux)  os_part="unknown-linux-gnu" ;;
        Darwin) os_part="apple-darwin" ;;
        *)      err "unsupported OS '$os' — build from source: cargo install nidus --features cli" ;;
    esac
    case "$arch" in
        x86_64|amd64)  arch_part="x86_64" ;;
        arm64|aarch64) arch_part="aarch64" ;;
        *)             err "unsupported arch '$arch' — build from source: cargo install nidus --features cli" ;;
    esac
    printf '%s-%s' "$arch_part" "$os_part"
}

# Resolve the version: an explicit NIDUS_VERSION, or follow the /latest redirect.
resolve_version() {
    if [ -n "${NIDUS_VERSION:-}" ]; then
        printf '%s' "$NIDUS_VERSION"
        return
    fi
    # The /releases/latest URL redirects to /tag/vX.Y.Z — read the final path.
    latest_url="https://github.com/$REPO/releases/latest"
    if need curl; then
        resolved="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$latest_url")"
    else
        # wget prints the resolved location on stderr with --max-redirect.
        resolved="$(wget -q -S --max-redirect=0 "$latest_url" 2>&1 | awk '/Location:/{print $2}' | tail -1)"
    fi
    tag="${resolved##*/}"
    [ -n "$tag" ] || err "could not determine latest version — set NIDUS_VERSION"
    printf '%s' "$tag"
}

main() {
    target="$(detect_target)"
    version="$(resolve_version)"
    asset="nidus-${target}.tar.gz"
    url="https://github.com/$REPO/releases/download/${version}/${asset}"

    say "installing nidus ${version} (${target})"

    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT

    download "$url" "$tmp/$asset" || err "download failed: $url"

    # Verify the sidecar checksum when the release ships one. Older releases
    # predate checksums — warn and proceed rather than block the install.
    if download "${url}.sha256" "$tmp/$asset.sha256" 2>/dev/null; then
        verify_sha256 "$tmp/$asset" "$tmp/$asset.sha256"
        say "checksum ok"
    else
        say "warning: no published checksum for this release — skipping verification"
    fi

    tar -xzf "$tmp/$asset" -C "$tmp" || err "failed to extract $asset"
    [ -f "$tmp/nidus" ] || err "archive did not contain a 'nidus' binary"

    mkdir -p "$BIN_DIR"
    install -m 0755 "$tmp/nidus" "$BIN_DIR/nidus" 2>/dev/null \
        || { cp "$tmp/nidus" "$BIN_DIR/nidus" && chmod 0755 "$BIN_DIR/nidus"; }

    say "installed to $BIN_DIR/nidus"
    if ! printf '%s' ":$PATH:" | grep -q ":$BIN_DIR:"; then
        say "note: $BIN_DIR is not on your PATH — add it, e.g.:"
        printf '  export PATH="%s:$PATH"\n' "$BIN_DIR"
    fi
    "$BIN_DIR/nidus" --version 2>/dev/null || true
}

main "$@"
