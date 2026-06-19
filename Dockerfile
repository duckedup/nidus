# syntax=docker/dockerfile:1
#
# The `nidus serve` HTTP server as a container image, configured entirely through
# `NIDUS_*` environment variables (see `src/cli/mod.rs`). The image is built for the
# *shared-backend* contract — object-store persistence (`s3://…`/`gs://…`) plus a
# Redis-family memory tier (`redis://…`) — because a container has no durable local
# disk; `NIDUS_REQUIRE_REMOTE=true` (baked in below) makes that contract a hard
# precondition so a misconfigured local store fails fast instead of losing data on
# restart. This is the precursor to a Helm chart for running nidus on Kubernetes.
#
#   docker run --rm -p 7700:7700 \
#     -e NIDUS_DIM=384 \
#     -e NIDUS_PERSISTENCE=s3://my-bucket/store \
#     -e NIDUS_MEMORY=redis://my-redis:6379 \
#     -e NIDUS_TOKEN=$SECRET \
#     -e AWS_ACCESS_KEY_ID=… -e AWS_SECRET_ACCESS_KEY=… -e AWS_REGION=… \
#     duckedup/nidus:latest

# ── Build stage ─────────────────────────────────────────────────────────────
# Pinned to the BUILD platform (never emulated): we cross-compile to the requested
# TARGETARCH with a GNU cross-linker, so the slow part — the Rust release build with
# LTO — always runs natively. `cc-debian12` (runtime) and `bookworm` (build) share a
# glibc, so the dynamically-linked binary just runs.
FROM --platform=$BUILDPLATFORM rust:1-bookworm AS builder

# Cross toolchains for both supported target arches. ring (ureq/redis TLS) is a small
# C+asm compile and needs the target CC; cargo needs the matching linker. The
# `libc6-dev-*-cross` packages carry the per-arch libc headers (e.g. aarch64's
# `bits/libc-header-start.h`) — without them the cross gcc falls back to the host's
# `/usr/include` and the ring build fails with a missing-header error. Both arches'
# cross packages are installed so the build works whichever way it cross-compiles
# (amd64 runner → arm64, or vice versa); they live in arch-specific paths and coexist
# with the native toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc-x86-64-linux-gnu gcc-aarch64-linux-gnu \
        libc6-dev-amd64-cross libc6-dev-arm64-cross \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# TARGETARCH is provided by buildx (amd64 | arm64).
ARG TARGETARCH
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    set -eux; \
    case "$TARGETARCH" in \
      amd64) target=x86_64-unknown-linux-gnu; cc=x86_64-linux-gnu-gcc ;; \
      arm64) target=aarch64-unknown-linux-gnu; cc=aarch64-linux-gnu-gcc ;; \
      *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    rustup target add "$target"; \
    # cargo wants CARGO_TARGET_<UPPER_UNDERSCORE>_LINKER; the cc crate (ring's build)
    # wants CC_<lower_underscore> — same triple, two casings. Shell var names can't
    # hold the triple's hyphens, so underscore both.
    export CARGO_TARGET_$(echo "$target" | tr 'a-z-' 'A-Z_')_LINKER="$cc"; \
    export CC_$(echo "$target" | tr '-' '_')="$cc"; \
    cargo build --release --features cli --target "$target"; \
    # Copy out of the cache mount (which won't persist past this RUN).
    cp "target/$target/release/nidus" /usr/local/bin/nidus

# ── Runtime stage ─────────────────────────────────────────────────────────────
# distroless/cc: glibc + libgcc + ca-certificates (for TLS to S3/GCS/Redis), no shell,
# no package manager. `:nonroot` runs as uid 65532 — drop-in for a hardened k8s
# securityContext. There is no durable volume: all state lives in the remote backends.
FROM gcr.io/distroless/cc-debian12:nonroot

COPY --from=builder /usr/local/bin/nidus /usr/local/bin/nidus

# Container defaults. Everything else (NIDUS_DIM, NIDUS_PERSISTENCE, NIDUS_MEMORY,
# NIDUS_TOKEN, cloud credentials …) is supplied at run time.
ENV NIDUS_ADDR=0.0.0.0:7700 \
    NIDUS_DIR=/data \
    NIDUS_REQUIRE_REMOTE=true

EXPOSE 7700

# `/health` is unauthenticated — wire it to a Kubernetes readiness/liveness probe.
ENTRYPOINT ["/usr/local/bin/nidus", "serve"]
