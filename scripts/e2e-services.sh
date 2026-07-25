#!/usr/bin/env bash
# Start/stop the backing services the cluster e2e tests need: a real S3
# (minio) and a real Redis-family memory tier (valkey).
#
# The single source of truth for these containers, called by BOTH `just
# e2e-services-up/down` and .github/workflows/integration.yml — so a local run
# and a CI run exercise the same setup and cannot drift apart.
#
# Not GitHub Actions `services:` containers, deliberately: those cannot pass a
# command to the image, and minio's official image needs `server /data`.
#
# Ports are non-default so this never collides with a developer's own
# minio/redis. They match the defaults in tests/e2e/cluster.rs; override there
# with NIDUS_E2E_* if you point the tests somewhere else.
set -euo pipefail

MINIO_NAME=nidus-e2e-minio
VALKEY_NAME=nidus-e2e-valkey
MINIO_PORT=${MINIO_PORT:-9100}
VALKEY_PORT=${VALKEY_PORT:-6479}
BUCKET=${NIDUS_E2E_S3_BUCKET:-nidus-test}

down() {
    docker rm -f "$MINIO_NAME" "$VALKEY_NAME" >/dev/null 2>&1 || true
}

up() {
    # Remove any leftovers first so `up` is idempotent.
    down

    docker run -d --name "$MINIO_NAME" -p "${MINIO_PORT}:9000" \
        -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
        quay.io/minio/minio:latest server /data >/dev/null
    docker run -d --name "$VALKEY_NAME" -p "${VALKEY_PORT}:6379" \
        valkey/valkey:8-alpine >/dev/null

    # Wait for readiness rather than sleeping: an unready service is the classic
    # source of flaky integration CI.
    echo "waiting for minio on :${MINIO_PORT} …"
    for _ in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:${MINIO_PORT}/minio/health/live" >/dev/null; then
            break
        fi
        sleep 0.5
    done
    curl -sf "http://127.0.0.1:${MINIO_PORT}/minio/health/live" >/dev/null || {
        echo "minio never became healthy; logs:" >&2
        docker logs "$MINIO_NAME" >&2 || true
        exit 1
    }

    echo "waiting for valkey on :${VALKEY_PORT} …"
    for _ in $(seq 1 60); do
        if docker exec "$VALKEY_NAME" valkey-cli ping 2>/dev/null | grep -q PONG; then
            break
        fi
        sleep 0.5
    done
    docker exec "$VALKEY_NAME" valkey-cli ping 2>/dev/null | grep -q PONG || {
        echo "valkey never answered PING; logs:" >&2
        docker logs "$VALKEY_NAME" >&2 || true
        exit 1
    }

    # minio starts with no buckets. Under its filesystem backend a top-level
    # directory IS a bucket, which creates one without needing the `mc` client.
    docker exec "$MINIO_NAME" mkdir -p "/data/${BUCKET}"

    echo "ready: minio :${MINIO_PORT} (minioadmin/minioadmin, bucket ${BUCKET}) + valkey :${VALKEY_PORT}"
}

case "${1:-}" in
    up) up ;;
    down) down ;;
    *)
        echo "usage: $0 {up|down}" >&2
        exit 2
        ;;
esac
