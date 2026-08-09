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
GCS_NAME=nidus-e2e-gcs
MINIO_PORT=${MINIO_PORT:-9100}
VALKEY_PORT=${VALKEY_PORT:-6479}
GCS_PORT=${GCS_PORT:-4650}
BUCKET=${NIDUS_E2E_S3_BUCKET:-nidus-test}
GCS_BUCKET=${NIDUS_E2E_GCS_BUCKET:-nidus-test}

down() {
    docker rm -f "$MINIO_NAME" "$VALKEY_NAME" "$GCS_NAME" >/dev/null 2>&1 || true
}

up() {
    # Remove any leftovers first so `up` is idempotent.
    down

    docker run -d --name "$MINIO_NAME" -p "${MINIO_PORT}:9000" \
        -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
        quay.io/minio/minio:latest server /data >/dev/null
    docker run -d --name "$VALKEY_NAME" -p "${VALKEY_PORT}:6379" \
        valkey/valkey:8-alpine >/dev/null
    # `-scheme http`: the emulator defaults to HTTPS with a self-signed cert, which the
    # backend's rustls would (rightly) refuse.
    docker run -d --name "$GCS_NAME" -p "${GCS_PORT}:4443" \
        fsouza/fake-gcs-server -scheme http >/dev/null

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

    echo "waiting for fake-gcs-server on :${GCS_PORT} …"
    for _ in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:${GCS_PORT}/storage/v1/b" >/dev/null; then
            break
        fi
        sleep 0.5
    done
    curl -sf "http://127.0.0.1:${GCS_PORT}/storage/v1/b" >/dev/null || {
        echo "fake-gcs-server never became ready; logs:" >&2
        docker logs "$GCS_NAME" >&2 || true
        exit 1
    }

    # minio starts with no buckets. Under its filesystem backend a top-level
    # directory IS a bucket, which creates one without needing the `mc` client.
    docker exec "$MINIO_NAME" mkdir -p "/data/${BUCKET}"
    # fake-gcs-server starts empty too; its JSON API creates the bucket directly.
    curl -sf -X POST -H 'content-type: application/json' \
        -d "{\"name\":\"${GCS_BUCKET}\"}" \
        "http://127.0.0.1:${GCS_PORT}/storage/v1/b?project=e2e" >/dev/null

    echo "ready: minio :${MINIO_PORT} (minioadmin/minioadmin, bucket ${BUCKET}) + valkey :${VALKEY_PORT}"
    echo "ready: fake-gcs-server :${GCS_PORT} (bucket ${GCS_BUCKET}) — run the gs:// lane with:"
    echo "  NIDUS_E2E_GCS_ENDPOINT='http://127.0.0.1:${GCS_PORT}'"
}

# ── Valkey CLUSTER (slot routing / MOVED-ASK), a separate leg ────────────────
#
# A single-node valkey exercises no slot routing at all, so nidus's cluster-mode
# tier client (redis-rs `cluster` feature) has never met a real slot map. This
# brings up a 3-master cluster so the SAME e2e suite can run against it, driven
# only by NIDUS_E2E_REDIS_URL — no second copy of the tests.
#
# LINUX ONLY, and deliberately loud about it rather than silently skipping.
# Cluster nodes gossip with each other at the addresses they announce, so those
# addresses must be reachable from BOTH the nodes and the test process. `--network
# host` is the only arrangement where 127.0.0.1:<port> means the same thing to
# everyone. On Docker Desktop (macOS/Windows) host networking does not bridge to
# the host, so the ports stay unreachable and the cluster cannot be driven.
CLUSTER_NODES=3
CLUSTER_BASE_PORT=${CLUSTER_BASE_PORT:-7100}

cluster_node_name() { echo "nidus-e2e-valkey-c$1"; }

cluster_ports() {
    for i in $(seq 0 $((CLUSTER_NODES - 1))); do
        echo $((CLUSTER_BASE_PORT + i))
    done
}

cluster_seeds() {
    local seeds=""
    for p in $(cluster_ports); do
        seeds="${seeds:+$seeds,}127.0.0.1:$p"
    done
    echo "$seeds"
}

down_cluster() {
    for i in $(seq 0 $((CLUSTER_NODES - 1))); do
        docker rm -f "$(cluster_node_name "$i")" >/dev/null 2>&1 || true
    done
}

up_cluster() {
    if [ "$(uname -s)" != "Linux" ]; then
        echo "valkey-cluster leg is Linux-only: it needs real host networking so the" >&2
        echo "nodes' announced addresses are reachable from both the cluster and the" >&2
        echo "tests. Docker Desktop's --network host does not bridge to the host, so" >&2
        echo "the cluster would come up unreachable. Runs in CI (ubuntu-latest)." >&2
        exit 1
    fi
    down_cluster

    local i=0
    for p in $(cluster_ports); do
        docker run -d --name "$(cluster_node_name "$i")" --network host \
            valkey/valkey:8-alpine \
            valkey-server --port "$p" --cluster-enabled yes \
            --cluster-config-file "nodes-$p.conf" --cluster-node-timeout 5000 \
            --appendonly no --save '' >/dev/null
        i=$((i + 1))
    done

    echo "waiting for ${CLUSTER_NODES} valkey nodes …"
    for p in $(cluster_ports); do
        for _ in $(seq 1 60); do
            if docker exec "$(cluster_node_name 0)" \
                valkey-cli -h 127.0.0.1 -p "$p" ping 2>/dev/null | grep -q PONG; then
                break
            fi
            sleep 0.5
        done
    done

    # `--cluster-replicas 0`: three masters is enough to exercise slot routing, and
    # replicas would only slow startup.
    docker exec "$(cluster_node_name 0)" sh -c \
        "valkey-cli --cluster create $(cluster_ports | sed 's/^/127.0.0.1:/' | tr '\n' ' ') \
         --cluster-replicas 0 --cluster-yes" >/dev/null

    echo "waiting for cluster_state:ok …"
    for _ in $(seq 1 60); do
        if docker exec "$(cluster_node_name 0)" \
            valkey-cli -h 127.0.0.1 -p "$CLUSTER_BASE_PORT" cluster info 2>/dev/null |
            grep -q "cluster_state:ok"; then
            break
        fi
        sleep 0.5
    done
    docker exec "$(cluster_node_name 0)" \
        valkey-cli -h 127.0.0.1 -p "$CLUSTER_BASE_PORT" cluster info 2>/dev/null |
        grep -q "cluster_state:ok" || {
        echo "valkey cluster never reached cluster_state:ok; logs:" >&2
        docker logs "$(cluster_node_name 0)" >&2 || true
        exit 1
    }

    echo "ready: valkey cluster on $(cluster_seeds)"
    echo "point the tests at it with:"
    echo "  NIDUS_E2E_REDIS_URL='valkey://$(cluster_seeds)?cluster=true'"
}

case "${1:-}" in
    up) up ;;
    down) down ;;
    up-cluster) up_cluster ;;
    down-cluster) down_cluster ;;
    # Print the URL the cluster leg should be driven with, so the justfile and the
    # workflow never hand-write (and drift on) the seed list.
    cluster-url) echo "valkey://$(cluster_seeds)?cluster=true" ;;
    *)
        echo "usage: $0 {up|down|up-cluster|down-cluster|cluster-url}" >&2
        exit 2
        ;;
esac
