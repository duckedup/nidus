---
title: Kubernetes (Helm)
description: Deploy nidus on Kubernetes with the official Helm chart — object-store + Redis backed, configured through values.
---

The [`charts/nidus`](https://github.com/duckedup/nidus/tree/main/charts/nidus) Helm
chart runs `nidus serve` on Kubernetes from the published
[`duckedup/nidus`](https://hub.docker.com/r/duckedup/nidus) image. It builds on the
[container image](/guides/http-server/#running-in-a-container): everything is
configured through `NIDUS_*` environment variables, and the pod is backed by *shared,
non-local* storage — an object store for the durable bytes and a Redis-family tier for
the working set — since a pod has no durable local disk.

## Prerequisites

- Kubernetes 1.23+ and Helm 3.8+
- An S3 or GCS bucket, with credentials nidus can use
- A reachable Redis (or Valkey/KeyDB/DragonflyDB) endpoint

## Install

```sh
helm install my-nidus oci://ghcr.io/duckedup/charts/nidus \
  --set nidus.dim=768 \
  --set nidus.persistence=s3://my-bucket/store \
  --set nidus.memory=redis://my-redis:6379 \
  --set auth.enabled=true --set auth.token="$(openssl rand -hex 32)" \
  --set credentials.inline.AWS_ACCESS_KEY_ID=AKIA... \
  --set credentials.inline.AWS_SECRET_ACCESS_KEY=... \
  --set credentials.inline.AWS_REGION=us-east-1
```

`nidus.dim`, `nidus.persistence`, and `nidus.memory` are required; the chart fails at
render time (with a clear message) if any is missing or not a remote backend, rather
than letting the pod crash-loop.

A `values.yaml` is usually cleaner than a wall of `--set`:

```yaml
nidus:
  dim: 768
  persistence: s3://my-bucket/store
  memory: redis://my-redis:6379

auth:
  enabled: true
  token: "change-me"

credentials:
  inline:
    AWS_ACCESS_KEY_ID: "AKIA..."
    AWS_SECRET_ACCESS_KEY: "..."
    AWS_REGION: "us-east-1"

resources:
  requests:
    cpu: 500m
    memory: 512Mi
```

```sh
helm install my-nidus oci://ghcr.io/duckedup/charts/nidus -f values.yaml
```

## Single writer

`nidus serve` is a **single writer** — it holds an exclusive lock on the shared
backend. Keep `replicaCount: 1`; extra replicas lose the lock race and crash-loop
(the server does not yet expose multi-instance cluster mode). The Deployment uses the
`Recreate` strategy on purpose: a rollout terminates the old writer first, and because
the image handles `SIGTERM` it flushes and releases its lock on the way out, so the
replacement acquires it immediately instead of waiting out the lock TTL.

## Credentials

In order of preference:

1. **Workload identity** — annotate the ServiceAccount
   (`serviceAccount.annotations`) for IRSA (EKS) or GKE Workload Identity and leave
   `credentials` empty. No static keys in the cluster.
2. **Existing Secrets** — `credentials.existingSecrets: [my-aws-creds]` loads them via
   `envFrom`; `auth.existingSecret` / `auth.existingSecretKey` supply the token.
   Works with SealedSecrets, External Secrets, etc.
3. **Inline** — `credentials.inline` and `auth.token` are written to a chart-managed
   Secret. Fine for a quick start; prefer 1 or 2 in production.

## Verify

```sh
kubectl port-forward svc/my-nidus 7700:7700
curl http://127.0.0.1:7700/health    # -> ok
```

The liveness and readiness probes use the same unauthenticated `/health` endpoint.
For the full value reference, see the chart's
[`values.yaml`](https://github.com/duckedup/nidus/blob/main/charts/nidus/values.yaml)
and [README](https://github.com/duckedup/nidus/blob/main/charts/nidus/README.md).
