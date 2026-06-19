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

## Authenticating to the backends

### Keyless (recommended on EKS / GKE)

The cleanest option: bind the ServiceAccount to a cloud role and leave `credentials` empty.

```yaml
serviceAccount:
  annotations:
    # EKS / IRSA (S3):
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/nidus
    # GKE Workload Identity (GCS):
    # iam.gke.io/gcp-service-account: nidus@my-project.iam.gserviceaccount.com
```

nidus exchanges the pod's injected web-identity token at STS (S3) or reads the GKE/GCE
metadata server (GCS), refreshing the temporary credentials automatically. ECS/Fargate task
roles and EC2 instance roles work the same way — no long-lived keys in the cluster.

On a cluster **without** the EKS webhook (self-hosted Kubernetes federated to AWS IAM via an
OIDC provider), enable `awsWebIdentity` and the chart projects the ServiceAccount token and
wires `AWS_ROLE_ARN` / `AWS_WEB_IDENTITY_TOKEN_FILE` itself:

```yaml
awsWebIdentity:
  enabled: true
  roleArn: arn:aws:iam::123456789012:role/nidus
  audience: sts.amazonaws.com   # must match the IAM OIDC provider's audience
```

### Static keys

Otherwise supply keys explicitly:

- **S3** — `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (plus optional
  `AWS_SESSION_TOKEN`, `AWS_REGION`, and `AWS_ENDPOINT_URL` for R2/MinIO), via
  `credentials.inline` or an existing Secret.
- **GCS** — a service-account key as `GOOGLE_APPLICATION_CREDENTIALS_JSON` (the key JSON
  inline). Put it in a Secret and list it in `credentials.existingSecrets`.
- **Redis** — credentials go in the URL (`rediss://user:pass@host:6380`; `rediss://`
  for TLS). When the URL has a password, source it from a Secret with
  `nidus.memorySecret` so it stays out of the rendered manifest:

  ```sh
  kubectl create secret generic nidus-redis \
    --from-literal=NIDUS_MEMORY="rediss://default:s3cr3t@redis.example.com:6380"
  ```
  ```yaml
  nidus:
    memory: ""
    memorySecret:
      name: nidus-redis
      key: NIDUS_MEMORY
  ```

Prefer **existing Secrets** (`credentials.existingSecrets`, `auth.existingSecret`,
`nidus.memorySecret`) over inline values in production — they integrate with
SealedSecrets, the External Secrets Operator, and similar. Inline values
(`credentials.inline`, `auth.token`) are written to a chart-managed Secret and are
handy for a quick start. The library guides cover the same credentials for the
[object stores](/guides/storage-backends/) and the [memory tier](/guides/memory-stores/).

## Verify

```sh
kubectl port-forward svc/my-nidus 7700:7700
curl http://127.0.0.1:7700/health    # -> ok
```

The liveness and readiness probes use the same unauthenticated `/health` endpoint.
For the full value reference, see the chart's
[`values.yaml`](https://github.com/duckedup/nidus/blob/main/charts/nidus/values.yaml)
and [README](https://github.com/duckedup/nidus/blob/main/charts/nidus/README.md).
