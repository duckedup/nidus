# nidus Helm chart

Run the [nidus](https://github.com/duckedup/nidus) vector store as an HTTP service on
Kubernetes. The chart deploys `nidus serve` from the published
[`duckedup/nidus`](https://hub.docker.com/r/duckedup/nidus) image, configured through
`NIDUS_*` environment variables.

nidus in a container has **no durable local disk**, so this chart requires a *shared,
non-local* store: an **object store** (S3/GCS) for the durable bytes and a
**Redis-family tier** for the in-RAM working set. The chart fails fast at render time
if either is missing.

## Prerequisites

- Kubernetes 1.23+ and Helm 3.8+
- An S3 or GCS bucket nidus can read/write, with credentials
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

Or from a local checkout: `helm install my-nidus ./charts/nidus -f my-values.yaml`.

A minimal `values.yaml`:

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
```

## Single writer

`nidus serve` is a **single writer**: it holds an exclusive lock on the shared backend.
Keep `replicaCount: 1`. Extra replicas lose the lock race and crash-loop — the server
does not yet expose multi-instance cluster mode. The deployment uses the `Recreate`
strategy so a rollout terminates the old writer (which releases its lock on `SIGTERM`)
before the replacement starts.

## Credentials

Three ways to supply backend credentials and the auth token, in order of preference:

1. **Workload identity** (no static keys): annotate the ServiceAccount
   (`serviceAccount.annotations`) for IRSA (EKS) / Workload Identity (GKE) and leave
   `credentials` empty.
2. **Existing Secrets** you manage (SealedSecrets, External Secrets, …):
   `credentials.existingSecrets: [my-aws-creds]` (loaded via `envFrom`) and
   `auth.existingSecret` / `auth.existingSecretKey` for the token.
3. **Inline** (`credentials.inline`, `auth.token`): written to a chart-managed Secret.
   Convenient for trying it out; prefer 1 or 2 in production.

## Common values

| Key | Default | Description |
| --- | --- | --- |
| `nidus.dim` | `0` | Embedding dimension. **Required.** |
| `nidus.persistence` | `""` | `s3://…` or `gs://…`. **Required.** |
| `nidus.memory` | `""` | `redis://…` (or valkey/keydb/dragonfly). **Required.** |
| `nidus.distance` | `""` | `cosine` (default) \| `euclidean` \| `dot`. |
| `nidus.ann` | `""` | `""` (exact) \| `hnsw` \| `ivf`, plus `nidus.annParams.*`. |
| `image.repository` / `image.tag` | `duckedup/nidus` / appVersion | Image to run. |
| `auth.enabled` / `auth.token` | `false` / `""` | Bearer-token auth. |
| `replicaCount` | `1` | Keep at 1 (single writer). |
| `service.type` / `service.port` | `ClusterIP` / `7700` | Service exposure. |
| `ingress.enabled` | `false` | Expose via Ingress. |
| `resources` | `{}` | Pod resource requests/limits. |

See [`values.yaml`](./values.yaml) for the full set.

## Health and probes

The liveness/readiness probes hit the unauthenticated `GET /health`. The same endpoint
is handy from outside:

```sh
kubectl port-forward svc/my-nidus 7700:7700
curl http://127.0.0.1:7700/health   # -> ok
```
