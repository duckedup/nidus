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

## Authenticating to the backends

### Keyless (recommended on EKS / GKE)

Leave `credentials` empty and bind the ServiceAccount to a cloud role via
`serviceAccount.annotations`:

```yaml
serviceAccount:
  annotations:
    # EKS / IRSA (S3):
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/nidus
    # GKE Workload Identity (GCS):
    # iam.gke.io/gcp-service-account: nidus@my-project.iam.gserviceaccount.com
```

nidus exchanges the injected web-identity token at STS (S3) or reads the GKE/GCE metadata
server (GCS), and refreshes the temporary credentials automatically. ECS/Fargate task roles
and EC2 instance roles are picked up the same way. No long-lived keys in the cluster.

### Static keys

**S3 (`s3://`)** — `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (plus optional
`AWS_SESSION_TOKEN`, `AWS_REGION`, and `AWS_ENDPOINT_URL` for R2/MinIO):

```yaml
credentials:
  inline:           # or keep these in an existing Secret (see below)
    AWS_ACCESS_KEY_ID: "AKIA..."
    AWS_SECRET_ACCESS_KEY: "..."
    AWS_REGION: "us-east-1"
```

**GCS (`gs://`)** — a service-account key as `GOOGLE_APPLICATION_CREDENTIALS_JSON`
(the key JSON inline). Keep the JSON in a Secret and reference it:

```sh
kubectl create secret generic gcs-key \
  --from-literal=GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat sa-key.json)"
```
```yaml
credentials:
  existingSecrets: [gcs-key]
```

**Redis (`redis://` / `rediss://`)** — credentials live in the URL userinfo, and TLS
uses the `rediss://` scheme. When the URL holds a password, set `memorySecret` so it
never appears in the rendered manifest:

```sh
kubectl create secret generic nidus-redis \
  --from-literal=NIDUS_MEMORY="rediss://default:s3cr3t@redis.example.com:6380"
```
```yaml
nidus:
  memory: ""                 # leave empty when sourcing from the Secret
  memorySecret:
    name: nidus-redis
    key: NIDUS_MEMORY
```

### Where to put credentials

- **Existing Secrets** (recommended): `credentials.existingSecrets: [my-creds]` loads a
  Secret via `envFrom`; `auth.existingSecret` supplies the token; `nidus.memorySecret`
  supplies the Redis URL. Works with SealedSecrets, External Secrets Operator, etc.
- **Inline** (`credentials.inline`, `auth.token`): written to a chart-managed Secret —
  convenient for a quick start; prefer existing Secrets in production.

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
