---
title: Kubernetes (Helm)
description: "Deploy nidus on Kubernetes with the official Helm chart: object-store + Redis backed, configured through values."
---

The [`charts/nidus`](https://github.com/duckedup/nidus/tree/main/charts/nidus) Helm
chart runs `nidus serve` on Kubernetes from the published
[`duckedup/nidus`](https://hub.docker.com/r/duckedup/nidus) image. It builds on the
[container image](/guides/http-server/#running-in-a-container): everything is
configured through `NIDUS_*` environment variables, and the pod is backed by *shared,
non-local* storage (an object store for the durable bytes and a Redis-family tier for
the working set), since a pod has no durable local disk.

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

`nidus serve` is a **single writer**: exactly one instance holds the writer handle
on the shared backend at a time. What the extra replicas do depends on how you
configure them:

- **Default** (`replicaCount: 1`): one writer, no standby. Simplest, and correct.
- **Hot standby**: set `replicaCount` greater than 1 *and* `nidus.waitForLease:
  true`. The losers stay up waiting for the writer handle instead of exiting, and
  one is promoted within about `nidus.lockTtl` of the writer dying. They report NOT
  ready while waiting, so the Service routes only to the active writer.
- **`replicaCount` greater than 1 without `waitForLease`: don't.** The extra pods
  lose the lock race and crash-loop, which is an alert, not a design.

Standby promotion needs cluster mode (`nidus.cluster: true`), which in turn needs a
shared object store *and* a shared memory tier: a local-disk store is single-node by
definition.

### Read-only readers

A reader replica does not compete for the writer lease at all; it just needs to
stay current with what the writer commits. Two knobs on `nidus` control that:

- `refreshInterval` refreshes every N seconds so the reader stays current without a
  sidecar calling `POST /refresh` (`0` means never, the default).
- `maxStaleness` fails readiness if the reader ever falls more than N seconds
  behind (`0` means no bound, the default).

Set both on reader replicas: the interval keeps them fresh, and the bound takes a
reader out of the Service if refreshing ever stops.

### Rolling updates

The Deployment defaults to `updateStrategy.type: Recreate`, not `RollingUpdate`: the
old writer must terminate, releasing its lock on `SIGTERM`, before the replacement
starts, or the new pod hits a held-lock error.

With `nidus.waitForLease: true`, `RollingUpdate` becomes viable, since the incoming
pod waits for the handle rather than failing: the old writer releases on `SIGTERM`
and the new one is promoted. `Recreate` stays the default because it is correct in
every configuration, and because a rolling update trades a brief write outage for
one that is briefer but harder to reason about. Change the strategy deliberately,
not by default.

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
roles and EC2 instance roles work the same way: no long-lived keys in the cluster.

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

- **S3**: `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (plus optional
  `AWS_SESSION_TOKEN`, `AWS_REGION`, and `AWS_ENDPOINT_URL` for R2/MinIO), via
  `credentials.inline` or an existing Secret.
- **GCS**: a service-account key as `GOOGLE_APPLICATION_CREDENTIALS_JSON` (the key JSON
  inline). Put it in a Secret and list it in `credentials.existingSecrets`.
- **Redis**: credentials go in the URL (`rediss://user:pass@host:6380`; `rediss://`
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
`nidus.memorySecret`) over inline values in production: they integrate with
SealedSecrets, the External Secrets Operator, and similar. Inline values
(`credentials.inline`, `auth.token`) are written to a chart-managed Secret and are
handy for a quick start. The library guides cover the same credentials for the
[object stores](/guides/storage-backends/) and the [memory tier](/guides/in-memory-tier/).

## Ingress and TLS

nidus serves plain HTTP: there is no in-process TLS. This is where TLS terminates.
The chart's `ClusterIP` Service is safe as long as it stays in-cluster, but the
moment you expose it with an Ingress, an empty `tls: []` publishes the bearer token
and every vector in cleartext to anything on the path. Populate `tls` whenever
`ingress.enabled` is true.

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "256m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
  hosts:
    - host: nidus.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - hosts: [nidus.example.com]
      secretName: nidus-tls   # cert-manager, or a Secret you manage
```

Two more things worth setting on the ingress rather than in nidus:

- A **proxy body-size limit** matching `nidus.maxBodyBytes`. Most ingress
  controllers default to 1 MiB and will reject an upsert long before nidus sees it
  (`nginx.ingress.kubernetes.io/proxy-body-size` on nginx).
- A **proxy read timeout** at least as long as `nidus.writeTimeout`, or the proxy
  will cut off a legitimate large upsert mid-batch
  (`nginx.ingress.kubernetes.io/proxy-read-timeout` on nginx).

Keep `/metrics` off any public host: it exposes traffic shape (never collection
names or data), and it is deliberately unauthenticated so a scraper is not reported
as down.

## Verify

```sh
kubectl port-forward svc/my-nidus 7700:7700
curl http://127.0.0.1:7700/health    # -> ok
```

The liveness probe uses the unauthenticated `/health` endpoint; the readiness probe
uses `/ready`, which stays false until the store is open (and, in cluster mode, while
a standby waits for the lease or a reader falls past its staleness bound).
For the full value reference, see the chart's
[`values.yaml`](https://github.com/duckedup/nidus/blob/main/charts/nidus/values.yaml)
and [README](https://github.com/duckedup/nidus/blob/main/charts/nidus/README.md).
