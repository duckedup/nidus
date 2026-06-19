{{/*
Expand the name of the chart.
*/}}
{{- define "nidus.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name. Truncated at 63 chars for DNS-1123 label limits.
*/}}
{{- define "nidus.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "nidus.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "nidus.labels" -}}
helm.sh/chart: {{ include "nidus.chart" . }}
{{ include "nidus.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "nidus.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nidus.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
ServiceAccount name to use.
*/}}
{{- define "nidus.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nidus.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Resolved image reference (tag falls back to the chart appVersion).
*/}}
{{- define "nidus.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end }}

{{/*
Whether the chart manages its own Secret (for an inline token and/or inline creds).
*/}}
{{- define "nidus.createSecret" -}}
{{- $inlineToken := and .Values.auth.enabled .Values.auth.token (not .Values.auth.existingSecret) -}}
{{- if or $inlineToken (gt (len (keys .Values.credentials.inline)) 0) -}}
true
{{- end -}}
{{- end }}

{{/*
Validate required, remote-only configuration. The image runs with
NIDUS_REQUIRE_REMOTE=true, so anything less fails the pod at startup — surface it
here at render time with an actionable message instead.
*/}}
{{- define "nidus.validate" -}}
{{- if not (.Values.nidus.dim) -}}
{{- fail "nidus.dim is required: set it to your embedding dimension (e.g. --set nidus.dim=768)" -}}
{{- end -}}
{{- $p := .Values.nidus.persistence -}}
{{- if not (or (hasPrefix "s3://" $p) (hasPrefix "gs://" $p) (hasPrefix "gcs://" $p)) -}}
{{- fail "nidus.persistence must be an object store: s3://<bucket>/<prefix> or gs://<bucket>/<prefix>" -}}
{{- end -}}
{{- if not .Values.nidus.memorySecret.name -}}
{{- $m := .Values.nidus.memory | lower -}}
{{- if not (or (hasPrefix "redis://" $m) (hasPrefix "rediss://" $m) (hasPrefix "valkey://" $m) (hasPrefix "valkeys://" $m) (hasPrefix "keydb://" $m) (hasPrefix "dragonfly://" $m)) -}}
{{- fail "nidus.memory must be a Redis-family URL (redis://, rediss://, valkey://, keydb://, dragonfly://), or set nidus.memorySecret.name to source it from a Secret" -}}
{{- end -}}
{{- end -}}
{{- end }}
