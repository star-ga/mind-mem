# mind-mem REST API

The REST API mirrors the MCP tool surface over HTTP/JSON. Start it with:

```bash
mm serve                        # 127.0.0.1:8080
mm serve --port 9000 --host 0.0.0.0
```

Or from Python:

```python
from mind_mem.api.rest import run
run(host="127.0.0.1", port=8080, workspace="/path/to/workspace")
```

Requires the `api` extra:

```bash
pip install 'mind-mem[api]'
```

## Base URL

```
http://127.0.0.1:8080
```

## Authentication

Set `MIND_MEM_TOKEN` for a user-scope token, `MIND_MEM_ADMIN_TOKEN` for admin scope.

```bash
export MIND_MEM_TOKEN="$(openssl rand -hex 32)"
export MIND_MEM_ADMIN_TOKEN="$(openssl rand -hex 32)"
```

Pass the token as a Bearer header:

```bash
curl -H "Authorization: Bearer $MIND_MEM_TOKEN" http://127.0.0.1:8080/v1/health
```

When no tokens are configured all endpoints are open (development mode).

Admin-scope endpoints (`/v1/propose_update`, `/v1/approve_apply`, `/v1/rollback_proposal`)
require `MIND_MEM_ADMIN_TOKEN`.

## Rate limiting

Per-client sliding window (default 120 req/min). Exceeding returns `429` with a `Retry-After` header.

## OpenAPI

Interactive docs at `/docs` (Swagger UI), schema at `/openapi.json`.

```bash
curl http://127.0.0.1:8080/openapi.json | jq .paths
```

---

## Endpoints

### GET /v1/health

Workspace status and schema version.

```bash
curl http://127.0.0.1:8080/v1/health
```

```json
{
  "_schema_version": "1.0",
  "status": "ok",
  "workspace": "/home/n/.openclaw/workspace",
  "workspace_exists": true,
  "schema_version": "2.1.0",
  "api_version": "3.2.0"
}
```

---

### GET /v1/metrics

Prometheus exposition format (requires `mind-mem[otel]`). Returns `404` if
`prometheus_client` is not installed.

```bash
curl http://127.0.0.1:8080/v1/metrics
```

---

### POST /v1/recall

Search memory. Mirrors the `recall` MCP tool.

```bash
curl -s -X POST http://127.0.0.1:8080/v1/recall \
  -H "Authorization: Bearer $MIND_MEM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "PostgreSQL schema decisions", "limit": 5, "backend": "auto"}'
```

Request body:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | string | required | 1–8192 chars |
| `limit` | int | 10 | 1–200 |
| `active_only` | bool | false | Only non-superseded blocks |
| `backend` | string | `"auto"` | `auto` \| `bm25` \| `hybrid` |

---

### GET /v1/block/{block_id}

Retrieve a single block by ID. Returns `404` if not found.

```bash
curl -s http://127.0.0.1:8080/v1/block/D-20240101-001 \
  -H "Authorization: Bearer $MIND_MEM_TOKEN"
```

---

### POST /v1/propose_update

Stage a new decision or task. **Admin scope required.**

```bash
curl -s -X POST http://127.0.0.1:8080/v1/propose_update \
  -H "Authorization: Bearer $MIND_MEM_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "block_type": "decision",
    "statement": "Adopt Rust for the hot-path inference kernel.",
    "rationale": "2x throughput improvement measured in benchmarks.",
    "tags": "performance,infra",
    "confidence": "high"
  }'
```

Request body:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `block_type` | string | required | `decision` \| `task` |
| `statement` | string | required | 1–500 chars |
| `rationale` | string | `""` | Up to 2000 chars |
| `tags` | string | `""` | Comma-separated |
| `confidence` | string | `"medium"` | `low` \| `medium` \| `high` |

---

### POST /v1/approve_apply

Apply a staged proposal. **Admin scope required.**

```bash
curl -s -X POST http://127.0.0.1:8080/v1/approve_apply \
  -H "Authorization: Bearer $MIND_MEM_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"proposal_id": "P-20240101-001", "dry_run": true}'
```

Set `dry_run: false` to commit.

---

### POST /v1/rollback_proposal

Roll back an applied proposal using its receipt timestamp.
**Admin scope required.**

```bash
curl -s -X POST http://127.0.0.1:8080/v1/rollback_proposal \
  -H "Authorization: Bearer $MIND_MEM_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"receipt_ts": "20240101-120000"}'
```

`receipt_ts` format: `YYYYMMDD-HHMMSS` (from the `approve_apply` response).

---

### GET /v1/scan

Run integrity scan (contradictions, drift, pending signals).

```bash
curl -s http://127.0.0.1:8080/v1/scan \
  -H "Authorization: Bearer $MIND_MEM_TOKEN"
```

---

### GET /v1/contradictions

List detected contradictions with resolution analysis.

```bash
curl -s http://127.0.0.1:8080/v1/contradictions \
  -H "Authorization: Bearer $MIND_MEM_TOKEN"
```
