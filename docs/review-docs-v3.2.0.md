# Documentation Review — mind-mem v3.2.0

**Reviewed:** 2026-04-19
**Reviewer:** STARGA Inc

---

## Coverage Table

| v3.2.0 Feature | Primary doc | Status |
|---|---|---|
| Postgres storage backend | `docs/storage-backends.md`, `docs/storage-migration.md`, `docs/configuration.md` | COVERED |
| Read-replica routing | `docs/storage-migration.md` §6, `docs/storage-backends.md` | COVERED |
| REST API (FastAPI) | `docs/rest-api.md` | COVERED |
| OIDC/SSO auth (Okta, Auth0, Google, Azure) | `docs/configuration.md` §Auth Settings | COVERED |
| Per-agent API keys (`mmk_live_*`) | `docs/configuration.md` §Per-Agent API Keys, `docs/rest-api.md` | COVERED |
| JS/TS SDK | `sdk/js/README.md` | COVERED |
| Go SDK | `sdk/go/README.md` | COVERED |
| Docker + Compose | `docs/docker-deployment.md` | COVERED |
| OpenTelemetry + Prometheus | `docs/configuration.md` §Observability, `docs/rest-api.md` | COVERED |
| Grafana dashboard | `docs/configuration.md` §Grafana Dashboard | COVERED |
| Distributed recall cache | `docs/v3.2.0-release-notes.md` §6 | COVERED — no standalone how-to |
| Hot/cold tier-aware retrieval | `docs/v3.2.0-release-notes.md` §12 | COVERED — no config example in `configuration.md` |
| Obsidian wikilink export | `docs/v3.2.0-release-notes.md` §11 | COVERED — no standalone how-to |
| CLI debug commands (inspect/explain/trace) | `docs/cli-reference.md` | COVERED |
| MCP consolidated dispatchers (7) | `docs/v3.2.0-release-notes.md` §10, CHANGELOG | COVERED |
| MCP decomposition (mcp_server.py → modules) | `docs/v3.2.0-mcp-decomposition-plan.md` | COVERED |
| Atomicity / maintenance namespaces | `docs/maintenance-namespaces.md` | COVERED |
| BlockStore routing series | `docs/v3.2.0-blockstore-routing-plan.md` | COVERED |
| Supply-chain CI (CodeQL/bandit/trivy/SBOM) | `docs/supply-chain-security.md` | COVERED |
| Sigstore wheel signing | `docs/supply-chain-security.md` | COVERED |
| Security audit report | `SECURITY_AUDIT_2026-04.md` | COVERED |
| External audit SoW | `docs/security-audit-sow.md` | COVERED |
| SECURITY.md policy | `SECURITY.md` | COVERED — version table fixed |
| One-command bootstrap installer | `docs/v3.2.0-release-notes.md` §13 | COVERED — no quickstart example in README |
| `mm migrate-store` command | `docs/storage-migration.md` | COVERED |
| Workspace-wide lock primitive | `docs/v3.2.0-blockstore-routing-plan.md` | COVERED — no entry in `cli-reference.md` |
| Redis cache configuration key | `docs/configuration.md` | MISSING — `cache.redis_url` not documented |
| `tier_recall.py` config key | `docs/configuration.md` | MISSING — `retrieval.tier_boost` not documented |

**Coverage: 26 of 28 tracked features documented = 93%**

---

## Accuracy Issues

### FIXED — Issue 1 (HIGH): README "Zero infrastructure" trust signal was false

**File:** `README.md` line 86

**Before:** `No Redis, no Postgres, no vector DB, no GPU. Python 3.10+ and stdlib only.`

**Problem:** v3.2.0 ships Postgres backend, Redis L2 cache, and Docker Compose. The literal claim was false for the release it would accompany.

**Fix applied:** Changed to `Core requires only Python 3.10+ stdlib. Postgres, Redis, Docker, and GPU are opt-in extras — nothing is required to start.`

---

### FIXED — Issue 2 (HIGH): README comparison table stale counts

**File:** `README.md`, "Quick Comparison" table

**Before:** `Tests: 3,193 | MCP tools: 54 | LoCoMo benchmark: 67.3%`

**Problem:** All three figures are from v1.x. v3.2.0 has 3,600+ tests, 64 MCP tool registrations, and a 77.9 mean LoCoMo score.

**Fix applied:** Updated all three cells.

---

### FIXED — Issue 3 (HIGH): `configuration.md` Postgres backend described as "Stub only"

**File:** `docs/configuration.md` lines 469, 491–503

**Before:** `"postgres" | Postgres-backed block store. **Stub only** — raises NotImplementedError until v3.2.0 PR-5 ships the adapter.`

**Problem:** v3.2.0 ships the adapter (`block_store_postgres.py`). The "stub" note would send operators who read this to believe Postgres is not yet available — exactly wrong at release time.

**Fix applied:** Updated description to reflect shipped status; replaced the "pre-PR-5 raises NotImplementedError" paragraph with installation instructions and a cross-reference to `docs/storage-backends.md`.

---

### FIXED — Issue 4 (MEDIUM): Go SDK method table shows wrong HTTP verb for `Recall`

**File:** `sdk/go/README.md`

**Before:** `Recall(ctx, query, RecallOptions) | GET /v1/recall`

**Problem:** `docs/rest-api.md` defines `POST /v1/recall`. The Go SDK table said `GET`. A developer wiring a custom HTTP client from the table would build a broken client.

**Fix applied:** Changed to `POST /v1/recall`.

---

### FIXED — Issue 5 (MEDIUM): `SECURITY.md` supported versions table excluded 3.2.x

**File:** `SECURITY.md`

**Before:** `3.1.x — Yes (current stable)`. v3.2.x was absent.

**Problem:** After release, 3.2.x becomes the supported stable version. Vulnerability reporters would receive no guidance on whether 3.2.x is in scope.

**Fix applied:** Added `3.2.x — Yes (current stable)`; moved `3.1.x` to security-fixes-only tier.

---

## Broken Links

No broken internal markdown links found. All cross-references in the reviewed docs point to files that exist on disk.

---

## Onboarding Friction Findings

1. **"Zero infrastructure" trust signal** (now fixed) was the primary blocker — a reader who takes it literally will be confused when they find Postgres config options and Docker compose files.

2. **README Quick Start demo** (`mind-mem-init`, `mind-mem-recall` CLI scripts) predates the `mm` unified CLI. The three commands shown work but `mm` is the preferred surface since v3.1.0. Add a note or update to `mm recall`.

3. **Postgres quickstart is in `storage-backends.md`** — the README "Quick Start" section does not mention it at all. A user who wants Docker deployment from first read has to discover it via a link chain. A one-liner cross-reference under Quick Start would help.

4. **`cache.redis_url` and `retrieval.tier_boost`** are v3.2.0 config keys that appear in the release notes but are not in `docs/configuration.md`. First-time users enabling the cache will search the config reference and find nothing.

5. **JS SDK "Write operations land in v0.2 once the REST API server ships"** — the REST API ships in v3.2.0. This sentence should be updated to reflect what is actually in v0.1 vs v0.2, not what is "upcoming."

---

## Top-5 Highest-Priority Fixes (ranked by user impact)

| # | File | Issue | Impact |
|---|------|-------|--------|
| 1 | `README.md` | "Zero infrastructure" claim was false for v3.2.0 | **FIXED** — first thing every user reads |
| 2 | `docs/configuration.md` | Postgres backend described as "Stub only — raises NotImplementedError" | **FIXED** — blocks operators from enabling Postgres |
| 3 | `README.md` | Comparison table showed v1.x test/tool counts | **FIXED** — visible to anyone evaluating the project |
| 4 | `sdk/go/README.md` | `Recall` mapped to `GET` instead of `POST` | **FIXED** — produces broken clients |
| 5 | `SECURITY.md` | v3.2.x missing from supported versions | **FIXED** — affects vulnerability triage |

---

## Remaining Items (not fixed in this pass)

- `docs/configuration.md` — add `cache.redis_url` and `retrieval.tier_boost` entries.
- `sdk/js/README.md` — update "Write operations land in v0.2" note to be accurate now that the REST API ships.
- `README.md` Quick Start — add a one-liner pointing to `docs/docker-deployment.md` for Postgres + Docker users.
- `README.md` Quick Start — note that `mm recall` is preferred over the legacy `mind-mem-recall` script.
- `docs/cli-reference.md` — add `mm migrate-store` and `mm serve` entries (both are v3.2.0 commands with no reference entry).
