"""Tests for the mind-mem REST API layer (v3.2.0).

Uses FastAPI TestClient (HTTPX-backed) — no real server started.
All tests run against create_app(tmp_workspace) to keep isolation.
"""

from __future__ import annotations

import textwrap
from typing import Any, Generator

import pytest

# ---------------------------------------------------------------------------
# Availability guard — skip whole module if fastapi not installed
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed; skipping REST API tests")

from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api.rest import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Any) -> str:
    """Minimal workspace fixture used by all tests."""
    for subdir in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / subdir).mkdir()
    decisions_md = tmp_path / "decisions" / "DECISIONS.md"
    decisions_md.write_text(
        textwrap.dedent("""\
            [D-20240101-001]
            type: decision
            status: active
            statement: Use PostgreSQL as primary database.
            ---
        """),
        encoding="utf-8",
    )
    return str(tmp_path)


@pytest.fixture()
def client(workspace: str, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """TestClient with no auth configured."""
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture()
def authed_client(workspace: str, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """TestClient with user token configured."""
    monkeypatch.setenv("MIND_MEM_TOKEN", "test-user-token-for-mind-mem-rest-api")
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture()
def admin_client(workspace: str, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """TestClient with both user + admin tokens configured."""
    monkeypatch.setenv("MIND_MEM_TOKEN", "test-user-token-for-mind-mem-rest-api")
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", "test-admin-token-for-mind-mem-rest-api")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

USER_TOKEN = "test-user-token-for-mind-mem-rest-api"
ADMIN_TOKEN = "test-admin-token-for-mind-mem-rest-api"
_AUTH_HEADER = {"Authorization": f"Bearer {USER_TOKEN}"}
_ADMIN_HEADER = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def _json(resp: Any) -> Any:
    return resp.json()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200

    def test_health_contains_schema_version(self, client: TestClient) -> None:
        data = _json(client.get("/v1/health"))
        assert "_schema_version" in data
        assert data["_schema_version"] == "1.0"

    def test_health_reports_workspace_exists(self, client: TestClient, workspace: str) -> None:
        data = _json(client.get("/v1/health"))
        assert data["workspace_exists"] is True

    def test_health_api_version(self, client: TestClient) -> None:
        data = _json(client.get("/v1/health"))
        assert data["api_version"] == "3.2.0"


# ---------------------------------------------------------------------------
# Recall endpoint
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_returns_envelope_with_schema_version(self, client: TestClient) -> None:
        resp = client.post("/v1/recall", json={"query": "PostgreSQL"})
        assert resp.status_code == 200
        data = _json(resp)
        assert "_schema_version" in data

    def test_recall_rejects_empty_query(self, client: TestClient) -> None:
        resp = client.post("/v1/recall", json={"query": ""})
        assert resp.status_code == 422

    def test_recall_rejects_invalid_backend(self, client: TestClient) -> None:
        resp = client.post("/v1/recall", json={"query": "test", "backend": "elasticsearch"})
        assert resp.status_code == 422

    def test_recall_accepts_valid_backends(self, client: TestClient) -> None:
        for backend in ("auto", "bm25", "hybrid"):
            resp = client.post("/v1/recall", json={"query": "test", "backend": backend})
            # 200 or 500 (workspace may be minimal) — not 422
            assert resp.status_code in (200, 500)

    def test_recall_rejects_limit_out_of_range(self, client: TestClient) -> None:
        resp = client.post("/v1/recall", json={"query": "test", "limit": 0})
        assert resp.status_code == 422
        resp2 = client.post("/v1/recall", json={"query": "test", "limit": 999})
        assert resp2.status_code == 422

    def test_recall_with_auth_token_passes(self, authed_client: TestClient) -> None:
        resp = authed_client.post("/v1/recall", json={"query": "test"}, headers=_AUTH_HEADER)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Get block endpoint
# ---------------------------------------------------------------------------


class TestGetBlock:
    def test_get_block_existing_returns_200(self, client: TestClient) -> None:
        resp = client.get("/v1/block/D-20240101-001")
        assert resp.status_code == 200
        data = _json(resp)
        assert data.get("found") is True

    def test_get_block_nonexistent_returns_404(self, client: TestClient) -> None:
        resp = client.get("/v1/block/D-99999999-999")
        assert resp.status_code == 404

    def test_get_block_invalid_format_returns_200_or_200_with_error(self, client: TestClient) -> None:
        # Invalid block ID — tool returns error envelope, REST layer still 200
        # unless "found" is False in which case it raises 404
        resp = client.get("/v1/block/invalid-id")
        # Either 200 (with error envelope) or 404
        assert resp.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Auth: 401 without token
# ---------------------------------------------------------------------------


class TestAuthRequired:
    """Auth-required endpoints must return 401 when no token is provided
    and MIND_MEM_TOKEN is configured."""

    def test_recall_returns_401_without_token(self, authed_client: TestClient) -> None:
        resp = authed_client.post("/v1/recall", json={"query": "test"})
        assert resp.status_code == 401

    def test_get_block_returns_401_without_token(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/v1/block/D-20240101-001")
        assert resp.status_code == 401

    def test_scan_returns_401_without_token(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/v1/scan")
        assert resp.status_code == 401

    def test_contradictions_returns_401_without_token(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/v1/contradictions")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Admin scope: 403 with user-scope token
# ---------------------------------------------------------------------------


class TestAdminRequired:
    """Admin endpoints return 403 when called with a user-scope-only token."""

    def test_propose_update_with_user_token_returns_403(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/v1/propose_update",
            json={
                "block_type": "decision",
                "statement": "Use Redis for caching.",
                "rationale": "REST test rationale — decision audit requirement",
                "confidence": "high",
            },
            headers=_AUTH_HEADER,  # user token, not admin
        )
        assert resp.status_code == 403

    def test_approve_apply_with_user_token_returns_403(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/v1/approve_apply",
            json={"proposal_id": "P-20240101-001", "dry_run": True},
            headers=_AUTH_HEADER,
        )
        assert resp.status_code == 403

    def test_rollback_proposal_with_user_token_returns_403(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/v1/rollback_proposal",
            json={"receipt_ts": "20240101-120000"},
            headers=_AUTH_HEADER,
        )
        assert resp.status_code == 403

    def test_propose_update_with_admin_token_passes_auth(self, admin_client: TestClient) -> None:
        resp = admin_client.post(
            "/v1/propose_update",
            json={
                "block_type": "decision",
                "statement": "Use Redis for caching.",
                "rationale": "REST test rationale — decision audit requirement",
                "confidence": "high",
            },
            headers=_ADMIN_HEADER,
        )
        # Auth passes — may fail for other reasons (workspace) but not 401/403
        assert resp.status_code not in (401, 403)


# ---------------------------------------------------------------------------
# Propose update validation
# ---------------------------------------------------------------------------


class TestProposeUpdateValidation:
    def test_invalid_block_type_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/propose_update",
            json={
                "block_type": "note",
                "statement": "Some statement",
            },
        )
        assert resp.status_code == 422

    def test_invalid_confidence_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/propose_update",
            json={
                "block_type": "decision",
                "statement": "Some statement",
                "rationale": "REST test rationale — decision audit requirement",
                "confidence": "extreme",
            },
        )
        assert resp.status_code == 422

    def test_empty_statement_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/propose_update",
            json={
                "block_type": "decision",
                "statement": "",
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_rate_limit_triggers_429(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """429 is returned after exceeding per-client limit."""
        monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
        monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)

        from mind_mem.mcp.infra.rate_limit import SlidingWindowRateLimiter, _rate_limiters, _rate_limiters_lock

        # Inject a pre-exhausted limiter for the anonymous client bucket
        tight_limiter = SlidingWindowRateLimiter(max_calls=1, window_seconds=60)
        tight_limiter.allow()  # consume the single slot

        with _rate_limiters_lock:
            # Keys are last-16-chars of token; anonymous = "anonymous"
            _rate_limiters["anonymous"] = tight_limiter

        app = create_app(workspace)
        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post("/v1/recall", json={"query": "test"})
            assert resp.status_code == 429
            assert "Retry-After" in resp.headers


# ---------------------------------------------------------------------------
# OpenAPI schema
# ---------------------------------------------------------------------------


class TestOpenAPI:
    def test_openapi_json_reachable(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_json_is_valid_json(self, client: TestClient) -> None:
        data = _json(client.get("/openapi.json"))
        assert "openapi" in data
        assert "paths" in data

    def test_openapi_contains_all_documented_endpoints(self, client: TestClient) -> None:
        paths = _json(client.get("/openapi.json"))["paths"]
        expected = {
            "/v1/recall",
            "/v1/block/{block_id}",
            "/v1/propose_update",
            "/v1/approve_apply",
            "/v1/rollback_proposal",
            "/v1/scan",
            "/v1/contradictions",
            "/v1/health",
            "/v1/metrics",
        }
        for path in expected:
            assert path in paths, f"Missing documented endpoint: {path}"

    def test_openapi_tags_present(self, client: TestClient) -> None:
        data = _json(client.get("/openapi.json"))
        all_tags: set[str] = set()
        for path_item in data["paths"].values():
            for op in path_item.values():
                if isinstance(op, dict):
                    all_tags.update(op.get("tags", []))
        assert "recall" in all_tags
        assert "governance" in all_tags
        assert "observability" in all_tags
