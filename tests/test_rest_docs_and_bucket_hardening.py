# Copyright 2026 STARGA, Inc.
"""N-12 + N-13: two REST hardening items from the 2026-04-28 audits.

**N-12 — rate-limit bucket key.** The sliding-window bucket was keyed on
``token[-16:]``, so a suffix of live credential material travelled into
the limiter's dict keys and every log line, metric label, and 429
diagnostic that echoes a bucket. A digest identifies a client exactly as
well without carrying any of the secret.

**N-13 — OpenAPI docs gating.** ``/docs``, ``/redoc`` and
``/openapi.json`` were served unconditionally. On loopback that is a
development convenience; on a routable bind it hands an unauthenticated
scanner a complete map of the admin surface. Gate: docs stay on for a
loopback bind, and switch off once the server is authenticated *and*
bound somewhere reachable.
"""

from __future__ import annotations

import hashlib

from mind_mem.api.rest import _client_id_from_token, _docs_enabled


class TestBucketKeyIsADigest:
    def test_bucket_key_contains_no_token_material(self):
        token = "sk-live-ABCDEFGHIJKLMNOP-tail1234567890"
        bucket = _client_id_from_token(token)
        assert token not in bucket
        # The old scheme leaked exactly this.
        assert token[-16:] not in bucket

    def test_bucket_key_is_the_sha256_prefix(self):
        token = "some-token-value"
        assert _client_id_from_token(token) == hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]

    def test_bucket_key_is_stable_and_distinct(self):
        a, b = "token-aaa", "token-bbb"
        assert _client_id_from_token(a) == _client_id_from_token(a)
        assert _client_id_from_token(a) != _client_id_from_token(b)

    def test_short_token_is_still_hashed(self):
        """The old branch returned a short token VERBATIM as the key."""
        assert _client_id_from_token("abc") != "abc"
        assert len(_client_id_from_token("abc")) == 16

    def test_no_token_is_still_anonymous(self):
        assert _client_id_from_token(None) == "anonymous"


class TestDocsGating:
    def test_docs_on_for_loopback_even_when_authenticated(self, monkeypatch):
        monkeypatch.delenv("MIND_MEM_API_DOCS", raising=False)
        monkeypatch.setenv("MIND_MEM_TOKEN", "t" * 20)
        for host in ("127.0.0.1", "localhost", "::1"):
            assert _docs_enabled(host) is True

    def test_docs_off_for_routable_bind_when_authenticated(self, monkeypatch):
        monkeypatch.delenv("MIND_MEM_API_DOCS", raising=False)
        monkeypatch.setenv("MIND_MEM_TOKEN", "t" * 20)
        for host in ("0.0.0.0", "::", "10.0.0.5", "203.0.113.9"):
            assert _docs_enabled(host) is False

    def test_unknown_bind_keeps_docs(self, monkeypatch):
        """create_app() called directly (tests, ASGI factories) must not
        silently lose its schema -- there is no bind to judge."""
        monkeypatch.delenv("MIND_MEM_API_DOCS", raising=False)
        monkeypatch.delenv("MIND_MEM_BIND_HOST", raising=False)
        monkeypatch.setenv("MIND_MEM_TOKEN", "t" * 20)
        assert _docs_enabled(None) is True

    def test_explicit_override_wins_both_ways(self, monkeypatch):
        monkeypatch.setenv("MIND_MEM_TOKEN", "t" * 20)
        monkeypatch.setenv("MIND_MEM_API_DOCS", "off")
        assert _docs_enabled("127.0.0.1") is False
        monkeypatch.setenv("MIND_MEM_API_DOCS", "on")
        assert _docs_enabled("0.0.0.0") is True

    def test_app_actually_drops_the_routes_when_gated(self, monkeypatch):
        """The gate must reach the app, not just the helper.

        A helper that returns False while FastAPI still mounts /docs is
        the vacuous-guard shape: the check runs, passes, and changes
        nothing.
        """
        from mind_mem.api.rest import create_app

        monkeypatch.delenv("MIND_MEM_API_DOCS", raising=False)
        monkeypatch.setenv("MIND_MEM_TOKEN", "t" * 20)
        monkeypatch.setenv("MIND_MEM_BIND_HOST", "0.0.0.0")
        app = create_app()
        assert app.docs_url is None
        assert app.redoc_url is None
        assert app.openapi_url is None
        served = {getattr(r, "path", None) for r in app.routes}
        assert "/openapi.json" not in served
        assert "/docs" not in served

    def test_app_keeps_routes_on_loopback(self, monkeypatch):
        from mind_mem.api.rest import create_app

        monkeypatch.delenv("MIND_MEM_API_DOCS", raising=False)
        monkeypatch.setenv("MIND_MEM_BIND_HOST", "127.0.0.1")
        app = create_app()
        assert app.openapi_url == "/openapi.json"
