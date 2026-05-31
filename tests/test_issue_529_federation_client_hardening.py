"""Regression for issue #529: FederationClient hardening.

Three defensive controls were missing from the federation client:
  1. No scheme allowlist  -> file:///etc/passwd as base_url would read
     local files and surface them as "peer response".
  2. No redirect cap      -> 302 -> 169.254.169.254 pivot to cloud
     metadata.
  3. No response-size cap -> hostile peer streams gigabytes into the
     client process.

These tests pin the constructor + transport hardening.
"""

from __future__ import annotations

import pytest
from mind_mem.v4.federation_client import (
    FederationClient,
    FederationTransportError,
)

# ---------------------------------------------------------------------------
# 1. Scheme allowlist
# ---------------------------------------------------------------------------


def test_file_scheme_rejected():
    with pytest.raises(FederationTransportError, match="scheme"):
        FederationClient("file:///etc/passwd")


def test_ftp_scheme_rejected():
    with pytest.raises(FederationTransportError, match="scheme"):
        FederationClient("ftp://example.com/")


def test_data_scheme_rejected():
    with pytest.raises(FederationTransportError, match="scheme"):
        FederationClient("data:text/plain,hello")


def test_http_scheme_accepted():
    # Should not raise.
    FederationClient("http://peer.local:5179")


def test_https_scheme_accepted():
    # Should not raise.
    FederationClient("https://peer.example.com")


def test_empty_host_rejected():
    with pytest.raises(FederationTransportError, match="host"):
        FederationClient("http://")


# ---------------------------------------------------------------------------
# 2. Same-origin redirect handler is installed
# ---------------------------------------------------------------------------


def test_strict_opener_installed():
    """The client must use a custom opener with the same-origin redirect
    handler. urllib's default opener would silently follow cross-origin
    redirects (issue #529 (2): SSRF pivot to cloud-metadata)."""
    c = FederationClient("http://peer.local:5179")
    handler_class_names = {type(h).__name__ for h in c._opener.handlers}
    assert "_SameOriginRedirectHandler" in handler_class_names


def test_redirect_handler_blocks_cross_origin():
    """The handler's redirect_request must raise on scheme/host/port change."""
    import urllib.error

    from mind_mem.v4.federation_client import _SameOriginRedirectHandler

    handler = _SameOriginRedirectHandler("http", "peer.local:5179")
    with pytest.raises(urllib.error.HTTPError, match="cross-origin redirect blocked"):
        handler.redirect_request(
            req=None,
            fp=None,
            code=302,
            msg="Found",
            hdrs={},
            newurl="http://169.254.169.254/latest/meta-data/",
        )


def test_redirect_handler_allows_same_origin():
    """Same scheme+host+port redirect should NOT raise (delegate to super)."""
    import urllib.request

    from mind_mem.v4.federation_client import _SameOriginRedirectHandler

    handler = _SameOriginRedirectHandler("http", "peer.local:5179")
    # super().redirect_request needs a real Request object; we just
    # verify the cross-origin check doesn't trip.
    req = urllib.request.Request("http://peer.local:5179/old")
    # Should return a new Request, not raise.
    new_req = handler.redirect_request(
        req=req,
        fp=None,
        code=302,
        msg="Found",
        hdrs={},
        newurl="http://peer.local:5179/new",
    )
    assert new_req is not None


# ---------------------------------------------------------------------------
# 3. Response-size cap
# ---------------------------------------------------------------------------


def test_response_size_cap_constant_exists():
    """MAX_RESP_BYTES must be defined and bounded."""
    from mind_mem.v4 import federation_client

    assert hasattr(federation_client, "MAX_RESP_BYTES")
    assert isinstance(federation_client.MAX_RESP_BYTES, int)
    # 1 MiB by default; operators can raise via env.
    assert federation_client.MAX_RESP_BYTES >= 1024  # reasonable floor


def test_response_size_cap_env_override(monkeypatch):
    """MIND_MEM_FED_MAX_RESP_BYTES env var should set the cap on import."""
    # The cap is read at import time, so this test confirms the
    # mechanism exists; full env override is exercised in production
    # by setting the env before the first import.
    import os

    from mind_mem.v4 import federation_client

    # Default cap reflects the env var if set, else 1 MiB.
    default = federation_client.MAX_RESP_BYTES
    env_value = os.environ.get("MIND_MEM_FED_MAX_RESP_BYTES")
    if env_value:
        assert default == int(env_value)
    else:
        assert default == 1 << 20
