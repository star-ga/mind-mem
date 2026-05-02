"""v3.7.0 H4: HTTP / REST auth must fail CLOSED by default.

Pre-v3.7.0 the helpers returned ``True`` whenever no token was
configured, which left every mutating MCP tool reachable
unauthenticated when an operator forgot to set ``MIND_MEM_TOKEN``.
The new contract is documented in
``mind_mem.mcp.infra.http_auth.verify_token``:

* Token configured + matching header → True
* Token configured + missing/wrong header → False
* No token + ``MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1`` → True
* No token + opt-in absent → False (fail-closed)

The CLI flag ``--allow-unauthenticated-localhost`` on
``mind-mem-mcp`` and ``mm serve`` is the only blessed way to set
that env var; it additionally enforces a loopback bind. These
tests cover both the helper-level contract and the startup checks.
"""

from __future__ import annotations

import os
import unittest


class TestVerifyTokenFailClosed(unittest.TestCase):
    """``verify_token`` is the shared helper used by every HTTP path."""

    def setUp(self):
        self._snapshot = {k: os.environ.get(k) for k in ("MIND_MEM_TOKEN", "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST")}
        for k in self._snapshot:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_no_token_no_optin_rejects(self):
        from mind_mem.mcp.infra.http_auth import verify_token

        self.assertFalse(verify_token({}))
        self.assertFalse(verify_token({"Authorization": "Bearer anything"}))

    def test_no_token_loopback_optin_allows(self):
        from mind_mem.mcp.infra.http_auth import verify_token

        os.environ["MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST"] = "1"
        self.assertTrue(verify_token({}))
        self.assertTrue(verify_token({"Authorization": "Bearer anything"}))

    def test_token_configured_requires_match(self):
        from mind_mem.mcp.infra.http_auth import verify_token

        os.environ["MIND_MEM_TOKEN"] = "right"
        self.assertTrue(verify_token({"Authorization": "Bearer right"}))
        self.assertFalse(verify_token({"Authorization": "Bearer wrong"}))
        self.assertFalse(verify_token({}))

    def test_optin_string_values(self):
        from mind_mem.mcp.infra.http_auth import _unauthenticated_explicitly_allowed

        for truthy in ("1", "true", "TRUE", "yes", "On"):
            os.environ["MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST"] = truthy
            self.assertTrue(_unauthenticated_explicitly_allowed(), truthy)
        for falsy in ("0", "no", "false", "off", ""):
            os.environ["MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST"] = falsy
            self.assertFalse(_unauthenticated_explicitly_allowed(), falsy)


class TestMCPHttpStartupRefusesWithoutAuth(unittest.TestCase):
    """``mind-mem-mcp --transport http`` must refuse to listen unauth.

    The audit's exit-criteria for H4: (a) no token + no flag → refuse
    to start, (b) flag + ``0.0.0.0`` → refuse, (c) flag + ``127.0.0.1``
    → start unauthenticated. We invoke the CLI as a subprocess so
    parser + enforcement both run end-to-end.
    """

    def setUp(self):
        self._snapshot = {
            k: os.environ.get(k)
            for k in (
                "MIND_MEM_TOKEN",
                "MIND_MEM_ADMIN_TOKEN",
                "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST",
                "OIDC_ISSUER",
                "OIDC_AUDIENCE",
            )
        }
        for k in self._snapshot:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _enforce(self, host, allow_flag):
        from mind_mem.mcp.server import _enforce_http_auth_or_localhost

        return _enforce_http_auth_or_localhost(host, allow_flag)

    def test_no_token_no_flag_refuses_to_start(self):
        with self.assertRaises(SystemExit) as cm:
            self._enforce("127.0.0.1", allow_flag=False)
        self.assertIn("refusing to start", str(cm.exception))

    def test_flag_with_0_0_0_0_refuses_to_start(self):
        with self.assertRaises(SystemExit) as cm:
            self._enforce("0.0.0.0", allow_flag=True)
        self.assertIn("loopback bind", str(cm.exception))

    def test_flag_with_127_0_0_1_starts(self):
        # Should not raise.
        self._enforce("127.0.0.1", allow_flag=True)

    def test_flag_with_localhost_starts(self):
        self._enforce("localhost", allow_flag=True)

    def test_flag_with_ipv6_loopback_starts(self):
        self._enforce("::1", allow_flag=True)

    def test_token_configured_starts_without_flag(self):
        os.environ["MIND_MEM_TOKEN"] = "x" * 32
        self._enforce("0.0.0.0", allow_flag=False)

    def test_admin_token_alone_starts(self):
        os.environ["MIND_MEM_ADMIN_TOKEN"] = "x" * 32
        self._enforce("0.0.0.0", allow_flag=False)

    def test_oidc_alone_starts(self):
        os.environ["OIDC_ISSUER"] = "https://idp.example.com"
        os.environ["OIDC_AUDIENCE"] = "mind-mem"
        self._enforce("0.0.0.0", allow_flag=False)


class TestRestRunFailClosed(unittest.TestCase):
    """``mm serve`` (``mind_mem.api.rest.run``) wears the same gate."""

    def setUp(self):
        self._snapshot = {
            k: os.environ.get(k)
            for k in (
                "MIND_MEM_TOKEN",
                "MIND_MEM_ADMIN_TOKEN",
                "MIND_MEM_API_KEY_DB",
                "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST",
                "OIDC_ISSUER",
                "OIDC_AUDIENCE",
            )
        }
        for k in self._snapshot:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _enforce(self, host, allow):
        try:
            from mind_mem.api.rest import _enforce_fail_closed
        except ImportError:
            self.skipTest("mind-mem[api] extra not installed")
        return _enforce_fail_closed(host, allow)

    def test_no_auth_no_flag_refuses(self):
        with self.assertRaises(SystemExit) as cm:
            self._enforce("127.0.0.1", allow=False)
        self.assertIn("refusing to start", str(cm.exception))

    def test_flag_routable_host_refuses(self):
        with self.assertRaises(SystemExit) as cm:
            self._enforce("0.0.0.0", allow=True)
        self.assertIn("loopback", str(cm.exception))

    def test_flag_loopback_starts(self):
        self._enforce("127.0.0.1", allow=True)
        self._enforce("localhost", allow=True)
        self._enforce("::1", allow=True)


if __name__ == "__main__":
    unittest.main()
