"""The MCP HTTP startup gate must agree with what actually enforces auth.

The v3.7.0 H4 gate in ``mind_mem.mcp.server`` accepted three
alternatives, but only two of them were ever enforced. It treated the
mere presence of ``OIDC_ISSUER`` + ``OIDC_AUDIENCE`` as "authentication
is configured" and returned early — which also skipped the loopback-bind
check underneath it. Nothing under ``mind_mem.mcp`` can verify a JWT:
the OIDC verifier lives in the separate ``mind_mem.api.rest`` FastAPI
app, and the MCP transport authenticates solely through the
``StaticTokenVerifier`` map built by ``_build_http_auth_tokens()`` from
``MIND_MEM_TOKEN`` / ``MIND_MEM_ADMIN_TOKEN``.

So ``OIDC_ISSUER=... OIDC_AUDIENCE=... mind-mem-mcp --transport http
--host 0.0.0.0`` passed the gate, left ``mcp.auth`` as ``None``, and
published every tool on every interface with no credential check — while
the startup log reported the deployment as authenticated.

These tests pin the invariant that closes that class of drift: the gate
admits a configuration **if and only if** the verifier map it is
standing in front of is non-empty.
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings
from unittest import mock

_ENV_KEYS = (
    "MIND_MEM_TOKEN",
    "MIND_MEM_ADMIN_TOKEN",
    "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST",
    "OIDC_ISSUER",
    "OIDC_AUDIENCE",
)

_GOOD_TOKEN = "t" * 32
_OIDC_ENV = {"OIDC_ISSUER": "https://idp.example.com", "OIDC_AUDIENCE": "mind-mem"}


class _EnvIsolated(unittest.TestCase):
    """Clears every auth-relevant env var and restores it afterwards."""

    def setUp(self) -> None:
        self._snapshot = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @staticmethod
    def _enforce(host: str, allow_flag: bool) -> None:
        from mind_mem.mcp.server import _enforce_http_auth_or_localhost

        _enforce_http_auth_or_localhost(host, allow_flag)


class TestOIDCIsNotMCPAuth(_EnvIsolated):
    """OIDC env vars configure the REST app, never the MCP transport."""

    def test_oidc_env_does_not_open_a_routable_listener(self) -> None:
        os.environ.update(_OIDC_ENV)
        with self.assertRaises(SystemExit) as cm:
            self._enforce("0.0.0.0", allow_flag=False)
        self.assertIn("refusing to start", str(cm.exception))

    def test_oidc_env_does_not_even_satisfy_a_loopback_bind(self) -> None:
        os.environ.update(_OIDC_ENV)
        with self.assertRaises(SystemExit):
            self._enforce("127.0.0.1", allow_flag=False)

    def test_oidc_env_still_forces_the_loopback_check_under_the_optin(self) -> None:
        # The early return on OIDC also jumped over the loopback-bind
        # check, so the opt-in flag stopped constraining the host.
        os.environ.update(_OIDC_ENV)
        with self.assertRaises(SystemExit) as cm:
            self._enforce("0.0.0.0", allow_flag=True)
        self.assertIn("loopback bind", str(cm.exception))

    def test_oidc_plus_real_token_is_accepted(self) -> None:
        # Positive control: OIDC is ignored, the token is what counts.
        os.environ.update(_OIDC_ENV)
        os.environ["MIND_MEM_TOKEN"] = _GOOD_TOKEN
        self._enforce("0.0.0.0", allow_flag=False)


class TestEmptyTokenIsNotAuth(_EnvIsolated):
    """An empty token never reaches the verifier map, so it is not auth."""

    def test_empty_user_token_refused(self) -> None:
        os.environ["MIND_MEM_TOKEN"] = ""
        with self.assertRaises(SystemExit) as cm:
            self._enforce("0.0.0.0", allow_flag=False)
        self.assertIn("refusing to start", str(cm.exception))

    def test_empty_admin_token_refused(self) -> None:
        os.environ["MIND_MEM_ADMIN_TOKEN"] = ""
        with self.assertRaises(SystemExit):
            self._enforce("0.0.0.0", allow_flag=False)


class TestGateAgreesWithVerifierMap(_EnvIsolated):
    """The anti-drift invariant, stated directly over a config matrix.

    Whatever the gate lets through on a routable host without the
    loopback opt-in must be exactly what ``_build_http_auth_tokens()``
    would arm the ``StaticTokenVerifier`` with. Any future "third way"
    to satisfy the gate that the verifier does not understand trips
    this test.
    """

    _MATRIX: tuple[dict[str, str], ...] = (
        {},
        dict(_OIDC_ENV),
        {"MIND_MEM_TOKEN": ""},
        {"MIND_MEM_ADMIN_TOKEN": ""},
        {"MIND_MEM_TOKEN": "", **_OIDC_ENV},
        {"MIND_MEM_TOKEN": _GOOD_TOKEN},
        {"MIND_MEM_ADMIN_TOKEN": _GOOD_TOKEN},
        {"MIND_MEM_TOKEN": _GOOD_TOKEN, "MIND_MEM_ADMIN_TOKEN": "a" * 32},
        {"MIND_MEM_TOKEN": _GOOD_TOKEN, **_OIDC_ENV},
    )

    def test_accepts_iff_verifier_map_is_non_empty(self) -> None:
        from mind_mem.mcp.infra.http_auth import _build_http_auth_tokens

        for env in self._MATRIX:
            with self.subTest(env=env):
                for k in _ENV_KEYS:
                    os.environ.pop(k, None)
                os.environ.update(env)

                enforceable = bool(_build_http_auth_tokens())
                try:
                    self._enforce("0.0.0.0", allow_flag=False)
                    gate_accepted = True
                except SystemExit:
                    gate_accepted = False

                self.assertEqual(
                    gate_accepted,
                    enforceable,
                    f"gate accepted={gate_accepted} but StaticTokenVerifier would be armed={enforceable} for {env}",
                )


class TestMainNeverBindsUnauthenticated(_EnvIsolated):
    """End-to-end probe: the listener must not open at all.

    Asserting on the helper alone would not prove the exploit is closed —
    ``main()`` is what actually calls ``mcp.run``. Here we drive the real
    entry point with the vulnerable environment and assert the transport
    is never started.
    """

    def test_oidc_env_and_wildcard_host_never_reaches_mcp_run(self) -> None:
        from mind_mem.mcp import server as srv

        os.environ.update(_OIDC_ENV)
        argv = ["mind-mem-mcp", "--transport", "http", "--host", "0.0.0.0"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(srv.mcp, "run") as run:
            with self.assertRaises(SystemExit):
                srv.main()
        run.assert_not_called()
        self.assertIsNone(getattr(srv.mcp, "auth", None))

    def test_loopback_optin_still_starts_and_marks_itself_unauthenticated(self) -> None:
        # Guard against over-fixing: the blessed unauthenticated dev flow
        # must still work, and must still set the opt-in env var that
        # ``verify_token`` reads.
        from mind_mem.mcp import server as srv

        argv = [
            "mind-mem-mcp",
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--allow-unauthenticated-localhost",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(srv.mcp, "run") as run:
            srv.main()
        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
        self.assertIsNone(getattr(srv.mcp, "auth", None))
        self.assertEqual(os.environ.get("MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST"), "1")

    def test_token_arms_the_verifier_before_the_listener_opens(self) -> None:
        # The other half of the mechanism: when the gate does let a
        # routable bind through, a verifier really is installed.
        from mind_mem.mcp import server as srv

        os.environ["MIND_MEM_TOKEN"] = _GOOD_TOKEN
        argv = ["mind-mem-mcp", "--transport", "http", "--host", "0.0.0.0"]
        previous_auth = getattr(srv.mcp, "auth", None)
        try:
            with mock.patch.object(sys, "argv", argv), mock.patch.object(srv.mcp, "run") as run:
                with warnings.catch_warnings():
                    # main() warns that a user-scope-only token was supplied;
                    # that advisory is not what this test is about.
                    warnings.simplefilter("ignore", UserWarning)
                    srv.main()
            self.assertEqual(run.call_args.kwargs["host"], "0.0.0.0")
            self.assertIsNotNone(srv.mcp.auth)
            self.assertEqual(type(srv.mcp.auth).__name__, "StaticTokenVerifier")
        finally:
            srv.mcp.auth = previous_auth


if __name__ == "__main__":
    unittest.main()
