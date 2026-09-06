"""Certificate pinning is opt-in, and when it is on it refuses — RM-2290 / RM-2382.

Two roadmap items ask for pinning: *TLS 1.3 minimum + cert pinning* and
*mTLS + certificate pinning on ``FederationClient``*. The floor half and the
mutual-TLS half already shipped (``test_network_tls_floor.py``); this file is
the pinning half.

What is asserted, and why each assertion can fail:

* the fingerprint is computed by a stdlib DER walk (mind-mem's core has no
  third-party dependencies), so it is checked against ``cryptography``'s own
  SubjectPublicKeyInfo serializer — an **independent oracle**, not a
  re-implementation of the same walk.
* a **mismatched pin is refused, and the request never leaves**. The peer
  records every request that reaches a handler; a refusal that happened
  after the request was sent would leave a row there.
* a **matching pin completes the request** (positive control). Without it a
  client that refuses everything would pass the negative test.
* **no pin configured is the behaviour that shipped before pinning existed** —
  the stock ``HTTPSHandler``, no pin verification, same handshake.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import socket
import ssl
import urllib.request
from pathlib import Path

import pytest
from _tls_certs import mint_ca_and_certs, serving, spki_fingerprint

from mind_mem.v4 import tls_floor
from mind_mem.v4.federation_client import FederationClient, FederationTransportError


@pytest.fixture()
def certs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    pytest.importorskip(
        "cryptography",
        reason="cryptography is needed to mint the throwaway CA these pinning tests use",
    )
    return mint_ca_and_certs(tmp_path)


def _der(pem_path: Path) -> bytes:
    return ssl.PEM_cert_to_DER_cert(pem_path.read_text(encoding="ascii"))


# ---------------------------------------------------------------------------
# The fingerprint itself
# ---------------------------------------------------------------------------


class TestSpkiFingerprint:
    def test_matches_the_cryptography_oracle_for_every_cert_in_the_chain(self, certs: tuple[Path, ...]) -> None:
        """The stdlib DER walk agrees with a real X.509 library, on three certs."""
        ca, server_cert, _key, client_cert, _ck = certs
        for path in (ca, server_cert, client_cert):
            assert tls_floor.spki_sha256(_der(path)) == spki_fingerprint(path), path.name

    def test_two_different_keys_do_not_share_a_fingerprint(self, certs: tuple[Path, ...]) -> None:
        """Otherwise every pin would match every peer and the check is theatre."""
        _ca, server_cert, _key, client_cert, _ck = certs
        assert tls_floor.spki_sha256(_der(server_cert)) != tls_floor.spki_sha256(_der(client_cert))

    def test_it_is_the_public_key_not_the_certificate_that_is_hashed(self, certs: tuple[Path, ...]) -> None:
        """SPKI, not cert-DER: a renewal that keeps the key must keep the pin.

        Hashing the whole certificate would be one line and would break on
        every routine renewal — the operational trap the recorded decision
        names. This pins that we did not take the one-line route.
        """
        _ca, server_cert, _key, _cc, _ck = certs
        der = _der(server_cert)
        assert tls_floor.spki_sha256(der) != hashlib.sha256(der).hexdigest()

    def test_garbage_is_refused_rather_than_hashed(self) -> None:
        with pytest.raises(ValueError):
            tls_floor.spki_sha256(b"this is not a certificate")

    def test_a_truncated_certificate_is_refused(self, certs: tuple[Path, ...]) -> None:
        _ca, server_cert, _key, _cc, _ck = certs
        der = _der(server_cert)
        with pytest.raises(ValueError):
            tls_floor.spki_sha256(der[: len(der) // 2])


class TestPinNormalisation:
    def test_the_accepted_spellings_all_mean_the_same_pin(self, certs: tuple[Path, ...]) -> None:
        _ca, server_cert, _key, _cc, _ck = certs
        hex_pin = spki_fingerprint(server_cert)
        digest = bytes.fromhex(hex_pin)
        colonised = ":".join(hex_pin[i : i + 2] for i in range(0, len(hex_pin), 2))
        b64 = base64.b64encode(digest).decode("ascii")
        spellings = [hex_pin, hex_pin.upper(), colonised, f"sha256:{hex_pin}", f"sha256//{b64}", b64]
        assert tls_floor.normalise_pins(spellings) == frozenset({hex_pin})

    def test_a_single_string_is_a_pin_set_of_one(self) -> None:
        pin = "a" * 64
        assert tls_floor.normalise_pins(pin) == frozenset({pin})

    def test_a_malformed_pin_is_refused(self) -> None:
        with pytest.raises(ValueError):
            tls_floor.normalise_pins("not-a-fingerprint")

    def test_a_short_hex_digest_is_refused_rather_than_padded(self) -> None:
        with pytest.raises(ValueError):
            tls_floor.normalise_pins("abcd")

    def test_an_empty_pin_set_is_refused(self) -> None:
        """An empty set would match nothing while looking configured."""
        with pytest.raises(ValueError):
            tls_floor.normalise_pins([])


# ---------------------------------------------------------------------------
# Wiring on the client the product actually uses
# ---------------------------------------------------------------------------


class TestFederationClientPinWiring:
    def test_a_pin_installs_a_pin_verifying_handler(self) -> None:
        client = FederationClient("https://peer.example.com", pinned_pubkey_sha256="0" * 64)
        assert client._pins == frozenset({"0" * 64})
        https = [h for h in client._opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
        assert https, "the strict opener lost its HTTPSHandler"
        assert type(https[0]) is not urllib.request.HTTPSHandler, "the pin was accepted but nothing verifies it"

    def test_the_scheme_allowlist_and_redirect_cap_survive_pinning(self) -> None:
        """Issue #529's guards are not weakened by adding a pin."""
        client = FederationClient("https://peer.example.com", pinned_pubkey_sha256="0" * 64)
        redirect = [h for h in client._opener.handlers if type(h).__name__ == "_SameOriginRedirectHandler"]
        assert redirect, "the same-origin redirect handler is gone"
        with pytest.raises(FederationTransportError, match="not allowed"):
            FederationClient("file:///etc/passwd", pinned_pubkey_sha256="0" * 64)

    def test_a_pin_on_a_plain_http_url_is_refused_not_ignored(self) -> None:
        with pytest.raises(FederationTransportError, match="silently ignored"):
            FederationClient("http://peer.local:8765", pinned_pubkey_sha256="0" * 64)

    def test_a_malformed_pin_is_refused_at_construction(self) -> None:
        with pytest.raises(FederationTransportError, match="pin"):
            FederationClient("https://peer.example.com", pinned_pubkey_sha256="nonsense")


# ---------------------------------------------------------------------------
# Real handshakes
# ---------------------------------------------------------------------------


class TestPinnedHandshakes:
    def test_matching_pin_completes_the_request(self, certs: tuple[Path, ...]) -> None:
        """POSITIVE CONTROL: the pin lets the peer it names through."""
        ca, cert, key, _cc, _ck = certs
        with serving(tls_floor.server_context(str(cert), str(key))) as server:
            client = FederationClient(
                f"https://127.0.0.1:{server.server_port}",
                cafile=str(ca),
                pinned_pubkey_sha256=spki_fingerprint(cert),
                timeout=10.0,
            )
            assert client.get_vclock("block-42") == {"peer-a": 7}
            assert server.requests == ["/federation/vclock/block-42"]

    def test_mismatched_pin_is_refused_and_the_request_never_leaves(self, certs: tuple[Path, ...]) -> None:
        """The CA trusts this peer; the pin does not. The pin wins.

        This is the whole point of pinning: a certificate that passes
        ordinary trust-store verification — a mis-issuance, or a
        TLS-intercepting proxy whose root the machine trusts — is still
        refused. The pinned value is a *real other key* (the client
        certificate's), not a made-up digest, so the refusal cannot be an
        artefact of an unparseable pin.
        """
        ca, cert, key, client_cert, _ck = certs
        with serving(tls_floor.server_context(str(cert), str(key))) as server:
            url = f"https://127.0.0.1:{server.server_port}"

            # Control: the same CA, the same peer, no pin — this connects.
            trusted = FederationClient(url, cafile=str(ca), timeout=10.0)
            assert trusted.get_vclock("block-42") == {"peer-a": 7}
            assert server.requests == ["/federation/vclock/block-42"]

            pinned = FederationClient(
                url,
                cafile=str(ca),
                pinned_pubkey_sha256=spki_fingerprint(client_cert),
                timeout=10.0,
            )
            before = len(server.requests)
            with pytest.raises(FederationTransportError, match="pin"):
                pinned.get_vclock("block-42")
            assert len(server.requests) == before, "a request reached the peer over an unpinned connection"

    def test_a_rotation_set_that_names_the_next_key_still_connects(self, certs: tuple[Path, ...]) -> None:
        """Pinning several keys is how a renewal stops being an outage."""
        ca, cert, key, client_cert, _ck = certs
        with serving(tls_floor.server_context(str(cert), str(key))) as server:
            client = FederationClient(
                f"https://127.0.0.1:{server.server_port}",
                cafile=str(ca),
                pinned_pubkey_sha256=[spki_fingerprint(client_cert), spki_fingerprint(cert)],
                timeout=10.0,
            )
            assert client.get_vclock("block-42") == {"peer-a": 7}

    def test_the_floor_still_applies_to_a_pinned_client(self, certs: tuple[Path, ...]) -> None:
        """Pinning is additional to the floor, never instead of it."""
        ca, cert, key, _cc, _ck = certs
        legacy = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        legacy.load_cert_chain(str(cert), str(key))
        legacy.maximum_version = ssl.TLSVersion.TLSv1_2

        with serving(legacy) as server:
            client = FederationClient(
                f"https://127.0.0.1:{server.server_port}",
                cafile=str(ca),
                pinned_pubkey_sha256=spki_fingerprint(cert),
                timeout=10.0,
            )
            with pytest.raises(FederationTransportError):
                client.get_vclock("block-42")
            assert server.requests == []


# ---------------------------------------------------------------------------
# The inbound half: the stdlib HTTP transport
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "intelligence" / "state").mkdir(parents=True)
    (ws / "decisions").mkdir(parents=True)
    config = {"version": "3.9.0", "workspace_path": str(ws), "block_store": {"backend": "markdown"}}
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    return str(ws)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _status_over(port: int, ctx: ssl.SSLContext | None) -> tuple[int, str | None]:
    """GET /status, returning ``(http status, negotiated TLS version)``."""
    conn: http.client.HTTPConnection
    if ctx is None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    else:
        conn = http.client.HTTPSConnection("127.0.0.1", port, timeout=10, context=ctx)
    try:
        conn.request("GET", "/status")
        # Read the negotiated version here, while the socket is still
        # open: the server answers "Connection: close", so by the time
        # the body has been read http.client has dropped ``conn.sock``.
        version = conn.sock.version() if isinstance(conn.sock, ssl.SSLSocket) else None
        resp = conn.getresponse()
        resp.read()
        return resp.status, version
    finally:
        conn.close()


class TestHttpTransportTlsListener:
    def test_plain_listener_is_unchanged(self, workspace: str) -> None:
        """CONTROL: no TLS arguments means the transport that shipped."""
        from mind_mem.http_transport import serve_http

        port = _free_port()
        _thread, stop = serve_http(
            workspace=workspace,
            host="127.0.0.1",
            port=port,
            allow_unauthenticated_localhost=True,
        )
        try:
            assert _status_over(port, None) == (200, None)
        finally:
            stop()

    def test_tls_listener_serves_over_tls13(self, workspace: str, certs: tuple[Path, ...]) -> None:
        from mind_mem.http_transport import serve_http

        ca, cert, key, _cc, _ck = certs
        port = _free_port()
        _thread, stop = serve_http(
            workspace=workspace,
            host="127.0.0.1",
            port=port,
            allow_unauthenticated_localhost=True,
            tls_certfile=str(cert),
            tls_keyfile=str(key),
        )
        try:
            assert _status_over(port, tls_floor.client_context(cafile=str(ca))) == (200, "TLSv1.3")
        finally:
            stop()

    def test_a_tls12_only_client_cannot_reach_the_tls_listener(self, workspace: str, certs: tuple[Path, ...]) -> None:
        """The listener's floor, proven from the client side.

        The same client with no ceiling succeeds in the test above, so a
        refusal here is the floor's doing and not a broken listener.
        """
        from mind_mem.http_transport import serve_http

        ca, cert, key, _cc, _ck = certs
        port = _free_port()
        _thread, stop = serve_http(
            workspace=workspace,
            host="127.0.0.1",
            port=port,
            allow_unauthenticated_localhost=True,
            tls_certfile=str(cert),
            tls_keyfile=str(key),
        )
        try:
            capped = ssl.create_default_context(cafile=str(ca))
            capped.maximum_version = ssl.TLSVersion.TLSv1_2
            with pytest.raises(OSError):
                _status_over(port, capped)
        finally:
            stop()

    def test_client_ca_makes_a_client_certificate_mandatory(self, workspace: str, certs: tuple[Path, ...]) -> None:
        from mind_mem.http_transport import serve_http

        ca, cert, key, client_cert, client_key = certs
        port = _free_port()
        _thread, stop = serve_http(
            workspace=workspace,
            host="127.0.0.1",
            port=port,
            allow_unauthenticated_localhost=True,
            tls_certfile=str(cert),
            tls_keyfile=str(key),
            tls_client_ca=str(ca),
        )
        try:
            with pytest.raises(OSError):
                _status_over(port, tls_floor.client_context(cafile=str(ca)))
            # POSITIVE CONTROL: same listener, with a certificate.
            authenticated = tls_floor.client_context(
                cafile=str(ca),
                client_cert=str(client_cert),
                client_key=str(client_key),
            )
            assert _status_over(port, authenticated) == (200, "TLSv1.3")
        finally:
            stop()

    def test_half_configured_tls_is_refused_before_the_bind(self, workspace: str) -> None:
        from mind_mem.http_transport import serve_http

        with pytest.raises(ValueError, match="tls_certfile"):
            serve_http(workspace=workspace, allow_unauthenticated_localhost=True, tls_keyfile="/nonexistent/key.pem")
        with pytest.raises(ValueError, match="mutual TLS"):
            serve_http(workspace=workspace, allow_unauthenticated_localhost=True, tls_client_ca="/nonexistent/ca.pem")


# ---------------------------------------------------------------------------
# Reachability: the operator's actual entry point
# ---------------------------------------------------------------------------


class _FakeThread:
    def join(self) -> None:
        return


class TestCliReachesTheTlsListener:
    """An option nobody can reach from the CLI is not shipped hardening.

    These walk the real path — ``argv -> build_parser -> args.func ->
    serve_http`` — and capture what the last leg was actually called with.
    """

    def test_http_serve_passes_its_tls_flags_through(
        self,
        workspace: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mind_mem import http_transport, mm_cli

        seen: dict[str, object] = {}

        def _fake_serve_http(**kwargs: object) -> tuple[_FakeThread, object]:
            seen.update(kwargs)
            return _FakeThread(), (lambda: None)

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.setenv(mm_cli.TLS_KEY_PASSWORD_ENV, "hunter2")
        monkeypatch.setattr(http_transport, "serve_http", _fake_serve_http)

        args = mm_cli.build_parser().parse_args(
            [
                "http-serve",
                "--allow-unauthenticated-localhost",
                "--tls-certfile",
                "/etc/mind-mem/peer.crt",
                "--tls-keyfile",
                "/etc/mind-mem/peer.key",
                "--tls-client-ca",
                "/etc/mind-mem/ca.pem",
            ]
        )
        assert args.func(args) == 0
        assert seen["tls_certfile"] == "/etc/mind-mem/peer.crt"
        assert seen["tls_keyfile"] == "/etc/mind-mem/peer.key"
        assert seen["tls_client_ca"] == "/etc/mind-mem/ca.pem"
        # The passphrase comes from the environment, never from argv.
        assert seen["tls_keyfile_password"] == "hunter2"

    def test_http_serve_without_tls_flags_asks_for_no_tls(
        self,
        workspace: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CONTROL: the default invocation is still the plain listener."""
        from mind_mem import http_transport, mm_cli

        seen: dict[str, object] = {}

        def _fake_serve_http(**kwargs: object) -> tuple[_FakeThread, object]:
            seen.update(kwargs)
            return _FakeThread(), (lambda: None)

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.delenv(mm_cli.TLS_KEY_PASSWORD_ENV, raising=False)
        monkeypatch.setattr(http_transport, "serve_http", _fake_serve_http)

        args = mm_cli.build_parser().parse_args(["http-serve", "--allow-unauthenticated-localhost"])
        assert args.func(args) == 0
        assert seen["tls_certfile"] is None
        assert seen["tls_keyfile"] is None
        assert seen["tls_client_ca"] is None
        assert seen["tls_keyfile_password"] is None

    def test_serve_passes_its_tls_flags_to_the_rest_listener(
        self,
        workspace: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("uvicorn", reason="the REST listener is only installed with the [api] extra")
        pytest.importorskip("fastapi", reason="the REST listener is only installed with the [api] extra")
        from mind_mem import mm_cli
        from mind_mem.api import rest

        seen: dict[str, object] = {}

        def _fake_run(**kwargs: object) -> None:
            seen.update(kwargs)

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.setattr(rest, "run", _fake_run)

        args = mm_cli.build_parser().parse_args(
            ["serve", "--tls-certfile", "/etc/mind-mem/api.crt", "--tls-client-ca", "/etc/mind-mem/ca.pem"]
        )
        assert args.func(args) == 0
        assert seen["tls_certfile"] == "/etc/mind-mem/api.crt"
        assert seen["tls_client_ca"] == "/etc/mind-mem/ca.pem"

    def test_no_tls_passphrase_flag_exists_on_either_command(self) -> None:
        """A key passphrase on argv is readable by every process on the box.

        argparse gives every registered option a ``dest`` on the parsed
        namespace, so the absence of one is the absence of the flag.
        """
        from mind_mem import mm_cli

        parser = mm_cli.build_parser()
        for argv in (["serve"], ["http-serve", "--allow-unauthenticated-localhost"]):
            args = parser.parse_args(argv)
            assert hasattr(args, "tls_certfile"), f"{argv[0]} lost its TLS flags"
            assert not hasattr(args, "tls_keyfile_password"), f"{argv[0]} takes a key passphrase on the command line"
            assert not hasattr(args, "tls_key_password"), f"{argv[0]} takes a key passphrase on the command line"
