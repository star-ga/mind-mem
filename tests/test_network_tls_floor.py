"""The TLS 1.3 floor holds by construction — roadmap v4.0.0 Group D (RM-2290).

The roadmap item asked for a TLS 1.3 minimum *and* certificate pinning.
The floor is unconditional and lives here. Pinning is **opt-in** and lives in
``test_network_cert_pinning.py``; what this file pins about it is the part
that belongs to the floor — that a client which configures no pin behaves
exactly as it did before pinning existed.

"By construction" is the load-bearing claim, and it is what these tests
attack: the floor lives on an :class:`ssl.SSLContext` that exists *before*
any socket does, so a peer that can only speak TLS 1.2 fails the handshake
and there is no connection to inspect afterwards. Two controls make that
falsifiable rather than decorative:

* a **positive control** — the same client, same CA, same certificate,
  against a TLS 1.3-capable peer, completes the request and gets its JSON.
  A floor that rejected everything would pass a negative test and be
  useless.
* a **server-side control** — the TLS-1.2-only peer that our client
  refuses is proven *functional* in the same test, by connecting to it
  with a floorless stdlib context and succeeding. That rules out "the
  test server was broken" as the reason for the refusal.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import inspect
import socket
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from _tls_certs import mint_ca_and_certs, serving

from mind_mem.v4 import tls_floor
from mind_mem.v4.federation_client import FederationClient, FederationTransportError

# ---------------------------------------------------------------------------
# Unit: the contexts mind-mem builds
# ---------------------------------------------------------------------------


class TestContextsCarryTheFloor:
    def test_client_context_minimum_version_is_tls13(self) -> None:
        ctx = tls_floor.client_context()
        assert ctx.minimum_version is ssl.TLSVersion.TLSv1_3
        assert tls_floor.context_meets_floor(ctx)

    def test_client_context_keeps_verification_and_hostname_checks(self) -> None:
        """Raising the floor must not have turned verification off."""
        ctx = tls_floor.client_context()
        assert ctx.verify_mode is ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_client_key_without_cert_is_refused(self) -> None:
        with pytest.raises(ValueError, match="client_cert"):
            tls_floor.client_context(client_key="/nonexistent/key.pem")

    def test_floor_is_refused_when_the_interpreter_cannot_enforce_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No TLS 1.3 in OpenSSL means no context at all, not a quiet downgrade."""
        # Positive control first: unpatched, this interpreter builds one.
        assert tls_floor.client_context() is not None
        monkeypatch.setattr(ssl, "HAS_TLSv1_3", False)
        with pytest.raises(tls_floor.TlsFloorUnavailable):
            tls_floor.client_context()

    def test_no_environment_variable_can_lower_the_floor(self) -> None:
        """A floor an operator can switch off is not a floor.

        Pins the absence of an override knob in the enforcement module —
        the cheapest way to reverse this hardening would be to add one.
        """
        source = inspect.getsource(tls_floor)
        assert "os.environ" not in source
        assert "getenv" not in source
        assert "\nimport os" not in source


class TestPinningIsOptInOnTopOfTheFloor:
    def test_decision_is_recorded_with_its_default_and_its_alternative(self) -> None:
        text = tls_floor.CERT_PINNING_DECISION
        assert "OPT-IN" in text
        assert "off by default" in text
        # A recorded decision that does not say what to use instead is a
        # gap with better prose.
        assert "mutual TLS" in text

    def test_a_client_with_no_pin_is_the_client_that_shipped_before_pinning(self) -> None:
        """The floor is unconditional; the pin is not. Nothing pins by default."""
        client = FederationClient("https://peer.example.com")
        assert client._pins is None
        https = [h for h in client._opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
        assert [type(h) for h in https] == [urllib.request.HTTPSHandler], (
            "a client that configured no pin got something other than the stock HTTPSHandler"
        )


# ---------------------------------------------------------------------------
# Wiring: the client the product actually uses
# ---------------------------------------------------------------------------


class TestFederationClientIsWiredToTheFloor:
    def test_https_client_installs_the_floored_context(self) -> None:
        client = FederationClient("https://peer.example.com")
        assert client._ssl_context is not None
        assert tls_floor.context_meets_floor(client._ssl_context)
        https = [h for h in client._opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
        assert https, "the strict opener has no HTTPSHandler, so urllib would fall back to a floorless default context"
        assert https[0]._context is client._ssl_context

    def test_plain_http_client_is_unchanged(self) -> None:
        """http:// peers (the loopback default) must behave exactly as before."""
        client = FederationClient("http://peer.local:8765")
        assert client._ssl_context is None

    def test_tls_options_on_a_plain_http_url_are_refused_not_ignored(self) -> None:
        with pytest.raises(FederationTransportError, match="silently ignored"):
            FederationClient("http://peer.local:8765", cafile="/etc/ssl/certs/ca.pem")


# ---------------------------------------------------------------------------
# Real handshakes
# ---------------------------------------------------------------------------


@pytest.fixture()
def certs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    pytest.importorskip(
        "cryptography",
        reason="cryptography is needed to mint the throwaway CA these handshake tests use",
    )
    return mint_ca_and_certs(tmp_path)


class TestRealHandshakes:
    def test_conforming_peer_succeeds(self, certs: tuple[Path, ...]) -> None:
        """POSITIVE CONTROL: the floor lets a TLS 1.3 peer through.

        Without this, every other assertion in this file is satisfied by a
        client that refuses to connect to anything.
        """
        ca, cert, key, _cc, _ck = certs
        with serving(tls_floor.server_context(str(cert), str(key))) as server:
            client = FederationClient(f"https://127.0.0.1:{server.server_port}", cafile=str(ca), timeout=10.0)
            assert client.get_vclock("block-42") == {"peer-a": 7}
            assert server.requests == ["/federation/vclock/block-42"]

    def test_tls12_only_peer_cannot_be_reached_and_never_sees_a_request(self, certs: tuple[Path, ...]) -> None:
        """The floor refuses the connection; the request is never sent.

        The peer is proven functional inside this same test by a floorless
        stdlib client that talks to it successfully — so the refusal is the
        floor's doing and not a broken fixture.
        """
        ca, cert, key, _cc, _ck = certs
        legacy = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        legacy.load_cert_chain(str(cert), str(key))
        legacy.maximum_version = ssl.TLSVersion.TLSv1_2

        with serving(legacy) as server:
            # Server-side control: a client with no floor completes the
            # handshake against this exact listener.
            floorless = ssl.create_default_context(cafile=str(ca))
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=10) as raw:
                with floorless.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
                    assert tls.version() == "TLSv1.2"

            client = FederationClient(f"https://127.0.0.1:{server.server_port}", cafile=str(ca), timeout=10.0)
            before = len(server.requests)
            with pytest.raises(FederationTransportError):
                client.get_vclock("block-42")
            assert len(server.requests) == before, "a request reached the peer over a sub-floor connection"

    def test_mutual_tls_binds_the_client_identity(self, certs: tuple[Path, ...]) -> None:
        """mTLS is the offered replacement for pinning, so it has to work.

        Roadmap item *mTLS + certificate pinning on FederationClient*
        (RM-2382): the mTLS half ships, the pinning half is declined.
        """
        ca, cert, key, client_cert, client_key = certs
        with serving(tls_floor.server_context(str(cert), str(key), client_ca=str(ca))) as server:
            url = f"https://127.0.0.1:{server.server_port}"

            # No client certificate -> the listener rejects the connection.
            anonymous = FederationClient(url, cafile=str(ca), timeout=10.0)
            with pytest.raises(FederationTransportError):
                anonymous.get_vclock("block-42")
            assert server.requests == []

            # POSITIVE CONTROL: same listener, same CA, with a certificate.
            authenticated = FederationClient(
                url,
                cafile=str(ca),
                client_cert=str(client_cert),
                client_key=str(client_key),
                timeout=10.0,
            )
            assert authenticated.get_vclock("block-42") == {"peer-a": 7}
            assert server.requests == ["/federation/vclock/block-42"]


# ---------------------------------------------------------------------------
# The inbound half: the server bind
# ---------------------------------------------------------------------------


async def _never_served(scope: Any, receive: Any, send: Any) -> None:  # pragma: no cover
    raise AssertionError("this app is never invoked; these tests only build the config")


class TestRestListenerFloor:
    def test_uvicorn_default_has_no_floor_but_ours_does(self, certs: tuple[Path, ...]) -> None:
        """The floor is mind-mem's doing, not something uvicorn already did.

        The stock config is loaded here as the control: if uvicorn ever
        starts flooring its own listener this assertion goes red and the
        wrapper can be reconsidered, rather than sitting there claiming
        credit for someone else's default.
        """
        uvicorn = pytest.importorskip("uvicorn", reason="uvicorn is only installed with the [api] extra")
        from mind_mem.api.rest import _floored_uvicorn_config

        _ca, cert, key, _cc, _ck = certs

        stock = uvicorn.Config(_never_served, host="127.0.0.1", port=0, ssl_certfile=str(cert), ssl_keyfile=str(key))
        stock.load()
        assert stock.ssl is not None
        assert stock.ssl.minimum_version is not ssl.TLSVersion.TLSv1_3

        floored = _floored_uvicorn_config(
            _never_served,
            "127.0.0.1",
            0,
            tls_certfile=str(cert),
            tls_keyfile=str(key),
            tls_keyfile_password=None,
            tls_client_ca=None,
        )
        assert floored.ssl is not None
        assert floored.ssl.minimum_version is ssl.TLSVersion.TLSv1_3

    def test_client_ca_switches_the_listener_to_mutual_tls(self, certs: tuple[Path, ...]) -> None:
        pytest.importorskip("uvicorn", reason="uvicorn is only installed with the [api] extra")
        from mind_mem.api.rest import _floored_uvicorn_config

        ca, cert, key, _cc, _ck = certs

        floored = _floored_uvicorn_config(
            _never_served,
            "127.0.0.1",
            0,
            tls_certfile=str(cert),
            tls_keyfile=str(key),
            tls_keyfile_password=None,
            tls_client_ca=str(ca),
        )
        assert floored.ssl is not None
        assert floored.ssl.verify_mode is ssl.CERT_REQUIRED
        assert floored.ssl.minimum_version is ssl.TLSVersion.TLSv1_3

    def test_run_refuses_half_configured_tls(self) -> None:
        pytest.importorskip("uvicorn", reason="uvicorn is only installed with the [api] extra")
        from mind_mem.api.rest import run

        with pytest.raises(ValueError, match="tls_certfile"):
            run(tls_keyfile="/nonexistent/key.pem")
        with pytest.raises(ValueError, match="mutual TLS"):
            run(tls_client_ca="/nonexistent/ca.pem")
