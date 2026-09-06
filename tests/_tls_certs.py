"""Throwaway CA, leaf certificates and a recording TLS peer for the TLS tests.

Extracted from ``test_network_tls_floor.py`` so the pinning tests can mint
the same chain and talk to the same kind of peer without importing a test
module. Nothing here touches a real trust store: everything is minted into
the caller's ``tmp_path`` and expires in a day.

:func:`spki_fingerprint` is deliberately computed with ``cryptography``
rather than with :func:`mind_mem.v4.tls_floor.spki_sha256`. It is the
independent oracle the stdlib DER walk is checked against — a fingerprint
helper that shared the parser under test could only ever agree with itself.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator


def mint_ca_and_certs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Return (ca_pem, server_cert, server_key, client_cert, client_key).

    Everything is minted into ``tmp_path`` and lives for a day; nothing
    here touches a real trust store.
    """
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    not_before = now - datetime.timedelta(minutes=5)
    not_after = now + datetime.timedelta(days=1)

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "mind-mem test CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        # Newer OpenSSL builds refuse a chain whose CA carries no subject
        # key identifier ("Missing Authority Key Identifier"), so the
        # throwaway CA is minted with the same extensions a real one has.
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    def _leaf(common_name: str, *, loopback_san: bool) -> tuple[Any, Any]:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        builder = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(not_before)
            .not_valid_after(not_after)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        )
        if loopback_san:
            # The IP SAN is deliberate: connecting by literal 127.0.0.1
            # keeps name resolution (and a machine whose "localhost"
            # answers ::1 first) out of the test.
            builder = builder.add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    ]
                ),
                critical=False,
            )
        return key, builder.sign(ca_key, hashes.SHA256())

    server_key, server_cert = _leaf("localhost", loopback_san=True)
    client_key, client_cert = _leaf("mind-mem test client", loopback_san=False)

    pem = serialization.Encoding.PEM
    fmt = serialization.PrivateFormat.TraditionalOpenSSL
    no_enc = serialization.NoEncryption()

    def _write(name: str, blob: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(blob)
        return path

    return (
        _write("ca.pem", ca_cert.public_bytes(pem)),
        _write("server.crt", server_cert.public_bytes(pem)),
        _write("server.key", server_key.private_bytes(pem, fmt, no_enc)),
        _write("client.crt", client_cert.public_bytes(pem)),
        _write("client.key", client_key.private_bytes(pem, fmt, no_enc)),
    )


def spki_fingerprint(cert_path: Path) -> str:
    """SHA-256 of the certificate's SubjectPublicKeyInfo, lowercase hex.

    Computed through ``cryptography``'s own DER serializer so it can serve
    as the oracle for the stdlib parser in ``mind_mem.v4.tls_floor``.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    spki = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(spki).hexdigest()


class RecordingHandler(BaseHTTPRequestHandler):
    """Answers the one federation route these tests call, and records the hit.

    The recording is the load-bearing part: a request that was never sent —
    because a floor or a pin refused the connection first — leaves this list
    empty, which is how "the refusal happened before anything leaked" is
    asserted rather than assumed.
    """

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self.server.requests.append(self.path)  # type: ignore[attr-defined]
        body = json.dumps({"version_vector": {"peer-a": 7}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return


class TlsServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, ctx: ssl.SSLContext) -> None:
        super().__init__(("127.0.0.1", 0), RecordingHandler)
        #: Paths of the requests that reached a handler. A handshake the
        #: listener refuses never appears here — which is the point.
        self.requests: list[str] = []
        self.socket = ctx.wrap_socket(self.socket, server_side=True)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Refused connections are an expected outcome in half these tests;
        # swallow rather than print a traceback into the pytest log.
        return


@contextlib.contextmanager
def serving(ctx: ssl.SSLContext) -> Iterator[TlsServer]:
    """Run a :class:`TlsServer` on a background thread for the block."""
    server = TlsServer(ctx)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
