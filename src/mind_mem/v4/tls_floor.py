"""TLS 1.3 floor and certificate pinning for mind-mem's own network surfaces.

Roadmap ``v4.0.0`` Group D, item *TLS 1.3 minimum + cert pinning*. Both
halves live here, and they are not the same kind of thing:

* the **floor** is unconditional. There is no way to ask for a context
  below it, and no environment variable that lowers it.
* **pinning** is opt-in and off by default — see
  :data:`CERT_PINNING_DECISION` for why that asymmetry is deliberate. A
  caller that configures no pin gets exactly the connection it got
  before pinning existed.

The floor is enforced **by construction**, not by inspection. Every
helper in this module returns an :class:`ssl.SSLContext` whose
``minimum_version`` is already ``TLSv1_3`` *before* the context is handed
to a socket, so a peer that can only speak TLS 1.2 fails the handshake
and no connection exists to inspect. Nothing here reads
``SSLSocket.version()`` after the fact and logs a complaint: a
post-connection check has already leaked the request.

Three ways the floor can fail, and what each does:

* the interpreter's OpenSSL has no TLS 1.3 →
  :class:`TlsFloorUnavailable` at context-construction time. The caller
  gets no context, so it cannot open a floorless listener or client.
* ``minimum_version`` cannot be assigned (a protocol constant that is not
  a TLS protocol) → :class:`TlsFloorUnavailable`.
* the assignment silently does not take → read-back in
  :func:`_apply_floor` raises :class:`TlsFloorUnavailable`.

All three fail closed. There is no environment variable that lowers the
floor, because a floor an operator can switch off is not a floor.

Pinning fails closed too, but at a different moment: the check runs after
the handshake and *before* the first request byte is written, so a peer
whose key is not in the pin set never receives the request it would have
been sent.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import ssl
import urllib.request
from typing import Iterable

__all__ = [
    "CERT_PINNING_DECISION",
    "CertificatePinMismatch",
    "TLS_FLOOR",
    "TlsFloorUnavailable",
    "apply_floor",
    "client_context",
    "context_meets_floor",
    "normalise_pins",
    "pinned_https_handler",
    "server_context",
    "spki_sha256",
    "verify_pinned_peer",
]

#: The floor. Not configurable.
TLS_FLOOR: ssl.TLSVersion = ssl.TLSVersion.TLSv1_3

#: Why pinning is a switch and the floor is not. Recorded here rather
#: than left as an unexplained asymmetry: the roadmap text for this item
#: and for the ``mTLS + certificate pinning on FederationClient`` item
#: both ask for pinning, and an operator reading either one deserves to
#: know what turning it on costs them.
CERT_PINNING_DECISION = (
    "certificate pinning in mind-mem is OPT-IN and off by default. Nothing pins "
    "unless an operator passes pinned_pubkey_sha256=, and a client that passes "
    "nothing behaves exactly as it did before pinning existed. The reason it is not "
    "the default is operational: for a product whose usual deployment is loopback, a "
    "pinned key store turns every routine peer-certificate renewal into a coordinated "
    "outage, and an operator surprised by that disables the pin — which is worse than "
    "never having had one. Two rules make it survivable: pin the public key (SPKI), "
    "not the certificate, so a renewal that keeps the key keeps working; and pin the "
    "incoming key alongside the current one BEFORE rotating, never after. Mutual TLS "
    "(client_cert/client_key plus a private CA via cafile) remains the recommended way "
    "to bind a peer's identity. Pinning is the additional defence mutual TLS does not "
    "give you: it detects a certificate that a trusted CA should never have issued, "
    "which is what a TLS-intercepting proxy on the path looks like."
)


class TlsFloorUnavailable(RuntimeError):
    """This interpreter cannot enforce the TLS 1.3 floor.

    Raised at context-construction time so a caller can never end up
    holding a usable-but-floorless context.
    """


class CertificatePinMismatch(ssl.SSLError):
    """The peer's public key is not one of the configured pins.

    An :class:`OSError` subclass (via :class:`ssl.SSLError`) so it travels
    the same path as any other connection failure: :mod:`urllib` wraps it
    into :class:`urllib.error.URLError`, and a caller that already handles
    network errors keeps handling it. Raised before the request is written.
    """


def _apply_floor(ctx: ssl.SSLContext) -> ssl.SSLContext:
    """Raise *ctx*'s minimum protocol version to :data:`TLS_FLOOR`.

    Verifies the assignment took effect by reading it back. A context
    that reports anything below the floor after assignment is refused
    rather than returned.
    """
    if not ssl.HAS_TLSv1_3:
        raise TlsFloorUnavailable(
            "this interpreter's OpenSSL build has no TLS 1.3 (ssl.HAS_TLSv1_3 is False); "
            "refusing to build a context that would negotiate TLS 1.2 or lower"
        )
    try:
        ctx.minimum_version = TLS_FLOOR
    except (ValueError, AttributeError) as exc:  # pragma: no cover - platform dependent
        raise TlsFloorUnavailable(f"cannot set minimum_version={TLS_FLOOR!r} on this SSLContext: {exc}") from exc
    if ctx.minimum_version != TLS_FLOOR:  # pragma: no cover - defensive read-back
        raise TlsFloorUnavailable(
            f"SSLContext.minimum_version did not take: asked for {TLS_FLOOR!r}, context reports {ctx.minimum_version!r}"
        )
    return ctx


def apply_floor(ctx: ssl.SSLContext) -> ssl.SSLContext:
    """Raise the floor on a context somebody else built.

    :func:`client_context` and :func:`server_context` cover the contexts
    mind-mem constructs. This is for the ones it does not: uvicorn builds
    its listener context itself, and the only way to give that listener a
    floor is to raise it on uvicorn's object before the socket exists.
    Same read-back, same fail-closed behaviour.
    """
    return _apply_floor(ctx)


def context_meets_floor(ctx: ssl.SSLContext) -> bool:
    """Return whether *ctx* already carries the floor.

    A predicate for tests and for callers that were handed a context by
    someone else (uvicorn builds its own, for one). It is **not** the
    enforcement mechanism — :func:`_apply_floor` is.
    """
    return getattr(ctx, "minimum_version", None) == TLS_FLOOR


def client_context(
    *,
    cafile: str | None = None,
    capath: str | None = None,
    cadata: str | bytes | None = None,
    client_cert: str | None = None,
    client_key: str | None = None,
    client_key_password: str | None = None,
) -> ssl.SSLContext:
    """Build an outbound TLS context with the floor already applied.

    Hostname checking and certificate verification stay on — this starts
    from :func:`ssl.create_default_context`, which sets both, and the
    floor is raised on top. Passing ``cafile``/``capath``/``cadata``
    replaces the system trust store with the operator's own CA, which is
    the supported way to bind a federation peer's identity (see
    :data:`CERT_PINNING_DECISION`).

    ``client_cert`` (with optional ``client_key``) turns the context into
    the client half of mutual TLS.
    """
    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=cafile, capath=capath, cadata=cadata)
    _apply_floor(ctx)
    if client_cert:
        ctx.load_cert_chain(certfile=client_cert, keyfile=client_key, password=client_key_password)
    elif client_key:
        raise ValueError("client_key was given without client_cert; mutual TLS needs the certificate chain too")
    return ctx


def server_context(
    certfile: str,
    keyfile: str | None = None,
    *,
    keyfile_password: str | None = None,
    client_ca: str | None = None,
) -> ssl.SSLContext:
    """Build an inbound TLS context with the floor already applied.

    ``client_ca`` switches the listener to mutual TLS: peers must present
    a certificate signed by that CA (``CERT_REQUIRED``) or the handshake
    fails. Without it the listener is ordinary server-authenticated TLS.
    """
    ctx = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
    _apply_floor(ctx)
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile, password=keyfile_password)
    if client_ca:
        ctx.load_verify_locations(cafile=client_ca)
        ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


# ---------------------------------------------------------------------------
# Certificate pinning (opt-in — see CERT_PINNING_DECISION)
# ---------------------------------------------------------------------------

_DER_SEQUENCE = 0x30
_DER_EXPLICIT_0 = 0xA0
#: tbsCertificate fields between the optional version and the key:
#: serialNumber, signature, issuer, validity, subject.
_TBS_FIELDS_BEFORE_KEY = 5
#: Longest DER length this parser accepts, in bytes of length header. Four
#: bytes is a 4 GiB element; a certificate that needs more is not one.
_MAX_DER_LENGTH_BYTES = 4


def _der_element(buf: bytes, pos: int) -> tuple[int, int, int]:
    """Return ``(tag, value_start, element_end)`` for the DER TLV at *pos*.

    ``element_end`` is where the next sibling begins, so walking a
    SEQUENCE is a loop over this function. Every bound is checked: a
    truncated or over-long certificate raises rather than reading past
    the buffer or returning a digest of the wrong bytes.
    """
    if pos + 2 > len(buf):
        raise ValueError("truncated DER: no tag and length at this position")
    tag = buf[pos]
    if tag & 0x1F == 0x1F:
        raise ValueError("unsupported DER: multi-byte tag")
    first = buf[pos + 1]
    if first < 0x80:
        length = first
        start = pos + 2
    else:
        count = first & 0x7F
        if count == 0 or count > _MAX_DER_LENGTH_BYTES:
            raise ValueError(f"unsupported DER length form: {count} length bytes")
        if pos + 2 + count > len(buf):
            raise ValueError("truncated DER: length bytes run past the buffer")
        length = int.from_bytes(buf[pos + 2 : pos + 2 + count], "big")
        start = pos + 2 + count
    end = start + length
    if end > len(buf):
        raise ValueError("truncated DER: element runs past the buffer")
    return tag, start, end


def _subject_public_key_info(der_cert: bytes) -> bytes:
    """Return the SubjectPublicKeyInfo element of a DER X.509 certificate.

    Stdlib only, on purpose: mind-mem's core has no third-party runtime
    dependencies, and requiring ``cryptography`` to verify a pin would
    make the security feature the optional one. The returned bytes are
    the complete SEQUENCE (header included), which is what
    ``cryptography``'s ``PublicFormat.SubjectPublicKeyInfo`` serializes
    and what every SPKI fingerprint in the wild is taken over.
    """
    tag, cert_start, _cert_end = _der_element(der_cert, 0)
    if tag != _DER_SEQUENCE:
        raise ValueError("not a DER certificate: the outer element is not a SEQUENCE")
    tag, tbs_start, tbs_end = _der_element(der_cert, cert_start)
    if tag != _DER_SEQUENCE:
        raise ValueError("not a DER certificate: tbsCertificate is not a SEQUENCE")

    pos = tbs_start
    tag, _value_start, element_end = _der_element(der_cert, pos)
    if tag == _DER_EXPLICIT_0:  # [0] EXPLICIT version, optional (absent in v1)
        pos = element_end
    for _ in range(_TBS_FIELDS_BEFORE_KEY):
        _tag, _value_start, element_end = _der_element(der_cert, pos)
        pos = element_end
        if pos >= tbs_end:
            raise ValueError("malformed DER certificate: tbsCertificate ends before the public key")
    tag, _value_start, element_end = _der_element(der_cert, pos)
    if tag != _DER_SEQUENCE:
        raise ValueError("malformed DER certificate: subjectPublicKeyInfo is not a SEQUENCE")
    if element_end > tbs_end:
        raise ValueError("malformed DER certificate: subjectPublicKeyInfo overruns tbsCertificate")
    return der_cert[pos:element_end]


def spki_sha256(der_cert: bytes) -> str:
    """SHA-256 of a DER certificate's public key, as lowercase hex.

    The *key*, not the certificate: a renewal that keeps the key keeps
    the fingerprint, which is what makes a pin something an operator can
    live with. Raises :class:`ValueError` on anything that is not a
    parseable certificate — a pin check must never compare a digest of
    bytes it did not understand.
    """
    return hashlib.sha256(_subject_public_key_info(der_cert)).hexdigest()


def _normalise_pin(value: str) -> str:
    """Return one pin in canonical form (lowercase hex), or raise.

    Accepts the spellings operators actually have in front of them: raw
    hex, colon-separated hex (``openssl``), and base64 with or without
    the ``sha256//`` prefix (``curl --pinnedpubkey``).
    """
    raw = value.strip()
    lowered = raw.lower()
    for prefix in ("sha256//", "sha256/", "sha256:", "sha-256:"):
        if lowered.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    compact = raw.replace(":", "").replace(" ", "")
    if len(compact) == 64:
        try:
            return bytes.fromhex(compact).hex()
        except ValueError as exc:
            raise ValueError(f"certificate pin {value!r} is 64 characters but not hexadecimal") from exc
    try:
        digest = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        digest = b""
    if len(digest) == 32:
        return digest.hex()
    raise ValueError(
        f"certificate pin {value!r} is neither 64 hex characters nor base64 for 32 bytes; "
        "expected a SHA-256 fingerprint of the peer's SubjectPublicKeyInfo"
    )


def normalise_pins(pins: str | Iterable[str]) -> frozenset[str]:
    """Return the canonical pin set for one string or a collection of them.

    A collection is a rotation set: pin the incoming key alongside the
    current one and the renewal is not an outage. An empty collection is
    refused — it would match no peer while looking configured.
    """
    values = [pins] if isinstance(pins, str) else list(pins)
    if not values:
        raise ValueError("an empty certificate-pin set would refuse every peer; pass at least one fingerprint")
    return frozenset(_normalise_pin(value) for value in values)


def verify_pinned_peer(sock: ssl.SSLSocket, pins: frozenset[str]) -> None:
    """Raise unless the peer on *sock* presents a key named by *pins*.

    Call this after the handshake and before the first byte of the
    request. Certificate *verification* has already happened by then (the
    contexts this module builds all set ``CERT_REQUIRED``); the pin is the
    additional question of whether the verified certificate is the one
    this deployment expects.
    """
    der = sock.getpeercert(binary_form=True)
    if not der:
        raise CertificatePinMismatch("the peer presented no certificate, so there is nothing to pin against")
    actual = spki_sha256(der)
    if actual not in pins:
        raise CertificatePinMismatch(
            f"certificate pin mismatch: the peer's public-key SHA-256 is {actual}, "
            f"which is not one of the {len(pins)} pinned fingerprint(s) {sorted(pins)}"
        )


def _pinned_connection_class(pins: frozenset[str]) -> type[http.client.HTTPSConnection]:
    """Build an HTTPS connection class that checks *pins* on connect.

    The check lives in ``connect()`` because that is the last moment
    before :meth:`http.client.HTTPConnection.send` writes the request
    line: a mismatch closes the socket and raises, and the peer never
    sees the request. A post-response check would already have leaked it.
    """

    class _PinnedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self) -> None:
            super().connect()
            sock = self.sock
            if not isinstance(sock, ssl.SSLSocket):  # pragma: no cover - HTTPSConnection always wraps
                self.close()
                raise CertificatePinMismatch("the connection is not TLS, so there is no peer certificate to pin against")
            try:
                verify_pinned_peer(sock, pins)
            except CertificatePinMismatch:
                self.close()
                raise

    return _PinnedHTTPSConnection


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """A urllib HTTPS handler whose connections must match a pin set."""

    def __init__(self, context: ssl.SSLContext, pins: frozenset[str]) -> None:
        super().__init__(context=context)
        self._ssl_context = context
        self._pins = pins
        self._connection_class = _pinned_connection_class(pins)

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(self._connection_class, req, context=self._ssl_context)


def pinned_https_handler(*, context: ssl.SSLContext, pins: frozenset[str]) -> urllib.request.HTTPSHandler:
    """Return an HTTPS handler that enforces *pins* under *context*.

    Drop-in for :class:`urllib.request.HTTPSHandler` in an opener. The
    floor still comes from *context*; pinning is additional to it, never
    a substitute — a pinned peer that can only speak TLS 1.2 still fails
    the handshake before the pin is ever consulted.
    """
    if not pins:
        raise ValueError("pinned_https_handler needs at least one pin; use a plain HTTPSHandler for no pinning")
    return _PinnedHTTPSHandler(context, pins)
