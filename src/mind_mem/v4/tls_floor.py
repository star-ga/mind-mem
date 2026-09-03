"""TLS 1.3 floor for mind-mem's own network surfaces.

Roadmap ``v4.0.0`` Group D, item *TLS 1.3 minimum + cert pinning*. The
floor half is implemented here. The pinning half is **deliberately not
implemented** — see :data:`CERT_PINNING_DECISION`.

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

Copyright STARGA, Inc.
"""

from __future__ import annotations

import ssl

__all__ = [
    "CERT_PINNING_DECISION",
    "TLS_FLOOR",
    "TlsFloorUnavailable",
    "apply_floor",
    "client_context",
    "context_meets_floor",
    "server_context",
]

#: The floor. Not configurable.
TLS_FLOOR: ssl.TLSVersion = ssl.TLSVersion.TLSv1_3

#: Why ``pinned_pubkey_sha256`` does not exist on any surface in this
#: package. Recorded here rather than left as an unexplained gap: the
#: roadmap text for this item and for the ``mTLS + certificate pinning on
#: FederationClient`` item both ask for pinning, and the architecture
#: ruling for 5.0.2 declined it.
CERT_PINNING_DECISION = (
    "certificate pinning is intentionally NOT implemented in mind-mem. "
    "The architecture ruling for 5.0.2 kept the TLS 1.3 floor and dropped pinning: "
    "for a product whose default deployment is loopback, a pinned SPKI store is an "
    "operational trap — it turns every routine peer-certificate renewal into a "
    "coordinated outage, and operators respond by disabling the pin, which is worse "
    "than never having had it. Use mutual TLS (client_cert/client_key plus a private "
    "CA via cafile) to bind peer identity instead; that survives renewal."
)


class TlsFloorUnavailable(RuntimeError):
    """This interpreter cannot enforce the TLS 1.3 floor.

    Raised at context-construction time so a caller can never end up
    holding a usable-but-floorless context.
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
