"""Request-scoped audit attribution for mind-mem's network transports.

Roadmap ``v4.0.0`` Group D, item *Audit headers* (``X-MindMem-Request-Id``,
``X-MindMem-Actor``, ``X-MindMem-Purpose``). The three headers were being
parsed and echoed by the REST middleware and then dropped: they were
written onto ``request.state`` and nothing anywhere read them back, so
neither the audit chain nor a log line ever carried them. "End-to-end"
was unmet even inside REST, and the gRPC leg parsed nothing at all.

This module is the propagation primitive the transports share. It sits at the package root rather than under
``mind_mem.api`` on purpose: ``mind_mem.api.__init__`` imports the
FastAPI app, and the stdlib-only federation client must not acquire a
FastAPI dependency just to stamp three headers on an outbound request.

It is a
:class:`contextvars.ContextVar` holding one **mutable** :class:`AuditContext`
per in-flight request, and the mutability is load-bearing:

    A sync FastAPI dependency runs in a threadpool worker under a *copy*
    of the request's context. ``ContextVar.set`` inside that copy is
    invisible to the endpoint, which runs under its own copy. The one
    thing both copies share is the *object* the parent bound — so the
    dependency records the authenticated identity by mutating that
    object, exactly as it stashes OIDC scopes on the shared ``Request``.

Two kinds of identity travel here and they are never merged:

``actor_claimed``
    Whatever the caller put in ``X-MindMem-Actor``. Unauthenticated,
    unverified, attacker-controlled. Provenance only.
``agent_authenticated``
    The identity the transport's own auth resolved. ``None`` until auth
    has run, and never assignable from a header.

The distinction is why :func:`record_authenticated_agent` exists as a
separate call instead of the header parser filling both in.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

__all__ = [
    "HEADER_ACTOR",
    "HEADER_PURPOSE",
    "HEADER_REQUEST_ID",
    "MAX_FIELD_LEN",
    "MAX_REQUEST_ID_LEN",
    "AuditContext",
    "bind_audit_context",
    "context_from_headers",
    "current_audit_context",
    "log_scope",
    "outbound_audit_headers",
    "record_authenticated_agent",
    "sanitize_header_value",
]

#: Canonical spellings. HTTP header names are case-insensitive, but these
#: are what mind-mem *emits*, and a caller grepping their proxy logs wants
#: one spelling to grep for.
HEADER_REQUEST_ID = "X-MindMem-Request-Id"
HEADER_ACTOR = "X-MindMem-Actor"
HEADER_PURPOSE = "X-MindMem-Purpose"

#: Length caps. A request id is a correlation token, not a payload; actor
#: and purpose are short prose. Both are bounded before anything logs,
#: echoes, or persists them.
MAX_REQUEST_ID_LEN = 64
MAX_FIELD_LEN = 256

_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_header_value(raw: Optional[str], *, default: str = "", max_len: int = MAX_FIELD_LEN) -> str:
    """Return *raw* with CR/LF/NUL/control bytes stripped, length-bounded.

    These values are echoed into response headers, into structured logs,
    and (for the actor) into a governed block's provenance field, so a
    caller who can smuggle a newline through gets header splitting and
    log forging for free.

    Leads with explicit ``.replace`` calls before the regex so the stock
    CodeQL ``py/log-injection`` and ``py/header-injection`` queries
    recognise this function as a sanitiser node.
    """
    if not raw:
        return default
    cleaned = raw.replace("\r", "").replace("\n", "")
    cleaned = _CTRL.sub("", cleaned)
    cleaned = cleaned[:max_len]
    # An input made *entirely* of control characters sanitises to the
    # empty string. That is an absent value, not a present empty one, so
    # it takes the default like any other absence.
    return cleaned or default


@dataclass
class AuditContext:
    """Attribution for one in-flight request. Mutable by design — see module docstring."""

    #: Correlation token. Server-assigned when the caller sent none.
    request_id: str
    #: ``X-MindMem-Actor`` as the caller sent it. A claim, never a fact.
    actor_claimed: str = ""
    #: ``X-MindMem-Purpose``. Always a claim.
    purpose: str = ""
    #: Which transport bound this context ("rest", "grpc", ...). Names the
    #: leg in a log line so a mixed deployment can tell them apart.
    transport: str = ""
    #: Identity the transport's own authentication resolved. ``None`` means
    #: auth has not run or did not identify anyone; it is NEVER filled from
    #: a header.
    agent_authenticated: Optional[str] = field(default=None)

    def log_bindings(self) -> dict[str, str]:
        """Key/value bindings for :mod:`mind_mem.v4.logging_context`.

        Absent fields are omitted rather than emitted empty, so a log line
        never claims an unattributed request had an actor of ``""``.
        """
        out = {"correlation_id": self.request_id, "request_id": self.request_id}
        if self.transport:
            out["transport"] = self.transport
        if self.actor_claimed:
            out["actor_claimed"] = self.actor_claimed
        if self.purpose:
            out["purpose"] = self.purpose
        if self.agent_authenticated:
            out["agent"] = self.agent_authenticated
        return out

    def outbound_headers(self) -> dict[str, str]:
        """Headers that carry this attribution to the *next* hop.

        The request id propagates so one correlation token spans a
        federation fan-out. The actor sent onward is the authenticated
        identity when there is one — a peer that receives our claim is
        better served by the identity we verified than by the one our
        caller asserted.
        """
        headers = {HEADER_REQUEST_ID: self.request_id}
        actor = self.agent_authenticated or self.actor_claimed
        if actor:
            headers[HEADER_ACTOR] = actor
        if self.purpose:
            headers[HEADER_PURPOSE] = self.purpose
        return headers


_AUDIT_CONTEXT: ContextVar[Optional[AuditContext]] = ContextVar("mindmem_audit_context", default=None)


def current_audit_context() -> Optional[AuditContext]:
    """Return the context bound for this request, or ``None`` outside one."""
    return _AUDIT_CONTEXT.get()


def context_from_headers(
    getter: Callable[[str], Optional[str]],
    *,
    transport: str,
) -> AuditContext:
    """Build a context from a case-insensitive header lookup.

    ``getter`` takes a lowercase header name and returns the raw value or
    ``None``; both Starlette's ``request.headers`` and a gRPC metadata
    dict adapt to it in one lambda. A missing or unusable request id is
    replaced by a fresh UUID-4 so every request is correlatable, including
    the ones from callers that send nothing.
    """
    return AuditContext(
        request_id=sanitize_header_value(
            getter("x-mindmem-request-id"),
            default=str(uuid.uuid4()),
            max_len=MAX_REQUEST_ID_LEN,
        ),
        actor_claimed=sanitize_header_value(getter("x-mindmem-actor")),
        purpose=sanitize_header_value(getter("x-mindmem-purpose")),
        transport=transport,
    )


@contextmanager
def log_scope(**bindings: str) -> Iterator[None]:
    """Push *bindings* onto the structured-log context, when it is armed.

    ``v4.logging_context`` defaults OFF, and a probe that decides whether
    a feature is on must not itself be observable when the answer is no.
    :func:`mind_mem.observability.log_context_active` is a pure in-memory
    attribute read — no config parse, no file read — so with the flag off
    this pushes nothing, pops nothing, and imports nothing.
    """
    from mind_mem.observability import log_context_active  # noqa: PLC0415

    if not log_context_active():
        yield
        return
    from mind_mem.v4.logging_context import LogContext  # noqa: PLC0415

    token = LogContext.push(**bindings)
    try:
        yield
    finally:
        LogContext.pop(token)


@contextmanager
def bind_audit_context(ctx: AuditContext) -> Iterator[AuditContext]:
    """Bind *ctx* for the duration of the block.

    Also pushes the context's bindings onto the ``v4.logging_context``
    stack via :func:`log_scope`, so every structured log record emitted
    while serving the request carries the correlation id and the
    attribution. Those bindings are a *snapshot*: the authenticated
    identity is not known yet at bind time, so the frame that resolves it
    pushes it — see ``rest._acting_as``.
    """
    token: Token[Optional[AuditContext]] = _AUDIT_CONTEXT.set(ctx)
    try:
        with log_scope(**ctx.log_bindings()):
            yield ctx
    finally:
        _AUDIT_CONTEXT.reset(token)


def record_authenticated_agent(agent_id: str) -> bool:
    """Record the identity authentication resolved, on the bound context.

    Mutates the shared object rather than setting a ContextVar, because
    the caller is a sync FastAPI dependency running in a threadpool copy
    of the request context — a ``set`` there reaches nothing.

    Returns:
        True when a context was bound and updated, False when this ran
        outside a request (a direct handler call in a test, an MCP tool).
        The boolean is returned rather than swallowed so a caller that
        *must* have attribution can tell that it got none.
    """
    ctx = _AUDIT_CONTEXT.get()
    if ctx is None:
        return False
    ctx.agent_authenticated = agent_id
    return True


def outbound_audit_headers() -> dict[str, str]:
    """Headers propagating the current attribution to the next hop.

    Empty when no context is bound, so a client used outside a served
    request sends exactly what it sent before.
    """
    ctx = _AUDIT_CONTEXT.get()
    if ctx is None:
        return {}
    return ctx.outbound_headers()
