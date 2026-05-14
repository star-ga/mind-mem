"""Federation wire-transport client for mind-mem v4.

Stdlib-only HTTP client for the federation endpoints exposed by
:mod:`mind_mem.http_transport` (``/federation/vclock``,
``/federation/conflicts``, ``/federation/write``,
``/federation/resolve``). Uses bearer-token auth via the existing
``X-MindMem-Token`` convention so two mind-mem hosts on the same
operator's network can exchange version vectors and resolve conflicts
without speaking MCP.

The transport is HTTPS-capable (``base_url`` may use ``https://``) but
TLS configuration is left to the underlying ``urllib`` / ``ssl``
machinery; the client does not currently pin certificates. Use a
reverse proxy or stunnel for TLS termination in production until the
roadmap's Group-D mTLS layer lands.

Example::

    from mind_mem.v4.federation_client import FederationClient

    client = FederationClient("http://peer.local:8765", token="secret")
    new_version = client.push_write("block-42", agent_id="alice")
    if new_version.conflict:
        print(f"conflict: {new_version.conflict}")
    resolution = client.resolve_conflict(
        "block-42",
        strategy="higher_version",
    )
    print(resolution.winner_agent, resolution.winner_version)

All methods raise:
    :class:`FederationAuthError`     on HTTP 401
    :class:`FederationFlagDisabled`  on HTTP 503 (v4.federation off)
    :class:`FederationTransportError` on any other non-2xx response or
                                      network error.
"""

from __future__ import annotations

import base64
import json
import os as _os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

# Issue #529: response-size cap mirrors the server-side body cap so a
# hostile or buggy peer can't stream gigabytes into the client process.
# Operators can raise this via env if they have a specific need.
MAX_RESP_BYTES = int(_os.environ.get("MIND_MEM_FED_MAX_RESP_BYTES", str(1 << 20)))  # 1 MiB
_ALLOWED_SCHEMES = frozenset({"http", "https"})

__all__ = [
    "FederationClient",
    "FederationAuthError",
    "FederationFlagDisabled",
    "FederationTransportError",
    "WriteResult",
    "ResolutionResult",
    "ConflictView",
]


class FederationTransportError(RuntimeError):
    """Network or HTTP-level failure talking to the federation endpoint."""


class FederationAuthError(FederationTransportError):
    """HTTP 401 — bearer token missing or wrong."""


class FederationFlagDisabled(FederationTransportError):
    """HTTP 503 — the remote host has ``v4.federation`` disabled."""


@dataclass(frozen=True)
class ConflictView:
    """A single outstanding conflict surfaced by the remote."""

    block_id: str
    left_agent: str
    left_version: int
    right_agent: str
    right_version: int


@dataclass(frozen=True)
class WriteResult:
    """Result of a federation /write call.

    ``conflict`` is ``None`` when the write did not create a new
    divergence — typically because only one agent has writes for the
    block, or because every agent's version vector is identical.
    """

    block_id: str
    agent_id: str
    version: int
    conflict: ConflictView | None


@dataclass(frozen=True)
class ResolutionResult:
    """Result of a federation /resolve call."""

    block_id: str
    winner_agent: str
    winner_version: int
    strategy: str
    merged_payload: bytes | None


class FederationClient:
    """Talk to a remote mind-mem host's federation surface over HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        # Issue #529 (1): scheme allowlist. urllib.request.urlopen
        # otherwise handles file://, ftp://, etc. so a base_url from a
        # config file / env var / peer-discovery handshake could turn
        # this client into a local-file or arbitrary-protocol reader.
        parts = urlsplit(base_url)
        if parts.scheme not in _ALLOWED_SCHEMES:
            raise FederationTransportError(
                f"federation base_url scheme {parts.scheme!r} not allowed; must be one of {sorted(_ALLOWED_SCHEMES)}"
            )
        if not parts.netloc:
            raise FederationTransportError(f"federation base_url has no host: {base_url!r}")
        self._base = base_url.rstrip("/")
        self._base_scheme = parts.scheme
        self._base_netloc = parts.netloc
        self._token = token
        self._timeout = timeout
        # Issue #529 (2): same-origin redirect cap. Build an opener that
        # rejects redirects to a different scheme/host/port — blocks the
        # SSRF pivot to cloud metadata endpoints
        # (169.254.169.254, metadata.google.internal, etc.).
        self._opener = _build_strict_opener(self._base_scheme, self._base_netloc)

    # ------------------------------------------------------------------
    # Reader API
    # ------------------------------------------------------------------

    def get_vclock(self, block_id: str) -> dict[str, int]:
        """Return the per-agent version vector for ``block_id`` on the remote."""
        path = f"/federation/vclock/{quote(block_id, safe='')}"
        payload = self._request("GET", path)
        vec = payload.get("version_vector", {})
        if not isinstance(vec, dict):
            raise FederationTransportError(f"unexpected response shape for vclock: {payload!r}")
        return {str(k): int(v) for k, v in vec.items()}

    def list_conflicts(self, *, limit: int = 100) -> list[ConflictView]:
        """Return up to ``limit`` outstanding conflicts on the remote."""
        path = f"/federation/conflicts?limit={int(limit)}"
        payload = self._request("GET", path)
        raw = payload.get("conflicts", [])
        if not isinstance(raw, list):
            raise FederationTransportError(f"unexpected response shape for conflicts: {payload!r}")
        out: list[ConflictView] = []
        for entry in raw:
            try:
                out.append(
                    ConflictView(
                        block_id=str(entry["block_id"]),
                        left_agent=str(entry["left_agent"]),
                        left_version=int(entry["left_version"]),
                        right_agent=str(entry["right_agent"]),
                        right_version=int(entry["right_version"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise FederationTransportError(f"malformed conflict entry: {entry!r} ({exc})") from exc
        return out

    # ------------------------------------------------------------------
    # Writer API
    # ------------------------------------------------------------------

    def push_write(self, block_id: str, *, agent_id: str) -> WriteResult:
        """Record an agent write on the remote and surface any conflict.

        The remote bumps its (block_id, agent_id) entry in
        ``block_tier_vclock`` atomically and re-runs
        :func:`federation.detect_conflict` against the resulting
        vector. If two agents now have diverging versions, the
        response carries a :class:`ConflictView`.
        """
        body = {"block_id": block_id, "agent_id": agent_id}
        payload = self._request("POST", "/federation/write", body=body)
        conflict_raw = payload.get("conflict")
        conflict = None
        if isinstance(conflict_raw, dict):
            try:
                conflict = ConflictView(
                    block_id=str(payload["block_id"]),
                    left_agent=str(conflict_raw["left_agent"]),
                    left_version=int(conflict_raw["left_version"]),
                    right_agent=str(conflict_raw["right_agent"]),
                    right_version=int(conflict_raw["right_version"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise FederationTransportError(f"malformed conflict in /write response: {payload!r} ({exc})") from exc
        return WriteResult(
            block_id=str(payload["block_id"]),
            agent_id=str(payload["agent_id"]),
            version=int(payload["version"]),
            conflict=conflict,
        )

    def resolve_conflict(
        self,
        block_id: str,
        *,
        strategy: str,
        merged_payload: bytes | None = None,
    ) -> ResolutionResult:
        """Pick a winner for the most-recent open conflict on ``block_id``.

        ``strategy`` must be one of the
        :class:`mind_mem.v4.federation.MergeStrategy` values:
        ``"last_writer_wins"``, ``"higher_version"``, or
        ``"three_way_merge"``. For ``"three_way_merge"`` you MUST supply
        ``merged_payload`` (the bytes that should be persisted as the
        merged content); the remote does not invoke a server-side
        merger callable.
        """
        body: dict[str, Any] = {"block_id": block_id, "strategy": strategy}
        if merged_payload is not None:
            body["merged_payload"] = base64.b64encode(merged_payload).decode("ascii")
        payload = self._request("POST", "/federation/resolve", body=body)
        merged_b64 = payload.get("merged_payload")
        merged = None
        if isinstance(merged_b64, str) and merged_b64:
            try:
                merged = base64.b64decode(merged_b64)
            except Exception as exc:
                raise FederationTransportError(f"resolve response merged_payload is not valid base64: {exc}") from exc
        return ResolutionResult(
            block_id=str(payload["block_id"]),
            winner_agent=str(payload["winner_agent"]),
            winner_version=int(payload["winner_version"]),
            strategy=str(payload["strategy"]),
            merged_payload=merged,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._base + path
        data: bytes | None = None
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._token is not None:
            headers["X-MindMem-Token"] = self._token
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, method=method, data=data, headers=headers)
        try:
            # Use the strict opener (issue #529 (2): same-origin redirect
            # cap) instead of the default urlopen.
            with self._opener.open(req, timeout=self._timeout) as resp:
                # Issue #529 (3): response-size cap. Read one byte more
                # than the cap; if we got it, the peer was over the limit
                # and we reject without buffering the rest.
                raw = resp.read(MAX_RESP_BYTES + 1)
                if len(raw) > MAX_RESP_BYTES:
                    raise FederationTransportError(f"federation response exceeds {MAX_RESP_BYTES} bytes")
        except urllib.error.HTTPError as exc:
            self._raise_for_status(exc.code, _safe_read(exc))
        except urllib.error.URLError as exc:
            raise FederationTransportError(f"network error: {exc.reason}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise FederationTransportError(f"non-JSON response from federation endpoint: {exc}") from exc
        if not isinstance(payload, dict):
            raise FederationTransportError(f"federation endpoint returned non-object: {payload!r}")
        return payload

    @staticmethod
    def _raise_for_status(status: int, raw: bytes) -> None:
        try:
            body = json.loads(raw.decode("utf-8"))
            msg = body.get("error") or str(body)
        except Exception:
            msg = raw.decode("utf-8", errors="replace") or f"HTTP {status}"
        if status == 401:
            raise FederationAuthError(f"HTTP 401: {msg}")
        if status == 503:
            raise FederationFlagDisabled(f"HTTP 503: {msg}")
        raise FederationTransportError(f"HTTP {status}: {msg}")


def _safe_read(exc: urllib.error.HTTPError) -> bytes:
    try:
        return exc.read() or b""
    except Exception:
        return b""


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects that change scheme/host/port. Issue #529 (2).

    A hostile peer answering 302 -> http://169.254.169.254/... would
    otherwise let the federation client read cloud-metadata-service
    credentials and surface them as "peer response" to caller code.
    """

    def __init__(self, allowed_scheme: str, allowed_netloc: str) -> None:
        self._allowed_scheme = allowed_scheme
        self._allowed_netloc = allowed_netloc

    def redirect_request(self, req, fp, code, msg, hdrs, newurl):  # type: ignore[override]
        parts = urlsplit(newurl)
        if parts.scheme != self._allowed_scheme or parts.netloc != self._allowed_netloc:
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"cross-origin redirect blocked: {self._allowed_scheme}://{self._allowed_netloc} -> {parts.scheme}://{parts.netloc}",
                hdrs,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, hdrs, newurl)


def _build_strict_opener(scheme: str, netloc: str) -> urllib.request.OpenerDirector:
    """Build an opener with the same-origin redirect handler installed."""
    opener = urllib.request.build_opener(_SameOriginRedirectHandler(scheme, netloc))
    return opener
