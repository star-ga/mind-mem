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
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

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
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

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
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
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
