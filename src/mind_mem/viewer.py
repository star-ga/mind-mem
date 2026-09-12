"""``mm view`` — a read-only local viewer over the governed corpus.

ROADMAP ("Local visual viewer"): "`mm view` web UI not yet shipped. Stack target: stdlib
HTTP + minimal JS/D3." The flag registry recorded the honest open question beside the
flag: "No viewer surface ships in this package. Wiring question: the flag may belong to a
client, not to mind-mem; if so it should move rather than be deleted." It belongs here,
because the viewer reads the governed corpus and the rules deciding what is *servable*
live here.

THE LOAD-BEARING PROPERTY IS THE ADMISSION FILTER, NOT THE HTML. This repo has already
paid for the alternative once: a block loader selected ``id, status, tags, json_blob``
and never filtered on ``status``, so quarantined and pending content surfaced verbatim
through a user-scope tool -- the column was right there in the SELECT, which is exactly
why it read as safe. So every block this module shows passes through
:func:`admissibility.admit_corpus`, the shared authority, rather than a status check
hand-rolled here that would drift the moment a status is added.

The payload is built from the ADMITTED list only. Not just the block array: counts,
type histogram and the recent list are all derived from the same filtered sequence,
because a summary rebuilt from the unfiltered corpus would leak withheld text while the
block list looked clean.

FAIL-CLOSED IN THREE MORE PLACES.
* A non-loopback bind is REFUSED. This surface has no authentication, so on a routable
  address it publishes the governed corpus to the network.
* Flag-gated OFF (``v4.viewer``) and probed with ``is_enabled_quiet``: ``is_enabled``
  warns on a malformed config, and a probe that logs on an OFF path makes the flag-off
  build observably different from one that never had the feature.
* NO WRITE PATH AT ALL -- no block-store, apply-engine, governance-gate or capture
  import, asserted over the import graph by the tests rather than promised here.

Stdlib only: ``http.server`` and ``json``. No dependency, no build step, and the page is
a single inline document so there is nothing to serve from disk and no path to traverse.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .admissibility import admit_corpus

__all__ = [
    "VIEWER_FLAG",
    "ViewerRefused",
    "build_payload",
    "resolve_bind",
    "render_page",
    "serve",
]

#: Bare flag name as it appears in ``ALL_V4_FLAGS`` (operators write ``v4.viewer``).
#: A LITERAL at the probe below, so the flag registry's consumer scan can see it -- a
#: computed name reads as "declared wired, 0 consumers" to the registry and to anyone
#: grepping.
VIEWER_FLAG = "viewer"

#: Loopback only. Names rather than a regex, because "starts with 127." would admit
#: 127.0.0.1.evil.example via a resolver and the point is to be boring here.
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

_MIN_PORT = 1024
_MAX_PORT = 65535

#: Blocks shown in the recent list. A viewer is for orientation, not for exporting the
#: corpus one page at a time.
_RECENT_LIMIT = 50


class ViewerRefused(RuntimeError):
    """The viewer refused to start. Always names what it refused and why."""


def _load_corpus(workspace: str) -> list[dict[str, Any]]:
    """The raw parsed corpus. Separated so tests can supply one, and so the admission
    filter below is visibly applied to whatever comes back."""
    from .graph_ingest import _load_corpus as load

    return list(load(workspace))


def resolve_bind(host: str, port: int) -> tuple[str, int]:
    """Validate the bind address, or raise. Loopback only, unprivileged port only."""
    candidate = str(host or "").strip()
    if candidate.lower() not in _LOOPBACK:
        raise ViewerRefused(
            f"refusing to bind {candidate!r}: the viewer has no authentication, so only a "
            f"loopback address is allowed ({', '.join(sorted(_LOOPBACK))}). On a routable "
            f"address this would publish the governed corpus to the network."
        )
    try:
        numeric = int(port)
    except (TypeError, ValueError) as exc:
        raise ViewerRefused(f"port {port!r} is not a number") from exc
    if not (_MIN_PORT <= numeric <= _MAX_PORT):
        raise ViewerRefused(
            f"refusing port {numeric}: use {_MIN_PORT}-{_MAX_PORT}. A privileged port "
            f"would need root for a read-only local page."
        )
    return candidate, numeric


def build_payload(workspace: str) -> dict[str, Any]:
    """The viewer's whole data model, built ONLY from admitted blocks.

    Every derived figure -- the total, the per-type histogram, the recent list -- comes
    from the same filtered sequence. Deriving any of them from the raw corpus would leak
    withheld content through a count or a summary while the block list looked clean.
    """
    admitted = admit_corpus(_load_corpus(workspace))

    blocks: list[dict[str, Any]] = []
    for block in admitted:
        text = str(
            block.get("Excerpt") or block.get("excerpt") or block.get("Statement") or ""
        )
        blocks.append(
            {
                "id": str(block.get("_id") or ""),
                "type": str(block.get("Type") or "unknown"),
                "status": str(block.get("Status") or ""),
                "date": str(block.get("Date") or ""),
                "excerpt": text[:300],
            }
        )

    by_type = Counter(b["type"] for b in blocks)
    return {
        "admitted": len(blocks),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "recent": sorted(blocks, key=lambda b: b["date"], reverse=True)[:_RECENT_LIMIT],
        "blocks": blocks,
        "note": (
            "read-only; shows only blocks the shared admission filter admits — pending "
            "and quarantined content is unresolvable here by construction, not filtered "
            "out downstream"
        ),
    }


def render_page(payload: dict[str, Any]) -> str:
    """One self-contained HTML document. No external asset, so nothing is served from
    disk and there is no path to traverse."""
    data = json.dumps(payload, indent=1).replace("</", "<\\/")
    rows = "".join(
        f"<tr><td>{_esc(b['id'])}</td><td>{_esc(b['type'])}</td>"
        f"<td>{_esc(b['date'])}</td><td>{_esc(b['excerpt'])}</td></tr>"
        for b in payload["recent"]
    )
    counts = "".join(
        f"<li><b>{_esc(k)}</b>: {v}</li>" for k, v in payload["by_type"].items()
    )
    return (
        "<!doctype html><meta charset=utf-8><title>mind-mem viewer</title>"
        "<style>body{font:14px system-ui;margin:2rem;max-width:70rem}"
        "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;"
        "padding:.4rem;text-align:left;vertical-align:top}"
        "code{background:#f4f4f4;padding:.1rem .3rem}</style>"
        f"<h1>mind-mem</h1><p>{_esc(payload['note'])}</p>"
        f"<p><b>{payload['admitted']}</b> admitted blocks</p><ul>{counts}</ul>"
        f"<h2>Recent</h2><table><tr><th>id</th><th>type</th><th>date</th>"
        f"<th>excerpt</th></tr>{rows}</table>"
        f"<h2>Raw</h2><pre><code>{_esc(data)}</code></pre>"
    )


def _esc(value: object) -> str:
    """Escape for HTML text. Block text is operator-authored but it is still TEXT, and a
    viewer that renders it as markup would execute whatever a captured document
    contained."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _flag_on() -> bool:
    """Silent probe. Never ``is_enabled`` -- see the module docstring."""
    try:
        from .v4.feature_flags import is_enabled_quiet

        return bool(is_enabled_quiet("viewer"))
    except Exception:  # pragma: no cover — a probe must never raise
        return False


def serve(workspace: str, host: str = "127.0.0.1", port: int = 8900) -> None:
    """Serve the viewer until interrupted. Read-only; raises rather than degrading."""
    if not _flag_on():
        raise ViewerRefused(
            'the viewer is disabled. Enable it in mind-mem.json: "v4": {"viewer": '
            '{"enabled": true}} — it is off by default because it serves the corpus '
            "without authentication."
        )
    bind_host, bind_port = resolve_bind(host, port)

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self) -> None:  # noqa: N802 — stdlib naming
            payload = build_payload(workspace)
            if self.path.rstrip("/") in ("/data", "/data.json"):
                body = json.dumps(payload, indent=1).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            else:
                body = render_page(payload).encode("utf-8")
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # No framing, no sniffing, no referrer: a local page showing governed
            # content should not be embeddable or leak its URL onward.
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            """Silent. The viewer is a read surface and its access log would be a second
            place governed block ids accumulate outside the evidence chain."""

    with ThreadingHTTPServer((bind_host, bind_port), _Handler) as httpd:
        print(f"mind-mem viewer on http://{bind_host}:{bind_port}/  (read-only, Ctrl+C to stop)")
        httpd.serve_forever()
