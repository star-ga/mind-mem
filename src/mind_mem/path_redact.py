# Copyright 2026 STARGA, Inc.
"""The one place absolute host paths are removed from user-scope replies.

A bound or user-scope caller must never learn where the workspace or the
operator's home directory sits on the host. Every tool that has to name a
location uses this module rather than editing a string by hand:

* :func:`workspace_label` -- the workspace as a short, non-absolute label;
* :func:`redact_host_paths` -- rewrite any text so that the resolved forms of
  paths the caller supplied come back in the caller's OWN spelling, the
  workspace becomes workspace-relative, and ``$HOME`` becomes ``~``.

Pure functions: no I/O beyond reading ``$HOME`` and no logging, so a flag-off
probe may call them freely.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping

__all__ = ["redact_host_paths", "supplied_pairs", "workspace_label"]

# A path is only "that path" when the next character cannot extend a path
# component, so ``/a/ws`` does not match inside ``/a/ws-other``.
_END = r"(?![\w.\-])"


def workspace_label(workspace: str) -> str:
    """The workspace's directory name -- never the absolute path."""
    return os.path.basename(str(workspace).rstrip("/\\")) or "."


def supplied_pairs(*supplied: str) -> dict[str, str]:
    """``{host spelling: caller spelling}`` for each path the caller passed.

    Both the ``~``-expanded and the fully resolved forms map back to the
    caller's own text, so a message built from either is echoed exactly as
    supplied (trailing slashes, ``//`` and ``~`` included).
    """
    out: dict[str, str] = {}
    for raw in supplied:
        if not isinstance(raw, str) or not raw:
            continue
        expanded = os.path.expanduser(raw)
        for form in (expanded, os.path.realpath(expanded)):
            if form and form != raw:
                out[form.rstrip("/") or form] = raw
    return out


def _forms(path: str) -> list[str]:
    seen: list[str] = []
    for form in (str(path), os.path.realpath(str(path))):
        form = form.rstrip("/")
        if len(form) > 1 and form not in seen:
            seen.append(form)
    return seen


def _literal(text: str) -> Callable[[re.Match[str]], str]:
    """A re.sub replacement that inserts *text* verbatim (no backslash or group expansion)."""
    return lambda _m: text


def redact_host_paths(
    text: str,
    *,
    workspace: str | None = None,
    supplied: Mapping[str, str] | None = None,
) -> str:
    """Return *text* with absolute host paths removed.

    Order matters and is deliberate: caller-supplied paths first (they are
    echoed verbatim), then the workspace (becomes workspace-relative; the bare
    root becomes ``.``), then ``$HOME`` (becomes ``~``).
    """
    if not text:
        return text
    out = text
    for host, mine in sorted((supplied or {}).items(), key=lambda kv: -len(kv[0])):
        if not host or host == "/":
            continue
        keep = mine if mine.endswith("/") else mine + "/"
        out = re.sub(re.escape(host) + "/", _literal(keep), out)
        out = re.sub(re.escape(host) + _END, _literal(mine), out)
    if workspace:
        for form in sorted(_forms(workspace), key=len, reverse=True):
            out = re.sub(re.escape(form) + "/", "", out)
            out = re.sub(re.escape(form) + _END, ".", out)
    home = os.path.expanduser("~")
    if home and home != "~":
        for form in sorted(_forms(home), key=len, reverse=True):
            out = re.sub(re.escape(form) + _END, "~", out)
    return out
