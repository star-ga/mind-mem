# Copyright 2026 STARGA, Inc.
"""World-staleness configuration — the default-OFF gate and its knobs.

The external-grounding checker (:mod:`mind_mem.world_staleness`) is
inert until ``v4.world_staleness.enabled`` is ``true``. That decision,
and every tunable that goes with it, is resolved and validated here so
the checker itself never reads raw config.

Workspace-local ``mind-mem.json`` is authoritative; a workspace that
does not mention the flag falls back to the process-level resolver in
:mod:`mind_mem.v4.feature_flags` (``MIND_MEM_CONFIG`` / cwd / user
config), so a CLI caller can flip the flag without editing a workspace.

Every knob is validated at this boundary. A malformed value falls back
to its default with a warning instead of raising: a config typo must
never be able to take ``scan()`` down.

Stdlib only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Final, Sequence

from .observability import get_logger
from .world_symbol_probe import DEFAULT_MAX_FILE_BYTES

__all__ = [
    "DEFAULT_MAX_REPORTED",
    "FEATURE_FLAG",
    "WorldStalenessConfig",
    "is_world_staleness_enabled",
    "resolve_world_config",
]

_log = get_logger("world_staleness")

FEATURE_FLAG: Final = "world_staleness"

#: Default cap on how many anchors a single ``scan()`` summary lists.
DEFAULT_MAX_REPORTED: Final = 50


@dataclass(frozen=True)
class WorldStalenessConfig:
    """Resolved, validated knobs for one workspace."""

    enabled: bool = False
    roots: tuple[str, ...] = ()
    missing_roots: tuple[str, ...] = ()
    inline: bool = True
    max_ref_drift: int = 0
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_reported: int = DEFAULT_MAX_REPORTED


def _read_flag_block(workspace: str) -> dict[str, Any]:
    """Return the ``v4.world_staleness`` sub-config for *workspace*.

    Workspace-local ``mind-mem.json`` wins; when it carries no such key
    the process-level resolver (``MIND_MEM_CONFIG`` / cwd / user config)
    is consulted, so a CLI caller can flip the flag without editing a
    workspace. Never raises — an unreadable config means OFF.

    "Unreadable" is where a *present* workspace file differs from an absent
    one, and the difference is the whole guarantee. A workspace with no
    ``mind-mem.json``, or one that simply does not mention the flag, is a
    workspace with no opinion, and the process-level resolver supplies one.
    A workspace file that exists and cannot be parsed is a workspace whose
    opinion could not be READ, and falling through to the environment there
    would let ``MIND_MEM_CONFIG`` turn the feature ON against an
    authoritative statement nobody has seen. It returns ``{}`` instead, which
    resolves to OFF — the direction the docstring has always promised.
    """
    path = os.path.join(os.path.abspath(workspace), "mind-mem.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            _log.warning("world_staleness_config_unreadable", path=path, error=str(exc))
            return {}
        v4 = data.get("v4") if isinstance(data, dict) else None
        if isinstance(v4, dict) and isinstance(v4.get(FEATURE_FLAG), dict):
            return dict(v4[FEATURE_FLAG])
    from .v4.feature_flags import flag_config

    sub = flag_config(FEATURE_FLAG)
    return dict(sub) if isinstance(sub, dict) else {}


def _coerce_int(raw: Any, default: int, *, minimum: int, knob: str) -> int:
    """Validated int for one knob, falling back to *default* WITH a warning.

    The fallback is the documented behaviour — a config typo must never take
    ``scan()`` down. The warning is what makes it honest: without it,
    ``"max_file_bytes": "10MB"`` or ``"max_reported": 0`` reverts in silence
    and the operator goes on believing the value they wrote is in force.
    *knob* names the offending key so the log line points at the line to fix.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        if raw is not None:
            _log.warning("world_staleness_config_invalid", knob=knob, value=repr(raw), fallback=default)
        return default
    if value < minimum:
        _log.warning(
            "world_staleness_config_below_minimum",
            knob=knob,
            value=value,
            minimum=minimum,
            fallback=default,
        )
        return default
    return value


def resolve_world_config(workspace: str) -> WorldStalenessConfig:
    """Resolve the world-staleness config for *workspace*.

    Every knob is validated here, at the boundary. A malformed value
    falls back to its default with a warning rather than raising, so a
    config typo can never take ``scan()`` down.
    """
    if not workspace:
        raise ValueError("workspace must be non-empty")
    ws = os.path.abspath(workspace)
    sub = _read_flag_block(ws)
    enabled = sub.get("enabled") is True

    raw_roots = sub.get("roots")
    candidates: list[str] = []
    if isinstance(raw_roots, str):
        candidates = [raw_roots]
    elif isinstance(raw_roots, Sequence):
        candidates = [r for r in raw_roots if isinstance(r, str) and r.strip()]
    if not candidates:
        candidates = [ws]

    roots: list[str] = []
    missing: list[str] = []
    for candidate in candidates:
        resolved = candidate if os.path.isabs(candidate) else os.path.join(ws, candidate)
        resolved = os.path.normpath(resolved)
        if os.path.isdir(resolved):
            if resolved not in roots:
                roots.append(resolved)
        elif resolved not in missing:
            missing.append(resolved)

    inline = sub.get("inline")
    return WorldStalenessConfig(
        enabled=enabled,
        roots=tuple(roots),
        missing_roots=tuple(missing),
        inline=True if inline is None else bool(inline),
        max_ref_drift=_coerce_int(sub.get("max_ref_drift"), 0, minimum=0, knob="max_ref_drift"),
        max_file_bytes=_coerce_int(sub.get("max_file_bytes"), DEFAULT_MAX_FILE_BYTES, minimum=1, knob="max_file_bytes"),
        max_reported=_coerce_int(sub.get("max_reported"), DEFAULT_MAX_REPORTED, minimum=1, knob="max_reported"),
    )


def is_world_staleness_enabled(workspace: str) -> bool:
    """True when the ``v4.world_staleness`` flag is ON for *workspace*."""
    try:
        return resolve_world_config(workspace).enabled
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        _log.warning("world_staleness_flag_check_failed", error=str(exc))
        return False
