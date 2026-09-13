"""``mind-mem.json`` config loading + configurable limits.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-1). Provides:

* :data:`_DEFAULT_LIMITS` — the hard-coded ceilings used when
  ``mind-mem.json`` doesn't override them.
* :func:`_get_limits` — resolve + validate the ``"limits"`` block
  from config, preserving defaults for any missing / malformed
  entries.
* :func:`_load_config` — parse ``mind-mem.json`` with graceful
  JSON-error fallback to ``init_workspace.DEFAULT_CONFIG``.
* :func:`_load_extra_categories` — project-level extra-category
  registry read from the ``"categories.extra_categories"`` block.
* :data:`QUERY_TIMEOUT_SECONDS` — module-level default sourced
  from ``_DEFAULT_LIMITS`` at import time.

Behavior is bit-for-bit identical to the pre-move version — the
log category ``mcp_server`` is preserved so log-based assertions
keep working, and the ``init_workspace.DEFAULT_CONFIG`` fallback
is still imported lazily inside the exception path.
"""

from __future__ import annotations

import json
import os

from mind_mem.observability import get_logger

from .workspace import _workspace

_log = get_logger("mcp_server")


_DEFAULT_LIMITS = {
    "max_recall_results": 100,
    "max_similar_results": 50,
    "max_prefetch_results": 20,
    "max_category_results": 10,
    "query_timeout_seconds": 30,
    "rate_limit_calls_per_minute": 120,
}


def _load_config(ws: str) -> dict:
    """Return the request's bound config if one governs *ws*, else load mind-mem.json (#26).

    THE SECOND CHOKEPOINT. ``_recall_core._get_config`` is not the only loader the ranking
    consults: ``_recall_impl_uncached`` builds the hybrid backend from ``_load_config(ws)``, so
    binding only the engine's loader still let ``HybridBackend.from_config`` be constructed from
    whatever was on disk when the leg ran. An independent control caught exactly that — the
    recorded metadata claimed the captured config while the backend had been built from a later
    one. Both loaders have to honour the binding or neither claim is worth anything.

    A copy is returned, not the bound mapping itself: callers here mutate the dict they get
    (defaults are merged into it), and letting that reach the context would make the "immutable
    request context" a claim rather than a fact.

    On JSONDecodeError, logs line/column and returns DEFAULT_CONFIG.
    """
    from mind_mem.request_context import context_config_for

    _bound = context_config_for(ws)
    if _bound is not None:
        return dict(_bound)

    config_path = os.path.join(ws, "mind-mem.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return dict(json.load(f))
    except json.JSONDecodeError as exc:
        _log.warning(
            "config_json_decode_error",
            path=config_path,
            line=exc.lineno,
            column=exc.colno,
            msg=str(exc),
        )
        # Fall back to built-in defaults
        from mind_mem.init_workspace import DEFAULT_CONFIG

        return dict(DEFAULT_CONFIG)
    except (OSError, UnicodeDecodeError) as exc:
        _log.warning("config_read_error", path=config_path, error=str(exc))
        return {}


def _load_extra_categories(ws: str) -> dict:
    """Load extra_categories from mind-mem.json config."""
    cfg = _load_config(ws)
    return dict(cfg.get("categories", {}).get("extra_categories", {}))


def _get_limits(ws: str | None = None) -> dict:
    """Return the limits dict from config, falling back to defaults."""
    if ws is None:
        try:
            ws = _workspace()
        except Exception:
            return dict(_DEFAULT_LIMITS)
    cfg = _load_config(ws)
    limits = cfg.get("limits", {})
    result = dict(_DEFAULT_LIMITS)
    for key in _DEFAULT_LIMITS:
        if key in limits:
            try:
                result[key] = int(limits[key])
            except (TypeError, ValueError):
                pass  # keep default
    return result


_VALID_QUALITY_GATE_MODES: frozenset[str] = frozenset({"off", "advisory", "strict"})
_DEFAULT_QUALITY_GATE_MODE = "advisory"


def _get_quality_gate_mode(ws: str | None = None) -> str:
    """Return the quality-gate mode from config, defaulting to ``"advisory"``.

    Reads ``quality_gate.mode`` from ``mind-mem.json``.  Any unrecognised
    value is silently replaced with ``"advisory"`` so a typo never disables
    the gate unexpectedly.

    Valid values: ``"off"`` | ``"advisory"`` | ``"strict"``.
    """
    if ws is None:
        try:
            ws = _workspace()
        except Exception:
            return _DEFAULT_QUALITY_GATE_MODE
    cfg = _load_config(ws)
    mode = cfg.get("quality_gate", {}).get("mode", _DEFAULT_QUALITY_GATE_MODE)
    if mode not in _VALID_QUALITY_GATE_MODES:
        _log.warning(
            "quality_gate_mode_invalid",
            value=mode,
            fallback=_DEFAULT_QUALITY_GATE_MODE,
        )
        return _DEFAULT_QUALITY_GATE_MODE
    return str(mode)


# Per-query timeout in seconds (read from config at call time via _get_limits)
QUERY_TIMEOUT_SECONDS = _DEFAULT_LIMITS["query_timeout_seconds"]
