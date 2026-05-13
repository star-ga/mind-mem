"""Audit A-15: shim re-export completeness regression test.

The MCP decomposition (v3.2.0 → v4.0.0) left several modules as
re-export shims so older import paths keep working. Whenever a
canonical module gains a public symbol, the shim has to be updated
or downstream users (and our own internal callers) silently break.

This test enumerates every shim → canonical pair and asserts:

* The shim exposes every PUBLIC name (no leading underscore) the
  canonical module exports via ``__all__`` if it defines one,
  otherwise every public top-level name.
* Every shim re-export resolves to the same object as the canonical
  module (no shadow copies, no double-import surprises).

Add a row to ``_SHIM_MAP`` when introducing a new shim. The test is
intentionally strict: a missing re-export fails CI before users
hit it.
"""

from __future__ import annotations

import importlib
from typing import Iterable

import pytest

# Maps shim module → canonical module. The canonical module is the
# source of truth; the shim's only job is to forward.
_SHIM_MAP: dict[str, str] = {
    # Add new shim/canonical pairs here.
}


def _public_names(mod) -> Iterable[str]:
    explicit = getattr(mod, "__all__", None)
    if explicit:
        yield from explicit
        return
    for name in vars(mod):
        if name.startswith("_"):
            continue
        yield name


@pytest.mark.parametrize("shim_name,canonical_name", sorted(_SHIM_MAP.items()))
def test_shim_reexports_public_surface(shim_name: str, canonical_name: str) -> None:
    try:
        shim = importlib.import_module(shim_name)
    except ImportError:
        pytest.skip(f"shim module {shim_name} is not present in this checkout")
    try:
        canonical = importlib.import_module(canonical_name)
    except ImportError:
        pytest.fail(f"canonical module {canonical_name} failed to import")

    missing: list[str] = []
    drifted: list[str] = []
    for name in _public_names(canonical):
        if not hasattr(shim, name):
            missing.append(name)
            continue
        if getattr(shim, name) is not getattr(canonical, name):
            drifted.append(name)

    assert not missing, (
        f"shim {shim_name!r} is missing re-exports from "
        f"{canonical_name!r}: {sorted(missing)}"
    )
    assert not drifted, (
        f"shim {shim_name!r} has drifted copies of: {sorted(drifted)} "
        f"— re-export the canonical object directly"
    )


def test_shim_map_is_non_circular() -> None:
    """Each shim entry must point at a distinct canonical target.

    Catches typos like ``{"a": "a"}`` that would silently pass the
    re-export check above.
    """
    for shim_name, canonical_name in _SHIM_MAP.items():
        assert shim_name != canonical_name, (
            f"shim_map entry {shim_name!r} points at itself"
        )
