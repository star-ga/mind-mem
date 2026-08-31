# Copyright 2026 STARGA, Inc.
"""The authorisation-scope vocabulary, defined exactly once.

This module exists because the same word was being answered differently in two
places. ``api/rest.py`` accepted only the literal ``"admin"`` while
``mcp/infra/acl.py`` treated ``{"admin", "full", "mind-mem:admin"}`` as admin.
While the REST admin gate was being skipped entirely that divergence was
invisible; closing the gate made it reachable and would have locked every
``full``-scoped key out of the REST admin endpoints.

The first repair imported the set from ``acl`` behind a ``try``/``except
ImportError`` that fell back to a *duplicated literal* — which reinstates the
defect the moment either copy is edited, and no test pinned them equal. Hence a
module with no dependencies of its own: both surfaces import it directly, there
is no fallback to drift from, and a REST-only install (no MCP extra) still
resolves it.
"""

from __future__ import annotations

#: Scopes that grant administrative authority. Any one of them is sufficient.
ADMIN_SCOPES = frozenset({"admin", "full", "mind-mem:admin"})

__all__ = ["ADMIN_SCOPES"]
