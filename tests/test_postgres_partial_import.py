# Copyright 2026 STARGA, Inc.
"""A partially-successful optional import must not cache a broken state.

`_require_psycopg` imports two modules and raises a clear, actionable
ImportError naming `pip install "mind-mem[postgres]"` when either is
missing. Its cache short-circuit tested only the FIRST of the two:

    if _psycopg is not None:
        return _psycopg, _psycopg_pool

So when `psycopg` is installed but `psycopg_pool` is not, the first call
sets `_psycopg`, then raises on the second import -- leaving the module
global set. Every later call short-circuits and returns `(psycopg, None)`
WITHOUT raising, and the caller constructs `ConnectionPool(...)` where
ConnectionPool is None:

    TypeError: 'NoneType' object is not callable

Observed live: `mind-mem-recall` printed
"recall: backend error ('NoneType' object is not callable), falling back
to scan" and silently degraded to a linear scan, so recall kept answering
with unrelated blocks instead of naming the missing dependency.

The failure is worse than a crash: the guard EXISTS and produces exactly
the right message on the first call, then never fires again.
"""

from __future__ import annotations

import builtins

import pytest


@pytest.mark.unit
def test_a_failed_pool_import_does_not_poison_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem import block_store_postgres as bsp

    monkeypatch.setattr(bsp, "_psycopg", None, raising=False)
    monkeypatch.setattr(bsp, "_psycopg_pool", None, raising=False)

    # Stub BOTH sides rather than relying on what this interpreter happens
    # to have installed. The first version of this test assumed psycopg was
    # importable; it passed on 3.12 (where it is) and failed on 3.14 (where
    # it is not), because the FIRST import then failed and the message named
    # psycopg instead of psycopg_pool. That is the same "test asserts the
    # runner, not the product" defect this suite has been clearing all day.
    import sys
    import types

    monkeypatch.setitem(sys.modules, "psycopg", types.ModuleType("psycopg"))
    real_import = builtins.__import__

    def no_pool(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psycopg_pool":
            raise ModuleNotFoundError("No module named 'psycopg_pool'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_pool)

    # First call: must raise, naming the remedy.
    with pytest.raises(ImportError, match=r"psycopg_pool"):
        bsp._require_psycopg()

    # SECOND call is the regression: it used to short-circuit on the
    # already-set _psycopg and hand back a None pool, so the caller got a
    # TypeError instead of this ImportError.
    with pytest.raises(ImportError, match=r"psycopg_pool"):
        bsp._require_psycopg()

    # And the partial state must not have been cached.
    assert bsp._psycopg_pool is None
    assert bsp._psycopg is None, "a partial import must not leave _psycopg set"


@pytest.mark.unit
def test_a_successful_import_still_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fix must not disable the cache it is correcting."""
    # Stubbed for the same reason as above -- this asserts the CACHE, which
    # is product behaviour, not whether the driver is installed here.
    import sys
    import types

    from mind_mem import block_store_postgres as bsp

    monkeypatch.setitem(sys.modules, "psycopg", types.ModuleType("psycopg"))
    pool_mod = types.ModuleType("psycopg_pool")
    pool_mod.ConnectionPool = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg_pool", pool_mod)

    monkeypatch.setattr(bsp, "_psycopg", None, raising=False)
    monkeypatch.setattr(bsp, "_psycopg_pool", None, raising=False)

    a = bsp._require_psycopg()
    b = bsp._require_psycopg()
    assert a[0] is b[0] and a[1] is b[1]
    assert bsp._psycopg is not None and bsp._psycopg_pool is not None
