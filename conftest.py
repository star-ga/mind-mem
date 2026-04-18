"""Shared pytest fixtures for mind-mem test suite."""

import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mind_mem.init_workspace import init as init_workspace  # noqa: E402


def _sqlite_has_load_extension() -> bool:
    """Some Python builds (notably the python.org macOS installer) omit the
    sqlite ``--enable-loadable-sqlite-extensions`` build flag, which means
    `sqlite_vec.load(conn)` cannot run. Tests that exercise the vector
    backend depend on this capability and must be skipped in that case —
    the production code already degrades to BM25-only recall on the same
    platforms, so skipping mirrors real-world behaviour.
    """
    try:
        conn = sqlite3.connect(":memory:")
        try:
            if not hasattr(conn, "enable_load_extension"):
                return False
            conn.enable_load_extension(True)
            conn.enable_load_extension(False)
            return True
        finally:
            conn.close()
    except (sqlite3.NotSupportedError, AttributeError):
        return False


_HAS_LOAD_EXT = _sqlite_has_load_extension()


def pytest_collection_modifyitems(config, items):
    """Skip vector-backend tests on Python builds without sqlite loadable
    extensions. Identifying heuristic: the test module imports
    `sqlite_vec` or the test id contains `niah` / `vector` / `semantic`."""
    if _HAS_LOAD_EXT:
        return
    skip_marker = pytest.mark.skip(reason="sqlite without loadable extensions — vector backend unavailable")
    for item in items:
        # Module-level skip: anything that mentions sqlite_vec in its
        # source. Fall back to name-based heuristic so we still catch
        # parametric suites like test_niah.py that load sqlite_vec
        # transitively via the recall path.
        module_file = getattr(item.module, "__file__", None)
        if module_file:
            try:
                with open(module_file, "r", encoding="utf-8") as f:
                    source = f.read()
            except OSError:
                source = ""
        else:
            source = ""
        lowered_id = item.nodeid.lower()
        needs_vector = (
            "sqlite_vec" in source
            or "niah" in lowered_id
            or "_vector" in lowered_id
            or "semantic" in lowered_id
        )
        if needs_vector:
            item.add_marker(skip_marker)


@pytest.fixture
def workspace(tmp_path):
    """Create and return an initialized mind-mem workspace."""
    ws = str(tmp_path / "workspace")
    os.makedirs(ws)
    init_workspace(ws)
    return ws
