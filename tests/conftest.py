"""Shared test fixtures.

Currently one thing: a real governance admission for tests that exercise
``BlockStore.write_block`` directly.

Since the write invariant landed, ``write_block`` refuses any write with no
open admission. That is the point of it. Storage-layer unit tests still need
to call it directly, so they open a **real** admission through the gate --
a genuine chain entry, not a stubbed receipt.

There is deliberately no test-only constructor, no wildcard receipt and no
environment escape hatch in ``src/``. An invariant with a bypass reserved for
tests is not an invariant, and this project has already been bitten twice by
checks that reported success over work they never inspected.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys
import tempfile
from typing import Iterator

import pytest

# The two benchmark entry-point scripts moved out of the wheel in 5.0.0
# (src/mind_mem/bench/{locomo_suite,recompaction_bench}.py -> benchmarks/).
# They are still real, tested logic -- they reproduce the published LoCoMo and
# recompaction numbers -- so their tests keep running; they just import by
# module name from benchmarks/ instead of through the package.
_BENCHMARKS = pathlib.Path(__file__).resolve().parents[1] / "benchmarks"
if _BENCHMARKS.is_dir() and str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))


@pytest.fixture
def admitted(tmp_path) -> Iterator[None]:
    """Open a real governance admission for the duration of one test.

    Uses ``admit_proposal``, whose receipt covers any block id for its scope --
    the same ambient authority ``apply_engine`` runs under while applying an
    approved proposal. It is a real ``admit`` call writing a real chain entry;
    nothing here forges a receipt.
    """
    from mind_mem.governance_gate import get_gate

    workspace = str(tmp_path)
    gate = get_gate(workspace)
    if gate is None:  # pragma: no cover - defensive
        yield
        return
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            gate.admit_proposal(
                proposal_id="TEST-ADMISSION",
                content="[]",
                actor="pytest",
            )
        )
        yield


@pytest.fixture
def admit_delete():
    """Open a real ``DELETE`` scope around one block id, for storage tests.

    The delete-side twin of :func:`admitted`, and needed for the same reason:
    since the delete invariant landed, ``BlockStore.delete_block`` refuses any
    removal with no ``admit_delete`` scope open, and a WRITE receipt is
    explicitly not transferable to a delete. Storage-layer tests that remove a
    row directly therefore have to open the real scope, exactly as the MCP and
    HTTP delete doors do.

    Measured 2026-09-02: six such tests were red on the ``postgres backend``
    job -- the only job that runs them, because they gate on
    ``MIND_MEM_TEST_PG_DSN`` and skip everywhere else, which is why a fully
    green local gate never saw them.

    Like :func:`admitted`, this mints a genuine receipt through the gate and
    writes a genuine chain entry. There is no test-only bypass, and there must
    not be one: an invariant with an escape hatch reserved for tests is not an
    invariant.
    """
    from mind_mem.governance_gate import get_gate

    @contextlib.contextmanager
    def _scope(workspace: str, block_id: str, *, rationale: str = "storage-layer test removal") -> Iterator[None]:
        gate = get_gate(workspace)
        with gate.admit_delete(str(block_id), rationale=rationale, actor="pytest"):
            yield

    return _scope


@pytest.fixture(autouse=True, scope="session")
def _child_processes_speak_utf8() -> Iterator[None]:
    """Make every subprocess this suite spawns write UTF-8.

    Many tests run ``mm_cli`` through ``subprocess.run(..., encoding="utf-8")``.
    On Windows a child Python writes stdout in the console codepage (cp1252),
    so any non-ASCII character the CLI prints comes back as bytes the parent
    cannot decode. The decode happens on subprocess's reader THREAD, where the
    UnicodeDecodeError does not propagate -- ``run`` returns with
    ``stdout=None`` and the test dies on ``"..." in None`` with a TypeError
    that says nothing about encoding. Measured 2026-08-29: an em dash in
    ``mm resume`` output (0x97 in cp1252) failed every Windows and macOS matrix
    row while passing on Linux, whose default is already UTF-8.

    Asking the child for UTF-8 is the other half of the parent already asking to
    decode UTF-8; without it the pair only agrees by accident of platform.
    Ten test modules spawn ``mm_cli`` and none of them set this, so it belongs
    here rather than in one helper.
    """
    import os

    previous = {k: os.environ.get(k) for k in ("PYTHONUTF8", "PYTHONIOENCODING")}
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _contain_bare_tempfiles(tmp_path_factory, monkeypatch):
    """Keep every bare ``tempfile`` call inside pytest's own tmp tree.

    115 call sites in this suite reach for ``tempfile.mkdtemp(prefix="mm_...")``
    without a ``dir=`` and without cleanup, so each one leaves a workspace
    behind in ``/tmp`` forever. Measured on this machine: **12,141 orphaned
    directories holding 469,413 inodes** — 57% of every inode on the tmpfs.
    That is not a tidiness problem. When the inode table fills, ``shutil.copy2``
    starts failing inside ``init_workspace``, and the suite reports dozens of
    unrelated-looking errors and ERRORs at collection; the run that found this
    died mid-traceback because pytest could no longer write its own output.

    Redirecting ``tempfile.tempdir`` is a single fix for all of them, and for
    every future one: ``mkdtemp``/``NamedTemporaryFile``/``mkstemp`` consult it
    when no explicit ``dir=`` is given, so unchanged call sites now land under
    ``tmp_path`` and pytest reaps them on its normal schedule. A call site that
    passes ``dir=`` explicitly is unaffected.

    The alternative — editing 115 sites — fixes today's leaks and none of
    tomorrow's, because nothing would stop the 116th.
    """
    # Deliberately NOT inside ``tmp_path``: several tests scan their own
    # ``tmp_path`` for stray files and would report our scratch as a leak
    # from the code under test. A sibling directory isolates without
    # polluting the thing being measured.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path_factory.mktemp("bare-tempfiles")))
    yield


@pytest.fixture(autouse=True)
def _isolate_operator_home_state(tmp_path_factory, monkeypatch):
    """Never let a test read or write the operator's real ``~/.mind-mem``.

    ``model_gate._registry_path`` falls back to ``~/.mind-mem`` when
    ``MIND_MEM_GATE_REGISTRY`` is unset, and ``_promotion_ledger_path`` sits
    beside it. So a plain ``WeightRegistry()`` in a test wrote the operator's
    live ``~/.mind-mem/model_promotions.json`` — 29 KB of it on this machine,
    last modified *by a test run*. Two sessions running the suite at once then
    fought over one file, which is why
    ``test_v28_completion::test_weight_registry_revert`` failed in full runs
    and passed alone.

    The docstring on ``_promotion_ledger_path`` says no test can write the
    operator's real ledger by forgetting a second env var — true, and beside
    the point: the default path needs no forgetting at all. Both env vars are
    pinned here so the guarantee holds without every test remembering.

    This is containment, not a behaviour change: a test that sets either
    variable itself still wins, because monkeypatch.setenv here runs first.
    """
    # Sibling of ``tmp_path``, not a child, for the same reason as above:
    # a test scanning its own tmp_path must not see this directory.
    home = tmp_path_factory.mktemp("operator-home")
    monkeypatch.setenv("MIND_MEM_GATE_REGISTRY", str(home / "model_gate.json"))
    monkeypatch.setenv("MIND_MEM_PROMOTION_LEDGER", str(home / "model_promotions.json"))
    yield
