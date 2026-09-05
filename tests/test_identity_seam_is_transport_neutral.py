# Copyright 2026 STARGA, Inc.
"""The caller-identity seam is core, and core never reaches into REST for it.

Closes finding M2 (ROADMAP items "FastAPI audit attribution" and "Audit
attribution through FastAPI sync deps" — one defect written twice).

``governance_gate`` and ``mcp.tools.encryption`` used to resolve the
acting identity with a lazy ``from mind_mem.api.rest import
current_agent_id`` wrapped in a bare ``except Exception``: the CORE
governance layer reaching into the OPTIONAL REST extra, with the failure
swallowed. Whenever that import did not resolve — FastAPI absent, module
renamed, anything — every audit record was silently stamped with a
fallback instead of the real actor. No error, no warning.

That is not hypothetical. It shipped: the import in ``encryption.py``
once named ``mcp.infra.observability``, which defines no such symbol, so
every decrypt audit record was written unattributed regardless of who
called. Only the module name was fixed at the time; the swallow that hid
it stayed, leaving the identical defect one rename away.

The receipt is the product's thesis. A receipt that silently misattributes
WHO is a receipt that lies about the one field an audit exists to
establish — so the guards below are not style checks.

Three things are proved here, in this order:

1. the ContextVar is genuinely SET and the readers genuinely SEE it
   (a bare ``assert reader() == "alice"`` passes for free if the var was
   never set and the default happened to match, so every such assertion
   is paired with proof the value is not the default);
2. the UNSET case produces the documented fallback AND its diagnostic;
3. core cannot re-acquire an import-time OR lazy dependency on
   ``api.rest`` for identity — an AST guard in the shape of the
   ``tests/_write_path_scan.py`` scans, each with a planted-violation
   positive control so a scan that has stopped looking fails loudly
   rather than passing empty.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import os
from typing import Iterator

import pytest
from _write_path_scan import SRC_ROOT, iter_source_files, parse, relpath

from mind_mem import audit_context as ac
from mind_mem import governance_gate as gg
from mind_mem.mcp.tools import encryption as enc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: The three readers of the seam, as callables. Named as a group because
#: the whole point of the fix is that they answer the SAME question with
#: the SAME value — they used to disagree ("system" vs "anonymous").
READERS = {
    "audit_context.current_agent": ac.current_agent,
    "governance_gate._current_agent": gg._current_agent,
    "encryption._current_actor": enc._current_actor,
}


@contextlib.contextmanager
def captured_audit_log() -> Iterator[list[logging.LogRecord]]:
    """Records emitted by the real ``mind-mem.audit_context`` logger.

    A handler on that logger rather than ``caplog``: ``StructuredLogger``
    sets ``propagate = False``, so nothing mind-mem logs ever reaches the
    root handler pytest installs. Asserting through ``caplog`` here would
    be asserting on an empty list for a reason unrelated to the code under
    test — the exact vacuous-pass shape these tests exist to refuse.
    """
    logger = logging.getLogger("mind-mem.audit_context")
    records: list[logging.LogRecord] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    sink = _Sink(level=logging.DEBUG)
    previous = logger.level
    logger.addHandler(sink)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(sink)
        logger.setLevel(previous)


@pytest.fixture
def unwarned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the once-per-process warning latch for this test."""
    monkeypatch.setattr(ac, "_unattributed_warned", False)


# ---------------------------------------------------------------------------
# 1. The var is SET, and the readers SEE it
# ---------------------------------------------------------------------------


class TestTheReadersSeeWhatWasSet:
    def test_the_identity_is_not_already_alice(self) -> None:
        """POSITIVE CONTROL for every ``== "alice"`` assertion below.

        Without this, an assertion that a reader returns "alice" would
        pass just as happily if nothing ever set the var and the default
        happened to be "alice".
        """
        assert ac.current_agent_id.get() == ac.UNATTRIBUTED
        assert ac.current_agent_id.get() != "alice"
        for name, read in READERS.items():
            assert read() != "alice", name

    def test_binding_sets_the_var_and_every_reader_sees_it(self) -> None:
        with ac.bind_current_agent("alice"):
            # The var was SET — proved directly, not inferred from a reader.
            assert ac.current_agent_id.get() == "alice"
            # ...and each reader saw that value.
            for name, read in READERS.items():
                assert read() == "alice", name

    def test_the_binding_is_scoped_and_unwinds(self) -> None:
        with ac.bind_current_agent("alice"):
            assert gg._current_agent() == "alice"
        assert ac.current_agent_id.get() == ac.UNATTRIBUTED
        assert gg._current_agent() == ac.UNATTRIBUTED

    def test_an_unidentified_bind_does_not_clobber_an_outer_identity(self) -> None:
        """A transport that could not identify its caller must not erase one."""
        with ac.bind_current_agent("alice"):
            with ac.bind_current_agent("") as inner:
                assert inner == ac.UNATTRIBUTED
                assert gg._current_agent() == "alice"
            with ac.bind_current_agent(ac.UNATTRIBUTED):
                assert gg._current_agent() == "alice"


# ---------------------------------------------------------------------------
# 2. The unset case: one documented fallback, and a diagnostic
# ---------------------------------------------------------------------------


class TestTheUnsetFallback:
    def test_every_reader_falls_back_to_the_same_value(self) -> None:
        """The two readers used to disagree: "system" and "anonymous".

        Both were the ``except`` arm of a swallowed import rather than a
        decision, and "system" was ONLY ever produced by the import
        failing — the working path already answered "anonymous". One
        value now, chosen deliberately, at the package root.
        """
        answers = {name: read() for name, read in READERS.items()}
        assert set(answers.values()) == {ac.UNATTRIBUTED}, answers
        assert "system" not in answers.values()

    def test_the_first_unattributed_call_warns(self, unwarned: None) -> None:
        with captured_audit_log() as records:
            assert ac.current_agent() == ac.UNATTRIBUTED
        warnings = [r for r in records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.msg for r in records]
        assert warnings[0].msg == "audit_attribution_absent"
        assert getattr(warnings[0], "data", {})["actor"] == ac.UNATTRIBUTED

    def test_later_unattributed_calls_drop_to_debug(self, unwarned: None) -> None:
        """One warning names the condition; a flood teaches operators to filter it.

        mind-mem is also used as a plain library and CLI where there is no
        transport and no authentication at all, so warning on every
        admission there would be noise, not signal.
        """
        with captured_audit_log() as records:
            ac.current_agent()
            ac.current_agent()
            ac.current_agent()
        assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG, logging.DEBUG]

    def test_an_attributed_call_emits_no_diagnostic(self, unwarned: None) -> None:
        """POSITIVE CONTROL for the two assertions above.

        If the diagnostic fired unconditionally, both would still pass.
        """
        with captured_audit_log() as records:
            with ac.bind_current_agent("alice"):
                assert ac.current_agent() == "alice"
        assert records == []

    def test_the_fallback_is_the_absence_of_an_identity(self) -> None:
        """It is a sentinel, not a principal: nothing may bind it as one."""
        with ac.bind_current_agent(ac.UNATTRIBUTED) as bound:
            assert bound == ac.UNATTRIBUTED
            assert ac.current_agent_id.get() == ac.UNATTRIBUTED


# ---------------------------------------------------------------------------
# 3. A transport that authenticated but did not bind is still attributed
# ---------------------------------------------------------------------------


class TestAuditContextFallback:
    def test_a_recorded_authenticated_agent_is_used(self, unwarned: None) -> None:
        ctx = ac.AuditContext(request_id="r-1", transport="grpc")
        with ac.bind_audit_context(ctx):
            assert ac.record_authenticated_agent("bob") is True
            # The ContextVar itself was NOT set -- this is the other leg.
            assert ac.current_agent_id.get() == ac.UNATTRIBUTED
            with captured_audit_log() as records:
                assert gg._current_agent() == "bob"
                assert enc._current_actor() == "bob"
            assert records == [], "an attributed call must emit no diagnostic"

    def test_a_context_without_auth_still_falls_back(self, unwarned: None) -> None:
        """POSITIVE CONTROL: the mere presence of a bound context is not identity."""
        ctx = ac.AuditContext(request_id="r-2", actor_claimed="mallory", transport="grpc")
        with ac.bind_audit_context(ctx):
            # A header claim is a claim. It is never promoted to identity.
            assert gg._current_agent() == ac.UNATTRIBUTED


# ---------------------------------------------------------------------------
# 4. REST attribution is unchanged
# ---------------------------------------------------------------------------

rest = pytest.importorskip("mind_mem.api.rest", reason="REST is an optional extra")


class TestRestKeepsWorkingExactlyAsBefore:
    def test_rest_reexports_the_same_object(self) -> None:
        """``mind_mem.api.rest.current_agent_id`` still resolves, same object.

        Kept because it is imported from there by
        ``tests/test_silent_failure_regressions.py`` and named in
        ``docs/configuration.md``; the alias means a ``.set`` through
        either name is visible through the other.
        """
        assert rest.current_agent_id is ac.current_agent_id

    def test_rest_unidentified_is_the_root_constant(self) -> None:
        assert rest._UNIDENTIFIED == ac.UNATTRIBUTED

    def test_a_set_through_rest_reaches_the_governance_reader(self) -> None:
        assert gg._current_agent() != "agent-7"  # positive control
        token = rest.current_agent_id.set("agent-7")
        try:
            assert ac.current_agent_id.get() == "agent-7"
            assert gg._current_agent() == "agent-7"
            assert enc._current_actor() == "agent-7"
        finally:
            rest.current_agent_id.reset(token)


# ---------------------------------------------------------------------------
# 5. AST guards -- core must not re-acquire the dependency
# ---------------------------------------------------------------------------

#: Names that carry, or stand in for, the acting identity. A core module
#: importing any of these FROM the REST layer is the defect returning.
#: Scoped to symbols rather than to the module, because two core modules
#: legitimately import ``run`` / ``create_app`` from ``api.rest``
#: (``mm_cli`` launches the server, ``spec.export_openapi`` derives the
#: OpenAPI document) and a blanket ban would have to be weakened to
#: accommodate them -- at which point it stops guarding anything.
IDENTITY_SYMBOLS = frozenset(
    {
        "UNATTRIBUTED",
        "_UNIDENTIFIED",
        "_current_actor",
        "_current_agent",
        "current_agent",
        "current_agent_id",
    }
)

#: Functions that resolve identity. Each must be reachable without an
#: exception handler, because after the inversion there is nothing
#: optional left for one to hide.
IDENTITY_READERS = frozenset(
    {
        "_current_actor",
        "_current_agent",
        "_report_unattributed",
        "current_agent",
    }
)


def core_source_files() -> tuple[str, ...]:
    """Every module under ``src/mind_mem`` except the REST layer itself."""
    api_dir = os.path.join(SRC_ROOT, "api") + os.sep
    return tuple(p for p in iter_source_files() if not p.startswith(api_dir))


def scan_identity_imports_from_rest(files: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    """``(file, symbol, lineno)`` for an identity symbol imported from ``api.rest``.

    ``ast.walk`` and not a module-level scan on purpose: the defect this
    replaces was a LAZY import inside a function body, which an
    import-time-only check would never have seen.
    """
    hits: list[tuple[str, str, int]] = []
    for path in files:
        tree = parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if not (module == "rest" or module.endswith("api.rest")):
                continue
            for alias in node.names:
                if alias.name in IDENTITY_SYMBOLS:
                    hits.append((relpath(path), alias.name, node.lineno))
    return tuple(sorted(hits))


def scan_api_package_imports(files: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    """``(file, module, lineno)`` for any import of the ``mind_mem.api`` package."""
    hits: list[tuple[str, str, int]] = []
    for path in files:
        tree = parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "api" or module.startswith(("api.", "mind_mem.api")):
                    hits.append((relpath(path), module, node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("mind_mem.api"):
                        hits.append((relpath(path), alias.name, node.lineno))
    return tuple(sorted(hits))


def scan_reader_exception_handlers(files: tuple[str, ...]) -> tuple[tuple[str, str, bool], ...]:
    """``(file, function, has_try)`` for every definition of an identity reader.

    A reader that is a plain alias assignment holds no logic and so
    appears in no row; the test asserts separately that the one real
    function is present, so an empty result is a failure and not a pass.
    """
    out: list[tuple[str, str, bool]] = []
    for path in files:
        tree = parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in IDENTITY_READERS:
                continue
            has_try = any(isinstance(inner, ast.Try) for inner in ast.walk(node))
            out.append((relpath(path), node.name, has_try))
    return tuple(sorted(out))


def _plant(tmp_path, body: str) -> tuple[str, ...]:
    target = tmp_path / "planted_violation.py"
    target.write_text(body, encoding="utf-8")
    return (str(target),)


class TestCoreDoesNotDependOnRestForIdentity:
    def test_no_core_module_imports_an_identity_symbol_from_api_rest(self) -> None:
        files = core_source_files()
        assert files, "the scan is not looking at anything"
        hits = scan_identity_imports_from_rest(files)
        assert hits == (), (
            f"core reached into the optional REST layer for identity again; identity is owned by mind_mem.audit_context: {hits}"
        )

    def test_the_import_scanner_finds_a_planted_violation(self, tmp_path) -> None:
        """POSITIVE CONTROL -- module-level form."""
        hits = scan_identity_imports_from_rest(_plant(tmp_path, "from mind_mem.api.rest import current_agent_id\n"))
        assert [h[1] for h in hits] == ["current_agent_id"]

    def test_the_import_scanner_finds_a_planted_LAZY_violation(self, tmp_path) -> None:
        """POSITIVE CONTROL -- the exact shape the defect had.

        A guarded import inside a function body. An import-time-only or
        top-of-file scan would report this file clean.
        """
        planted = (
            "def _current_agent() -> str:\n"
            "    try:\n"
            "        from mind_mem.api.rest import current_agent_id\n"
            "        return current_agent_id.get()\n"
            "    except Exception:\n"
            '        return "system"\n'
        )
        hits = scan_identity_imports_from_rest(_plant(tmp_path, planted))
        assert [h[1] for h in hits] == ["current_agent_id"]

    def test_the_import_scanner_ignores_the_sanctioned_non_identity_imports(self) -> None:
        """The guard is symbol-scoped, so ``run`` / ``create_app`` stay legal.

        Proves the scan is discriminating rather than simply finding
        nothing: these imports of ``api.rest`` DO exist in core today.
        """
        sanctioned = scan_api_package_imports(core_source_files())
        assert sanctioned, "expected mm_cli / spec.export_openapi to import api.rest"

    def test_audit_context_imports_nothing_from_the_api_package(self) -> None:
        """The inversion must not reverse: the owner cannot depend on a transport."""
        owner = os.path.join(SRC_ROOT, "audit_context.py")
        assert os.path.exists(owner)
        assert scan_api_package_imports((owner,)) == ()

    def test_the_api_scanner_finds_a_planted_violation(self, tmp_path) -> None:
        """POSITIVE CONTROL for the assertion above."""
        hits = scan_api_package_imports(_plant(tmp_path, "import mind_mem.api.rest\n"))
        assert [h[1] for h in hits] == ["mind_mem.api.rest"]

    def test_no_identity_reader_carries_an_exception_handler(self) -> None:
        """The swallow IS the bug -- not the module it happened to name."""
        rows = scan_reader_exception_handlers(iter_source_files())
        assert ("src/mind_mem/audit_context.py", "current_agent", False) in rows, (
            f"the scan found no definition of current_agent -- it is not looking at anything: {rows}"
        )
        swallowers = [r for r in rows if r[2]]
        assert swallowers == [], f"an identity reader grew an exception handler: {swallowers}"

    def test_the_swallow_scanner_finds_a_planted_try(self, tmp_path) -> None:
        """POSITIVE CONTROL for the assertion above."""
        planted = 'def current_agent() -> str:\n    try:\n        return _lookup()\n    except Exception:\n        return "system"\n'
        rows = scan_reader_exception_handlers(_plant(tmp_path, planted))
        assert [r[1:] for r in rows] == [("current_agent", True)]
