"""A skipped test reads as a pass — so the skip surface itself needs a gate.

Every summary line the suite prints (``9356 passed, 27 skipped``) and every
green check on GitHub counts a skip on the same side of the ledger as a pass.
That makes an *unjustified* skip a hole wearing coverage's clothes: the
assertion never executes, nobody is told, and the capability it guards is
untested while the dashboard says otherwise.

The 5.0.2 skip audit classified all 140 static skip constructs under
``tests/`` and found none of the worst kind — zero unconditional
``pytest.mark.skip``, zero ``xfail``, zero ``expectedFailure``.  Every skip
present is gated on a real, probed capability (a live Postgres DSN, root,
symlink support, ``/proc/self/fd``, a case-sensitive filesystem, a git
history, sqlite loadable extensions).  This module is the ratchet that keeps
it that way: the counts below are **at zero**, so nothing here tolerates a
backlog — it forbids the first regression.

What is banned, and why each form is worse than a failing test:

``@pytest.mark.skip`` / ``@unittest.skip`` (unconditional)
    Runs on no platform, no Python version, no CI row, ever.  The assertion
    has never executed and never will.

``xfail`` / ``expectedFailure``
    Records a known-broken behaviour as an expected outcome.  A green run
    then *includes* the defect.

``skipif(<constant truthy>)``
    The conditional form of the first one.  ``skipif(True, ...)`` and
    ``skipUnless(False, ...)`` read like environment gates in a diff and are
    unconditional skips in fact.

a skip with no reason
    Not a hole by itself, but it makes the next audit impossible: ``-rs``
    prints an empty line and the reader cannot tell (a) from (b).

Deliberately NOT banned: ``skipif`` on a real runtime probe,
``pytest.skip()`` inside a capability check, and ``pytest.importorskip``
(pytest synthesises the reason).  Those are how a suite honestly says "this
host cannot run this" — the audit's job is to prove each one runs *somewhere*
in the matrix, which is a judgement no scanner can make.

Scope is ``tests/``.  The one dynamic ``pytest.mark.skip`` in the repository
lives in the root ``conftest.py`` and is applied programmatically behind a
runtime sqlite probe (``_sqlite_has_load_extension``); it is a conditional
skip expressed as a marker object, not an unconditional decorator, and it is
out of this scanner's directory on purpose rather than by exemption.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

# Names whose *last* segment marks the construct, keyed by the finding kind.
# Matched on the dotted tail so ``pytest.mark.skip``, ``mark.skip`` and a
# ``from pytest import mark`` alias all resolve the same way.
_UNCONDITIONAL_SKIP_TAILS = frozenset({"mark.skip", "unittest.skip"})
_XFAIL_TAILS = frozenset({"mark.xfail", "pytest.xfail", "unittest.expectedFailure"})
_CONDITIONAL_SKIP_LAST = frozenset({"skipif", "skipIf", "skipUnless"})
# ``skipUnless(cond)`` skips when cond is FALSY; the other two skip when
# truthy. A constant argument makes either one unconditional.
_SKIPS_WHEN_FALSY = frozenset({"skipUnless"})


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int
    src: str


def _dotted(node: ast.AST) -> str | None:
    """``pytest.mark.skip`` -> "pytest.mark.skip"; a subscript/call -> None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _tail(dotted: str, n: int) -> str:
    return ".".join(dotted.split(".")[-n:])


def _is_constant_truthiness(node: ast.AST) -> tuple[bool, bool]:
    """(is_constant, truthiness) for a literal skip condition.

    ``skipif("sys.platform == 'win32'")`` — pytest's string form — is a
    non-empty string and therefore *literally* truthy, but pytest ``eval``s
    it, so it is a real condition and must not be flagged. Only genuine
    literals (True/False/1/0/None) are constant conditions here.
    """
    if isinstance(node, ast.Constant) and not isinstance(node.value, str):
        return True, bool(node.value)
    return False, False


def _has_reason(call: ast.Call, *, positional_index: int) -> bool:
    """A reason is a ``reason=`` kwarg or the documented positional slot."""
    if any(kw.arg == "reason" and _nonempty(kw.value) for kw in call.keywords):
        return True
    if len(call.args) > positional_index and _nonempty(call.args[positional_index]):
        return True
    # ``**kwargs`` forwarding: cannot prove absence, so do not claim it.
    return any(kw.arg is None for kw in call.keywords)


def _nonempty(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return bool(node.value)
    return True  # an f-string / name / call is a reason we cannot read but is present


def _scan_source(text: str, rel: str) -> list[Finding]:
    findings: list[Finding] = []
    tree = ast.parse(text)

    def record(kind: str, node: ast.AST) -> None:
        seg = ast.get_source_segment(text, node) or ""
        findings.append(Finding(kind, rel, getattr(node, "lineno", 0), " ".join(seg.split())[:200]))

    def check_marker(node: ast.AST) -> None:
        """``node`` is a decorator expression or a pytestmark value."""
        call = node if isinstance(node, ast.Call) else None
        target = call.func if call is not None else node
        dotted = _dotted(target)
        if dotted is None:
            return
        last = dotted.split(".")[-1]
        if _tail(dotted, 2) in _UNCONDITIONAL_SKIP_TAILS:
            record("unconditional-skip", node)
            return
        if _tail(dotted, 2) in _XFAIL_TAILS or last == "expectedFailure":
            record("xfail", node)
            return
        if last in _CONDITIONAL_SKIP_LAST and call is not None:
            if call.args:
                is_const, truthy = _is_constant_truthiness(call.args[0])
                skips = (not truthy) if last in _SKIPS_WHEN_FALSY else truthy
                if is_const and skips:
                    record("constant-skipif", node)
                    return
            # unittest.skipIf(cond, "reason") -> reason is positional slot 1
            if not _has_reason(call, positional_index=1):
                record("reasonless-skip", node)

    # ``ast.walk`` reaches a decorator's Call a second time through the generic
    # Call branch below. Without this set ``@pytest.mark.xfail(...)`` is counted
    # twice and a future finding count silently drifts. Measured: the control
    # fixture reported 4 xfail findings for 3 xfail forms before the guard.
    marker_nodes: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                marker_nodes.add(id(dec))
                check_marker(dec)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
                values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
                for v in values:
                    marker_nodes.add(id(v))
                    check_marker(v)
        elif isinstance(node, ast.Call):
            if id(node) in marker_nodes:
                continue
            dotted = _dotted(node.func)
            if dotted is None:
                continue
            last = dotted.split(".")[-1]
            if _tail(dotted, 2) in _XFAIL_TAILS:
                record("xfail", node)
            elif last == "skip" and _tail(dotted, 2) not in _UNCONDITIONAL_SKIP_TAILS:
                # ``pytest.skip("why")`` inside a runtime probe: allowed, but
                # it must say why. ``importorskip`` is exempt (pytest writes
                # the reason for it).
                if not _has_reason(node, positional_index=0):
                    record("reasonless-skip", node)

    return findings


@lru_cache(maxsize=8)
def scan_tree(root: Path) -> tuple[tuple[Finding, ...], int]:
    """Return (findings, files_scanned). Both halves matter: an empty finding
    list is only evidence when the file count proves the scan happened.

    Cached and tuple-valued because four tests below scan the same ~530-file
    tree and a full parse costs ~3s; without this the module alone would add
    roughly 10s to every one of the 15 matrix rows. Nothing mutates the tree
    between calls, and each control fixture gets a fresh tmp_path (its own
    cache key), so the cache cannot serve one tree's answer for another.
    """
    findings: list[Finding] = []
    files = 0
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files += 1
        findings.extend(_scan_source(path.read_text(encoding="utf-8"), str(path.relative_to(root))))
    return tuple(findings), files


# ---------------------------------------------------------------------------
# Positive control — the scanner must SEE a true positive of every banned form
# ---------------------------------------------------------------------------

# Written as a string and parsed at runtime so these forms are literals here,
# never real decorators on real tests.
_BAD_FIXTURE = textwrap.dedent(
    """
    import sys
    import unittest

    import pytest

    pytestmark = pytest.mark.skip(reason="module-wide unconditional")


    @pytest.mark.skip(reason="never runs")
    def test_unconditional() -> None:
        assert False


    @pytest.mark.xfail(reason="known broken")
    def test_xfail() -> None:
        assert False


    @unittest.skip("unconditional unittest form")
    class TestUnittestSkip(unittest.TestCase):
        pass


    @pytest.mark.skipif(True, reason="constant condition")
    def test_constant_true() -> None:
        assert False


    @unittest.skipUnless(False, "constant condition, inverted")
    def test_constant_unless() -> None:
        assert False


    @unittest.expectedFailure
    def test_expected_failure() -> None:
        assert False


    def test_runtime_xfail() -> None:
        pytest.xfail("bail out")


    @pytest.mark.skipif(sys.platform == "win32")
    def test_reasonless() -> None:
        assert True


    def test_reasonless_call() -> None:
        pytest.skip()
    """
)

# The false-positive control: every one of these is a legitimate skip and the
# scanner must stay silent on all of them. A scanner that flags these would
# make the ratchet unmaintainable and get exempted away within a release.
_GOOD_FIXTURE = textwrap.dedent(
    """
    import os
    import sys
    import unittest

    import pytest

    psycopg = pytest.importorskip("psycopg", reason="psycopg not installed")

    pytestmark = pytest.mark.skipif(not os.environ.get("DSN"), reason="no live DSN")


    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
    def test_posix_only() -> None:
        assert True


    @pytest.mark.skipif("sys.platform == 'win32'", reason="string condition form")
    def test_string_condition() -> None:
        assert True


    @unittest.skipUnless(os.path.isdir("/proc/self/fd"), "/proc/self/fd is Linux-only")
    class TestProcOnly(unittest.TestCase):
        pass


    def test_probe() -> None:
        if not os.path.isdir("/proc/self/fd"):
            pytest.skip("/proc/self/fd unavailable; cannot measure descriptors here")
        assert True


    @pytest.mark.parametrize("conclusion", ["failure", "skipped", "neutral"])
    def test_parametrized_over_skip_strings(conclusion: str) -> None:
        assert conclusion
    """
)


@pytest.fixture()
def control_tree(tmp_path: Path) -> Path:
    root = tmp_path / "control"
    root.mkdir()
    (root / "test_bad_forms.py").write_text(_BAD_FIXTURE, encoding="utf-8")
    (root / "test_good_forms.py").write_text(_GOOD_FIXTURE, encoding="utf-8")
    return root


class TestScannerHasTeeth:
    """Before any "zero findings" claim, prove the method can see a positive."""

    def test_every_banned_form_is_detected(self, control_tree: Path) -> None:
        findings, files = scan_tree(control_tree)
        assert files == 2, f"control tree not scanned: {files} files"
        bad = [f for f in findings if f.path == "test_bad_forms.py"]
        kinds = sorted({f.kind for f in bad})
        assert kinds == ["constant-skipif", "reasonless-skip", "unconditional-skip", "xfail"], kinds
        by_kind = {k: sum(1 for f in bad if f.kind == k) for k in kinds}
        # 2 unconditional decorators (pytest.mark.skip, unittest.skip) + the
        # module-level pytestmark; 3 xfail forms; 2 constant conditions;
        # 2 reasonless.
        assert by_kind == {
            "unconditional-skip": 3,
            "xfail": 3,
            "constant-skipif": 2,
            "reasonless-skip": 2,
        }, by_kind

    def test_legitimate_skips_are_not_flagged(self, control_tree: Path) -> None:
        """A false-positive control: the ratchet must not punish honest gates."""
        findings, _ = scan_tree(control_tree)
        good = [f for f in findings if f.path == "test_good_forms.py"]
        assert good == [], f"scanner flagged legitimate environment gates: {good}"


class TestTestSuiteHasNoVacuousSkips:
    def test_the_scan_actually_ran(self) -> None:
        """An empty finding list is only evidence when the search happened."""
        _, files = scan_tree(TESTS_DIR)
        # The suite has ~500 test modules; a floor of 100 catches a scan
        # pointed at the wrong directory or an rglob that stopped matching,
        # without pinning a number that legitimate churn would break.
        assert files >= 100, f"only {files} files scanned under {TESTS_DIR}"

    def test_the_scan_sees_the_real_skip_surface(self) -> None:
        """Non-vacuity, second leg: the parser reaches real skip constructs.

        ``test_the_scan_actually_ran`` proves files were opened; this proves
        they were understood. Counted against the same AST walk the ratchet
        uses, so a parser that silently stopped resolving ``pytest.mark.*``
        turns this red instead of turning the ratchet green.
        """
        seen = 0
        for path in sorted(TESTS_DIR.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    dotted = _dotted(node.func)
                    if dotted and dotted.split(".")[-1] in _CONDITIONAL_SKIP_LAST:
                        seen += 1
        assert seen >= 10, f"only {seen} skipif constructs resolved; the tail-matching is broken"

    def test_no_unconditional_skip_or_xfail(self) -> None:
        """Ratchet at zero. This is not a budget — it is a floor of none."""
        findings, _ = scan_tree(TESTS_DIR)
        offenders = [f for f in findings if f.kind in {"unconditional-skip", "xfail", "constant-skipif"}]
        assert offenders == [], "\n".join(
            f"{f.path}:{f.line} [{f.kind}] {f.src}\n"
            "  An unconditional skip/xfail runs on NO matrix row: the assertion never\n"
            "  executes and the summary still counts it beside the passes. Gate it on a\n"
            "  real capability probe, or fix the test."
            for f in offenders
        )

    def test_every_skip_states_a_reason(self) -> None:
        """``-rs`` must be able to answer 'why' for every skip in the suite."""
        findings, _ = scan_tree(TESTS_DIR)
        offenders = [f for f in findings if f.kind == "reasonless-skip"]
        assert offenders == [], "\n".join(f"{f.path}:{f.line} {f.src}" for f in offenders)
