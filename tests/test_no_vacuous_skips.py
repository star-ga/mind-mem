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

There are two ratchets here, and the second is the one that caught real
coverage loss.  ``TestTestSuiteHasNoVacuousSkips`` bans skips that run
nowhere *by construction*.  ``TestModuleScopeGatesAreSatisfiableSomewhere``
bans the far more common kind that runs nowhere *by configuration* — a
module-scope ``importorskip`` naming a dependency no CI job installs, which
withdraws an entire file while every run still reports green.  Its long
preamble records the two measured instances that motivated it.
"""

from __future__ import annotations

import ast
import re
import sys
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest
from _toml_compat import load_pyproject

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


# ---------------------------------------------------------------------------
# Second ratchet: a module-scope gate must be satisfiable on SOME CI job
# ---------------------------------------------------------------------------
#
# The ratchet above bans skips that run nowhere *by construction*. This one
# bans the kind that runs nowhere *by configuration*, which the 5.0.2 audit
# found is where the real coverage went:
#
#   * ``tests/test_mic_map_bench.py`` opened with a module-scope
#     ``pytest.importorskip("pytest_benchmark")``. No CI job installs the
#     ``[benchmark]`` extra on the matrix, so the import aborted the module on
#     every row -- taking with it eight throughput/size tests that need no
#     plugin at all. Twenty tests, zero executions, every run green.
#   * ``tests/test_iter_active_blocks.py`` opened with a module-scope
#     ``pytest.importorskip("psycopg")``. The five Markdown-backend tests above
#     it -- the DEFAULT backend -- were skipped on all 15 matrix rows.
#
# A module-scope ``importorskip`` is the highest-leverage skip in the suite:
# one line silently withdraws an entire file. So each one has to name a
# dependency that at least one CI job actually installs, for a file that job
# actually selects. Everything the audit classified as legitimate passes this:
# ``fastapi`` / ``httpx`` / ``jose`` / ``hypothesis`` / ``sentence_transformers``
# / ``cryptography`` ship in the ``[test]`` extra that every matrix row
# installs, and ``psycopg`` ships in ``[postgres]``, installed by the dedicated
# "postgres backend" job -- which selects its files by grepping tests/ for its
# DSN environment variable, so file selection is checked too, not just the
# extra.
#
# Scope is module-scope ``importorskip`` only, and that is a deliberate line,
# not an exemption: a module-level ``pytestmark = pytest.mark.skipif(<expr>)``
# withdraws a file just as thoroughly, but ``<expr>`` is arbitrary Python and
# mapping it back to a distribution would be guesswork. A gate that guesses
# gets argued with and then loosened. This one only asserts what it can read.

CI_WORKFLOW = TESTS_DIR.parent / ".github" / "workflows" / "ci.yml"
PYPROJECT = TESTS_DIR.parent / "pyproject.toml"

# Distribution name -> import name, only where they differ beyond ``-``/``_``.
# Kept explicit rather than resolved from installed metadata: this check has to
# give the same answer on a runner that does NOT have the package installed,
# which is precisely the situation it exists to reason about.
_IMPORT_NAME = {
    "python-jose": "jose",
    "pyyaml": "yaml",
    "opentelemetry-api": "opentelemetry",
    "opentelemetry-sdk": "opentelemetry",
    "opentelemetry-exporter-otlp": "opentelemetry",
}
# Keys are matched AFTER any ``[extra]`` suffix is stripped, so write
# ``psycopg``, never ``psycopg[binary]`` — the bracketed form would be a dead
# entry that reads as coverage. A requirement's bracketed extras also
# contribute their own names (``python-jose[cryptography]`` -> ``cryptography``,
# which is exactly right); a few of those are build flavours rather than
# importable modules (``psycopg[binary]`` -> ``binary``). That only ever makes
# the available-set slightly larger, and no real gate imports a name like that.

# The file-selection predicate of the "postgres backend" job. That job runs
# `grep -rl "MIND_MEM_TEST_PG_DSN" tests/`, so a file it never selects gets no
# benefit from the extra it installs.
_PG_JOB_SELECTOR = "MIND_MEM_TEST_PG_DSN"


def _requirement_import_names(spec: str) -> set[str]:
    """Import names a single requirement string makes available."""
    spec = spec.split(";")[0].strip()  # drop environment markers
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", " "):
        spec = spec.split(sep)[0]
    spec = spec.strip()
    names: set[str] = set()
    if "[" in spec:
        base, _, rest = spec.partition("[")
        for extra in rest.rstrip("]").split(","):
            extra = extra.strip()
            if extra:
                names.add(_IMPORT_NAME.get(extra.lower(), extra.replace("-", "_").lower()))
        spec = base
    key = spec.lower()
    names.add(_IMPORT_NAME.get(key, key.replace("-", "_")))
    return {n for n in names if n}


def _extra_import_names(extra: str) -> set[str]:
    """Import names installed by ``pip install -e '.[<extra>]'``."""
    data = load_pyproject()
    assert data is not None, "cannot read pyproject.toml; this gate cannot run vacuously"
    reqs = data["project"]["optional-dependencies"][extra]
    names: set[str] = set()
    for req in reqs:
        names |= _requirement_import_names(req)
    return names


@lru_cache(maxsize=1)
def _ci_installed_extras() -> frozenset[str]:
    """Extras named in any ``pip install -e ".[...]"`` line in ci.yml."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    found: set[str] = set()
    for match in re.finditer(r'pip install -e "?\.\[([^\]]+)\]', text):
        for extra in match.group(1).split(","):
            found.add(extra.strip())
    return frozenset(found)


def _module_scope_importorskips(path: Path) -> list[tuple[int, str]]:
    """``(lineno, module)`` for every module-scope ``importorskip`` in *path*."""
    out: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # not module scope: costs one test, not the file
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "importorskip"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                out.append((node.lineno, node.args[0].value))
    return out


def _unsatisfiable_module_gates(root: Path) -> tuple[list[str], int]:
    """(findings, files_scanned) for module-scope gates no CI job can satisfy.

    The glob is ``test_*.py`` because that is pytest's own ``python_files``
    setting: this gate reasons about what the default collector picks up, and a
    file it never collects cannot be withdrawn from a run it was never in.
    Exactly one file in the tree is excluded by that boundary --
    ``tests/red_team/behavioral_audit.py``, whose module-scope
    ``importorskip("inspect_petri")`` is genuinely unsatisfiable under ci.yml.
    It is named here rather than exempted: it is run by path from the separate
    (advisory, `continue-on-error`) red-team workflow, which installs the
    ``[red-team]`` extra, so its gate answers to that workflow's configuration
    and not to this one's model of the matrix.
    """
    matrix_names = _extra_import_names("test")
    pg_names = matrix_names | _extra_import_names("postgres")
    stdlib = set(sys.stdlib_module_names)
    findings: list[str] = []
    files = 0
    for path in sorted(root.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        gates = _module_scope_importorskips(path)
        if not gates:
            continue
        files += 1
        text = path.read_text(encoding="utf-8")
        available = pg_names if _PG_JOB_SELECTOR in text else matrix_names
        for lineno, module in gates:
            root_name = module.split(".")[0]
            if root_name in stdlib or root_name == "mind_mem" or root_name in available:
                continue
            findings.append(
                f"{path.relative_to(root)}:{lineno} module-scope importorskip({module!r}): "
                "no CI job installs it for this file, so the whole module is withdrawn on "
                "every row while the run reports green. Put the dependency in an extra a "
                "job installs, or move the gate onto the individual tests that need it."
            )
    return findings, files


class TestModuleScopeGatesAreSatisfiableSomewhere:
    def test_the_ci_workflow_is_readable_and_installs_what_we_think(self) -> None:
        """Non-vacuity: every assertion below rests on these three reads."""
        extras = _ci_installed_extras()
        assert "test" in extras, f"ci.yml no longer installs the [test] extra: {sorted(extras)}"
        assert "postgres" in extras, f"ci.yml no longer installs the [postgres] extra: {sorted(extras)}"
        assert _PG_JOB_SELECTOR in CI_WORKFLOW.read_text(encoding="utf-8"), (
            "the postgres job no longer selects files by that env var; this gate's "
            "file-selection model is stale and must be re-derived, not relaxed"
        )
        names = _extra_import_names("test")
        assert {"pytest", "fastapi", "jose", "sentence_transformers"} <= names, sorted(names)
        assert "psycopg" in _extra_import_names("postgres")
        # The two-tier model (matrix rows vs the one job with the extra) is only
        # meaningful while [postgres] supplies something [test] does not. If a
        # future release folds the driver into [test] that is a FIX, not a
        # breakage -- so this asserts the model is still needed, and says so,
        # rather than asserting the driver stayed out of [test].
        assert _extra_import_names("postgres") - names, (
            "the [postgres] extra now adds nothing beyond [test]; the file-selection leg of this "
            "gate is dead code and should be removed, not worked around"
        )

    def test_control_tree_detects_an_unsatisfiable_gate(self, tmp_path: Path) -> None:
        """Positive control: the scanner must see a true positive."""
        root = tmp_path / "gates"
        root.mkdir()
        (root / "test_bad_gate.py").write_text(
            'import pytest\n\ntorch = pytest.importorskip("torch")\n\n\ndef test_x() -> None:\n    assert True\n',
            encoding="utf-8",
        )
        (root / "test_good_gate.py").write_text(
            'import pytest\n\nfastapi = pytest.importorskip("fastapi")\n\n\ndef test_y() -> None:\n    assert True\n',
            encoding="utf-8",
        )
        (root / "test_good_pg.py").write_text(
            'import os\n\nimport pytest\n\npsycopg = pytest.importorskip("psycopg")\n'
            '_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN")\n\n\ndef test_z() -> None:\n    assert True\n',
            encoding="utf-8",
        )
        # Same driver as test_good_pg.py, but the file never mentions the env
        # var the "postgres backend" job greps for, so that job never selects
        # it and the extra it installs does this file no good. This is the
        # file-selection leg, and it is the exact shape of a real finding
        # (tests/test_mcp_db_error_backstop.py) -- kept as a control because
        # without it the leg could rot into a no-op unnoticed.
        (root / "test_unselected_pg.py").write_text(
            'import pytest\n\npsycopg = pytest.importorskip("psycopg")\n\n\ndef test_v() -> None:\n    assert True\n',
            encoding="utf-8",
        )
        (root / "test_in_function.py").write_text(
            'import pytest\n\n\ndef test_w() -> None:\n    pytest.importorskip("torch")\n    assert True\n',
            encoding="utf-8",
        )
        findings, files = _unsatisfiable_module_gates(root)
        assert files == 4, f"expected 4 gated files scanned, got {files}"
        flagged = sorted(f.split(":")[0] for f in findings)
        assert flagged == ["test_bad_gate.py", "test_unselected_pg.py"], findings

    def test_no_module_scope_gate_runs_nowhere(self) -> None:
        """Ratchet at zero over the real suite."""
        findings, files = _unsatisfiable_module_gates(TESTS_DIR)
        assert files >= 15, f"only {files} files with module-scope gates found; the scan is broken"
        assert findings == [], "\n".join(findings)
