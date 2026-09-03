#!/usr/bin/env python3
"""Where each counted doc claim gets its TRUE value from.

Split out of ``check_docs_alignment.py`` so the two halves stay separately
readable: this module answers "what is actually true", the other answers
"where is it claimed, and does the claim match". Nothing here reads a doc.

Every function raises :class:`AuthorityError` rather than returning a
plausible default. That is the whole contract: an authority that cannot be
computed must stop the gate, because an empty finding list from a checker
that never ran reads exactly like a clean bill of health.
"""

from __future__ import annotations

import ast
import fnmatch
import io
import re
import subprocess  # nosec B404 - fixed argv, no shell, repo-local commands only
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

from scripts import count_mcp_tools as cmt

# The checkpoint the published weights were trained from. HF revision
# ``v4.1.1`` is a full fine-tune of the ``v4.0.0`` corpus; both tags carry the
# same tool surface, so either resolves to the same authority value. Bump this
# ONLY when a retrain actually ships.
TRAINED_REVISION = "v4.1.1"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class AuthorityError(RuntimeError):
    """An authority could not be computed.

    Raised, never swallowed: a checker that cannot reach its authority has not
    passed, it has *not run*, and reporting "no drift" from an empty finding
    list is how a dead verifier reads as a clean bill of health.
    """


def _load_toml(path: Path) -> dict:
    """Parse a TOML file on every interpreter ``requires-python`` admits.

    ``tomllib`` entered the stdlib in 3.11 and ``requires-python`` is
    ``>=3.10``, so the bare ``import tomllib`` this replaced turned every
    pyproject-reading authority into an ``AuthorityError`` on both 3.10 rows
    of the matrix -- 21 red tests there, and none anywhere else, because the
    machine the code was written on runs 3.12.

    The fallback costs no new dependency: ``tomli`` is already declared in the
    ``test`` extra as ``tomli>=2.0,<3.0 ; python_version < '3.11'``, which is
    what every matrix row installs (verified in the 3.10 job's own pip log:
    ``tomli-2.4.1``), and ``tests/_toml_compat.py`` already resolves the same
    pair for the test side. On 3.11+ nothing changes -- ``tomllib`` wins and
    ``tomli`` is never even imported.

    Neither parser importable still raises :class:`AuthorityError` naming both,
    rather than returning ``{}``: an authority that could not be computed must
    stop the gate, and an empty parse would silently report zero dependencies
    and no classifiers as if they were the truth.
    """
    try:
        import tomllib as _toml  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        try:
            import tomli as _toml  # type: ignore[no-redef]  # noqa: PLC0415
        except ModuleNotFoundError as exc:  # pragma: no cover - no parser at all
            raise AuthorityError(
                f"no TOML parser to read {path}: tomllib is stdlib only on 3.11+ and tomli is not "
                f"installed ({exc}). Install the 'test' extra, which declares tomli for python<3.11."
            ) from exc
    try:
        with path.open("rb") as fh:
            return dict(_toml.load(fh))
    except (OSError, ValueError) as exc:
        raise AuthorityError(f"could not parse {path}: {exc}") from exc


# --------------------------------------------------------------------------
# Test count -- static, so it is a fact about the TREE and not about the
# machine reading it.
#
# Through 5.0.1 this authority was ``pytest --collect-only`` with the CI
# selector. Collection is a function of the environment: a module whose
# module-level ``pytest.importorskip`` misses its extra is dropped whole, so
# the number moved with what the host had installed. Measured 2026-09-03 on
# one commit: 11,726 on a workstation with every extra, 11,662 on the CI rows
# that install ``[test]`` alone (four Postgres modules dropped), and no single
# number a doc could state was true on both. ``--fix`` wrote the workstation's
# count twice and CI rejected it twice. Reproduced here by hiding ``psycopg``
# from the import system: 11,665 against 11,726 on the same tree.
#
# What the tree contains does not depend on who is looking. The count below is
# ``def test_*`` functions -- at module level or inside a ``Test*`` class,
# nested ``Test*`` classes included -- in every ``tests/**/test_*.py``, which
# is the ``[tool.pytest.ini_options]`` collection rule applied to SOURCE. It
# never imports a test module, so a missing extra cannot change it;
# ``tests/test_docs_alignment.py`` proves that by hiding ``psycopg`` and
# ``sqlite_vec`` from the import system and requiring the same number.
#
# It is a different number from a runner's, and the docs say so: parametrised
# cases are counted once (they are one function), stress-marked tests are
# counted (they are in the tree), ``tests/integration`` is counted (so is
# that). Every doc surface therefore says "test functions", never "tests" --
# a measurement that changes while the claim's wording stays turns a true
# sentence into a false one, and ``check_docs_alignment`` refuses the old
# spelling at suite scale for exactly that reason.
# --------------------------------------------------------------------------

#: pytest's own defaults, used only when ``pyproject.toml`` does not set the
#: key (it does; see ``[tool.pytest.ini_options]``).
_PYTEST_DEFAULTS = {
    "testpaths": ["tests"],
    "python_files": ["test_*.py", "*_test.py"],
    "python_classes": ["Test*"],
    "python_functions": ["test*"],
}


def _pytest_option(root: Path, key: str) -> tuple[str, ...]:
    """A ``[tool.pytest.ini_options]`` list, from pyproject or pytest's default.

    Read from the same file pytest reads, so the counting rule follows the
    collection rule by construction rather than by a copy that can drift.
    A string value (pytest accepts ``python_files = "test_*.py"``) is
    whitespace-split, exactly as pytest splits it.
    """
    options = _load_toml(root / "pyproject.toml").get("tool", {}).get("pytest", {}).get("ini_options", {})
    value = options.get(key, _PYTEST_DEFAULTS[key])
    if isinstance(value, str):
        value = value.split()
    return tuple(str(v) for v in value)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _test_functions_in(body: list[ast.stmt], classes: tuple[str, ...], functions: tuple[str, ...]) -> int:
    count = 0
    for item in body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _matches_any(item.name, functions):
            count += 1
        elif isinstance(item, ast.ClassDef) and _matches_any(item.name, classes):
            count += _test_functions_in(item.body, classes, functions)
    return count


def test_functions_in_source(
    source: str,
    filename: str = "<test module>",
    *,
    classes: tuple[str, ...] = tuple(_PYTEST_DEFAULTS["python_classes"]),
    functions: tuple[str, ...] = ("test_*",),
) -> int:
    """Test functions pytest would collect from *source*, counted without importing it.

    Module-level functions and methods of matching classes, nested matching
    classes included. A ``def`` nested inside a function is not a test and is
    not counted; neither is a method of a class pytest would not collect.
    """
    tree = ast.parse(source, filename=filename)
    return _test_functions_in(tree.body, classes, functions)


def static_test_count(root: Path | None = None) -> int:
    """Test functions the tree contains -- counted from source, never from a run.

    Raises :class:`AuthorityError` when there is nothing to count or a file
    cannot be parsed: a zero from a suite this size means the counting rule
    broke, and a broken authority must exit 2 rather than report every
    four-digit claim stale.
    """
    root = root or _project_root()
    file_patterns = _pytest_option(root, "python_files")
    classes = _pytest_option(root, "python_classes")
    functions = _pytest_option(root, "python_functions")
    files: list[Path] = []
    for testpath in _pytest_option(root, "testpaths"):
        base = root / testpath
        files.extend(p for p in sorted(base.rglob("*.py")) if p.is_file() and _matches_any(p.name, file_patterns))
    if not files:
        raise AuthorityError(f"no test files under {root} match {file_patterns} -- the test-count authority has nothing to count")
    total = 0
    for path in files:
        try:
            total += test_functions_in_source(path.read_text(encoding="utf-8"), str(path), classes=classes, functions=functions)
        except (OSError, SyntaxError) as exc:
            raise AuthorityError(f"could not parse {path}: {exc}") from exc
    if total == 0:
        raise AuthorityError(f"{len(files)} test file(s) under {root} define no test functions -- counting rule broke")
    return total


def live_tool_count() -> int:
    """Distinct MCP tool names the server registers today."""
    return cmt.count_tools()


def _safe_members(tar: tarfile.TarFile, dest: Path) -> Iterator[tarfile.TarInfo]:
    """Yield only members that land inside *dest*, as regular files or dirs.

    Refuses an absolute path, a path that escapes *dest* through ``..`` or a
    symlink, and any member that is not a file or a directory. Raising rather
    than skipping is deliberate: this reads a revision to COUNT something, so
    a surprising archive means the count would be wrong, and a wrong count
    quietly published on a model card is the failure this module exists to
    prevent.
    """
    root = dest.resolve()
    for member in tar:
        if member.issym() or member.islnk():
            raise AuthorityError(f"archive member {member.name!r} is a link; refusing to unpack")
        if not (member.isfile() or member.isdir()):
            raise AuthorityError(f"archive member {member.name!r} is not a file or directory")
        target = Path(member.name)
        if target.is_absolute() or target.drive or target.root:
            raise AuthorityError(f"archive member {member.name!r} is an absolute path")
        resolved = (root / target).resolve()
        if resolved != root and root not in resolved.parents:
            raise AuthorityError(f"archive member {member.name!r} escapes the extraction directory")
        yield member


def trained_tool_count(revision: str = TRAINED_REVISION, root: Path | None = None) -> int:
    """Distinct MCP tool names at the revision the weights were trained from.

    The same AST rule the live counter uses (``cmt._tool_names``), applied to
    blobs read out of git history -- one counting rule, two revisions, so the
    two numbers on the model card are comparable by construction. Writing a
    second counter here is how "84 vs 96" happened in the first place.
    """
    root = root or _project_root()
    paths = ["src/mind_mem/mcp/tools", "src/mind_mem/mcp_server.py"]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            archive = subprocess.run(  # nosec B603
                ["git", "archive", revision, *paths],
                cwd=root,
                capture_output=True,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AuthorityError(f"could not read revision {revision} from git: {exc}") from exc
        if archive.returncode != 0:
            raise AuthorityError(
                f"revision {revision!r} is not in this checkout "
                f"(shallow clone? the CI step needs actions/checkout with fetch-depth: 0): "
                f"{archive.stderr.decode('utf-8', 'replace').strip()}"
            )
        # ``tarfile`` rather than a ``tar`` subprocess: this runs on the
        # Windows matrix rows too, and the stdlib already does the job.
        #
        # Every member is checked before extraction, on EVERY version. The
        # previous form passed ``filter="data"`` on 3.12+ and nothing at all
        # on 3.10/3.11, resting on an argument about the caller: the archive
        # comes from ``git archive`` over this repository's own history, so
        # there is no external input. That argument is true today and is the
        # wrong shape -- it makes the safety a property of who calls this
        # rather than of what it does, and the two supported versions with no
        # protection are the two the argument was quietly covering for.
        # Code scanning called it (py/tarslip) and it was right to.
        try:
            with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r|") as tar:
                tar.extractall(tmp, members=_safe_members(tar, Path(tmp)))  # nosec B202
        except (OSError, tarfile.TarError, ValueError) as exc:
            raise AuthorityError(f"could not unpack revision {revision}: {exc}") from exc
        base = Path(tmp)
        files = sorted((base / "src/mind_mem/mcp/tools").glob("*.py"))
        monolith = base / "src/mind_mem/mcp_server.py"
        if monolith.exists():
            files.append(monolith)
        if not files:
            raise AuthorityError(f"revision {revision} exposed no MCP tool source files")
        names: set[str] = set()
        for path in files:
            names.update(cmt._tool_names(path))
    if not names:
        raise AuthorityError(f"revision {revision} registered no MCP tools -- counting rule broke")
    return len(names)


def client_count(root: Path | None = None) -> int:
    """Number of AI clients ``mm install-all`` wires, from the registry itself."""
    root = root or _project_root()
    src = root / "src"
    added = str(src) not in sys.path
    if added:
        sys.path.insert(0, str(src))
    try:
        from mind_mem.hook_installer import AGENT_REGISTRY  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - defensive
        raise AuthorityError(f"could not import hook_installer.AGENT_REGISTRY: {exc}") from exc
    finally:
        if added:
            sys.path.remove(str(src))
    return len(AGENT_REGISTRY)


def mcp_client_count(root: Path | None = None) -> int:
    """Clients ``mm install-all`` writes a native MCP entry for.

    An ``AgentSpec`` with an empty ``mcp_fmt`` is skipped with
    ``reason="no_mcp_format"``, so the non-empty ones ARE the supported set.
    Three clients (``copilot-cli``, ``grok-build``, ``vibe``) gained MCP
    writers without any of the three docs that enumerate them noticing.
    """
    root = root or _project_root()
    src = root / "src"
    added = str(src) not in sys.path
    if added:
        sys.path.insert(0, str(src))
    try:
        from mind_mem.hook_installer import AGENT_REGISTRY  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - defensive
        raise AuthorityError(f"could not import hook_installer.AGENT_REGISTRY: {exc}") from exc
    finally:
        if added:
            sys.path.remove(str(src))
    return sum(1 for spec in AGENT_REGISTRY.values() if getattr(spec, "mcp_fmt", ""))


def mind_kernel_count(root: Path | None = None) -> int:
    """Number of MIND scoring kernels shipped as ``.mind`` source.

    The README's comparison matrix and docs/migration.md both froze at "16
    MIND kernels" -- the count when that row was written -- while the
    directory grew to 26.
    """
    root = root or _project_root()
    kernels = sorted((root / "mind").glob("*.mind"))
    if not kernels:
        raise AuthorityError(f"no .mind kernels under {root / 'mind'}")
    return len(kernels)


def resource_count(root: Path | None = None) -> int:
    """Number of ``mcp.resource(...)`` registrations, counted from the AST."""
    root = root or _project_root()
    path = root / "src" / "mind_mem" / "mcp" / "resources.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise AuthorityError(f"could not parse {path}: {exc}") from exc
    uris: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resource"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            uris.add(node.args[0].value)
    if not uris:
        raise AuthorityError(f"no mcp.resource(...) registrations found in {path}")
    return len(uris)


def package_version(root: Path | None = None) -> str:
    """``__version__`` from ``src/mind_mem/__init__.py`` -- the version authority."""
    root = root or _project_root()
    path = root / "src" / "mind_mem" / "__init__.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthorityError(f"could not read {path}: {exc}") from exc
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        raise AuthorityError(f"no __version__ assignment in {path}")
    return m.group(1)


def core_dependency_count(root: Path | None = None) -> int:
    """Length of ``[project] dependencies`` -- the "core deps: zero" authority."""
    root = root or _project_root()
    path = root / "pyproject.toml"
    data = _load_toml(path)
    return len(data.get("project", {}).get("dependencies") or [])


# --------------------------------------------------------------------------
# CI shape -- the workflow directory is the authority for what CI runs
# --------------------------------------------------------------------------

# ``.github/workflows/*.yml`` is parsed by hand, not with pyyaml: pyyaml is not
# a dependency of this package (core deps are zero) and the release gates in
# ``tests/test_release_preflight_gates.py`` already hand-roll for the same
# reason. Every parse that does not find the shape it expects raises
# ``AuthorityError`` -- a matrix this cannot read is an authority that did not
# run, not an empty finding list.

_JOB_HEADER = re.compile(r"^  (?P<name>[A-Za-z0-9_-]+):\s*(#.*)?$")
_MATRIX_LIST = re.compile(r"^\s*(?P<key>os|python-version):\s*\[(?P<items>[^\]]*)\]\s*$")
_WORKFLOW_NAME = re.compile(r"^name:\s*(?P<name>.+?)\s*$", re.MULTILINE)


class CIMatrix:
    """The ``test`` job's OS × Python cross-product, read from ``ci.yml``."""

    __slots__ = ("python_versions", "operating_systems")

    def __init__(self, python_versions: tuple[str, ...], operating_systems: tuple[str, ...]) -> None:
        self.python_versions = python_versions
        self.operating_systems = operating_systems

    @property
    def job_count(self) -> int:
        return len(self.python_versions) * len(self.operating_systems)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CIMatrix(python_versions={self.python_versions!r}, operating_systems={self.operating_systems!r})"


def _split_yaml_list(items: str) -> tuple[str, ...]:
    return tuple(part.strip().strip("\"'") for part in items.split(",") if part.strip())


def ci_matrix(root: Path | None = None, job: str = "test") -> CIMatrix:
    """The OS and Python lists of one ``ci.yml`` job's ``strategy.matrix``.

    Scoped to a single job on purpose. ``ci.yml`` holds several jobs that pin
    one Python version and one that fans out; a whole-file scan for
    ``python-version:`` would collect the pins too and report a Python list the
    matrix never ran.
    """
    root = root or _project_root()
    path = root / ".github" / "workflows" / "ci.yml"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuthorityError(f"could not read {path}: {exc}") from exc

    start: int | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        header = _JOB_HEADER.match(line)
        if header is None:
            continue
        if header.group("name") == job:
            start = idx
        elif start is not None:
            end = idx
            break
    if start is None:
        raise AuthorityError(f"{path} has no job named {job!r}")

    found: dict[str, tuple[str, ...]] = {}
    for line in lines[start:end]:
        hit = _MATRIX_LIST.match(line)
        if hit is not None:
            found[hit.group("key")] = _split_yaml_list(hit.group("items"))
    missing = {"os", "python-version"} - found.keys()
    if missing:
        raise AuthorityError(
            f"{path} job {job!r}: no inline {sorted(missing)} matrix list found "
            f"(a block-style list would need a parser change, not a default)"
        )
    if not found["os"] or not found["python-version"]:
        raise AuthorityError(f"{path} job {job!r}: matrix list is empty")
    return CIMatrix(python_versions=found["python-version"], operating_systems=found["os"])


def workflow_inventory(root: Path | None = None) -> dict[str, str]:
    """``{filename: workflow name}`` for every file in ``.github/workflows``.

    The authority behind ``docs/ci-workflows.md``'s table. That table listed a
    "Security Review" workflow that does not exist, gave Benchmark a push/PR
    trigger it lost, and omitted two whole workflows -- drift a table of
    prose cannot notice about itself.
    """
    root = root or _project_root()
    directory = root / ".github" / "workflows"
    if not directory.is_dir():
        raise AuthorityError(f"no workflow directory at {directory}")
    out: dict[str, str] = {}
    for path in sorted(directory.glob("*.yml")) + sorted(directory.glob("*.yaml")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AuthorityError(f"could not read {path}: {exc}") from exc
        match = _WORKFLOW_NAME.search(text)
        if match is None:
            raise AuthorityError(f"{path} has no top-level 'name:' -- cannot be named in a doc table")
        out[path.name] = match.group("name")
    if not out:
        raise AuthorityError(f"{directory} contains no workflow files")
    return out


# --------------------------------------------------------------------------
# v4 feature flags -- ONE counting rule, two revisions (as for MCP tools)
# --------------------------------------------------------------------------

_FLAG_TUPLE = "ALL_V4_FLAGS"
_FLAGS_PATH = "src/mind_mem/v4/feature_flags.py"


def _count_flag_literal(source: str, where: str) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AuthorityError(f"could not parse {where}: {exc}") from exc
    for node in ast.walk(tree):
        target: str | None = None
        if isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = getattr(node.targets[0], "id", None)
        if target != _FLAG_TUPLE:
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List, ast.Set)):
            raise AuthorityError(f"{where}: {_FLAG_TUPLE} is not a literal sequence -- counting rule broke")
        return len(node.value.elts)
    raise AuthorityError(f"{where}: no {_FLAG_TUPLE} assignment found")


def live_flag_count(root: Path | None = None) -> int:
    """Number of v4 feature flags the tree declares today."""
    root = root or _project_root()
    path = root / _FLAGS_PATH
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthorityError(f"could not read {path}: {exc}") from exc
    return _count_flag_literal(source, str(path))


def trained_flag_count(revision: str = TRAINED_REVISION, root: Path | None = None) -> int:
    """Number of v4 feature flags at the revision the weights were trained from.

    The model card advertised a "35-flag inventory" through four releases. It
    matches no revision: ``v4.1.1`` declared 38 and the tree now declares 52.
    Same failure as the 84-vs-96 tool count -- a number asserted rather than
    measured -- so it gets the same treatment: one rule, two revisions.
    """
    root = root or _project_root()
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            ["git", "show", f"{revision}:{_FLAGS_PATH}"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuthorityError(f"could not read {_FLAGS_PATH} at {revision}: {exc}") from exc
    if proc.returncode != 0:
        raise AuthorityError(
            f"revision {revision!r} does not carry {_FLAGS_PATH} in this checkout "
            f"(shallow clone? CI needs actions/checkout with fetch-depth: 0): {proc.stderr.strip()}"
        )
    return _count_flag_literal(proc.stdout, f"{revision}:{_FLAGS_PATH}")


# --------------------------------------------------------------------------
# Eval probe surface -- the harness files are the authority for the scores
# --------------------------------------------------------------------------


def _probe_total(path: Path) -> int:
    """Sum the probe lists the harness in *path* actually benches.

    Derived from the code rather than from a hardcoded list of names: the
    module-level literal sequences that a ``_bench*`` function or ``main``
    references ARE the probe surface, so adding a category to the harness moves
    this number without anyone editing the counter.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise AuthorityError(f"could not parse {path}: {exc}") from exc
    sizes: dict[str, int] = {}
    for node in tree.body:
        target: str | None = None
        if isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = getattr(node.targets[0], "id", None)
        if target and target.isupper() and isinstance(node.value, (ast.List, ast.Tuple)):
            sizes[target] = len(node.value.elts)
    used: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not (node.name.startswith("_bench") or node.name == "main"):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and inner.id in sizes:
                used.add(inner.id)
    if not used:
        raise AuthorityError(f"{path}: no benched probe list found -- counting rule broke")
    return sum(sizes[name] for name in used)


def eval_probe_counts(root: Path | None = None) -> tuple[int, int]:
    """``(main probes, held-out probes)`` for the shipped 4b eval."""
    root = root or _project_root()
    main = _probe_total(root / "train" / "eval_harness.py")
    holdout = _probe_total(root / "train" / "eval_holdout.py")
    return main, holdout


# --------------------------------------------------------------------------
# Per-module facts -- "(N lines, M tests, X% coverage)" in a module doc header
# --------------------------------------------------------------------------


def module_line_count(rel: str, root: Path | None = None) -> int:
    """Lines in one source file -- the authority for a doc's "N lines"."""
    root = root or _project_root()
    path = root / rel
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except OSError as exc:
        raise AuthorityError(f"could not read {path}: {exc}") from exc


def module_test_count(rel: str, root: Path | None = None) -> int:
    """``def test_*`` functions in the test file that covers *rel*.

    Counted from the AST rather than by collecting, so this stays a cheap
    in-process authority: the doc claim is "this module has M tests", and a
    test function is what that means.
    """
    root = root or _project_root()
    path = root / "tests" / f"test_{Path(rel).stem}.py"
    if not path.is_file():
        raise AuthorityError(f"no test file at {path} for the module doc claim about {rel}")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise AuthorityError(f"could not parse {path}: {exc}") from exc
    count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
    if count == 0:
        raise AuthorityError(f"{path} declares no test functions -- counting rule broke")
    return count


# --------------------------------------------------------------------------
# Supported Python -- the most-repeated claim in the docs, and the one with
# the least behind it: nine live surfaces say "Python 3.10+" or
# "Python 3.10-3.14 supported" and none of them was checked against packaging
# metadata. ``requires-python`` is what pip enforces; the classifiers are what
# the index advertises, so both are read.
# --------------------------------------------------------------------------

_REQUIRES_FLOOR = re.compile(r">=\s*(?P<v>\d+\.\d+)")
_CLASSIFIER_VERSION = re.compile(r"^Programming Language :: Python :: (?P<v>\d+\.\d+)$")


def python_support(root: Path | None = None) -> tuple[str, str, str]:
    """``(requires-python floor, lowest classifier, highest classifier)``."""
    root = root or _project_root()
    path = root / "pyproject.toml"
    data = _load_toml(path)
    project = data.get("project", {})
    requires = str(project.get("requires-python", ""))
    floor = _REQUIRES_FLOOR.search(requires)
    if floor is None:
        raise AuthorityError(f"{path}: requires-python {requires!r} has no '>=' floor to hold a doc claim to")
    versions = sorted(
        (m.group("v") for m in (_CLASSIFIER_VERSION.match(c) for c in project.get("classifiers", [])) if m),
        key=lambda v: tuple(int(part) for part in v.split(".")),
    )
    if not versions:
        raise AuthorityError(f"{path}: no 'Programming Language :: Python :: X.Y' classifiers")
    return floor.group("v"), versions[0], versions[-1]


# --------------------------------------------------------------------------
# Storage backends -- what ``mind-mem-init --backend`` actually accepts
# --------------------------------------------------------------------------


def storage_backends(root: Path | None = None) -> tuple[str, ...]:
    """``init_workspace.SUPPORTED_BACKENDS`` -- the backends badge's authority.

    Read from the AST rather than imported, so this stays usable from a bare
    checkout. The badge said "markdown | postgres" while ``encrypted`` has been
    a first-class ``--backend`` choice with its own row in
    ``docs/storage-backends.md``: a badge can undersell a product as easily as
    it can oversell one, and both are the same defect.
    """
    root = root or _project_root()
    path = root / "src" / "mind_mem" / "init_workspace.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise AuthorityError(f"could not parse {path}: {exc}") from exc
    for node in tree.body:
        target: str | None = None
        if isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = getattr(node.targets[0], "id", None)
        if target != "SUPPORTED_BACKENDS":
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            raise AuthorityError(f"{path}: SUPPORTED_BACKENDS is not a literal sequence -- counting rule broke")
        names = tuple(e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str))
        if not names:
            raise AuthorityError(f"{path}: SUPPORTED_BACKENDS holds no string literals")
        return names
    raise AuthorityError(f"{path}: no SUPPORTED_BACKENDS assignment found")
