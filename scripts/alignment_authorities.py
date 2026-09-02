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
import io
import re
import subprocess  # nosec B404 - fixed argv, no shell, repo-local commands only
import sys
import tarfile
import tempfile
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


def collect_test_count(root: Path | None = None, python: str | None = None) -> int:
    """Number of tests CI collects, using CI's exact selector.

    ``ci.yml`` runs ``python3 -m pytest tests/ --ignore=tests/integration
    ... -m "not stress"``; the badge claims that number, so the badge's
    authority has to be that selector and nothing looser. Collection is
    environment-sensitive (a missing optional extra can drop a module), which
    is why the CI step pins ubuntu + 3.12 and why a failure here is loud
    rather than defaulted.
    """
    root = root or _project_root()
    exe = python or sys.executable
    cmd = [
        exe,
        "-m",
        "pytest",
        "tests/",
        "--ignore=tests/integration",
        "--collect-only",
        "-q",
        "-m",
        "not stress",
    ]
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=1800)  # nosec B603
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuthorityError(f"could not run pytest collection: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise AuthorityError(f"pytest collection exited {proc.returncode}:\n{tail}")
    return parse_collected(proc.stdout)


# "9701/10034 tests collected (333 deselected) in 13.19s" and the simpler
# "9701 tests collected in 13.19s" that pytest prints with no deselection.
_COLLECTED_RE = re.compile(r"^(\d+)(?:/\d+)?\s+tests?\s+collected\b", re.MULTILINE)


def parse_collected(stdout: str) -> int:
    """Pull the selected-test count out of ``pytest --collect-only -q`` output."""
    matches = _COLLECTED_RE.findall(stdout)
    if not matches:
        tail = "\n".join(stdout.splitlines()[-15:])
        raise AuthorityError(f"no 'N tests collected' line in pytest output:\n{tail}")
    return int(matches[-1])


def live_tool_count() -> int:
    """Distinct MCP tool names the server registers today."""
    return cmt.count_tools()


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
        # ``filter=`` landed in 3.12 and this package supports 3.10+, so it is
        # passed only where it exists. The archive is produced by ``git
        # archive`` from this repository's own history -- no external input.
        try:
            with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r|") as tar:
                if sys.version_info >= (3, 12):
                    tar.extractall(tmp, filter="data")  # nosec B202
                else:
                    tar.extractall(tmp)  # nosec B202
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
    try:
        import tomllib  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - py<3.11 only
        raise AuthorityError(f"tomllib unavailable: {exc}") from exc
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, ValueError) as exc:
        raise AuthorityError(f"could not parse {path}: {exc}") from exc
    return len(data.get("project", {}).get("dependencies") or [])
