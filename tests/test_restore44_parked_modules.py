"""The parked modules from the 5.0.0 restore: still here, still whole, still parked.

5.0.0 deleted 44 modules because nothing imported them. ``e4720dc`` restored all
44. Most now have a caller. Six do not, and each of those six carries a recorded
decision in ``scripts/reachability_baseline.txt`` of the form
``module  # waiting: <trigger>``.

That register is prose, and prose is unenforced. Three holes follow from it, and
this file closes all three.

**1. The shipped gate cannot tell WIRED from DELETED.**
``scripts/check_reachable_modules.py`` computes ``fixed = baseline - unreachable``
and prints it as ``NEWLY REACHABLE (wire or delete confirmed)``. A parked module
that is *deleted* leaves ``unreachable`` exactly as a parked module that is
*wired* does, the line reads as progress either way, and ``--check`` exits 0
because only ADDITIONS fail. So the next sweep can delete any of the six and
every gate in CI stays green. That is the 5.0.0 mistake with a green light.
``test_every_parked_module_still_exists`` is the missing half: a baseline entry
whose file has vanished is a deletion, and a deletion of a parked capability is
the one repair that is never correct.

**2. A gutted module is a deleted module.** Keeping the file while emptying its
public API passes an existence check and loses the capability just the same, so
each parked module's own ``__all__`` is checked against the module object.

**3. A trigger nobody evaluates never fires.** Each ``waiting:`` note names a
condition. Here each one becomes a predicate, asserted false, so the day it
turns true CI says so — by name, in the release that turned it. That matters
most for ``tenant_audit``, whose own note says it must be wired in the SAME
release that adds per-tenant identity, because retrofitting audit isolation
after tenant records have interleaved into one chain is the exact mistake audit
chains exist to prevent.

Every negative assertion below is paired with a positive control proving the
method can see the thing it reports absent, and the import scanner is the
shipped gate's own — a second scanner would be a second answer.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import json
import pathlib
import re
import types
from typing import Any, Iterable, Mapping

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "mind_mem"
BASELINE = REPO_ROOT / "scripts" / "reachability_baseline.txt"
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_reachable_modules.py"

#: The four modules this lane owns. ``bootstrap_corpus`` is NOT here: it is
#: wired, through the console script ``mind-mem-bootstrap``, and the tests for
#: that live in ``test_bootstrap_corpus_wiring.py``.
LANE_W_PARKED = ("governance_raft", "memory_mesh", "tenant_audit")

#: Request-handling surfaces. ``tenant_audit``'s trigger is phrased against
#: "the server surface", and these are it.
SERVER_SURFACE_ROOTS = (
    SRC / "http_transport.py",
    SRC / "api",
    SRC / "mcp",
)

#: Config sections that declare peer endpoints able to write governance state.
#: ``memory_mesh`` is the one the docs actually document; the others are listed
#: so a future rename or a federation peer list is seen by the same predicate
#: rather than needing a second one.
PEER_SECTIONS = ("memory_mesh", "federation", "mesh", "peers")


# ---------------------------------------------------------------------------
# Shared machinery — the shipped gate's scanner, not a second copy of it
# ---------------------------------------------------------------------------


def _load_gate() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_lane_w_reach_gate", GATE_SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {GATE_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def _parse_baseline(text: str) -> dict[str, str]:
    """``{module: trigger}``; a module with no ``waiting:`` note maps to ``""``."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        module, _, note = line.partition("#")
        module = module.strip()
        if not module:
            continue
        note = note.strip()
        out[module] = note[len("waiting:") :].strip() if note.lower().startswith("waiting:") else ""
    return out


def _baseline() -> dict[str, str]:
    return _parse_baseline(BASELINE.read_text(encoding="utf-8"))


def _rel(path: pathlib.Path) -> str:
    """Repo-relative when possible; absolute otherwise (tmp_path fixtures)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _module_file(dotted: str) -> pathlib.Path:
    """Where a dotted ``mind_mem`` submodule's source must live."""
    return SRC.joinpath(*dotted.split(".")).with_suffix(".py")


@functools.lru_cache(maxsize=None)
def _importers_of(target: str, *, root: pathlib.Path = SRC) -> frozenset[str]:
    """Modules under *root* that import ``mind_mem.<target>``.

    Uses the shipped gate's own AST helpers, so this answers the same question
    the reachability gate answers, on the same evidence.
    """
    found: set[str] = set()
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        me = ".".join(path.relative_to(SRC).with_suffix("").parts)
        if me == target:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for raw in GATE._imported_names(tree):
            if GATE._resolve(me, raw) == target:
                found.add(me)
                break
    return frozenset(found)


def _missing_public_api(module: Any) -> list[str]:
    """Names in ``__all__`` that the module object does not actually provide."""
    declared = getattr(module, "__all__", None)
    if not declared:
        return ["__all__"]
    return [name for name in declared if not hasattr(module, name)]


# ---------------------------------------------------------------------------
# 1. Deletion guard — what the shipped gate structurally cannot see
# ---------------------------------------------------------------------------


def test_the_baseline_register_is_readable_and_populated() -> None:
    """Anti-vacuity control for every assertion that iterates the register.

    An empty or unparseable baseline would make each ``for module in ...`` loop
    below iterate nothing and pass, which is precisely how a deletion sweep
    would look if it also emptied the register.
    """
    assert BASELINE.is_file(), f"the reachability register is gone: {BASELINE}"
    entries = _baseline()
    assert len(entries) >= len(LANE_W_PARKED), f"register holds {len(entries)} entries: {sorted(entries)}"
    for module in LANE_W_PARKED:
        assert module in entries, f"{module} left the register without being wired or explained"


@pytest.mark.parametrize("module", sorted(_baseline()))
def test_every_parked_module_still_exists(module: str) -> None:
    """A parked module whose file vanished was DELETED, not wired.

    The shipped gate reports both as ``NEWLY REACHABLE`` and exits 0. This is
    the assertion that separates them.
    """
    path = _module_file(module)
    assert path.is_file(), (
        f"{module} is in the reachability register but {path.relative_to(REPO_ROOT)} does not exist. "
        "A parked module leaves the register by being WIRED, never by being deleted — see the 5.0.0 "
        "incident, where 44 modules were removed on the evidence that nothing imported them."
    )


@pytest.mark.parametrize("module", sorted(_baseline()))
def test_every_parked_module_still_imports(module: str) -> None:
    """Present on disk is not enough; it must still load."""
    importlib.import_module(f"mind_mem.{module}")


def test_the_existence_check_can_see_an_absence() -> None:
    """Positive control: the method used above does report a missing file."""
    assert not _module_file("no_such_parked_module_zqx").is_file()
    assert not _module_file("storage.no_such_parked_module_zqx").is_file()
    # ...and it does find one that is really there, so it is not answering
    # "False" to everything.
    assert _module_file("memory_mesh").is_file()
    assert _module_file("storage.sharded_pg").is_file()


@pytest.mark.parametrize("module", sorted(_baseline()))
def test_every_parked_module_records_its_trigger(module: str) -> None:
    """A park with no recorded condition is an open question, not a decision."""
    trigger = _baseline()[module]
    assert trigger, f"{module} is parked with no `# waiting:` trigger — nobody can ever decide to un-park it"


def test_the_trigger_parser_can_see_a_missing_note() -> None:
    """Positive control for the parser used above."""
    parsed = _parse_baseline("with_note  # waiting: something happens\nbare_module\nother  # just a comment\n")
    assert parsed["with_note"] == "something happens"
    assert parsed["bare_module"] == ""
    assert parsed["other"] == ""


@pytest.mark.parametrize("module", LANE_W_PARKED)
def test_a_parked_module_keeps_its_whole_public_api(module: str) -> None:
    """Gutting a parked module loses the capability as surely as deleting it."""
    loaded = importlib.import_module(f"mind_mem.{module}")
    assert _missing_public_api(loaded) == []


def test_the_public_api_check_can_see_a_gutted_module() -> None:
    """Positive control: the checker reports names that are not there."""
    gutted = types.SimpleNamespace(__all__=["present", "gone"], present=1)
    assert _missing_public_api(gutted) == ["gone"]
    assert _missing_public_api(types.SimpleNamespace()) == ["__all__"]


# ---------------------------------------------------------------------------
# 2. Deletion is not free elsewhere — the integrity manifest names two of them
# ---------------------------------------------------------------------------


def _critical_modules() -> tuple[str, ...]:
    from mind_mem import protection

    return protection._CRITICAL_MODULES


def test_every_module_the_integrity_manifest_names_still_exists() -> None:
    """``protection._CRITICAL_MODULES`` names files, not imports.

    Two parked modules are in it (``tenant_audit.py``, ``governance_raft.py``),
    so deleting either leaves strict-integrity mode naming a file that is not
    there — a second, independent cost the import graph cannot see. The same
    dangling-reference shape was found for ``storage/sharded_pg.py`` during the
    restore and is checked here rather than left to the next audit.
    """
    missing = [name for name in _critical_modules() if not (SRC / name).is_file()]
    assert missing == [], f"protection._CRITICAL_MODULES names files that do not exist: {missing}"


def test_the_integrity_manifest_still_names_the_parked_modules() -> None:
    named = set(_critical_modules())
    assert {"tenant_audit.py", "governance_raft.py"} <= named


def test_the_manifest_check_can_see_a_missing_file() -> None:
    """Positive control for the check above."""
    probe = ("tenant_audit.py", "no_such_critical_module_zqx.py")
    missing = [name for name in probe if not (SRC / name).is_file()]
    assert missing == ["no_such_critical_module_zqx.py"]


# ---------------------------------------------------------------------------
# 3. bootstrap_corpus — the blind spot an import grep hides in
# ---------------------------------------------------------------------------


def test_bootstrap_corpus_is_reachable_only_through_its_console_script() -> None:
    """It has no importer and it is wired anyway.

    ``[project.scripts] mind-mem-bootstrap = "mind_mem.bootstrap_corpus:main"``
    is the whole call path. An import grep — the instrument that produced the
    5.0.0 verdict — sees nothing here and would report the module dead. Naming
    the route in a test is what stops that conclusion being drawn twice.
    """
    entry_points = GATE._entry_point_modules()
    assert "bootstrap_corpus" in entry_points, "the mind-mem-bootstrap console script is gone"
    assert _importers_of("bootstrap_corpus") == frozenset(), (
        "bootstrap_corpus gained an importer; that is fine, but this test documented the console "
        "script as its ONLY route — update the claim rather than the assertion"
    )
    assert "bootstrap_corpus" not in _baseline(), "a wired module must not sit in the parked register"


@pytest.mark.parametrize(
    ("source", "importer"),
    [
        ("import mind_mem.memory_mesh", "caller"),
        ("from mind_mem.memory_mesh import MemoryMesh", "caller"),
        ("from mind_mem import memory_mesh", "caller"),
        ("from . import memory_mesh", "caller"),
        ("from . import memory_mesh as _mesh", "caller"),
        ("from .memory_mesh import MemoryMesh", "caller"),
        ("from ..memory_mesh import MemoryMesh", "sub.caller"),
    ],
)
def test_the_import_scanner_resolves_every_form_a_caller_can_use(source: str, importer: str) -> None:
    """Each of these six spellings is a real caller. Any one it cannot see is
    a module it will report dead.

    The relative forms are the ones that were missed: ``_imported_names``
    recorded only ``node.module`` for a relative import, which is ``None`` for
    ``from . import memory_mesh`` — so the module being imported appeared
    nowhere in the scan. The absolute twin of that bug (``from mind_mem import
    usage_meter``) is why 5.0.0 deleted two modules out from under a console
    script; this is the same bug one dot to the left.
    """
    resolved = {GATE._resolve(importer, raw) for raw in GATE._imported_names(ast.parse(source))}
    assert "memory_mesh" in resolved, f"{source!r} from {importer!r} resolved to {sorted(resolved)}"


def test_the_import_scanner_does_not_resolve_what_is_not_there() -> None:
    """Negative control for the parametrisation above.

    Without it, a scanner that returned every module name for every input
    would pass all seven rows.
    """
    resolved = {GATE._resolve("caller", raw) for raw in GATE._imported_names(ast.parse("import os\nfrom pathlib import Path\n"))}
    assert "memory_mesh" not in resolved
    resolved_sibling = {GATE._resolve("caller", raw) for raw in GATE._imported_names(ast.parse("from . import tenant_audit"))}
    assert resolved_sibling == {"", "tenant_audit"}, sorted(resolved_sibling)


def test_the_entry_point_parser_and_import_scanner_both_work() -> None:
    """Positive control for both instruments used above.

    Without this, ``"bootstrap_corpus" in entry_points`` and ``_importers_of(...)
    == set()`` would both pass on a parser that returned nothing at all.
    """
    entry_points = GATE._entry_point_modules()
    assert {"mm_cli", "init_workspace", "recall"} <= entry_points, sorted(entry_points)
    # A module with importers really does report them.
    assert _importers_of("audit_chain"), "the import scanner found no importer of audit_chain — it is broken"
    assert _importers_of("no_such_module_zqx") == frozenset()


# ---------------------------------------------------------------------------
# 4. Trigger predicates — one per parked module this lane owns
# ---------------------------------------------------------------------------

# --- governance_raft: "a deployment config with >1 governance write endpoint"


def governance_write_endpoints(config: Mapping[str, Any]) -> list[str]:
    """Endpoints a deployment config declares as able to write governance state.

    The local node is always one. Every peer declared under a peer-sync section
    is another, because a peer that syncs the ``governance`` scope is by
    definition a second writer — the condition ``governance_raft`` waits on.
    """
    out = ["local"]
    for section in PEER_SECTIONS:
        block = config.get(section)
        peers: Iterable[Any]
        if isinstance(block, Mapping):
            peers = block.get("peers") or ()
        elif isinstance(block, list):
            peers = block
        else:
            continue
        for peer in peers:
            if isinstance(peer, Mapping):
                out.append(str(peer.get("endpoint") or peer.get("peer_id") or peer))
            else:
                out.append(str(peer))
    return out


def _shipped_configs() -> list[tuple[str, dict]]:
    """Every example deployment config the repo ships or documents."""
    found: list[tuple[str, dict]] = []
    example = REPO_ROOT / "mind-mem.example.json"
    if example.is_file():
        try:
            data = json.loads(example.read_text(encoding="utf-8"))
        except ValueError:  # pragma: no cover - a broken example is its own bug
            data = None
        if isinstance(data, dict):
            found.append((example.name, data))
    fence = re.compile(r"```json\n(.*?)```", re.S)
    for doc in sorted((REPO_ROOT / "docs").glob("*.md")):
        for index, block in enumerate(fence.findall(doc.read_text(encoding="utf-8"))):
            try:
                data = json.loads(block)
            except ValueError:
                continue
            if isinstance(data, dict):
                found.append((f"{doc.name}#json{index}", data))
    return found


def test_no_shipped_config_declares_a_second_governance_write_endpoint() -> None:
    """``governance_raft`` stays parked while there is one writer.

    Its note: *waiting: a SECOND writer to the governance store exists ...
    Checkable: a deployment config with >1 governance write endpoint.*
    """
    offenders: dict[str, list[str]] = {}
    for label, config in _shipped_configs():
        endpoints = governance_write_endpoints(config)
        if len(endpoints) > 1:
            offenders[label] = endpoints
    assert offenders == {}, (
        f"a shipped config now declares more than one governance write endpoint: {offenders}. "
        "That is governance_raft's recorded trigger — wire it (consensus before side effect) or "
        "record why last-write-wins is still sufficient."
    )


def test_the_shipped_config_corpus_is_real_and_not_empty() -> None:
    """Anti-vacuity control for the assertion above.

    "No config declares two endpoints" is worthless if no config was read, and
    nearly worthless if what was read is incidental JSON rather than deployment
    configuration. So the harvest must be non-empty AND must contain something
    recognisable as a mind-mem config.

    Deliberately NOT asserted here: that some harvested config carries a peer
    section. It does today — ``docs/setup.md`` documents a ``memory_mesh``
    block — but that block has no consumer in ``src/`` and removing it is one
    of the two correct repairs, so pinning it would make the right fix fail
    this test. The predicate's ability to SEE a peer section is proved on
    synthetic input instead, in the test below.
    """
    configs = _shipped_configs()
    assert configs, "no shipped config was harvested — the predicate above ran on nothing"
    recognisable = [label for label, cfg in configs if {"recall", "workspace_path", "governance_mode"} & set(cfg)]
    assert recognisable, f"nothing harvested looks like a mind-mem config; harvested: {[c[0] for c in configs]}"


def test_the_endpoint_predicate_counts_a_second_writer() -> None:
    """Positive control: it fires on a config that really declares two."""
    assert governance_write_endpoints({}) == ["local"]
    assert governance_write_endpoints({"memory_mesh": {"enabled": False, "peers": []}}) == ["local"]
    two = governance_write_endpoints(
        {
            "memory_mesh": {
                "enabled": True,
                "peers": [{"peer_id": "b", "endpoint": "http://b:8765", "scopes": ["governance"]}],
            }
        }
    )
    assert len(two) == 2 and "http://b:8765" in two
    assert len(governance_write_endpoints({"federation": {"peers": ["http://c:8765"]}})) == 2


# --- memory_mesh: "never a second parallel sync surface"


def test_memory_mesh_is_not_a_second_parallel_sync_surface() -> None:
    """Its note ends: *fold in or wire the delta, never a second parallel sync
    surface.* An importer of ``memory_mesh`` inside the package is that second
    surface appearing, so the overlap review against ``v4/federation`` is owed
    before it lands.
    """
    importers = _importers_of("memory_mesh")
    assert importers == frozenset(), (
        f"memory_mesh is now imported by {sorted(importers)}. Its recorded decision requires an overlap "
        "review against v4/federation.py FIRST — fold the delta into federation, or record why two "
        "peer-sync vocabularies are correct."
    )


def test_the_surface_memory_mesh_was_parked_in_favour_of_is_still_wired() -> None:
    """The park's justification is that ``v4/federation`` covers the ground.

    If federation ever loses its callers the justification is gone and the
    decision has to be retaken — so the claim is asserted, not assumed. This is
    also the positive control for the scanner used just above: it demonstrably
    finds importers when there are some.
    """
    assert _importers_of("v4.federation"), "v4/federation.py has no importer — memory_mesh's park no longer has a basis"


# --- tenant_audit: "the server surface gains authenticated per-tenant identity"


#: Markers for a per-tenant identity arriving on a request. A constant
#: ``tenant_id`` in a helper is not one; a header, claim or request field is.
_TENANT_IDENTITY = re.compile(
    r"X-MindMem-Tenant|tenant_id\s*=\s*(?:self\.)?(?:headers|request|claims|token|ctx|context)|"
    r"headers\.get\(\s*['\"][^'\"]*[Tt]enant|['\"]tenant['\"]\s*:\s*(?:headers|request|claims)",
)


def tenant_identity_markers(roots: Iterable[pathlib.Path]) -> list[str]:
    """``path:line`` for every per-tenant identity marker under *roots*."""
    hits: list[str] = []
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            if "__pycache__" in path.parts or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if _TENANT_IDENTITY.search(line):
                    hits.append(f"{_rel(path)}:{number}")
    return hits


def test_no_server_surface_carries_per_tenant_identity_yet() -> None:
    """The moment one does, ``tenant_audit`` must be wired in the SAME release.

    Its note says so, and gives the reason: retrofitting audit isolation after
    tenant records have interleaved into a single chain is unfixable after the
    fact — the chain is append-only, so the interleaving is permanent.
    """
    hits = tenant_identity_markers(SERVER_SURFACE_ROOTS)
    assert hits == [], (
        f"a server surface now resolves a per-tenant identity at {hits}. tenant_audit.get_chain() must be "
        "wired into the audit path in THIS release: once records from two tenants land in one chain, "
        "splitting them later is not possible."
    )


def test_the_tenant_identity_detector_fires_on_a_server_surface(tmp_path: pathlib.Path) -> None:
    """Positive control: the scan really can see the marker it reports absent.

    The fixture mirrors the shape of the real surfaces — a package directory
    with a request handler in it — so the negative assertion above is a
    measurement, not an artefact of a regex that matches nothing.
    """
    surface = tmp_path / "api"
    surface.mkdir()
    (surface / "__init__.py").write_text("", encoding="utf-8")
    (surface / "rest.py").write_text(
        "def handle(self):\n    tenant_id = self.headers.get('X-MindMem-Tenant', '')\n    return tenant_id\n",
        encoding="utf-8",
    )
    hits = tenant_identity_markers([surface])
    assert len(hits) == 1 and hits[0].endswith("rest.py:2"), hits
    # A file with no marker yields nothing, so the detector is not matching
    # every line it reads.
    quiet = tmp_path / "quiet.py"
    quiet.write_text("def handle(self):\n    return self.headers.get('X-MindMem-Token', '')\n", encoding="utf-8")
    assert tenant_identity_markers([quiet]) == []


def test_the_server_surfaces_the_detector_scans_are_real() -> None:
    """Anti-vacuity: a typo'd path would make the scan above read nothing."""
    for root in SERVER_SURFACE_ROOTS:
        assert root.exists(), f"server surface {root} does not exist — the tenant scan is reading nothing"
    scanned = [p for root in SERVER_SURFACE_ROOTS for p in ([root] if root.is_file() else root.rglob("*.py"))]
    assert len(scanned) > 5, f"only {len(scanned)} server-surface files scanned"
