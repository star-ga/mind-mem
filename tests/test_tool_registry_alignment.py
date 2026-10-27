"""The MCP tool registry, its re-exports, and the docs that describe it agree.

Each test here pins one place the surface used to drift from the code:

* ``mind_mem.mcp_server`` re-exported 70 of the 107 registered tools, and its
  ``recall`` was the six-parameter engine function rather than the 17-parameter
  wire tool a client actually calls.
* ``mind_mem.mcp.tools`` described "the 57-tool MCP surface".
* ``docs/api-reference.md`` named 48 of the 107 tools, listed one that is not
  an MCP tool at all (``baseline_snapshot``), and started the server with a
  ``--workspace`` flag the server's parser rejects.
* ``docs/mind-mem-4b-setup.md`` said the live server exposes 102.
* ``abstention_classifier`` claimed an MCP exposure no MCP module provides.

Every negative assertion is paired with a positive control so a check that
silently matched nothing cannot pass.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

from scripts import check_docs_alignment as cda
from scripts import check_tool_surface as cts

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mind_mem"
API_REF = ROOT / "docs" / "api-reference.md"

_HAS_FASTMCP = importlib.util.find_spec("fastmcp") is not None
needs_fastmcp = pytest.mark.skipif(not _HAS_FASTMCP, reason="fastmcp not installed")


def _live_tools() -> list:
    from mind_mem.mcp.server import mcp

    return list(asyncio.run(mcp.list_tools()))


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


@needs_fastmcp
class TestMcpServerReexports:
    def test_every_registered_tool_is_reexported_as_the_registered_function(self) -> None:
        import mind_mem.mcp_server as shim

        tools = _live_tools()
        # Positive control: the registry is the full surface, not an empty list.
        assert len(tools) >= 100
        missing = sorted(t.name for t in tools if not hasattr(shim, t.name))
        assert missing == [], f"mcp_server does not re-export: {missing}"
        different = sorted(t.name for t in tools if getattr(shim, t.name) is not t.fn)
        assert different == [], f"mcp_server re-exports a different function than the registry serves: {different}"

    def test_recall_has_the_wire_signature(self) -> None:
        import mind_mem.mcp_server as shim
        from mind_mem.mcp.tools import public

        assert inspect.signature(shim.recall) == inspect.signature(public.recall)
        assert "mode" in inspect.signature(shim.recall).parameters

    def test_registry_count_matches_the_counting_rule(self) -> None:
        from scripts import count_mcp_tools as cmt

        assert len(_live_tools()) == cmt.count_tools()


# ---------------------------------------------------------------------------
# Docstrings that state the surface size
# ---------------------------------------------------------------------------

_N_TOOL = re.compile(r"\b(\d{2,3})-tool\b")


def test_tools_package_docstring_states_no_stale_count() -> None:
    from scripts import count_mcp_tools as cmt

    doc = ast.get_docstring(ast.parse((SRC / "mcp" / "tools" / "__init__.py").read_text(encoding="utf-8"))) or ""
    assert doc, "mind_mem.mcp.tools has lost its docstring"
    live = cmt.count_tools()
    stale = [n for n in _N_TOOL.findall(doc) if int(n) != live]
    assert stale == [], f"mind_mem.mcp.tools docstring claims a {stale}-tool surface; live is {live}"


def test_abstention_classifier_claims_no_mcp_exposure_unless_mcp_imports_it() -> None:
    doc = ast.get_docstring(ast.parse((SRC / "abstention_classifier.py").read_text(encoding="utf-8"))) or ""
    assert "abstention" in doc.lower()  # positive control: the right docstring
    mcp_text = "\n".join(p.read_text(encoding="utf-8") for p in (SRC / "mcp").rglob("*.py"))
    mcp_reaches_it = bool(re.search(r"abstention_classifier|check_abstention", mcp_text))
    claims = re.search(r"exposed for (?:the )?production MCP|MCP recall path", doc)
    assert mcp_reaches_it or not claims, f"docstring claims MCP exposure but no module under mcp/ reaches it: {claims.group(0)!r}"


# ---------------------------------------------------------------------------
# docs/api-reference.md
# ---------------------------------------------------------------------------


class TestApiReferenceDrift:
    def test_detects_a_ghost_and_a_missing_name(self) -> None:
        """Positive control for the checker itself."""
        text = "## MCP Server (2 tools)\n\n| Tool | Description |\n|---|---|\n| `recall` | x |\n| `nope_tool` | y |\n"
        drift = cts.api_reference_drift(text, {"recall", "scan"})
        assert drift["ghosts"] == ["nope_tool"]
        assert drift["missing"] == ["scan"]

    def test_names_outside_the_mcp_section_are_not_tools(self) -> None:
        text = "## Python API\n\n| Fn | x |\n|---|---|\n| `freeze_baseline` | x |\n\n## MCP Server\n\n| `recall` | x |\n"
        assert cts.api_reference_drift(text, {"recall"}) == {"ghosts": [], "missing": []}

    def test_real_reference_has_no_ghost_and_no_missing_tool(self) -> None:
        real = set(cts.registered_tools())
        assert len(real) >= 100
        drift = cts.api_reference_drift(API_REF.read_text(encoding="utf-8"), real)
        assert drift == {"ghosts": [], "missing": []}


@needs_fastmcp
class TestApiReferenceIndex:
    def test_generated_index_is_current(self) -> None:
        text = API_REF.read_text(encoding="utf-8")
        current = cts.extract_api_index(text)
        assert current is not None, "docs/api-reference.md has no generated tool index"
        assert current == cts.render_api_index(cts.live_registry()), (
            "the generated tool index is stale; run `python3 scripts/check_tool_surface.py --write-api-index`"
        )

    def test_index_lists_every_live_tool_once(self) -> None:
        rows = cts.live_registry()
        rendered = cts.render_api_index(rows)
        names = re.findall(r"^\| `([a-z_][a-z0-9_]*)` \|", rendered, re.M)
        assert sorted(names) == sorted(r["name"] for r in rows)
        assert len(names) == len(set(names)) == len(_live_tools())

    def test_index_scope_follows_the_acl(self) -> None:
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS

        rows = {r["name"]: r for r in cts.live_registry()}
        assert any(r["scope"] == "admin" for r in rows.values())  # positive control
        for name, row in rows.items():
            assert row["scope"] == ("admin" if name in ADMIN_TOOLS else "user"), name

    def test_hand_table_parameters_are_real_parameters(self) -> None:
        rows = cts.live_registry()
        text = API_REF.read_text(encoding="utf-8")
        assert cts.hand_table_param_drift(text, rows) == []
        # Positive control: the pre-fix add_block_edge row is reported.
        bad = "## MCP Server\n\n| `add_block_edge` | x | `source_block_id`, `kind` (`cites`/`refines`) |\n"
        assert cts.hand_table_param_drift(bad, rows) == ["add_block_edge: `source_block_id`"]

    def test_check_doc_names_fails_on_a_stale_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: delete one row from a copy of the reference and the gate must go red."""
        text = API_REF.read_text(encoding="utf-8")
        index = cts.extract_api_index(text)
        assert index is not None
        mutated_index = re.sub(r"^\| `recall` \|.*\n", "", index, count=1, flags=re.M)
        assert mutated_index != index
        mutated = text.replace(index, mutated_index)
        doc = tmp_path / "api-reference.md"
        doc.write_text(mutated, encoding="utf-8")
        monkeypatch.setattr(cts, "API_REF", doc)
        assert cts.main(["--check-doc-names"]) == 1


# ---------------------------------------------------------------------------
# Server flags named in the docs exist
# ---------------------------------------------------------------------------

_SERVER_CMD = re.compile(r"(?:mcp_server\.py|mind-mem-mcp|-m\s+mind_mem\.mcp_server)((?:\s+--?[A-Za-z][\w-]*(?:[ =][^\s`\-][^\s`]*)?)+)")


def _server_flags() -> set[str]:
    text = (SRC / "mcp" / "server.py").read_text(encoding="utf-8")
    return set(re.findall(r"add_argument\(\s*\"(--[a-z][a-z-]*)\"", text)) | {"--help", "-h"}


def test_documented_server_flags_exist() -> None:
    flags = _server_flags()
    assert {"--transport", "--port"} <= flags  # positive control
    bad: list[str] = []
    seen = 0
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in _SERVER_CMD.finditer(line):
                for flag in re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", m.group(1)):
                    seen += 1
                    if flag not in flags:
                        bad.append(f"{path.relative_to(ROOT)}:{lineno}: {flag}")
    assert seen > 0  # the scan found documented invocations at all
    assert bad == [], f"docs pass flags the MCP server does not accept: {bad}"


# ---------------------------------------------------------------------------
# docs/mind-mem-4b-setup.md: "the live server now exposes **102**"
# ---------------------------------------------------------------------------


def _auth() -> cda.Authorities:
    return cda.Authorities(
        tests=9707,
        live_tools=107,
        trained_tools=83,
        clients=19,
        mcp_clients=11,
        resources=8,
        mind_kernels=26,
        version="5.0.1",
        core_deps=0,
        live_flags=52,
        trained_flags=38,
        eval_main_probes=111,
        eval_holdout_probes=22,
        ci_python_versions=("3.10", "3.11", "3.12", "3.13", "3.14"),
        ci_operating_systems=("ubuntu-latest", "macos-latest", "windows-latest"),
        workflows=(("ci.yml", "CI"), ("release.yml", "Release")),
        python_floor="3.10",
        python_classifier_min="3.10",
        python_classifier_max="3.14",
        backends=("markdown", "postgres", "encrypted"),
    )


def _scan(line: str) -> list:
    return cda.scan_line("docs/example.md", 1, line, _auth())


def test_bare_server_exposes_count_is_a_gated_claim() -> None:
    findings = _scan("> The live server now exposes **102**, so the model does not know")
    assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tools", "102", "107")]
    assert _scan("> The live server now exposes **107**, so the model does not know") == []
    # The spelling the gate already covered is not reported twice.
    assert len(_scan("The current server exposes 102 MCP tools.")) == 1
    # Unrelated "exposes N" prose is not a tool claim.
    assert _scan("The HTTP adapter exposes 4 routes.") == []


# ---------------------------------------------------------------------------
# Repo paths named by the tool docs exist
# ---------------------------------------------------------------------------

_REPO_PATH = re.compile(r"`((?:src|tests|scripts|train|docs|benchmarks)/[\w./-]+\.\w+)`")


@pytest.mark.parametrize("rel", ["docs/api-reference.md", "docs/mind-mem-4b-setup.md", "docs/troubleshooting.md"])
def test_tool_docs_name_only_existing_repo_paths(rel: str) -> None:
    text = (ROOT / rel).read_text(encoding="utf-8")
    missing = sorted({p for p in _REPO_PATH.findall(text) if not (ROOT / p).exists()})
    assert missing == [], f"{rel} names repo paths that do not exist: {missing}"


def test_roadmap_points_only_at_acceptance_files_that_exist() -> None:
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "mind-mem-4b" in text  # positive control: the retrain item is still there
    named = sorted({p for p in _REPO_PATH.findall(text) if p.startswith("train/")})
    missing = [p for p in named if not (ROOT / p).exists()]
    assert missing == [], f"ROADMAP.md names training files absent from this repo: {missing}"
