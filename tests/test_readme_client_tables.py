# Copyright 2026 STARGA, Inc.
"""The client tables in the README and the client docs must name exactly the clients the installer knows.

Authority: ``mind_mem.hook_installer.AGENT_REGISTRY`` (the same authority
``scripts/check_docs_alignment.py`` uses for the client counts). The counts
were gated, the tables were not: the README listed Claude Desktop (not in the
registry), OpenCode under ``./install.sh`` (which does not install it), and
omitted OpenCode and GitHub Copilot CLI from the vendor table. These tests
compare the set of ``mm install`` ids in each table with the registry, so a
client added to or dropped from the registry without a doc edit fails here.

Vendors are gated too. Nothing checked them, which is how NemoClaw (NVIDIA)
and NanoClaw (an independent project) were published as "OpenClaw variant".
``VENDORS`` below is the single place a vendor is stated; every vendor table
must agree with it, and it must cover exactly the registry.

Each negative assertion has a positive control: the same parser is run on a
document with one row removed, or one vendor changed, and must report it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mind_mem.hook_installer import AGENT_REGISTRY

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
INTEGRATIONS = ROOT / "docs" / "integrations.md"
CLIENT_DOC = ROOT / "docs" / "client-integrations.md"

# Who makes each registry client. Sources: each vendor's own site or repo
# (NemoClaw: nvidianews.nvidia.com/news/nvidia-announces-nemoclaw and
# docs.nvidia.com/nemoclaw; NanoClaw: nanoclaw.dev; OpenClaw: the OpenClaw
# Foundation per github.com/openclaw/openclaw).
VENDORS: dict[str, str] = {
    "claude-code": "Anthropic",
    "codex": "OpenAI",
    "grok-build": "xAI",
    "vibe": "Mistral AI",
    "opencode": "OpenCode (open source)",
    "gemini": "Google",
    "cursor": "Anysphere",
    "windsurf": "Cognition (formerly Codeium)",
    "aider": "Aider-AI (open source)",
    "openclaw": "OpenClaw Foundation (open source)",
    "nanoclaw": "NanoClaw / Qwibit (open source)",
    "nemoclaw": "NVIDIA",
    "continue": "Continue.dev",
    "cline": "Cline",
    "roo": "Roo Code",
    "zed": "Zed Industries",
    "copilot": "GitHub / Microsoft",
    "copilot-cli": "GitHub / Microsoft",
    "cody": "Sourcegraph",
    "qodo": "Qodo",
}

VENDOR_HEADER = "| Client | Vendor | `mm install` id |"

# (document, header prefix, column holding the id as inline code)
TABLES = {
    "readme-vendor": (README, VENDOR_HEADER, 2),
    "readme-config-location": (README, "| Client | Config Location | Format |", 0),
    "readme-config-file": (README, "| Client | Config File |", 0),
    "integrations-vendor": (INTEGRATIONS, VENDOR_HEADER, 2),
    "client-doc-vendor": (CLIENT_DOC, VENDOR_HEADER, 2),
}
VENDOR_TABLES = ("readme-vendor", "integrations-vendor", "client-doc-vendor")
MCP_TABLE = (CLIENT_DOC, "| Client | `mm install` id | Config path |", 1)
# Tables whose rows must spell each MCP-aware client's MCP file in full.
MCP_PATH_TABLES = ("readme-config-location", "readme-config-file")

_ID = re.compile(r"`([a-z][a-z0-9-]*)`")
_HEADING = re.compile(r"Native integration with (\d+) clients \((\d+) MCP-aware clients\)")
_INSTALL_ROW = re.compile(r"^\|\s*Install\s*\|\s*`mm install(?:-all --agent)? ([a-z][a-z0-9-]*)`")


def _table_rows(text: str, header_prefix: str) -> list[list[str]]:
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(header_prefix)]
    assert len(starts) == 1, f"expected exactly one table headed {header_prefix!r}, found {len(starts)}"
    rows = []
    for ln in lines[starts[0] + 2 :]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows


def _row_id(cells: list[str], col: int, where: str) -> str:
    found = _ID.findall(cells[col])
    assert found, f"{where}: row has no `id` in column {col}: {cells}"
    return found[-1]


def _ids(text: str, header: str, col: int, where: str = "") -> list[str]:
    return [_row_id(cells, col, where) for cells in _table_rows(text, header)]


def _vendors(text: str) -> dict[str, str]:
    return {_row_id(c, 2, "vendor"): c[1] for c in _table_rows(text, VENDOR_HEADER)}


def _install_ids(text: str) -> list[str]:
    return [m.group(1) for ln in text.splitlines() if (m := _INSTALL_ROW.match(ln))]


def _home(path_tmpl: str) -> str:
    return path_tmpl.replace("{home}", "~")


_CACHE: dict[Path, str] = {}


def _read(path: Path) -> str:
    if path not in _CACHE:
        _CACHE[path] = path.read_text(encoding="utf-8")
    return _CACHE[path]


def _mcp_aware() -> set[str]:
    return {k for k, v in AGENT_REGISTRY.items() if v.mcp_fmt}


def test_vendor_map_covers_exactly_the_registry() -> None:
    assert set(VENDORS) == set(AGENT_REGISTRY), (
        f"VENDORS missing {sorted(set(AGENT_REGISTRY) - set(VENDORS))}, extra {sorted(set(VENDORS) - set(AGENT_REGISTRY))}"
    )


@pytest.mark.parametrize("table", sorted(TABLES))
def test_table_names_exactly_the_registry(table: str) -> None:
    doc, header, col = TABLES[table]
    ids = _ids(_read(doc), header, col, table)
    assert len(ids) == len(set(ids)), f"{table}: duplicate rows {ids}"
    assert set(ids) == set(AGENT_REGISTRY), (
        f"{table}: missing {sorted(set(AGENT_REGISTRY) - set(ids))}, extra {sorted(set(ids) - set(AGENT_REGISTRY))}"
    )


@pytest.mark.parametrize("table", VENDOR_TABLES)
def test_vendor_column_matches_vendor_map(table: str) -> None:
    doc, _header, _col = TABLES[table]
    got = _vendors(_read(doc))
    wrong = {k: (v, VENDORS[k]) for k, v in got.items() if v != VENDORS[k]}
    assert not wrong, f"{table}: vendor (doc, expected) {wrong}"


@pytest.mark.parametrize("table", VENDOR_TABLES)
def test_vendor_table_marks_exactly_the_mcp_aware_clients(table: str) -> None:
    doc, header, col = TABLES[table]
    marked = {_row_id(c, col, table) for c in _table_rows(_read(doc), header) if c[3].startswith("Yes")}
    assert marked == _mcp_aware()


def test_client_doc_mcp_table_names_exactly_the_mcp_aware_clients() -> None:
    doc, header, col = MCP_TABLE
    rows = _table_rows(_read(doc), header)
    ids = [_row_id(c, col, "mcp-table") for c in rows]
    assert sorted(ids) == sorted(_mcp_aware())
    for cells in rows:
        spec = AGENT_REGISTRY[_row_id(cells, col, "mcp-table")]
        assert f"`{_home(spec.mcp_path_tmpl)}`" == cells[2], f"{spec.name}: {cells[2]!r}"


@pytest.mark.parametrize("table", MCP_PATH_TABLES)
def test_mcp_path_is_spelled_in_full(table: str) -> None:
    doc, header, col = TABLES[table]
    for cells in _table_rows(_read(doc), header):
        spec = AGENT_REGISTRY[_row_id(cells, col, table)]
        if spec.mcp_fmt:
            assert "no MCP writer" not in cells[1], cells
            assert f"`{_home(spec.mcp_path_tmpl)}`" in cells[1], f"{table}/{spec.name}: {cells[1]!r}"


def test_client_doc_has_one_install_section_per_registry_client() -> None:
    ids = _install_ids(_read(CLIENT_DOC))
    assert len(ids) == len(set(ids)), ids
    assert set(ids) == set(AGENT_REGISTRY), (
        f"missing {sorted(set(AGENT_REGISTRY) - set(ids))}, extra {sorted(set(ids) - set(AGENT_REGISTRY))}"
    )


@pytest.mark.parametrize("doc", [README, INTEGRATIONS], ids=["readme", "integrations"])
def test_heading_counts_match_registry(doc: Path) -> None:
    m = _HEADING.search(_read(doc))
    assert m, f"integration heading not found in {doc.name}"
    assert int(m.group(1)) == len(AGENT_REGISTRY)
    assert int(m.group(2)) == len(_mcp_aware())


def test_registry_descriptions_do_not_call_independent_projects_variants() -> None:
    for key in ("nanoclaw", "nemoclaw"):
        assert "variant" not in AGENT_REGISTRY[key].description.lower(), AGENT_REGISTRY[key].description
    assert "NVIDIA" in AGENT_REGISTRY["nemoclaw"].description


# --------------------------------------------------------------- controls


def _drop_row(text: str, header: str, col: int, victim: str) -> str:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(header))
    end = start + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    kept = [ln for i, ln in enumerate(lines) if not (start + 2 <= i < end and f"`{victim}`" in ln.split("|")[col + 1])]
    assert len(kept) == len(lines) - 1, "control did not remove exactly one row"
    return "\n".join(kept)


@pytest.mark.parametrize("table", sorted(TABLES))
def test_positive_control_a_dropped_row_is_detected(table: str) -> None:
    doc, header, col = TABLES[table]
    victim = next(iter(AGENT_REGISTRY))
    assert victim not in set(_ids(_drop_row(_read(doc), header, col, victim), header, col))


@pytest.mark.parametrize("table", VENDOR_TABLES)
def test_positive_control_a_wrong_vendor_is_detected(table: str) -> None:
    doc, _header, _col = TABLES[table]
    text = _read(doc)
    row = next(ln for ln in text.splitlines() if ln.startswith("| NemoClaw | NVIDIA | `nemoclaw` |"))
    mutated = text.replace(row, row.replace("| NVIDIA |", "| OpenClaw variant |"), 1)
    got = _vendors(mutated)
    assert got["nemoclaw"] != VENDORS["nemoclaw"]


def test_positive_control_a_vaguer_mcp_path_is_detected() -> None:
    text = _read(README)
    exact = _home(AGENT_REGISTRY["cline"].mcp_path_tmpl)
    vague = exact.replace("~/.vscode-server/data/User", "<vscode-user>")
    assert text.count(f"`{exact}`") >= 2
    mutated = text.replace(f"`{exact}`", f"`{vague}`")
    doc, header, col = TABLES["readme-config-file"]
    rows = {_row_id(c, col, "x"): c for c in _table_rows(mutated, header)}
    assert f"`{exact}`" not in rows["cline"][1]


def test_positive_control_a_missing_install_section_is_detected() -> None:
    text = _read(CLIENT_DOC)
    mutated = text.replace("| Install | `mm install vibe` |", "| Install | (removed) |")
    assert "vibe" not in set(_install_ids(mutated))
