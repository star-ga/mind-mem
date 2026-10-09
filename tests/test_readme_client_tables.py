# Copyright 2026 STARGA, Inc.
"""The README client tables must name exactly the clients the installer knows.

Authority: ``mind_mem.hook_installer.AGENT_REGISTRY`` (the same authority
``scripts/check_docs_alignment.py`` uses for the client counts). The counts
were gated, the tables were not: the README listed Claude Desktop (not in the
registry), OpenCode under ``./install.sh`` (which does not install it), and
omitted OpenCode and GitHub Copilot CLI from the vendor table. These tests
compare the set of ``mm install`` ids in each table with the registry, so a
client added to or dropped from the registry without a README edit fails here.

Each negative assertion has a positive control: the same parser is run on a
README with one row removed and must report the missing id.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mind_mem.hook_installer import AGENT_REGISTRY

README = Path(__file__).resolve().parent.parent / "README.md"

# Each table is identified by its header prefix; ``col`` is the cell that
# carries the registry id as inline code.
TABLES = {
    "vendor": ("| Client | Vendor | `mm install` id |", 2),
    "config_location": ("| Client | Config Location | Format |", 0),
    "config_file": ("| Client | Config File |", 0),
}

_ID = re.compile(r"`([a-z][a-z0-9-]*)`")
_HEADING = re.compile(r"Native integration with (\d+) clients \((\d+) MCP-aware clients\)")


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


def _ids(text: str, table: str) -> list[str]:
    header, col = TABLES[table]
    out = []
    for cells in _table_rows(text, header):
        found = _ID.findall(cells[col])
        assert found, f"{table}: row has no `id` in column {col}: {cells}"
        out.append(found[-1])
    return out


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("table", sorted(TABLES))
def test_table_names_exactly_the_registry(readme: str, table: str) -> None:
    ids = _ids(readme, table)
    assert len(ids) == len(set(ids)), f"{table}: duplicate rows {ids}"
    assert set(ids) == set(AGENT_REGISTRY), (
        f"{table}: missing {sorted(set(AGENT_REGISTRY) - set(ids))}, extra {sorted(set(ids) - set(AGENT_REGISTRY))}"
    )


def test_vendor_table_marks_exactly_the_mcp_aware_clients(readme: str) -> None:
    header, col = TABLES["vendor"]
    marked = {_ID.findall(c[col])[-1] for c in _table_rows(readme, header) if c[3].startswith("Yes")}
    assert marked == {k for k, v in AGENT_REGISTRY.items() if v.mcp_fmt}


def test_config_file_table_names_an_mcp_file_for_every_mcp_aware_client(readme: str) -> None:
    header, col = TABLES["config_file"]
    for cells in _table_rows(readme, header):
        spec = AGENT_REGISTRY[_ID.findall(cells[col])[-1]]
        if spec.mcp_fmt:
            assert "no MCP writer" not in cells[1], cells
            leaf = spec.mcp_path_tmpl.rsplit("/", 1)[-1]
            assert leaf in cells[1], f"{spec.name}: MCP file {leaf} not in {cells[1]!r}"


def test_heading_counts_match_registry(readme: str) -> None:
    m = _HEADING.search(readme)
    assert m, "integration heading not found"
    assert int(m.group(1)) == len(AGENT_REGISTRY)
    assert int(m.group(2)) == sum(1 for v in AGENT_REGISTRY.values() if v.mcp_fmt)


@pytest.mark.parametrize("table", sorted(TABLES))
def test_positive_control_a_dropped_row_is_detected(readme: str, table: str) -> None:
    header, col = TABLES[table]
    victim = next(iter(AGENT_REGISTRY))
    lines = readme.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(header))
    end = start + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    kept = [ln for i, ln in enumerate(lines) if not (start + 2 <= i < end and f"`{victim}`" in ln.split("|")[col + 1])]
    assert len(kept) == len(lines) - 1, "control did not remove exactly one row"
    assert victim not in set(_ids("\n".join(kept), table))
