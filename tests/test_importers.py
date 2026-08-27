"""Tests for the roadmap Group G migration importers (file-based subset).

Everything here runs offline against committed fixture dumps — the whole
point of shipping only the file-based subset is that an import never
opens a socket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem import mm_cli
from mind_mem._recall_constants import CORPUS_FILES
from mind_mem.block_parser import parse_file
from mind_mem.block_provenance import extract_provenance
from mind_mem.block_store import _BLOCK_PREFIX_MAP
from mind_mem.importers import (
    ALL_SYSTEMS,
    DEFERRED_SYSTEMS,
    IMPORTED_CORPUS_FILE,
    QUARANTINE_STATUS,
    QUARANTINE_TIER,
    SUPPORTED_SYSTEMS,
    ImportParseError,
    UnsupportedSystemError,
    provenance_token,
    resolve_system,
    run_import,
)
from mind_mem.importers.parsers import parse_payload

FIXTURES = Path(__file__).parent / "fixtures" / "importers"

CHROMA_DUMP = str(FIXTURES / "chroma_export.json")
MEM0_DUMP = str(FIXTURES / "mem0_export.json")
LETTA_DUMP = str(FIXTURES / "letta_agent.af.json")
NEAR_DUP_DUMP = str(FIXTURES / "chroma_near_duplicates.json")

# Record counts each committed fixture is expected to yield.
EXPECTED_RECORDS = {"chroma": 4, "mem0": 3, "letta": 5}


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "ws"
    for sub in ("memory", "decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"version": "4.9.1", "workspace_path": str(ws), "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    return str(ws)


def _imported_blocks(workspace: str) -> list[dict]:
    path = Path(workspace) / IMPORTED_CORPUS_FILE
    if not path.is_file():
        return []
    return parse_file(str(path))


# ---------------------------------------------------------------------------
# Registry + wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_supported_is_the_local_subset(self) -> None:
        assert SUPPORTED_SYSTEMS == ("agentmem", "chatjson", "chroma", "letta", "markdown", "mem0")

    def test_deferred_names_the_endpoint_backed_systems(self) -> None:
        assert set(DEFERRED_SYSTEMS) == {"pinecone", "qdrant", "weaviate"}
        assert not set(DEFERRED_SYSTEMS) & set(SUPPORTED_SYSTEMS)

    def test_cli_choices_stay_in_lockstep(self) -> None:
        assert mm_cli._IMPORT_SYSTEM_CHOICES == ALL_SYSTEMS

    def test_imp_prefix_is_mapped_in_both_prefix_maps(self) -> None:
        from mind_mem.mcp.tools.memory_ops import _BLOCK_PREFIX_MAP as MCP_MAP

        assert _BLOCK_PREFIX_MAP["IMP"] == ("memory", "IMPORTED.md")
        assert MCP_MAP["IMP"] == _BLOCK_PREFIX_MAP["IMP"]

    def test_imported_corpus_is_indexed(self) -> None:
        assert CORPUS_FILES["imported"] == IMPORTED_CORPUS_FILE


class TestResolveSystem:
    @pytest.mark.parametrize("raw", ["chroma", "  CHROMA ", "Mem0", "LETTA"])
    def test_normalizes_supported(self, raw: str) -> None:
        assert resolve_system(raw) in SUPPORTED_SYSTEMS

    @pytest.mark.parametrize("system", sorted(DEFERRED_SYSTEMS))
    def test_deferred_systems_are_refused_explicitly(self, system: str) -> None:
        with pytest.raises(UnsupportedSystemError) as excinfo:
            resolve_system(system)
        message = str(excinfo.value)
        assert system in message
        assert "DEFERRED" in message
        for supported in SUPPORTED_SYSTEMS:
            assert supported in message

    def test_unknown_system_is_refused(self) -> None:
        with pytest.raises(UnsupportedSystemError) as excinfo:
            resolve_system("faiss")
        assert "unsupported source system" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class TestParsers:
    def test_chroma_parallel_arrays(self) -> None:
        records = parse_payload("chroma", json.loads(Path(CHROMA_DUMP).read_text(encoding="utf-8")))
        assert len(records) == EXPECTED_RECORDS["chroma"]
        assert records[0].external_id == "chroma-0001"
        assert records[0].metadata["team"] == "platform"
        assert records[0].metadata["collection"] == "agent_memory"

    def test_chroma_rejects_length_mismatch(self) -> None:
        with pytest.raises(ImportParseError, match="inconsistent"):
            parse_payload("chroma", {"ids": ["a"], "documents": ["x", "y"]})

    def test_chroma_requires_documents(self) -> None:
        with pytest.raises(ImportParseError, match="documents"):
            parse_payload("chroma", {"ids": ["a"]})

    def test_mem0_drops_empty_memory(self) -> None:
        raw = json.loads(Path(MEM0_DUMP).read_text(encoding="utf-8"))
        assert len(raw["results"]) == 4  # fixture carries one empty placeholder
        records = parse_payload("mem0", raw)
        assert len(records) == EXPECTED_RECORDS["mem0"]
        assert records[0].metadata["user_id"] == "alice"
        assert records[0].created_at.startswith("2026-01-14")

    def test_mem0_requires_results(self) -> None:
        with pytest.raises(ImportParseError, match="results"):
            parse_payload("mem0", {"data": []})

    def test_letta_core_and_archival(self) -> None:
        records = parse_payload("letta", json.loads(Path(LETTA_DUMP).read_text(encoding="utf-8")))
        assert len(records) == EXPECTED_RECORDS["letta"]
        sections = [r.metadata["section"] for r in records]
        assert sections.count("core_memory") == 2
        assert sections.count("archival_memory") == 3
        assert records[0].external_id == "release-captain/core/persona"

    def test_letta_messages_are_not_imported(self) -> None:
        records = parse_payload("letta", json.loads(Path(LETTA_DUMP).read_text(encoding="utf-8")))
        joined = " ".join(r.text for r in records)
        assert "cut the release" not in joined
        assert "starting the reproducibility gate" not in joined

    def test_letta_rejects_a_dump_with_no_memory(self) -> None:
        with pytest.raises(ImportParseError, match="core_memory"):
            parse_payload("letta", {"name": "x", "messages": []})


# ---------------------------------------------------------------------------
# Acceptance gate 1 — N blocks, each carrying imported:<system> provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system,dump",
    [("chroma", CHROMA_DUMP), ("mem0", MEM0_DUMP), ("letta", LETTA_DUMP)],
)
def test_dump_imports_to_n_blocks_with_provenance(workspace: str, system: str, dump: str) -> None:
    result = run_import(workspace, system, dump)
    expected = EXPECTED_RECORDS[system]

    assert result.parsed == expected
    assert result.imported == expected
    assert len(result.block_ids) == expected
    assert len(set(result.block_ids)) == expected

    blocks = _imported_blocks(workspace)
    assert len(blocks) == expected

    token = provenance_token(system)
    for block in blocks:
        assert str(block["_id"]).startswith(f"IMP-{system}-")
        assert block["Source"] == token
        assert extract_provenance(block)["tool_id"] == token
        assert extract_provenance(block)["purpose"] == "migration-import"
        # External ingest is quarantined on arrival — never authoritative.
        assert block["Status"] == QUARANTINE_STATUS
        assert block["IngestTier"] == QUARANTINE_TIER
        assert block["ImportBatch"] == result.batch
        assert block["Statement"].strip()


def test_all_three_formats_coexist_in_one_workspace(workspace: str) -> None:
    total = 0
    for system, dump in (("chroma", CHROMA_DUMP), ("mem0", MEM0_DUMP), ("letta", LETTA_DUMP)):
        total += run_import(workspace, system, dump).imported
    blocks = _imported_blocks(workspace)
    assert total == sum(EXPECTED_RECORDS.values())
    assert len(blocks) == total
    assert len({b["_id"] for b in blocks}) == total


# ---------------------------------------------------------------------------
# Acceptance gate 2 — idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system,dump",
    [("chroma", CHROMA_DUMP), ("mem0", MEM0_DUMP), ("letta", LETTA_DUMP)],
)
def test_reimport_is_idempotent(workspace: str, system: str, dump: str) -> None:
    first = run_import(workspace, system, dump)
    corpus = Path(workspace) / IMPORTED_CORPUS_FILE
    after_first = corpus.read_bytes()

    second = run_import(workspace, system, dump)

    assert second.parsed == first.parsed
    assert second.imported == 0
    assert second.skipped_existing == first.parsed
    assert second.block_ids == ()
    # Byte-identical corpus: a re-import writes nothing at all.
    assert corpus.read_bytes() == after_first

    blocks = _imported_blocks(workspace)
    ids = [b["_id"] for b in blocks]
    assert len(ids) == first.parsed
    assert len(set(ids)) == len(ids)  # zero duplicates


def test_idempotent_across_a_third_run_and_other_systems(workspace: str) -> None:
    run_import(workspace, "mem0", MEM0_DUMP)
    run_import(workspace, "chroma", CHROMA_DUMP)
    run_import(workspace, "mem0", MEM0_DUMP)
    run_import(workspace, "chroma", CHROMA_DUMP)
    ids = [b["_id"] for b in _imported_blocks(workspace)]
    assert len(ids) == EXPECTED_RECORDS["mem0"] + EXPECTED_RECORDS["chroma"]
    assert len(set(ids)) == len(ids)


def test_dry_run_writes_nothing(workspace: str) -> None:
    result = run_import(workspace, "chroma", CHROMA_DUMP, dry_run=True)
    assert result.dry_run is True
    assert result.imported == EXPECTED_RECORDS["chroma"]
    assert not (Path(workspace) / IMPORTED_CORPUS_FILE).exists()


# ---------------------------------------------------------------------------
# Acceptance gate 3 — imported content is INERT until released
#
# The end-to-end "quarantined -> governed release -> recallable" proof
# lives in tests/test_importers_quarantine.py; these two pin the half
# that matters for every dump format: a fresh import answers nothing.
# ---------------------------------------------------------------------------


def test_imported_content_is_not_recallable_while_quarantined(workspace: str) -> None:
    from mind_mem.recall import recall

    run_import(workspace, "chroma", CHROMA_DUMP)
    hits = recall(workspace, "reproducibility gate staging deployment", limit=10)
    ids = {hit.get("id") or hit.get("_id") for hit in hits}
    assert not any(str(i).startswith("IMP-chroma-") for i in ids), ids


def test_recall_withholds_mem0_and_letta_content(workspace: str) -> None:
    from mind_mem.recall import recall

    run_import(workspace, "mem0", MEM0_DUMP)
    run_import(workspace, "letta", LETTA_DUMP)

    mem0_hits = recall(workspace, "billing service payment reconciliation escalation", limit=10)
    assert not any(str(h.get("id") or h.get("_id")).startswith("IMP-mem0-") for h in mem0_hits)

    letta_hits = recall(workspace, "migration held an exclusive lock rollback release", limit=10)
    assert not any(str(h.get("id") or h.get("_id")).startswith("IMP-letta-") for h in letta_hits)


def test_multiline_record_round_trips_through_the_parser(workspace: str) -> None:
    run_import(workspace, "chroma", CHROMA_DUMP)
    blocks = _imported_blocks(workspace)
    postmortem = [b for b in blocks if "Postmortem" in b["Statement"]]
    assert len(postmortem) == 1
    statement = postmortem[0]["Statement"]
    assert "Root cause was a stale connection pool after the failover." in statement
    assert "pool health checks were advisory only" in statement
    assert "the reconnect backoff was unbounded" in statement


# ---------------------------------------------------------------------------
# Acceptance gate 4 — unsupported / deferred systems fail explicitly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", sorted(DEFERRED_SYSTEMS))
def test_run_import_refuses_deferred_systems(workspace: str, system: str) -> None:
    with pytest.raises(UnsupportedSystemError) as excinfo:
        run_import(workspace, system, CHROMA_DUMP)
    assert "DEFERRED" in str(excinfo.value)
    assert not (Path(workspace) / IMPORTED_CORPUS_FILE).exists()


@pytest.mark.parametrize("system", sorted(DEFERRED_SYSTEMS))
def test_cli_exits_with_a_deferred_message(workspace: str, system: str, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    code = mm_cli.main(["import", "--from", system, CHROMA_DUMP])
    captured = capsys.readouterr()
    assert code == mm_cli.IMPORT_EXIT_UNSUPPORTED
    assert "DEFERRED" in captured.err
    assert system in captured.err
    assert ", ".join(SUPPORTED_SYSTEMS) in captured.err


def test_cli_rejects_a_completely_unknown_system(workspace: str, monkeypatch) -> None:
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    with pytest.raises(SystemExit) as excinfo:
        mm_cli.main(["import", "--from", "faiss", CHROMA_DUMP])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Dump-boundary validation
# ---------------------------------------------------------------------------


class TestDumpValidation:
    def test_missing_file(self, workspace: str, tmp_path: Path) -> None:
        with pytest.raises(ImportParseError, match="not found"):
            run_import(workspace, "chroma", str(tmp_path / "nope.json"))

    def test_directory_is_not_a_dump(self, workspace: str, tmp_path: Path) -> None:
        with pytest.raises(ImportParseError, match="not a regular file"):
            run_import(workspace, "chroma", str(tmp_path))

    def test_malformed_json(self, workspace: str, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ImportParseError, match="not valid JSON"):
            run_import(workspace, "chroma", str(bad))

    def test_cli_reports_a_malformed_dump(self, workspace: str, tmp_path: Path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        code = mm_cli.main(["import", "--from", "chroma", str(bad)])
        assert code == mm_cli.IMPORT_EXIT_BAD_DUMP
        assert "not valid JSON" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Opt-in near-duplicate collapse (default OFF)
# ---------------------------------------------------------------------------


class TestNearDuplicateFlag:
    def test_default_off_imports_every_record(self, workspace: str) -> None:
        result = run_import(workspace, "chroma", NEAR_DUP_DUMP)
        assert result.parsed == 2
        assert result.imported == 2
        assert result.skipped_near_duplicate == 0

    def test_flag_on_collapses_the_near_duplicate(self, workspace: str) -> None:
        result = run_import(workspace, "chroma", NEAR_DUP_DUMP, dedup_near=True)
        assert result.parsed == 2
        assert result.imported == 1
        assert result.skipped_near_duplicate == 1

    def test_flag_off_is_identical_to_the_no_flag_call(self, tmp_path: Path) -> None:
        """Flag-off must be byte-identical to the pre-flag behaviour."""
        corpora = []
        for name in ("a", "b"):
            ws = tmp_path / name
            (ws / "memory").mkdir(parents=True)
            (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
            kwargs = {"dedup_near": False} if name == "b" else {}
            run_import(str(ws), "chroma", NEAR_DUP_DUMP, **kwargs)  # type: ignore[arg-type]
            corpora.append((ws / IMPORTED_CORPUS_FILE).read_bytes())
        assert corpora[0] == corpora[1]


# ---------------------------------------------------------------------------
# CLI happy path
# ---------------------------------------------------------------------------


def test_cli_end_to_end(workspace: str, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    assert mm_cli.main(["import", "--from", "mem0", MEM0_DUMP]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["system"] == "mem0"
    assert payload["imported"] == EXPECTED_RECORDS["mem0"]
    assert payload["skipped_existing"] == 0
    assert len(payload["block_ids"]) == EXPECTED_RECORDS["mem0"]

    assert mm_cli.main(["import", "--from", "mem0", MEM0_DUMP]) == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["imported"] == 0
    assert payload2["skipped_existing"] == EXPECTED_RECORDS["mem0"]


def test_cli_dry_run(workspace: str, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    assert mm_cli.main(["import", "--from", "letta", LETTA_DUMP, "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["imported"] == EXPECTED_RECORDS["letta"]
    assert not (Path(workspace) / IMPORTED_CORPUS_FILE).exists()
