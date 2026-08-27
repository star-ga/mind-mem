"""Tests for the note-tree and transcript importers.

These are the formats agent memory actually lives in on disk: markdown
note trees, auto-memory directories, and chat-session transcripts.
Everything runs offline against committed fixtures — no socket, no
credential, no stub server.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mind_mem import mm_cli
from mind_mem.block_parser import parse_file
from mind_mem.block_provenance import extract_provenance
from mind_mem.importers import (
    ALL_SYSTEMS,
    DIRECTORY_SYSTEMS,
    IMPORTED_CORPUS_FILE,
    QUARANTINE_STATUS,
    QUARANTINE_TIER,
    SUPPORTED_SYSTEMS,
    ImportParseError,
    provenance_token,
    run_import,
)
from mind_mem.importers.fs_source import (
    MAX_TREE_FILES,
    SourceNote,
    load_note_tree,
    markdown_link_targets,
    parse_front_matter,
    wikilink_targets,
)
from mind_mem.importers.note_parsers import parse_chat_json, resolve_links
from mind_mem.importers.parsers import parse_payload

FIXTURES = Path(__file__).parent / "fixtures" / "importers"

VAULT = str(FIXTURES / "vault")
AGENT_MEMORY = str(FIXTURES / "agent_memory")
CHAT_SESSION = str(FIXTURES / "chat_session.json")

#: Records each committed fixture is expected to yield.
EXPECTED_RECORDS = {"markdown": 3, "agentmem": 5, "chatjson": 4}

SOURCES = {"markdown": VAULT, "agentmem": AGENT_MEMORY, "chatjson": CHAT_SESSION}
CASES = [(system, SOURCES[system]) for system in sorted(SOURCES)]


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


def _by_external_id(workspace: str) -> dict[str, dict]:
    return {str(block.get("ExternalId")): block for block in _imported_blocks(workspace)}


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_the_three_note_formats_are_supported(self) -> None:
        for system in ("markdown", "agentmem", "chatjson"):
            assert system in SUPPORTED_SYSTEMS

    def test_vector_store_and_service_dumps_are_still_supported(self) -> None:
        for system in ("chroma", "mem0", "letta"):
            assert system in SUPPORTED_SYSTEMS

    def test_cli_choices_stay_in_lockstep(self) -> None:
        assert mm_cli._IMPORT_SYSTEM_CHOICES == ALL_SYSTEMS

    def test_directory_systems_are_the_note_tree_ones(self) -> None:
        assert DIRECTORY_SYSTEMS == frozenset({"agentmem", "markdown"})

    def test_chroma_is_marked_low_value_with_an_upgrade_path(self) -> None:
        from mind_mem.importers.parsers import parse_chroma

        doc = parse_chroma.__doc__ or ""
        assert "deferred:" in doc
        assert "LOW-VALUE" in doc
        assert "upgrade path:" in doc


# ---------------------------------------------------------------------------
# Front matter / link extraction
# ---------------------------------------------------------------------------


class TestFrontMatter:
    def test_nested_keys_flatten_to_dotted_paths(self) -> None:
        fields, body = parse_front_matter(
            "---\nname: note_one\ndescription: a summary\nmetadata:\n  type: reference\n  node_type: memory\n---\nbody\n"
        )
        assert fields["name"] == "note_one"
        assert fields["description"] == "a summary"
        assert fields["metadata.type"] == "reference"
        assert fields["metadata.node_type"] == "memory"
        assert body.strip() == "body"

    def test_sequences_join_with_commas(self) -> None:
        fields, _ = parse_front_matter("---\ntags:\n  - alpha\n  - beta\n---\nbody\n")
        assert fields["tags"] == "alpha,beta"

    def test_no_front_matter_leaves_the_body_untouched(self) -> None:
        fields, body = parse_front_matter("# Heading\n\ntext\n")
        assert fields == {}
        assert body == "# Heading\n\ntext\n"

    def test_wikilinks_are_ordered_and_deduplicated(self) -> None:
        found = wikilink_targets("see [[b]] then [[a|alias]] then [[b]] and [[a#section]]")
        assert found == ("b", "a")

    def test_markdown_link_targets_reduce_to_stems(self) -> None:
        assert markdown_link_targets("- [label](notes/deep_note.md) and [x](other.md#top)") == (
            "deep_note",
            "other",
        )


# ---------------------------------------------------------------------------
# Note-tree loading (boundary validation + determinism)
# ---------------------------------------------------------------------------


class TestNoteTreeLoading:
    def test_walk_is_sorted_and_prunes_excluded_dirs(self) -> None:
        notes = load_note_tree(VAULT)
        paths = [note.relative_path for note in notes]
        assert paths == sorted(paths)
        assert not any(path.startswith("templates/") for path in paths)
        assert not any("/.obsidian/" in path or path.startswith(".obsidian/") for path in paths)

    def test_notes_carry_no_filesystem_timestamps(self) -> None:
        assert set(SourceNote.__dataclass_fields__) == {"relative_path", "front_matter", "body"}

    def test_two_copies_of_a_tree_load_identically(self, tmp_path: Path) -> None:
        copy = tmp_path / "copy"
        shutil.copytree(VAULT, copy)
        assert load_note_tree(VAULT) == load_note_tree(str(copy))

    def test_a_file_is_not_a_note_tree(self) -> None:
        with pytest.raises(ImportParseError, match="not a directory"):
            load_note_tree(CHAT_SESSION)

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ImportParseError, match="not found"):
            load_note_tree(str(tmp_path / "nope"))

    def test_a_directory_with_no_notes(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "data.txt").write_text("not a note", encoding="utf-8")
        with pytest.raises(ImportParseError, match="no markdown notes"):
            load_note_tree(str(empty))

    def test_non_utf8_note_is_an_explicit_error(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "bad.md").write_bytes(b"\xff\xfe not utf-8")
        with pytest.raises(ImportParseError, match="not valid UTF-8"):
            load_note_tree(str(tree))

    def test_oversized_note_is_skipped_not_fatal(self, tmp_path: Path, monkeypatch) -> None:
        from mind_mem.importers import fs_source

        monkeypatch.setattr(fs_source, "MAX_NOTE_BYTES", 64)
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "small.md").write_text("tiny note", encoding="utf-8")
        (tree / "huge.md").write_text("x" * 500, encoding="utf-8")
        notes = fs_source.load_note_tree(str(tree))
        assert [note.relative_path for note in notes] == ["small.md"]

    def test_file_count_ceiling_is_enforced(self, tmp_path: Path, monkeypatch) -> None:
        from mind_mem.importers import fs_source

        monkeypatch.setattr(fs_source, "MAX_TREE_FILES", 2)
        tree = tmp_path / "tree"
        tree.mkdir()
        for index in range(4):
            (tree / f"note{index}.md").write_text(f"note {index}", encoding="utf-8")
        with pytest.raises(ImportParseError, match="more than 2 notes"):
            fs_source.load_note_tree(str(tree))

    def test_default_ceilings_are_sane(self) -> None:
        assert MAX_TREE_FILES >= 1000


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class TestMarkdownVault:
    def test_records_and_skipped_empty_note(self) -> None:
        records = parse_payload("markdown", load_note_tree(VAULT))
        assert len(records) == EXPECTED_RECORDS["markdown"]
        # The placeholder note has front matter but no body — it is dropped.
        assert "notes/empty-note.md" not in {record.external_id for record in records}

    def test_front_matter_becomes_metadata(self) -> None:
        records = {r.external_id: r for r in parse_payload("markdown", load_note_tree(VAULT))}
        note = records["notes/architecture.md"]
        assert note.metadata["title"] == "Storage architecture"
        assert note.metadata["tags"] == "architecture,storage"
        assert note.metadata["status"] == "current"
        assert note.metadata["path"] == "notes/architecture.md"
        assert note.metadata["folder"] == "notes"

    def test_wikilinks_are_preserved_on_the_record(self) -> None:
        records = {r.external_id: r for r in parse_payload("markdown", load_note_tree(VAULT))}
        assert records["notes/architecture.md"].links == ("connection-pool-outage", "nightly-compaction")
        assert records["notes/incidents/connection-pool-outage.md"].links == ("architecture",)

    def test_link_names_resolve_to_notes_in_the_tree(self) -> None:
        records = parse_payload("markdown", load_note_tree(VAULT))
        index = resolve_links(records)
        assert index["architecture"] == "notes/architecture.md"
        assert index["connection-pool-outage"] == "notes/incidents/connection-pool-outage.md"
        # A link to a note that is not in the tree simply does not resolve.
        assert "nightly-compaction" not in index

    def test_created_front_matter_becomes_the_timestamp(self) -> None:
        records = {r.external_id: r for r in parse_payload("markdown", load_note_tree(VAULT))}
        assert records["notes/incidents/connection-pool-outage.md"].created_at == "2026-01-14T09:12:00Z"

    def test_rejects_a_json_payload(self) -> None:
        with pytest.raises(ImportParseError, match="expects a directory"):
            parse_payload("markdown", {"documents": ["x"]})


class TestAgentMemoryDirectory:
    def test_sections_are_classified_structurally(self) -> None:
        records = {r.external_id: r for r in parse_payload("agentmem", load_note_tree(AGENT_MEMORY))}
        assert records["MEMORY.md"].metadata["section"] == "index"
        assert records["OPERATING-MANUAL.md"].metadata["section"] == "instructions"
        assert records["reference_pool_health_checks.md"].metadata["section"] == "memory"
        assert records["memory/nested_retention_policy.md"].metadata["section"] == "memory"

    def test_name_description_and_nested_type_are_mapped(self) -> None:
        records = {r.external_id: r for r in parse_payload("agentmem", load_note_tree(AGENT_MEMORY))}
        note = records["reference_pool_health_checks.md"]
        assert note.metadata["name"] == "reference_pool_health_checks"
        assert note.metadata["description"].startswith("Advisory pool health checks")
        # metadata.type is surfaced flat AND kept under its dotted path.
        assert note.metadata["type"] == "reference"
        assert note.metadata["metadata.type"] == "reference"
        assert note.metadata["metadata.node_type"] == "memory"

    def test_index_links_by_path_and_by_wikilink(self) -> None:
        records = {r.external_id: r for r in parse_payload("agentmem", load_note_tree(AGENT_MEMORY))}
        links = records["MEMORY.md"].links
        assert "reference_pool_health_checks" in links
        assert "feedback_append_only_writes" in links
        assert "nested_retention_policy" in links

    def test_note_wikilinks_resolve_across_the_directory(self) -> None:
        records = parse_payload("agentmem", load_note_tree(AGENT_MEMORY))
        index = resolve_links(records)
        assert index["feedback_append_only_writes"] == "feedback_append_only_writes.md"
        assert index["nested_retention_policy"] == "memory/nested_retention_policy.md"

    def test_rejects_a_json_payload(self) -> None:
        with pytest.raises(ImportParseError, match="expects a directory"):
            parse_payload("agentmem", [{"role": "user", "content": "hi"}])


class TestChatTranscript:
    def _payload(self) -> dict:
        return json.loads(Path(CHAT_SESSION).read_text(encoding="utf-8"))

    def test_one_record_per_non_empty_turn(self) -> None:
        raw = self._payload()
        assert len(raw["messages"]) == 5  # fixture carries one empty turn
        records = parse_payload("chatjson", raw)
        assert len(records) == EXPECTED_RECORDS["chatjson"]

    def test_role_session_and_turn_are_metadata(self) -> None:
        records = parse_payload("chatjson", self._payload())
        assert [r.metadata["role"] for r in records] == ["system", "user", "assistant", "assistant"]
        assert {r.metadata["session"] for r in records} == {"reindex-triage-2026-01-16"}
        assert [r.metadata["turn"] for r in records] == ["0", "1", "2", "4"]

    def test_structured_content_parts_are_joined(self) -> None:
        records = parse_payload("chatjson", self._payload())
        assistant = records[2]
        assert "Start with the posting-list rebuild." in assistant.text
        assert "mixed" not in assistant.text
        assert "every query pays a merge cost" in assistant.text

    def test_turn_metadata_is_kept(self) -> None:
        records = parse_payload("chatjson", self._payload())
        assert records[3].metadata["resolution"] == "confirmed"
        assert records[3].metadata["component"] == "index"
        assert records[3].external_id == "turn-final"

    def test_bare_turn_list_is_accepted(self) -> None:
        records = parse_chat_json([{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}])
        assert [r.text for r in records] == ["first", "second"]
        assert [r.external_id for r in records] == ["session/0", "session/1"]

    def test_multi_session_wrapper_is_accepted(self) -> None:
        records = parse_chat_json(
            {
                "sessions": [
                    {"id": "s1", "messages": [{"role": "user", "content": "alpha"}]},
                    {"id": "s2", "messages": [{"role": "user", "content": "beta"}]},
                ]
            }
        )
        assert [r.metadata["session"] for r in records] == ["s1", "s2"]

    def test_missing_turn_array_is_an_explicit_error(self) -> None:
        with pytest.raises(ImportParseError, match="no turn array"):
            parse_chat_json({"session_id": "x", "notes": []})

    def test_wrong_turn_type_is_an_explicit_error(self) -> None:
        with pytest.raises(ImportParseError, match="must be a JSON object"):
            parse_chat_json({"messages": [{"role": "user", "content": "ok"}, "not-an-object"]})

    def test_non_list_turn_array_is_an_explicit_error(self) -> None:
        with pytest.raises(ImportParseError, match="must be a list"):
            parse_chat_json({"messages": {"role": "user"}})


# ---------------------------------------------------------------------------
# Acceptance gate — provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system,source", CASES)
def test_import_stamps_provenance_on_every_block(workspace: str, system: str, source: str) -> None:
    result = run_import(workspace, system, source)
    expected = EXPECTED_RECORDS[system]

    assert result.parsed == expected
    assert result.imported == expected
    assert len(set(result.block_ids)) == expected

    token = provenance_token(system)
    blocks = _imported_blocks(workspace)
    assert len(blocks) == expected
    for block in blocks:
        assert str(block["_id"]).startswith(f"IMP-{system}-")
        assert block["Source"] == token
        assert extract_provenance(block)["tool_id"] == token
        assert extract_provenance(block)["purpose"] == "migration-import"
        # External ingest is quarantined on arrival — never authoritative.
        assert block["Status"] == QUARANTINE_STATUS
        assert block["IngestTier"] == QUARANTINE_TIER
        assert block["ImportBatch"] == result.batch
        assert str(block["Statement"]).strip()


def test_every_new_format_coexists_in_one_workspace(workspace: str) -> None:
    total = sum(run_import(workspace, system, source).imported for system, source in CASES)
    ids = [block["_id"] for block in _imported_blocks(workspace)]
    assert total == sum(EXPECTED_RECORDS.values())
    assert len(ids) == total
    assert len(set(ids)) == total


# ---------------------------------------------------------------------------
# Acceptance gate — idempotency (zero duplicates on re-run)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system,source", CASES)
def test_reimport_is_idempotent(workspace: str, system: str, source: str) -> None:
    first = run_import(workspace, system, source)
    corpus = Path(workspace) / IMPORTED_CORPUS_FILE
    after_first = corpus.read_bytes()

    second = run_import(workspace, system, source)

    assert second.parsed == first.parsed
    assert second.imported == 0
    assert second.skipped_existing == first.parsed
    assert second.block_ids == ()
    assert corpus.read_bytes() == after_first

    ids = [block["_id"] for block in _imported_blocks(workspace)]
    assert len(ids) == first.parsed
    assert len(set(ids)) == len(ids)


def test_a_relocated_tree_imports_to_the_same_block_ids(tmp_path: Path) -> None:
    """Block ids follow content + relative path, never the absolute path."""
    ids = []
    for name in ("first", "second"):
        copy = tmp_path / name / "vault"
        shutil.copytree(VAULT, copy)
        ws = tmp_path / f"ws-{name}"
        (ws / "memory").mkdir(parents=True)
        (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
        ids.append(run_import(str(ws), "markdown", str(copy)).block_ids)
    assert ids[0] == ids[1]


def test_dry_run_writes_nothing(workspace: str) -> None:
    result = run_import(workspace, "agentmem", AGENT_MEMORY, dry_run=True)
    assert result.dry_run is True
    assert result.imported == EXPECTED_RECORDS["agentmem"]
    assert not (Path(workspace) / IMPORTED_CORPUS_FILE).exists()


# ---------------------------------------------------------------------------
# Acceptance gate — metadata + wikilinks reach the block (nothing dropped)
# ---------------------------------------------------------------------------


def test_vault_front_matter_and_links_land_on_the_block(workspace: str) -> None:
    run_import(workspace, "markdown", VAULT)
    block = _by_external_id(workspace)["notes/architecture.md"]
    metadata = str(block["Metadata"])
    assert "title=Storage architecture" in metadata
    assert "tags=architecture,storage" in metadata
    assert "path=notes/architecture.md" in metadata
    assert str(block["Links"]) == "connection-pool-outage, nightly-compaction"


def test_auto_memory_front_matter_lands_on_the_block(workspace: str) -> None:
    run_import(workspace, "agentmem", AGENT_MEMORY)
    blocks = _by_external_id(workspace)
    note = blocks["reference_pool_health_checks.md"]
    metadata = str(note["Metadata"])
    assert "name=reference_pool_health_checks" in metadata
    assert "metadata.type=reference" in metadata
    assert "type=reference" in metadata
    assert "section=memory" in metadata
    assert "description=Advisory pool health checks" in metadata
    assert str(note["Links"]) == "feedback_append_only_writes"
    assert "section=instructions" in str(blocks["OPERATING-MANUAL.md"]["Metadata"])
    assert "section=index" in str(blocks["MEMORY.md"]["Metadata"])


def test_transcript_role_lands_on_the_block(workspace: str) -> None:
    run_import(workspace, "chatjson", CHAT_SESSION)
    block = _by_external_id(workspace)["turn-final"]
    metadata = str(block["Metadata"])
    assert "role=assistant" in metadata
    assert "session=reindex-triage-2026-01-16" in metadata
    assert "resolution=confirmed" in metadata


def test_blocks_without_links_have_no_links_field(workspace: str) -> None:
    run_import(workspace, "chatjson", CHAT_SESSION)
    assert all("Links" not in block for block in _imported_blocks(workspace))


# ---------------------------------------------------------------------------
# Acceptance gate — imported content is recallable
# ---------------------------------------------------------------------------


def test_vault_content_is_withheld_until_released(workspace: str) -> None:
    from mind_mem.recall import recall

    run_import(workspace, "markdown", VAULT)
    hits = recall(workspace, "append-only block store canonical file prefix", limit=10)
    assert not any(str(hit.get("id") or hit.get("_id")).startswith("IMP-markdown-") for hit in hits)


def test_auto_memory_content_is_withheld_until_released(workspace: str) -> None:
    from mind_mem.recall import recall

    run_import(workspace, "agentmem", AGENT_MEMORY)
    hits = recall(workspace, "advisory health checks cannot evict a dead socket after failover", limit=10)
    assert not any(str(hit.get("id") or hit.get("_id")).startswith("IMP-agentmem-") for hit in hits)


def test_transcript_content_is_withheld_until_released(workspace: str) -> None:
    from mind_mem.recall import recall

    run_import(workspace, "chatjson", CHAT_SESSION)
    hits = recall(workspace, "posting-list rebuild shard segment format merge cost", limit=10)
    assert not any(str(hit.get("id") or hit.get("_id")).startswith("IMP-chatjson-") for hit in hits)


def test_multiline_note_body_round_trips(workspace: str) -> None:
    run_import(workspace, "agentmem", AGENT_MEMORY)
    statement = str(_by_external_id(workspace)["feedback_append_only_writes.md"]["Statement"])
    assert "Append-only writes keep the on-disk layout auditable." in statement
    assert "erases the evidence trail recall depends on" in statement
    assert "reviewed rather than applied straight to the file" in statement


# ---------------------------------------------------------------------------
# Acceptance gate — unsupported source shapes fail explicitly
# ---------------------------------------------------------------------------


class TestUnsupportedSources:
    def test_note_importer_given_a_json_file(self, workspace: str) -> None:
        with pytest.raises(ImportParseError, match="not a directory"):
            run_import(workspace, "markdown", CHAT_SESSION)

    def test_dump_importer_given_a_directory(self, workspace: str) -> None:
        with pytest.raises(ImportParseError, match="not a regular file"):
            run_import(workspace, "chatjson", VAULT)

    def test_transcript_importer_given_a_note_tree_dump(self, workspace: str, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"documents": ["x"]}), encoding="utf-8")
        with pytest.raises(ImportParseError, match="no turn array"):
            run_import(workspace, "chatjson", str(bad))

    def test_cli_reports_a_bad_source(self, workspace: str, monkeypatch, capsys) -> None:
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        code = mm_cli.main(["import", "--from", "markdown", CHAT_SESSION])
        assert code == mm_cli.IMPORT_EXIT_BAD_DUMP
        assert "not a directory" in capsys.readouterr().err

    def test_cli_rejects_an_unknown_format(self, workspace: str, monkeypatch) -> None:
        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        with pytest.raises(SystemExit) as excinfo:
            mm_cli.main(["import", "--from", "nosuchformat", VAULT])
        assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Opt-in link edges (default OFF, flag-off byte-identical)
# ---------------------------------------------------------------------------


def _fresh_ws(tmp_path: Path, name: str) -> str:
    ws = tmp_path / name
    (ws / "memory").mkdir(parents=True)
    (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
    return str(ws)


class TestLinkEdgeFlag:
    def test_default_off_materializes_no_edges(self, workspace: str) -> None:
        result = run_import(workspace, "agentmem", AGENT_MEMORY)
        assert result.linked_edges == 0
        assert "linked_edges" not in result.as_dict()

    def test_default_receipt_keys_are_stable(self, workspace: str) -> None:
        payload = run_import(workspace, "markdown", VAULT).as_dict()
        assert set(payload) == {
            "system",
            "source_path",
            "parsed",
            "imported",
            "skipped_existing",
            "skipped_near_duplicate",
            "block_ids",
            "dry_run",
            "batch",
            "status",
        }
        # The receipt states the quarantine out loud: a caller reading it
        # cannot mistake an import for content that is live in recall.
        assert payload["status"] == QUARANTINE_STATUS
        assert payload["batch"].startswith("IMPB-markdown-")

    def test_flag_off_is_byte_identical_to_the_no_flag_call(self, tmp_path: Path) -> None:
        corpora = []
        for name in ("a", "b"):
            ws = _fresh_ws(tmp_path, name)
            kwargs = {"link_edges": False} if name == "b" else {}
            run_import(ws, "agentmem", AGENT_MEMORY, **kwargs)  # type: ignore[arg-type]
            corpora.append((Path(ws) / IMPORTED_CORPUS_FILE).read_bytes())
        assert corpora[0] == corpora[1]

    def test_flag_on_leaves_the_corpus_byte_identical(self, tmp_path: Path) -> None:
        """Edges land in the lineage graph — never in the corpus."""
        corpora = []
        for name, flag in (("off", False), ("on", True)):
            ws = _fresh_ws(tmp_path, name)
            result = run_import(ws, "agentmem", AGENT_MEMORY, link_edges=flag)
            corpora.append((Path(ws) / IMPORTED_CORPUS_FILE).read_bytes())
            if flag:
                assert result.linked_edges > 0
                assert result.as_dict()["linked_edges"] == result.linked_edges
        assert corpora[0] == corpora[1]

    def test_flag_on_writes_resolvable_cites_edges(self, tmp_path: Path) -> None:
        from mind_mem.block_lineage import block_lineage

        ws = _fresh_ws(tmp_path, "edges")
        result = run_import(ws, "agentmem", AGENT_MEMORY, link_edges=True)
        source = next(b for b in _imported_blocks(ws) if b.get("ExternalId") == "reference_pool_health_checks.md")
        lineage = block_lineage(ws, str(source["_id"]), 1)
        neighbours = {edge.block_id for edge in lineage.edges}
        assert all(edge.kind == "cites" for edge in lineage.edges)
        target = next(b for b in _imported_blocks(ws) if b.get("ExternalId") == "feedback_append_only_writes.md")
        assert str(target["_id"]) in neighbours
        assert result.linked_edges >= 3

    def test_flag_on_is_deterministic(self, tmp_path: Path) -> None:
        counts = {
            run_import(_fresh_ws(tmp_path, f"det{index}"), "agentmem", AGENT_MEMORY, link_edges=True).linked_edges for index in range(2)
        }
        assert len(counts) == 1

    def test_dry_run_never_writes_edges(self, workspace: str) -> None:
        assert run_import(workspace, "agentmem", AGENT_MEMORY, link_edges=True, dry_run=True).linked_edges == 0


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system,source", CASES)
def test_cli_end_to_end_is_idempotent(workspace: str, system: str, source: str, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)

    assert mm_cli.main(["import", "--from", system, source]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["system"] == system
    assert payload["imported"] == EXPECTED_RECORDS[system]

    assert mm_cli.main(["import", "--from", system, source]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["imported"] == 0
    assert second["skipped_existing"] == EXPECTED_RECORDS[system]


def test_cli_link_edges_flag(workspace: str, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    assert mm_cli.main(["import", "--from", "markdown", VAULT, "--link-edges"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["linked_edges"] >= 2
