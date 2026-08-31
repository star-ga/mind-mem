"""Regressions for paths that used to report success while doing the wrong thing.

Every test here pins a call that returned a success-shaped answer -- an
``AuditEntry``, a PASS line, ``links_included`` -- from code that had
silently skipped, forked or overwritten the thing it claimed to handle.
Each one fails against the pre-fix source.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# audit_chain -- an unreadable ledger tail is not an empty ledger
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_append_refuses_to_fork_chain_over_corrupt_tail(tmp_path: Path) -> None:
    """A damaged last line must fail the append, not restart at seq 1.

    Before the fix ``_last_entry`` returned None for a corrupt tail
    exactly as it does for an empty chain, so ``append`` wrote seq 1 with
    the genesis prev_hash on top of the existing entries and handed the
    caller an ordinary AuditEntry. Every later entry then chained off the
    fork while every writer reported success.
    """
    from mind_mem.audit_chain import AuditChain, AuditChainCorruptedError

    chain = AuditChain(str(tmp_path))
    for _ in range(3):
        chain.append("create_block", "a.md", agent="tester", reason="setup")
    ok, errors = chain.verify()
    assert ok, errors

    chain_file = tmp_path / ".mind-mem-audit" / "chain.jsonl"
    with open(chain_file, "a", encoding="utf-8") as fh:
        fh.write("{\n")
    before = chain_file.read_text(encoding="utf-8")

    with pytest.raises(AuditChainCorruptedError):
        chain.append("update_field", "a.md", agent="tester", reason="after damage")

    # Nothing was written, so the damage stays a single locatable break
    # instead of the root of a second chain.
    assert chain_file.read_text(encoding="utf-8") == before


@pytest.mark.unit
def test_corrupt_chain_surfaces_as_an_import_refusal(tmp_path: Path) -> None:
    """The refusal must reach callers that already convert ledger failures.

    ``record_import_in_chain`` documents ImportQuarantineError as the way a
    caller learns an import could not be recorded; it catches
    OSError/ValueError, so the new error is an OSError rather than a fresh
    class that would slip past that handler and bypass the contract.
    """
    from mind_mem.audit_chain import AuditChain
    from mind_mem.importers.quarantine import ImportQuarantineError, record_import_in_chain

    AuditChain(str(tmp_path)).append("create_block", "a.md")
    with open(tmp_path / ".mind-mem-audit" / "chain.jsonl", "a", encoding="utf-8") as fh:
        fh.write("{\n")

    with pytest.raises(ImportQuarantineError):
        record_import_in_chain(
            str(tmp_path),
            system="notes",
            source_path="/src",
            batch="b1",
            block_ids=["N-1"],
            corpus_file="memory/NOTES.md",
        )


@pytest.mark.unit
def test_append_still_starts_a_fresh_chain_at_seq_one(tmp_path: Path) -> None:
    """The fail-closed guard must not break the genuinely-empty case."""
    from mind_mem.audit_chain import _GENESIS_HASH, AuditChain

    chain = AuditChain(str(tmp_path))
    entry = chain.append("create_block", "a.md")
    assert entry.seq == 1
    assert entry.prev_hash == _GENESIS_HASH


@pytest.mark.unit
def test_genesis_hash_is_the_documented_constant() -> None:
    """The anchor is 64 zeros; the old comment claimed it was a digest."""
    import hashlib

    from mind_mem.audit_chain import _GENESIS_HASH

    assert _GENESIS_HASH == "0" * 64
    assert _GENESIS_HASH != hashlib.sha256(b"mind-mem-genesis").hexdigest()


# ---------------------------------------------------------------------------
# importers.fs_source -- front matter nesting and the key cap
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_nested_sibling_key_does_not_overwrite_top_level_field() -> None:
    """Returning from depth 2 to depth 1 pops one level, not all of them.

    Before the fix the whole prefix was cleared, so ``metadata.description``
    was written as a bare ``description`` and silently replaced the real
    top-level one -- which parse_agent_memory reads as the block's
    authoritative description.
    """
    from mind_mem.importers.fs_source import parse_front_matter

    text = "---\ndescription: REAL top-level\nmetadata:\n  nested:\n    a: 1\n  description: injected-from-nested\n---\nbody\n"
    front, body = parse_front_matter(text)

    assert front["description"] == "REAL top-level"
    assert front["metadata.description"] == "injected-from-nested"
    assert front["metadata.nested.a"] == "1"
    assert body == "body\n"


@pytest.mark.unit
def test_sibling_scalar_after_nested_block_keeps_its_parent() -> None:
    from mind_mem.importers.fs_source import parse_front_matter

    text = "---\nmetadata:\n  type: note\n  sub:\n    x: 1\n  other: v\n---\n"
    front, _ = parse_front_matter(text)

    assert "other" not in front
    assert front["metadata.other"] == "v"
    assert front["metadata.type"] == "note"
    assert front["metadata.sub.x"] == "1"


@pytest.mark.unit
def test_list_keys_obey_the_front_matter_key_cap() -> None:
    """``k:`` / ``- item`` pairs mint keys and must be capped like scalars."""
    from mind_mem.importers.fs_source import _MAX_FRONT_MATTER_KEYS, parse_front_matter

    lines = ["---"]
    lines += [f"k{i}: v" for i in range(_MAX_FRONT_MATTER_KEYS)]
    for i in range(10):
        lines += [f"list{i}:", "  - a"]
    lines += ["---", ""]
    front, _ = parse_front_matter("\n".join(lines))

    assert len(front) == _MAX_FRONT_MATTER_KEYS


@pytest.mark.unit
def test_existing_list_key_can_still_grow_at_the_cap() -> None:
    """The cap bounds distinct keys, not items appended to one already there."""
    from mind_mem.importers.fs_source import _MAX_FRONT_MATTER_KEYS, parse_front_matter

    lines = ["---", "tags:", "  - a", "  - b"]
    lines += [f"k{i}: v" for i in range(_MAX_FRONT_MATTER_KEYS)]
    lines += ["---", ""]
    front, _ = parse_front_matter("\n".join(lines))

    assert front["tags"] == "a,b"


# ---------------------------------------------------------------------------
# telemetry -- opentelemetry-api and opentelemetry-sdk are two distributions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_tracing_noops_when_only_the_otel_api_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """api-without-sdk is a supported install shape; init_tracing must no-op.

    The old gate probed only ``opentelemetry`` / ``opentelemetry.trace``
    (both in opentelemetry-api) and then imported ``opentelemetry.sdk.*``
    (a separate distribution), so on such a host the documented usage
    example raised ModuleNotFoundError.
    """
    from mind_mem import telemetry

    monkeypatch.setattr(telemetry, "_HAS_OTEL", True)
    monkeypatch.setattr(telemetry, "_HAS_OTEL_SDK", False)
    # Guarantee the sdk import would blow up if it were attempted, and that
    # the idempotency short-circuit cannot be what makes this pass.
    monkeypatch.setattr(telemetry, "_otel_initialized", False)
    monkeypatch.setattr(telemetry, "_tracer", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk", None)

    telemetry.init_tracing()
    telemetry.init_tracing(endpoint="http://localhost:4317")

    assert telemetry._otel_initialized is False
    assert telemetry._tracer is None


@pytest.mark.unit
def test_sdk_probe_is_separate_from_the_api_probe() -> None:
    from mind_mem import telemetry

    expected = telemetry._HAS_OTEL and importlib.util.find_spec("opentelemetry.sdk.trace") is not None
    assert telemetry._HAS_OTEL_SDK is expected


# ---------------------------------------------------------------------------
# validate_py -- zero blocks is not a passing corpus
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_zero_block_corpus_files_assert_nothing(tmp_path: Path) -> None:
    """A file that parses to no blocks must not emit per-block PASS lines.

    Every per-block check is a comprehension over ``blocks`` and is
    vacuously true at length zero, so a wiped corpus used to report a full
    row of passes ("All 0 Decisions IDs match ...").
    """
    from mind_mem.init_workspace import init
    from mind_mem.validate_py import Validator

    ws = tmp_path / "ws"
    init(str(ws))

    validator = Validator(str(ws))
    rc = validator.run()

    report = "\n".join(validator.lines)
    assert "All 0 " not in report
    assert "in all 0 blocks" not in report
    assert "0 blocks" in report  # the emptiness itself is reported
    # A freshly initialised workspace legitimately holds no blocks, so the
    # emptiness is a warning and the exit status stays clean.
    assert rc == 0
    assert validator.issues == 0
    assert validator.warnings > 0


# ---------------------------------------------------------------------------
# mcp vault_sync -- the knowledge graph lives under memory/
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vault_sync_finds_the_knowledge_graph_in_its_real_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """include_links=True must actually emit links.

    The tool resolved ``<ws>/knowledge_graph.db``; every other tool (and
    the writer) uses ``<ws>/memory/knowledge_graph.db``, which is the only
    location the product ever creates. The isfile guard had no else, so a
    caller who asked for links got a link-free note and success.
    """
    from mind_mem.knowledge_graph import KnowledgeGraph, Predicate
    from mind_mem.mcp.tools.agent import vault_sync

    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    with KnowledgeGraph(str(ws / "memory" / "knowledge_graph.db")) as kg:
        kg.add_edge("D-20260101-001", Predicate.RELATED_TO, "PRJ-mind-mem", source_block_id="D-20260101-001")

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    monkeypatch.setenv("MIND_MEM_VAULT_ALLOWLIST", str(vault))

    payload: dict[str, Any] = json.loads(
        vault_sync(
            vault_root=str(vault),
            block_id="D-20260101-001",
            relative_path="notes/d1.md",
            body="text",
            include_links=True,
        )
    )

    assert "error" not in payload, payload
    assert payload["links_included"] is True
    written = Path(payload["written"]).read_text(encoding="utf-8")
    assert "## Links" in written
    assert "[[prj-mind-mem]]" in written  # the registry slugifies entity names


# ---------------------------------------------------------------------------
# frame_fields -- an unreadable source is not an absent one
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unreadable_frame_source_is_warned_about(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``os.path.isfile`` answers False to missing AND to permission-denied.

    The docstring promises an unreadable source is skipped *with a
    warning*; the isfile test skipped it silently, so a frames file the
    process cannot read was indistinguishable from a workspace that
    declares none.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores the permission bits this test relies on")

    from mind_mem import frame_fields

    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "FRAMES.md").write_text("[TF-1]\nTitle: x\n", encoding="utf-8")
    os.chmod(locked, 0o000)

    events: list[tuple[str, dict[str, Any]]] = []

    class _Recorder:
        def warning(self, event: str, **fields: Any) -> None:
            events.append((event, fields))

    monkeypatch.setattr(frame_fields, "_log", _Recorder())
    try:
        blocks = frame_fields.load_blocks(str(tmp_path), ["locked/FRAMES.md"], "TF-")
    finally:
        os.chmod(locked, stat.S_IRWXU)

    assert blocks == ()
    assert [name for name, _ in events] == ["frame_source_unreadable"]


@pytest.mark.unit
def test_absent_frame_source_stays_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No frames file is the default state, so it must not warn."""
    from mind_mem import frame_fields

    events: list[str] = []

    class _Recorder:
        def warning(self, event: str, **fields: Any) -> None:
            events.append(event)

    monkeypatch.setattr(frame_fields, "_log", _Recorder())
    assert frame_fields.load_blocks(str(tmp_path), ["frames/FRAMES.md"], "TF-") == ()
    assert events == []


# ---------------------------------------------------------------------------
# init_workspace -- an incomplete install must not exit 0
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_main_fails_when_package_data_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Missing templates were counted as creations and main() returned 0.

    ``mind-mem-init /ws && next-step`` then proceeded against a workspace
    with no corpus scaffold, because nothing in the return value
    distinguished a built workspace from a hollow one.
    """
    from mind_mem import init_workspace

    empty_templates = tmp_path / "no-templates"
    empty_templates.mkdir()
    monkeypatch.setattr(init_workspace, "TEMPLATE_DIR", str(empty_templates))

    ws = tmp_path / "ws"
    rc = init_workspace.main([str(ws)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "INCOMPLETE" in captured.err
    # The MISSING entries are no longer counted as things that were created.
    assert init_workspace.MISSING_MARKER not in captured.out


@pytest.mark.unit
def test_init_main_returns_zero_on_a_complete_install(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from mind_mem import init_workspace

    rc = init_workspace.main([str(tmp_path / "ws")])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Done." in captured.out


# ---------------------------------------------------------------------------
# v4 cognitive kernel -- the typo case the docstring claimed to cover
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unknown_kernel_name_names_the_valid_kernels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mistyped kernel name must say what the valid names are.

    The docstring promised the available kernels in the message; a bare
    ``KernelKind(kernel)`` raised ValueError("'x' is not a valid
    KernelKind") with no list, which is exactly the typo case it claimed
    to cover.
    """
    from mind_mem.v4.cognitive_kernel import FLAG, mind_recall

    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": {FLAG: {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))

    with pytest.raises(ValueError) as excinfo:
        mind_recall(str(tmp_path), "q", kernel="recnet_first")

    message = str(excinfo.value)
    assert "recnet_first" in message
    assert "valid kernels" in message
    assert "surprise_weighted" in message
