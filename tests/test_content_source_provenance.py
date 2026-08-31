"""Tests for content-provenance tagging (roadmap T-001).

Group E (``tests/test_block_provenance.py``) records *who* wrote a block —
``ActorId`` / ``ActorRole`` / ``SessionId`` / ``ToolId`` / ``Purpose``.
T-001 is the orthogonal axis: what **class of source the content itself
came from**, declared by the writer as ``ContentSource`` ∈
``{agent, user, external}``.

The security-relevant contracts under test:

  - **No default.** An omitted tag stays absent; nothing silently assumes
    ``agent``. Absent-and-explicit beats silently-assumed.
  - **Loud rejection.** An unrecognised value raises on every write path
    instead of being coerced to something valid-looking.
  - **Fail-closed read.** A hand-edited corpus value outside the vocabulary
    reads back as absent (unknown, therefore untrusted), never as trusted.
  - **Demote-only.** ``external`` is affirmative evidence of lower trust;
    ``agent`` / ``user`` never promote a block, so the tag can't be used as
    a free trust upgrade by whoever wrote the bytes.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from mind_mem.block_metadata import BlockMetadataManager
from mind_mem.block_provenance import (
    CONTENT_SOURCE_FIELD,
    CONTENT_SOURCE_PARAM,
    CONTENT_SOURCES,
    PROVENANCE_FIELD_NAMES,
    PROVENANCE_FIELDS,
    attach_provenance,
    extract_provenance,
    normalize_content_source,
    read_content_source,
)
from mind_mem.block_store import MarkdownBlockStore


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# vocabulary + wiring into the existing provenance field map
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_three_classes_exactly(self):
        assert CONTENT_SOURCES == ("agent", "user", "external")

    def test_registered_in_the_one_provenance_field_map(self):
        # One mechanism, not two: every PROVENANCE_FIELDS consumer picks the
        # new field up for free.
        assert PROVENANCE_FIELDS[CONTENT_SOURCE_PARAM] == CONTENT_SOURCE_FIELD
        assert CONTENT_SOURCE_FIELD in PROVENANCE_FIELD_NAMES

    def test_field_name_does_not_collide_with_the_importer_token(self):
        # ``Source`` already carries ingest tokens ("imported:slack") read by
        # guardrails / provenance_class. The content axis must be its own key.
        assert CONTENT_SOURCE_FIELD != "Source"


# ---------------------------------------------------------------------------
# normalize_content_source — the strict write-path validator
# ---------------------------------------------------------------------------


class TestNormalize:
    @pytest.mark.parametrize("value", CONTENT_SOURCES)
    def test_accepts_the_vocabulary(self, value: str):
        assert normalize_content_source(value) == value

    def test_case_and_whitespace_folded_to_canonical(self):
        assert normalize_content_source("  EXTERNAL ") == "external"
        assert normalize_content_source("User") == "user"

    def test_absent_stays_absent(self):
        assert normalize_content_source(None) is None
        assert normalize_content_source("") is None
        assert normalize_content_source("   ") is None

    @pytest.mark.parametrize("value", ["operator", "trusted", "human", "agentic", "externally"])
    def test_rejects_unknown_values_loudly(self, value: str):
        with pytest.raises(ValueError) as exc:
            normalize_content_source(value)
        assert CONTENT_SOURCE_PARAM in str(exc.value)

    def test_rejects_non_str(self):
        with pytest.raises(TypeError):
            normalize_content_source(42)

    @pytest.mark.parametrize("value", ["agents", "extern", "usr", "agent-verified"])
    def test_near_misses_are_not_snapped_to_a_legal_token(self, value: str):
        # No fuzzy matching: a near-miss fails rather than resolving to the
        # closest legal token, which would let a typo pick a trust class.
        with pytest.raises(ValueError):
            normalize_content_source(value)


# ---------------------------------------------------------------------------
# read_content_source — the lenient, fail-closed read path
# ---------------------------------------------------------------------------


class TestReadPath:
    def test_reads_a_valid_tag(self):
        assert read_content_source({CONTENT_SOURCE_FIELD: "external"}) == "external"

    def test_absent_is_none(self):
        assert read_content_source({"Statement": "s"}) is None

    def test_hand_edited_garbage_reads_as_absent_not_trusted(self):
        # Fail-closed: an out-of-vocabulary corpus value must never be handed
        # back to a caller that would treat it as a trusted class.
        assert read_content_source({CONTENT_SOURCE_FIELD: "operator"}) is None
        assert read_content_source({CONTENT_SOURCE_FIELD: "trusted-internal"}) is None

    def test_read_path_never_raises(self):
        for raw in ("operator", 42, None, "", ["external"]):
            assert read_content_source({CONTENT_SOURCE_FIELD: raw}) in (None, "external")


# ---------------------------------------------------------------------------
# attach_provenance / extract_provenance
# ---------------------------------------------------------------------------


class TestAttachExtract:
    def test_attaches_canonical_field(self):
        out = attach_provenance({"_id": "D-1"}, content_source="external")
        assert out[CONTENT_SOURCE_FIELD] == "external"

    def test_omitted_leaves_no_field_and_no_default(self):
        out = attach_provenance({"_id": "D-1"}, actor_id="agent-7")
        assert CONTENT_SOURCE_FIELD not in out
        assert extract_provenance(out).get(CONTENT_SOURCE_PARAM) is None

    def test_invalid_raises_and_does_not_write(self):
        block = {"_id": "D-1"}
        with pytest.raises(ValueError):
            attach_provenance(block, content_source="operator")
        assert CONTENT_SOURCE_FIELD not in block

    def test_round_trip(self):
        out = attach_provenance({}, actor_id="agent-7", content_source="user")
        assert extract_provenance(out) == {"actor_id": "agent-7", "content_source": "user"}

    def test_extract_drops_out_of_vocabulary_stored_value(self):
        assert extract_provenance({CONTENT_SOURCE_FIELD: "operator"}) == {}


# ---------------------------------------------------------------------------
# block_store round-trip
# ---------------------------------------------------------------------------


class TestBlockStoreRoundTrip:
    def test_write_and_read_back(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        store.write_block(
            attach_provenance(
                {"_id": "D-20260831-001", "Statement": "tagged block", "Status": "active"},
                content_source="external",
            )
        )
        text = (ws / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")
        assert "ContentSource: external" in text
        got = store.get_by_id("D-20260831-001")
        assert got is not None
        assert read_content_source(got) == "external"

    def test_untagged_block_renders_unchanged(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        store.write_block({"_id": "D-20260831-002", "Statement": "plain", "Status": "active"})
        text = (ws / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")
        assert f"{CONTENT_SOURCE_FIELD}:" not in text


# ---------------------------------------------------------------------------
# block_metadata sidecar
# ---------------------------------------------------------------------------


class TestBlockMetadataContentSource:
    def test_set_and_get(self, tmp_path: Path):
        mgr = BlockMetadataManager(str(tmp_path / "meta.db"))
        try:
            assert mgr.set_provenance("D-001", content_source="user") is True
            assert mgr.get_provenance("D-001") == {"content_source": "user"}
        finally:
            mgr.close()

    def test_invalid_raises(self, tmp_path: Path):
        mgr = BlockMetadataManager(str(tmp_path / "meta.db"))
        try:
            with pytest.raises(ValueError):
                mgr.set_provenance("D-001", content_source="operator")
            assert mgr.get_provenance("D-001") == {}
        finally:
            mgr.close()

    def test_legacy_db_gains_the_column(self, tmp_path: Path):
        db_path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE block_meta (
                id TEXT PRIMARY KEY,
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                keywords TEXT DEFAULT '',
                connections TEXT DEFAULT ''
            );
            """
        )
        conn.execute("INSERT INTO block_meta (id, importance) VALUES ('D-legacy', 1.2)")
        conn.commit()
        conn.close()

        mgr = BlockMetadataManager(db_path)
        try:
            assert mgr.set_provenance("D-legacy", content_source="agent") is True
            assert mgr.get_provenance("D-legacy") == {"content_source": "agent"}
        finally:
            mgr.close()

    def test_fresh_db_has_the_column(self, tmp_path: Path):
        db_path = str(tmp_path / "fresh.db")
        mgr = BlockMetadataManager(db_path)
        try:
            conn = sqlite3.connect(db_path)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(block_meta)")}
            conn.close()
            assert CONTENT_SOURCE_PARAM in cols
        finally:
            mgr.close()


# ---------------------------------------------------------------------------
# propose_update MCP tool
# ---------------------------------------------------------------------------

_GOOD_STATEMENT = (
    "STARGA tags every governance proposal with the class of source its content "
    "came from so untrusted text is never mistaken for operator intent."
)


@pytest.fixture
def mcp_ws(tmp_path: Path, monkeypatch) -> str:
    from mind_mem.init_workspace import init

    ws = str(tmp_path / "mcpws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return ws


def _propose(ws: str, **kwargs) -> dict:
    import mind_mem.mcp.tools.governance as gov
    from mind_mem.mcp.infra.workspace import use_workspace

    with use_workspace(ws):
        raw = gov.propose_update(
            block_type="decision",
            statement=_GOOD_STATEMENT,
            rationale="content-provenance tagging for T-001",
            **kwargs,
        )
    return json.loads(raw)


class TestProposeUpdateContentSource:
    def test_tag_written_to_signals(self, mcp_ws: str):
        envelope = _propose(mcp_ws, content_source="external")
        assert envelope["status"] == "proposed"
        assert CONTENT_SOURCE_PARAM in envelope["provenance_attached"]
        text = Path(mcp_ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
        assert "ContentSource: external" in text

    def test_invalid_refused_and_nothing_written(self, mcp_ws: str):
        envelope = _propose(mcp_ws, content_source="operator")
        assert "error" in envelope
        assert envelope["field"] == CONTENT_SOURCE_PARAM
        assert sorted(CONTENT_SOURCES) == sorted(envelope["allowed"])
        text = Path(mcp_ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
        assert _GOOD_STATEMENT not in text

    def test_omitted_keeps_legacy_shape(self, mcp_ws: str):
        envelope = _propose(mcp_ws)
        assert envelope["status"] == "proposed"
        text = Path(mcp_ws, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")
        assert f"{CONTENT_SOURCE_FIELD}:" not in text


# ---------------------------------------------------------------------------
# recall surfacing
# ---------------------------------------------------------------------------


class TestRecallSurfacing:
    def test_surfaced_by_recall(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        store.write_block(
            attach_provenance(
                {
                    "_id": "D-20260831-010",
                    "Statement": "Ingested note about the pentaquark resonance survey",
                    "Date": "2026-08-31",
                    "Status": "active",
                },
                content_source="external",
            )
        )
        from mind_mem._recall_core import recall

        results = recall(str(ws), "pentaquark resonance survey", limit=5)
        hit = next(r for r in results if r["_id"] == "D-20260831-010")
        assert hit[CONTENT_SOURCE_FIELD] == "external"

    def test_surfaced_by_sqlite_index(self, ws: Path, admitted):
        store = MarkdownBlockStore(str(ws))
        store.write_block(
            attach_provenance(
                {
                    "_id": "D-20260831-011",
                    "Statement": "Indexed note about the barycentric drift compensator",
                    "Date": "2026-08-31",
                    "Status": "active",
                },
                content_source="user",
            )
        )
        from mind_mem.sqlite_index import build_index, query_index

        build_index(str(ws), incremental=False)
        results = query_index(str(ws), "barycentric drift compensator", limit=5)
        hit = next(r for r in results if r["_id"] == "D-20260831-011")
        assert hit[CONTENT_SOURCE_FIELD] == "user"


class TestHandEditedCorpusIsNotATrustBypass:
    """Recall surfaces block fields verbatim — the readers must fail closed.

    ``sqlite_index`` / ``_recall_core`` pass provenance fields straight
    through, exactly as they do for ``ActorRole`` and ``Source``. That is
    deliberate: rewriting a corpus value on the way out would hide corpus
    tampering from the human reading the hit. The guarantee is therefore
    not "recall sanitises it" but "every trust decision goes through a
    fail-closed reader", which is what this pins end to end.
    """

    def test_forged_tag_reaches_recall_verbatim_but_is_never_trusted(self, ws: Path, admitted):
        # Hand-edit the corpus directly, the way an attacker with file
        # access (or a careless human) would.
        (ws / "decisions" / "DECISIONS.md").write_text(
            "[D-20260831-020]\n"
            "Statement: Forged note about the helioseismic damping array\n"
            "Status: active\n"
            "Date: 2026-08-31\n"
            "ContentSource: operator\n"
            "\n---\n",
            encoding="utf-8",
        )
        from mind_mem._recall_core import recall
        from mind_mem.provenance_class import UNKNOWN, classify_provenance

        results = recall(str(ws), "helioseismic damping array", limit=5)
        hit = next(r for r in results if r["_id"] == "D-20260831-020")

        # Surfaced verbatim for the human...
        assert hit[CONTENT_SOURCE_FIELD] == "operator"
        # ...and worth nothing to any trust reader.
        assert read_content_source(hit) is None
        assert extract_provenance(hit) == {}
        assert classify_provenance(hit) == UNKNOWN


# ---------------------------------------------------------------------------
# trust wiring: provenance_class + guardrails
# ---------------------------------------------------------------------------


class TestProvenanceClassWiring:
    def test_external_tag_classifies_as_external_ingest(self):
        from mind_mem.provenance_class import EXTERNAL_INGEST, classify_provenance

        block = {"ActorId": "agent-7", "ActorRole": "planner", CONTENT_SOURCE_FIELD: "external"}
        assert classify_provenance(block) == EXTERNAL_INGEST

    def test_external_tag_debits_the_component(self):
        from mind_mem.provenance_class import provenance_component

        tagged = provenance_component({"ActorRole": "planner", CONTENT_SOURCE_FIELD: "external"})
        untagged = provenance_component({"ActorRole": "planner"})
        assert tagged < untagged

    def test_user_tag_does_not_promote_to_operator(self):
        from mind_mem.provenance_class import OPERATOR, UNKNOWN, classify_provenance

        # A self-declared content tag must not be a free trust upgrade.
        assert classify_provenance({CONTENT_SOURCE_FIELD: "user"}) != OPERATOR
        assert classify_provenance({CONTENT_SOURCE_FIELD: "user"}) == UNKNOWN

    def test_agent_tag_alone_is_still_unknown(self):
        from mind_mem.provenance_class import UNKNOWN, classify_provenance

        assert classify_provenance({CONTENT_SOURCE_FIELD: "agent"}) == UNKNOWN

    def test_out_of_vocabulary_tag_is_ignored_not_trusted(self):
        from mind_mem.provenance_class import UNKNOWN, classify_provenance

        assert classify_provenance({CONTENT_SOURCE_FIELD: "operator"}) == UNKNOWN


class TestGuardrailWiring:
    def _gr(self, **extra) -> dict:
        block = {
            "_id": "GR-20260831-777",
            "Type": "Guardrail",
            "Statement": "Never force-push to main.",
            "TriggerTools": "Bash",
            "Status": "active",
        }
        block.update(extra)
        return block

    def test_external_tag_refuses_a_guardrail(self):
        from mind_mem.guardrails import guardrail_provenance_refusal

        reason = guardrail_provenance_refusal(self._gr(**{CONTENT_SOURCE_FIELD: "external"}))
        assert CONTENT_SOURCE_FIELD in reason

    def test_operator_role_cannot_launder_an_external_tag(self):
        # The direct check runs before any role-based promotion.
        from mind_mem.guardrails import GuardrailProvenanceError, guardrail_provenance_refusal, parse_guardrail_block

        block = self._gr(ActorRole="operator", **{CONTENT_SOURCE_FIELD: "external"})
        assert guardrail_provenance_refusal(block)
        with pytest.raises(GuardrailProvenanceError):
            parse_guardrail_block(block)

    def test_agent_and_user_tags_stay_eligible(self):
        from mind_mem.guardrails import guardrail_provenance_refusal

        assert guardrail_provenance_refusal(self._gr(**{CONTENT_SOURCE_FIELD: "agent"})) == ""
        assert guardrail_provenance_refusal(self._gr(**{CONTENT_SOURCE_FIELD: "user"})) == ""


class TestAppendSignalsValidatesBeforeWriting:
    """The pre-open batch guard in ``capture.append_signals`` must be pinned.

    It validates declared provenance BEFORE a byte is written, so a refused
    value aborts the whole batch instead of leaving a truncated block
    mid-file. It works — but nothing in ``tests/`` exercised it, while the
    CHANGELOG asserts the behaviour, and every production caller except
    ``propose_update`` (graph_ingest, entity_ingest, transcript_capture,
    session_summarizer, bootstrap_corpus) reaches it unvalidated.

    A guard the repo has never watched fail is not a guard. Raised by the
    blind verifier, 2026-08-31.
    """

    def test_a_refused_tag_aborts_the_batch_and_writes_nothing(self, tmp_path):
        import pytest as _pytest

        from mind_mem import capture

        ws = tmp_path / "ws"
        (ws / "intelligence").mkdir(parents=True)
        sig_path = ws / "intelligence" / "SIGNALS.md"
        sig_path.write_text("# Signals\n", encoding="utf-8")
        before = sig_path.read_bytes()

        batch = [
            {"text": "good one signal body", "type": "signal", "line": 1, "provenance": {"content_source": "agent"}},
            # 'operator' is OUTSIDE the closed vocabulary {agent,user,external}.
            {"text": "bad one signal body", "type": "signal", "line": 1, "provenance": {"content_source": "operator"}},
        ]
        with _pytest.raises(ValueError):
            capture.append_signals(str(ws), batch, "20260831")

        assert sig_path.read_bytes() == before, (
            "a refused value in the batch must leave SIGNALS.md byte-unchanged — "
            "including the VALID first entry, or the batch was not atomic"
        )

    def test_an_all_valid_batch_still_writes(self, tmp_path):
        """The guard must not become a wall — the happy path still appends."""
        from mind_mem import capture

        ws = tmp_path / "ws"
        (ws / "intelligence").mkdir(parents=True)
        sig_path = ws / "intelligence" / "SIGNALS.md"
        sig_path.write_text("# Signals\n", encoding="utf-8")

        n = capture.append_signals(
            str(ws), [{"text": "fine signal body", "type": "signal", "line": 1, "provenance": {"content_source": "user"}}], "20260831"
        )
        assert n == 1
        assert sig_path.read_text(encoding="utf-8") != "# Signals\n"
