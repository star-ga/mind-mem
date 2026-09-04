# Copyright 2026 STARGA, Inc.
"""A list-valued ``Tags`` field used to brick the reindex for the whole workspace.

``MarkdownBlockStore.write_block`` accepts a block whose ``Tags`` is a list.
``_render_block`` writes any list-valued field in the corpus's list form::

    Tags:
    - probe

…and the block parser reads that back as ``["probe"]``. Every consumer
downstream of the parser was written for the string form.

**Measured on 5.0.2 at 2697baf**, one governed write followed by
``build_index``::

    write_block({..., "Tags": ["probe"]})   -> landed
    parse -> Tags: ['probe']
    build_index(ws) -> AttributeError: 'list' object has no attribute 'split'
        _recall_detection.py:620 in _parse_speaker_from_tags
        via sqlite_index.py:802 in _insert_block

One malformed block, and the reindex for **every** block in the workspace
raises. Not one block skipped — the whole index build.

The fix is at the layer the shape belongs to:
:func:`~mind_mem._recall_detection.normalise_tags` renders the field as the
comma-separated string its readers expect, ``_parse_speaker_from_tags``
normalises before splitting, and the five sibling readers that put the raw
value into a result payload, a SQLite parameter or an FTS5 row normalise
too. A guard added only where the traceback pointed would have left the
list in five other places, two of which are parameter bindings.

Every assertion here is paired: the reindex is shown succeeding beside the
mutation twin that reproduces the crash (:class:`TestMutationTwin`), and the
string form is shown behaving **identically** before and after, because a
fix that changed the existing corpus's speaker extraction would be a
regression wearing a fix's clothes.

**Honest scope.** The measured reach is the store API — ``write_block`` with
a list-valued ``Tags``. The sanctioned ``propose_update`` path passes tags
as a comma-separated string (``governance.py:304``), so whether a proposal
can produce this block is UNVERIFIED and is not claimed here.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem._recall_context import _block_to_result
from mind_mem._recall_detection import _parse_speaker_from_tags, normalise_tags
from mind_mem.governance_gate import get_gate
from mind_mem.init_workspace import init as init_workspace
from mind_mem.sqlite_index import _extract_fts_fields, build_index
from mind_mem.storage import get_block_store

BLOCK_ID = "D-20260903-001"


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init_workspace(ws)
    yield ws


def _write(workspace: str, tags: Any) -> None:
    """Land one block through the governed path, carrying *tags* verbatim."""
    store = get_block_store(workspace)
    with get_gate(workspace).admit_proposal("P-20260903-001", "probe"):
        store.write_block(
            {
                "_id": BLOCK_ID,
                "Statement": "a probe block about pineapple protocol rollout",
                "Status": "active",
                "Type": "Decision",
                "Tags": tags,
            }
        )


def _index_row(workspace: str) -> dict[str, Any]:
    path = os.path.join(workspace, ".mind-mem-index", "recall.db")
    assert os.path.isfile(path), f"no index at {path}; build_index wrote nothing"
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM blocks WHERE id = ?", (BLOCK_ID,)).fetchone()
    finally:
        conn.close()
    assert row is not None, f"{BLOCK_ID} is not in the index at all"
    return dict(row)


# ---------------------------------------------------------------------------
# The probe — the exact shape that bricked the reindex
# ---------------------------------------------------------------------------


class TestTheListFormRoundTrips:
    @pytest.mark.unit
    def test_the_store_really_writes_and_parses_the_list_form(self, workspace: str) -> None:
        """The premise, checked. Without it every test below is hypothetical."""
        _write(workspace, ["probe"])

        rendered = Path(workspace, "decisions", "DECISIONS.md").read_text(encoding="utf-8")
        assert "Tags:\n- probe" in rendered, f"the store no longer renders list form; re-measure. Got:\n{rendered}"

        block = get_block_store(workspace).get_by_id(BLOCK_ID)
        assert block is not None
        assert block["Tags"] == ["probe"], f"the parser no longer returns a list ({block['Tags']!r}); this file is measuring nothing"

    @pytest.mark.unit
    def test_build_index_survives_the_list_form(self, workspace: str) -> None:
        """The defect: this raised AttributeError before the fix."""
        _write(workspace, ["probe"])

        build_index(workspace)

        row = _index_row(workspace)
        assert row["tags"] == "probe", f"the index holds {row['tags']!r}; a list reached the SQLite binding"
        assert isinstance(row["tags"], str)

    @pytest.mark.unit
    def test_one_malformed_block_does_not_brick_its_neighbours(self, workspace: str) -> None:
        """The blast radius, which is what made this worth fixing.

        The crash was inside the per-block insert of a whole-workspace loop,
        so the *other* blocks never got indexed either. Two blocks, one
        malformed; both must be in the index.
        """
        store = get_block_store(workspace)
        with get_gate(workspace).admit_proposal("P-20260903-002", "probe"):
            store.write_block({"_id": BLOCK_ID, "Statement": "list tags here", "Status": "active", "Type": "Decision", "Tags": ["probe"]})
            store.write_block(
                {
                    "_id": "D-20260903-002",
                    "Statement": "an innocent neighbour",
                    "Status": "active",
                    "Type": "Decision",
                    "Tags": "FACT, Caroline",
                }
            )

        build_index(workspace)

        path = os.path.join(workspace, ".mind-mem-index", "recall.db")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            ids = {r[0] for r in conn.execute("SELECT id FROM blocks WHERE id LIKE 'D-%'")}
        finally:
            conn.close()
        assert {BLOCK_ID, "D-20260903-002"} <= ids, f"the neighbour was lost with the malformed block: {sorted(ids)}"

    @pytest.mark.unit
    def test_the_index_json_blob_still_carries_the_block(self, workspace: str) -> None:
        """Normalising the payload must not have eaten the stored block."""
        _write(workspace, ["probe", "rollout"])
        build_index(workspace)

        blob = json.loads(_index_row(workspace)["json_blob"])
        assert blob["_id"] == BLOCK_ID
        assert blob["Tags"] == ["probe", "rollout"], "the indexed block no longer records what the corpus holds"


# ---------------------------------------------------------------------------
# The normaliser — total, and a no-op on the shape the corpus already holds
# ---------------------------------------------------------------------------


class TestNormaliseTags:
    @pytest.mark.unit
    @pytest.mark.parametrize("text", ["", "FACT, Caroline", "  spaced , out  ", "single", "a,b,c"])
    def test_a_string_is_returned_unchanged(self, text: str) -> None:
        """Byte-identical for every corpus that never held a list.

        ``is`` and not ``==``: the guarantee is that the existing path does
        not merely produce an equal value, it produces the same object.
        """
        assert normalise_tags(text) is text

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (["probe"], "probe"),
            (["FACT", "Caroline"], "FACT, Caroline"),
            (("FACT", "Caroline"), "FACT, Caroline"),
            ([" FACT ", " Caroline "], "FACT, Caroline"),
            (["FACT", "", None, "Caroline"], "FACT, None, Caroline"),
            ([], ""),
            (None, ""),
            (0, ""),
            ({}, ""),
            (42, "42"),
        ],
    )
    def test_every_other_shape_folds_into_the_string_form(self, value: object, expected: str) -> None:
        """Total: a reader of an already-stored block cannot reject its input.

        ``None`` inside a list stringifies rather than vanishing — dropping
        it would silently renumber the tag positions the speaker parser
        counts from, which is a worse failure than an ugly tag.

        Every falsy value reads as the empty field, which is what the
        ``block.get("Tags", "") or ""`` at the call sites already did — so
        the readers that used to spell it that way are unchanged for every
        input they could previously see.
        """
        assert normalise_tags(value) == expected

    @pytest.mark.unit
    def test_the_result_is_always_a_string(self) -> None:
        for value in ("", "x", [], ["a"], ("a",), None, 0, 1.5, {"a": 1}):
            assert isinstance(normalise_tags(value), str), f"{value!r} normalised to a non-string"


class TestTheSpeakerParserIsUnchangedOnStrings:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("tags", "speaker"),
        [
            ("FACT, Caroline", "Caroline"),
            ("EVENT, Nikolai, extra", "Nikolai"),
            ("FACT", ""),
            ("", ""),
            ("FACT, lowercase", ""),
            ("FACT, PLAN", ""),
        ],
    )
    def test_the_string_form_behaves_exactly_as_before(self, tags: str, speaker: str) -> None:
        """The regression guard: the fix must not move the existing corpus."""
        assert _parse_speaker_from_tags(tags) == speaker

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("tags", "speaker"),
        [
            (["FACT", "Caroline"], "Caroline"),
            (["probe"], ""),
            ([], ""),
            (None, ""),
        ],
    )
    def test_the_list_form_answers_the_same_question(self, tags: object, speaker: str) -> None:
        """A list and the string it renders as must give one answer."""
        assert _parse_speaker_from_tags(tags) == speaker

    @pytest.mark.unit
    def test_the_two_shapes_agree(self) -> None:
        """Stated as an equality rather than two separate expectations."""
        for parts in (["FACT", "Caroline"], ["EVENT", "Nikolai", "extra"], ["FACT"], []):
            assert _parse_speaker_from_tags(parts) == _parse_speaker_from_tags(", ".join(parts))


# ---------------------------------------------------------------------------
# The sibling readers — the ones a call-site-only guard would have missed
# ---------------------------------------------------------------------------


class TestTheSiblingReadersNormaliseToo:
    @pytest.mark.unit
    def test_the_fts_row_holds_a_string(self) -> None:
        """``_extract_fts_fields`` binds ``tags`` into an FTS5 insert."""
        fields = _extract_fts_fields({"_id": BLOCK_ID, "Statement": "s", "Tags": ["probe", "rollout"]})
        assert fields["tags"] == "probe, rollout"
        assert isinstance(fields["tags"], str)

    @pytest.mark.unit
    def test_a_result_payload_carries_a_string(self) -> None:
        """``_block_to_result`` puts ``tags`` in front of a caller."""
        result = _block_to_result({"_id": BLOCK_ID, "Statement": "s", "Tags": ["probe"]})
        assert result["tags"] == "probe"
        assert result["speaker"] == ""

    @pytest.mark.unit
    def test_the_postgres_hit_shape_carries_a_string(self) -> None:
        """``_pg_block_to_hit`` is the fourth reader, on another backend."""
        from mind_mem._recall_core import _pg_block_to_hit

        hit = _pg_block_to_hit({"_id": BLOCK_ID, "Statement": "s", "Tags": ["FACT", "Caroline"]}, 1.0)
        assert hit["tags"] == "FACT, Caroline"
        assert hit["speaker"] == "Caroline"


# ---------------------------------------------------------------------------
# Mutation twin — a gate never observed failing is not a gate
# ---------------------------------------------------------------------------


class TestMutationTwin:
    @pytest.mark.unit
    def test_removing_the_normalisation_puts_the_crash_back(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Restore the pre-fix shape and the measured AttributeError returns.

        Both bindings are patched, because the fix is two layers deep on
        purpose: ``sqlite_index`` normalises before it binds, and the parser
        normalises before it splits. Reverting one alone would leave the
        other holding — which is the point of fixing both, and the reason
        this twin has to revert both to reproduce.
        """
        import mind_mem.sqlite_index as sqlite_index

        def _pre_fix_parser(tags_str: Any) -> str:
            if not tags_str:
                return ""
            parts = [t.strip() for t in tags_str.split(",")]
            return parts[1] if len(parts) > 1 else ""

        monkeypatch.setattr(sqlite_index, "normalise_tags", lambda value: value)
        monkeypatch.setattr(sqlite_index, "_parse_speaker_from_tags", _pre_fix_parser)

        _write(workspace, ["probe"])
        with pytest.raises(AttributeError, match="'list' object has no attribute 'split'"):
            build_index(workspace)

    @pytest.mark.unit
    def test_and_the_unmutated_build_succeeds_on_the_same_block(self, workspace: str) -> None:
        """The other half of the twin, on the same input, in the same file."""
        _write(workspace, ["probe"])
        build_index(workspace)
        assert _index_row(workspace)["tags"] == "probe"
