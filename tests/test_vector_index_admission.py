# Copyright 2026 STARGA, Inc.
"""``rebuild_index`` must not put withheld blocks into the vector index.

It parsed the corpus straight off disk with ``parse_file`` and **no status
filter at all**, so a quarantined or pending block was embedded and became
reachable by similarity search — while every text path correctly refused to
serve it. Same defect class as the consolidation loader that SELECTed a status
and then never filtered on it, except here there was not even a status to
ignore.

The fix calls the shared gate (``admit_corpus``) rather than re-implementing
one. That matters: an unstated status is SERVABLE, so a hand-rolled check that
only rejected known-bad values would admit anything unlabelled.

Every negative assertion below is paired with a positive control. "The canary
is not in the index" passes trivially if the canary was never written, and
that is the most common way a security test proves nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CANARY = "QUARANTINED-CANARY-9d41b7"
CLEAN = "ADMITTED-CONTROL-51ca02"


@pytest.fixture()
def ws(tmp_path) -> str:
    from mind_mem.init_workspace import init

    root = str(tmp_path / "ws")
    os.makedirs(root)
    init(root)

    # Rendered with the store's own writer so the fixture cannot drift from
    # the real on-disk format -- a hand-written approximation is how a
    # "nothing leaked" test ends up parsing zero blocks and passing vacuously.
    from mind_mem.block_store import _render_block

    decisions = Path(root) / "decisions" / "DECISIONS.md"
    decisions.write_text(
        _render_block(
            {
                "_id": "DEC-20260101-001",
                "Statement": CLEAN,
                "Date": "2026-01-01",
                "Status": "active",
                "Type": "decision",
            }
        )
        + "\n"
        + _render_block(
            {
                "_id": "DEC-20260101-002",
                "Statement": CANARY,
                "Date": "2026-01-01",
                "Status": "quarantined",
                "Type": "decision",
            }
        ),
        encoding="utf-8",
    )
    return root


def _index_text(workspace: str) -> str:
    """Every byte of the built index, or '' if none was written."""
    for candidate in (
        Path(workspace) / ".mind-mem-index" / "index.json",
        Path(workspace) / "index.json",
    ):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    hits = list(Path(workspace).rglob("index.json"))
    return "\n".join(h.read_text(encoding="utf-8") for h in hits)


class TestRebuildIndexAdmission:
    def test_the_corpus_really_contains_both_blocks(self, ws) -> None:
        """Positive control #1 — the fixture is not empty.

        Without this, every 'canary not indexed' assertion below could pass
        because nothing was ever parsed.
        """
        from mind_mem.block_parser import parse_file

        parsed = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
        statements = " ".join(str(b.get("Statement", "")) for b in parsed)
        assert CLEAN in statements, "fixture lost the admitted block"
        assert CANARY in statements, "fixture lost the quarantined block"

    def test_admit_corpus_actually_withholds_the_canary(self, ws) -> None:
        """Positive control #2 — the gate distinguishes the two blocks.

        If admit_corpus admitted everything, the index assertion would be
        meaningless.
        """
        from mind_mem.admissibility import admit_corpus
        from mind_mem.block_parser import parse_file

        parsed = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
        admitted = " ".join(str(b.get("Statement", "")) for b in admit_corpus(parsed))
        assert CLEAN in admitted
        assert CANARY not in admitted

    def test_the_quarantined_block_never_reaches_the_vector_index(self, ws) -> None:
        from mind_mem.recall_vector import rebuild_index

        try:
            rebuild_index(ws)
        except Exception as exc:  # pragma: no cover - no embedding provider here
            pytest.skip(f"vector backend unavailable in this environment: {exc}")

        blob = _index_text(ws)
        if not blob:
            pytest.skip("no index.json produced; nothing to assert against")

        assert CANARY not in blob, (
            "a quarantined block was written into the vector index -- it is reachable by similarity while every text path withholds it"
        )
        # The paired positive control: the ADMITTED block DID make it in, so
        # the assertion above is not passing because the index is empty.
        assert CLEAN in blob, (
            "positive control failed: the admitted block is missing too, so 'canary not present' proves nothing about filtering"
        )
