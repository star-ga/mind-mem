"""`backfill` must find a block's text, and must SAY when it cannot.

MEASURED 2026-09-11 against the live 2776-block corpus while running the roadmap's
"Run the backfill over the live corpus" yield measurement:

    corpus blocks loaded: 2776
    blocks_scanned: 34

34 of 2776 — 1.2%. The other 2742 were skipped and the run reported clean.

Cause: the text lookup was `block.get("excerpt") or block.get("content") or
block.get("Statement")` — two lowercase keys and one capitalised. The live corpus
carries **`Excerpt`** (capital E) on 2692 blocks. A case mismatch on one key silently
removed 98.8% of the corpus from every backfill that has ever run, and the yield number
such a run produces looks exactly like a real measurement of the extractor's value.

Two fixes, because either alone leaves the hole:

  1. The lookup is CASE-INSENSITIVE over the known text-bearing field names, so
     `Excerpt` and `excerpt` are the same field — which is what every reader already
     assumed.
  2. `backfill` reports `blocks_without_text`. A scan that drops most of its input must
     say so: "scanned 34" alone cannot be distinguished from "the corpus has 34
     blocks", and that indistinguishability is what let this survive.
"""

from __future__ import annotations

from mind_mem.graph_ingest import backfill

TEXT = "This decision supersedes Retry Policy V1."


def _extract(text: str) -> list[dict]:
    return [{"subject": "X", "predicate": "supersedes", "object": "Y"}] if text.strip() else []


def test_a_capitalised_Excerpt_is_found(tmp_path):
    """The exact field 2692 live blocks use, and the one that was invisible."""
    corpus = [{"_id": "DEC-1", "Excerpt": TEXT}]
    got = backfill(str(tmp_path), corpus=corpus, extract_fn=_extract)
    assert got["blocks_examined"] == 1, got
    assert got["edges_extracted"] == 1, got


def test_every_case_variant_of_every_text_field_is_found(tmp_path):
    """One key was capitalised and one was not, which is how the mismatch hid. Pinning
    all of them keeps the next added field from reintroducing it."""
    for key in ("excerpt", "Excerpt", "content", "Content", "statement", "Statement"):
        corpus = [{"_id": "DEC-1", key: TEXT}]
        got = backfill(str(tmp_path), corpus=corpus, extract_fn=_extract)
        assert got["blocks_examined"] == 1, (key, got)


def test_a_block_with_no_text_at_all_is_COUNTED_not_just_skipped(tmp_path):
    """The structural half. A silent skip is why a 1.2% scan read as a clean run."""
    corpus = [
        {"_id": "DEC-1", "Excerpt": TEXT},
        {"_id": "B-1", "Range": "2026-02-09 .. 2026-02-15"},  # a real live shape
        {"_id": "B-2"},
    ]
    got = backfill(str(tmp_path), corpus=corpus, extract_fn=_extract)
    # `blocks_scanned` counts blocks with IDS and always did; the number that says
    # what was READ is `blocks_examined`. Reporting only the first is what let a
    # 1.2% scan read as a clean run on the live corpus.
    assert got["blocks_scanned"] == 3, got
    assert got["blocks_examined"] == 1, got
    assert got["blocks_without_text"] == 2, got
    assert got["blocks_scanned"] == got["blocks_examined"] + got["blocks_without_text"]


def test_blocks_without_text_is_zero_when_every_block_has_text(tmp_path):
    """POSITIVE CONTROL: the counter must be able to read zero, or it carries no
    information and a reader learns nothing from seeing it."""
    corpus = [{"_id": "DEC-1", "Excerpt": TEXT}, {"_id": "DEC-2", "content": TEXT}]
    got = backfill(str(tmp_path), corpus=corpus, extract_fn=_extract)
    assert got["blocks_without_text"] == 0, got
    assert got["blocks_scanned"] == 2, got
    assert got["blocks_examined"] == 2, got


def test_the_first_non_empty_field_wins_deterministically(tmp_path):
    """A block carrying two text fields must resolve the same way on every run: the
    extractor's input decides what edges get proposed, so a reader comparing two runs
    must see a real change rather than a dict-order difference."""
    corpus = [{"_id": "DEC-1", "Excerpt": "first", "content": "second"}]
    seen: list[str] = []
    backfill(str(tmp_path), corpus=corpus, extract_fn=lambda t: seen.append(t) or [])
    again: list[str] = []
    backfill(str(tmp_path), corpus=corpus, extract_fn=lambda t: again.append(t) or [])
    assert seen == again, (seen, again)
    assert seen and seen[0] in ("first", "second")


def test_a_whitespace_only_field_counts_as_no_text(tmp_path):
    """Whitespace is not text; treating it as text hands the extractor nothing and
    inflates blocks_scanned, which is the number this fix exists to make honest."""
    corpus = [{"_id": "DEC-1", "Excerpt": "   \n  "}]
    got = backfill(str(tmp_path), corpus=corpus, extract_fn=_extract)
    assert got["blocks_examined"] == 0, got
    assert got["blocks_without_text"] == 1, got
    # And the yield must not divide by a block it never read.
    assert got["edges_per_block"] == 0.0, got
