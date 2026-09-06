"""A fact card must stay SMALL. It must not become a copy of its parent.

There is a real loss here, and it is worth stating precisely because it will
tempt the same wrong cure again. Extraction is lossy in the retrieval-fatal
direction: it keeps the atomic clause and discards the surrounding words, and
the discarded words are frequently the ones a question uses. Measured on
LongMemEval-S question ad7109d1 -- a turn reading "my internet speed has been
really good ... I upgraded to 500 Mbps" yields the card "upgraded to 500 Mbps
about three weeks ago", carrying not one of "speed", "internet" or "plan".

d487549 answered that by copying the whole parent statement into the card's
searchable Context. Measured on one haystack, that took the fact surface from
14,426 indexed characters to 448,559 (31x) and the database from 4 KB to
5.2 MB, because a card's searchable text grew from 37 characters to 1,221. It
also stopped the layer working: a card containing its parent's words can only
match when the parent already matched, so the small-to-big injection branch had
nothing left to contribute -- on a 488-parent corpus, injections went 338 to 0.
No recall gain was measured in exchange.

The parent's language is already searchable on the parent surface, which is
queried separately. These tests hold the card to being small.
"""

import os
import sqlite3
import tempfile

from mind_mem.extractor import extract_facts
from mind_mem.sqlite_index import build_index

TURN = (
    "I did notice that my internet speed has been really good lately, "
    "especially when I'm streaming movies on Netflix. I upgraded to 500 Mbps "
    "about three weeks ago, and it's made a huge difference."
)


def _workspace(tmp: str) -> str:
    d = os.path.join(tmp, "decisions")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(f"[SESSION-1__t0]\nStatement: {TURN}\nDate: 2023-05-20\nStatus: active\n")
    with open(os.path.join(tmp, "mind-mem.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    return tmp


def _db(ws: str) -> sqlite3.Connection:
    con = sqlite3.connect(os.path.join(ws, ".mind-mem-index", "recall.db"))
    con.row_factory = sqlite3.Row
    return con


class TestExtractionIsLossy:
    """The premise. Pinned so the guard below cannot outlive the problem."""

    def test_the_card_drops_the_words_the_question_uses(self) -> None:
        cards = [c["content"] for c in extract_facts(TURN, speaker="user")]
        assert cards, "no cards extracted; the premise cannot be checked"
        joined = " ".join(cards).lower()
        assert "500 mbps" in joined, cards
        assert "internet" not in joined, f"extraction now keeps the context word, so this no longer demonstrates the loss: {cards}"


class TestFactCardStaysSmall:
    def test_a_card_does_not_carry_its_parent_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _workspace(tmp)
            build_index(ws, incremental=False)
            con = _db(ws)
            try:
                rows = con.execute("SELECT all_text FROM blocks_fts_facts").fetchall()
            finally:
                con.close()
            assert rows, "no fact cards indexed; the assertion would be vacuous"
            # A marker the PARENT has and no card legitimately extracts.
            # "streaming movies on Netflix" would be wrong here: extraction
            # mints "user is streaming movies on Netflix" as a card in its own
            # right, so finding it proves nothing about copying.
            marker = "I did notice that"
            assert marker in TURN, "the marker must be present in the parent to mean anything"
            for r in rows:
                assert marker not in r["all_text"], (
                    "a card is carrying its parent's whole statement again: "
                    "that costs ~31x the fact index and kills small-to-big "
                    f"injection: {r['all_text'][:120]!r}"
                )

    def test_the_fact_surface_stays_far_smaller_than_the_parent_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _workspace(tmp)
            build_index(ws, incremental=False)
            con = _db(ws)
            try:
                p = con.execute("SELECT SUM(LENGTH(all_text)) n FROM blocks_fts").fetchone()["n"] or 1
                f = con.execute("SELECT SUM(LENGTH(all_text)) n FROM blocks_fts_facts").fetchone()["n"] or 0
            finally:
                con.close()
            assert f < p, (
                f"the fact surface ({f} chars) is no longer smaller than the parent surface ({p} chars) -- cards have stopped being atomic"
            )

    def test_cards_are_still_indexed_and_searchable(self) -> None:
        """Small must not become absent."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _workspace(tmp)
            build_index(ws, incremental=False)
            con = _db(ws)
            try:
                hit = con.execute(
                    "SELECT COUNT(*) c FROM blocks_fts_facts WHERE blocks_fts_facts MATCH ?",
                    ("Mbps",),
                ).fetchone()["c"]
            finally:
                con.close()
            assert hit >= 1, "the card that holds the answer is not searchable at all"
