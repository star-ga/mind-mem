"""A fact card must stay reachable by the language it was extracted from.

Extraction is lossy in the retrieval-fatal direction: it keeps the atomic
clause and drops the surrounding words, and the dropped words are frequently
the ones a question uses. Measured on LongMemEval-S question ad7109d1 -- a turn
reading "my internet speed has been really good ... I upgraded to 500 Mbps"
yields the card "upgraded to 500 Mbps about three weeks ago", carrying not one
of "speed", "internet" or "plan". The question "What speed is my new internet
plan?" could not reach the single card in the corpus that answered it, so
small-to-big retrieval had no parent to lift.
"""

import os
import sqlite3
import tempfile

from mind_mem.sqlite_index import _extract_fts_fields, build_index

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


class TestFactCardCarriesItsSourceTurn:
    def test_the_card_itself_does_not_contain_the_query_terms(self) -> None:
        """The premise. If this ever stops holding, the fix below is moot."""
        from mind_mem.extractor import extract_facts

        cards = [c["content"] for c in extract_facts(TURN, speaker="user")]
        assert cards, "no cards extracted; the premise cannot be checked"
        joined = " ".join(cards).lower()
        assert "500 mbps" in joined, cards
        assert "internet" not in joined, f"extraction kept the context word, so this corpus no longer demonstrates the loss: {cards}"

    def test_context_makes_the_card_reachable_by_the_source_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _workspace(tmp)
            build_index(ws, incremental=False)
            con = _db(ws)
            try:
                rows = con.execute(
                    "SELECT block_id FROM blocks_fts_facts WHERE blocks_fts_facts MATCH ?",
                    ("internet",),
                ).fetchall()
            finally:
                con.close()
            assert rows, "no fact card matches 'internet', so a question using that word cannot reach the card that answers it"

    def test_context_is_populated_from_the_parent_statement(self) -> None:
        fields = _extract_fts_fields({"Statement": "upgraded to 500 Mbps", "Context": TURN})
        assert "internet" in fields["context"]
        assert fields["statement"] == "upgraded to 500 Mbps", (
            "the atomic clause must stay the statement; context is the lowest-weighted column precisely so it cannot outrank it"
        )

    def test_the_parent_surface_is_unaffected(self) -> None:
        """Context belongs to the fact card only; parents keep their own text."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _workspace(tmp)
            build_index(ws, incremental=False)
            con = _db(ws)
            try:
                n_parent = con.execute("SELECT COUNT(*) c FROM blocks_fts").fetchone()["c"]
                n_fact = con.execute("SELECT COUNT(*) c FROM blocks_fts_facts").fetchone()["c"]
            finally:
                con.close()
            assert n_parent >= 1
            assert n_fact >= 1
