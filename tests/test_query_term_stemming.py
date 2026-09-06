"""The query and the index must agree on what a word stems to.

The FTS index is built with FTS5's real Porter stemmer
(``tokenize='porter unicode61'``), but the query is pre-stemmed by mind-mem's
own simplified stemmer before it reaches MATCH, so FTS5 stems an
already-stemmed token a second time and the two token spaces stop agreeing.

Measured over 470 LongMemEval-S questions: 165 of 1592 distinct query words
land on a term the index does not contain, affecting 332 of 4420 served query
tokens. Two shapes, one cause -- the index space is Porter-over-raw-text with
no lemma table:

    over-truncation   speed->spe, creamer->cream, theater->theat, sister->sist
    irregular lemma   bought->buy, got->get, spent->spend

The worst outcome is not a bad ranking but an empty one: "Which theater did we
visit?" returned NO ROWS at all, because MATCH "theat" matches nothing while
MATCH "theater" matches the document.

The fix ADDS the raw words rather than replacing the processed ones: dropping
the processed tokens measured net-positive on affected questions but lost one
on an unaffected control, because the lemma and month expansions do real work.
"""

import os
import tempfile

import pytest

from mind_mem._recall_tokenization import tokenize
from mind_mem.sqlite_index import build_index, query_index

DOCS = [
    ("D-20260521-000001", "I upgraded to 500 Mbps and my internet speed is much better now"),
    ("D-20260522-000002", "I redeemed a coupon on coffee creamer at the store yesterday"),
    ("D-20260523-000003", "we went to the theater to see a play last weekend"),
    ("D-20260524-000004", "my sister called about the dinner plans in Denver"),
    ("D-20260525-000005", "totally unrelated filler about gardening and compost bins"),
]


@pytest.fixture(scope="module")
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "decisions")
        os.makedirs(d, exist_ok=True)
        rows = [f"[{bid}]\nStatement: {txt}\nDate: 2026-05-2{i}\nStatus: active\n" for i, (bid, txt) in enumerate(DOCS, 1)]
        with open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8") as fh:
            fh.write("\n---\n\n".join(rows))
        with open(os.path.join(tmp, "mind-mem.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        build_index(tmp, incremental=False)
        yield tmp


class TestTheSimplifiedStemmerStillOverTruncates:
    """The premise. If this stops holding, the guards below are moot."""

    @pytest.mark.parametrize(
        "word,truncated",
        [("speed", "spe"), ("creamer", "cream"), ("theater", "theat"), ("sister", "sist")],
    )
    def test_the_query_stemmer_produces_a_term_the_index_lacks(self, word: str, truncated: str) -> None:
        assert truncated in tokenize(f"the {word} here"), (
            f"{word!r} no longer stems to {truncated!r}; this corpus no longer demonstrates the mismatch"
        )


class TestOverTruncatedWordsStillRetrieve:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("What speed is my new internet plan?", "D-20260521-000001"),
            ("Where did I redeem a coupon on coffee creamer?", "D-20260522-000002"),
            ("Which theater did we visit?", "D-20260523-000003"),
            ("What did my sister say about dinner in Denver?", "D-20260524-000004"),
        ],
    )
    def test_the_right_document_ranks_first(self, workspace, query: str, expected: str) -> None:
        res = query_index(workspace, query, limit=3)
        assert res, f"no rows at all for {query!r} -- the query term matches nothing in the index"
        assert res[0].get("_id") == expected, f"{query!r} ranked {[r.get('_id') for r in res]}, expected {expected} first"

    def test_an_unmatched_query_still_returns_nothing(self, workspace) -> None:
        """The union must widen the token space, not match everything."""
        res = query_index(workspace, "zebra helicopter oscilloscope", limit=3)
        assert res == [], f"unrelated query matched something: {[r.get('_id') for r in res]}"

    def test_a_stopword_only_query_is_still_empty(self, workspace) -> None:
        assert query_index(workspace, "the and of is", limit=3) == []
