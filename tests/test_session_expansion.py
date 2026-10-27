"""Session expansion returns whole admitted sessions, ranked by hits then session BM25, within a budget."""

from __future__ import annotations

from mind_mem.session_expansion import expand_sessions, rank_sessions, render_transcript, session_of


def _corpus():
    rows = {}
    for s, texts, date in (
        (1, ["[Ana] I adopted a greyhound.", "[Bo] What is its name?", "[Ana] Bramble."], "1 May 2023"),
        (2, ["[Ana] The cello lessons start soon.", "[Bo] Nice!"], "9 May 2023"),
        (3, ["[Bo] I painted the fence.", "[Ana] Looks great."], "20 May 2023"),
    ):
        for t, text in enumerate(texts, 1):
            rows[f"DIA-D{s}-{t}"] = {"Statement": text, "Tags": f"session-{s}, x", "Date": date, "DiaID": f"D{s}:{t}", "speaker": text[1:4]}
    return rows


def test_session_of_prefers_explicit_field_then_tag_then_date():
    assert session_of({"Session": "abc", "Tags": "session-2", "Date": "d"}) == "S:abc"
    assert session_of({"Tags": "session-2, x", "Date": "d"}) == "T:2"
    assert session_of({"Date": "2024-01-02"}) == "D:2024-01-02"
    assert session_of({}) == ""


def test_hit_session_comes_first_then_lexical():
    corpus = _corpus()
    order = rank_sessions("When did Bo paint the fence?", [{"_id": "DIA-D2-1"}], corpus)
    assert order[0] == "T:2"  # the hit's session
    assert order[1] == "T:3"  # best lexical match among the rest


def test_whole_session_pulls_in_the_neighbouring_answer():
    corpus = _corpus()
    items = expand_sessions("What is the name of Ana's dog?", [{"_id": "DIA-D1-1"}], corpus, budget_chars=90)
    ids = [i["_id"] for i in items]
    assert ids == ["DIA-D1-1", "DIA-D1-2", "DIA-D1-3"]  # the answer is in a turn recall did not return
    assert items[2]["excerpt"] == "[Ana] Bramble." and items[2]["DiaID"] == "D1:3"


def test_budget_skips_sessions_that_do_not_fit_and_keeps_corpus_order():
    corpus = _corpus()
    items = expand_sessions("fence", [{"_id": "DIA-D3-1"}, {"_id": "DIA-D1-1"}], corpus, budget_chars=60)
    assert {i["session"] for i in items} == {"T:3"}  # session 1 (82 chars) does not fit the remainder
    big = expand_sessions("fence", [{"_id": "DIA-D3-1"}, {"_id": "DIA-D1-1"}], corpus, budget_chars=10_000)
    assert [i["_id"] for i in big] == list(corpus)  # everything, in corpus (chronological) order


def test_disabled_budget_returns_nothing():
    assert expand_sessions("q", [{"_id": "DIA-D1-1"}], _corpus(), budget_chars=0) == []


def test_only_corpus_blocks_can_appear():
    corpus = _corpus()
    items = expand_sessions("q", [{"_id": "NOT-IN-CORPUS"}], corpus, budget_chars=10_000)
    assert {i["_id"] for i in items} <= set(corpus)


def test_render_transcript_has_session_headers_and_sources():
    items = expand_sessions("q", [{"_id": "DIA-D2-1"}], _corpus(), budget_chars=50)
    text = render_transcript(items)
    assert text.splitlines()[0] == "--- session 2 (9 May 2023) ---"
    assert "D2:1 [Ana] The cello lessons start soon." in text
