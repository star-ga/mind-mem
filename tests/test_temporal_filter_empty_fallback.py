"""A temporal hard-filter window that excludes every candidate falls back to the unfiltered ranking."""

from __future__ import annotations

from datetime import date

from mind_mem.recall import recall

CORPUS = """\
[D-20230520-001]
Statement: user: I spent ten hours watching nature documentaries on the streaming service.
Status: active
Date: 2023-05-20

---

[D-20230521-001]
Statement: user: The garden needs weeding.
Status: active
Date: 2023-05-21
"""


def _ws(tmp_path):
    ws = tmp_path / "ws"
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text(CORPUS, encoding="utf-8")
    (ws / "mind-mem.json").write_text('{"recall": {"backend": "bm25"}}', encoding="utf-8")
    return str(ws)


def test_window_that_removes_everything_keeps_the_ranking(tmp_path):
    ws = _ws(tmp_path)
    hits = recall(ws, "How many hours did I spend watching documentaries last month?", limit=5, scoring_instant=date(2023, 5, 30))
    assert hits and hits[0]["_id"] == "D-20230520-001"


def test_window_that_keeps_something_still_filters(tmp_path):
    ws = _ws(tmp_path)
    hits = recall(ws, "What did I watch on 2023-05-20?", limit=5, scoring_instant=date(2023, 5, 30))
    assert all(h.get("Date", "2023-05-20")[:10] == "2023-05-20" for h in hits if h.get("Date"))
