"""chat_with_memory(session_expansion_chars=N) answers from whole admitted sessions; 0 leaves it unchanged."""

from __future__ import annotations

from mind_mem.chat_memory import ChatRequest, chat_with_memory

DECISIONS = """\
[D-20260301-001]
Statement: user: We adopted a greyhound last week.
Status: active
Date: 2026-03-01
Tags: session-1

---

[D-20260301-002]
Statement: assistant: Congratulations! What is the dog called?
Status: active
Date: 2026-03-01
Tags: session-1

---

[D-20260301-003]
Statement: user: Her name is Bramble.
Status: active
Date: 2026-03-01
Tags: session-1

---

[D-20260402-001]
Statement: user: The boiler needs a service.
Status: active
Date: 2026-04-02
Tags: session-2
"""


def _ws(tmp_path) -> str:
    ws = tmp_path / "ws"
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text(DECISIONS, encoding="utf-8")
    (ws / "mind-mem.json").write_text('{"recall": {"backend": "bm25"}}', encoding="utf-8")
    return str(ws)


def _one_hit(ws, question, limit):
    return [{"_id": "D-20260301-001", "excerpt": "user: We adopted a greyhound last week.", "score": 1.0, "file": "decisions/DECISIONS.md"}]


def _capture(seen):
    def gen(request: ChatRequest) -> str:
        seen.append([e.block_id for e in request.evidence])
        return f"Noted [[{request.evidence[0].block_id}]]."

    return gen


def test_expansion_brings_in_the_rest_of_the_session(tmp_path):
    ws, seen = _ws(tmp_path), []
    chat_with_memory(ws, "What is the dog called?", recall_fn=_one_hit, generator=_capture(seen), session_expansion_chars=4000)
    assert "D-20260301-003" in seen[0]  # the answer turn recall did not return
    assert seen[0][:3] == ["D-20260301-001", "D-20260301-002", "D-20260301-003"]


def test_budget_zero_is_the_unexpanded_path(tmp_path):
    ws, seen = _ws(tmp_path), []
    chat_with_memory(ws, "What is the dog called?", recall_fn=_one_hit, generator=_capture(seen))
    assert seen[0] == ["D-20260301-001"]
