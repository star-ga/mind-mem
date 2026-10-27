"""chat_with_memory(computed_facts=True) puts code-computed dates/options in the prompt; off is byte-identical."""

from __future__ import annotations

from mind_mem.chat_memory import ChatRequest, chat_with_memory

DECISIONS = """\
[D-20230813-042]
Statement: I laid the patio yesterday.
Status: active
Date: 2023-08-13

---

[D-20230901-001]
Statement: I avoid anything that needs cash on the day.
Status: active
Date: 2023-09-01
"""


def _ws(tmp_path) -> str:
    ws = tmp_path / "ws"
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text(DECISIONS, encoding="utf-8")
    (ws / "mind-mem.json").write_text('{"recall": {"backend": "bm25"}}', encoding="utf-8")
    return str(ws)


def _hits(ws, q, limit):
    return [
        {
            "_id": "D-20230813-042",
            "excerpt": "I laid the patio yesterday.",
            "score": 1.0,
            "Date": "2023-08-13",
            "file": "decisions/DECISIONS.md",
        },
        {
            "_id": "D-20230901-001",
            "excerpt": "I avoid anything that needs cash on the day.",
            "score": 0.5,
            "Date": "2023-09-01",
            "file": "decisions/DECISIONS.md",
        },
    ]


def test_off_is_byte_identical_and_on_adds_the_block(tmp_path):
    ws = _ws(tmp_path)
    seen: list[str] = []

    def gen(req: ChatRequest) -> str:
        seen.append(req.prompt)
        return f"Done [[{req.evidence[0].block_id}]]."

    q = "Today is 2023-12-09. How many weeks ago was laying the patio?"
    chat_with_memory(ws, q, recall_fn=_hits, generator=gen)
    chat_with_memory(ws, q, recall_fn=_hits, generator=gen, computed_facts=False)
    chat_with_memory(ws, q, recall_fn=_hits, generator=gen, computed_facts=True)
    assert seen[0] == seen[1]
    assert "Computed from the evidence" not in seen[0]
    assert "119 days (weeks = days // 7 = 17)" in seen[2]


def test_choice_answer_without_an_option_is_retried_once(tmp_path):
    ws = _ws(tmp_path)
    prompts: list[str] = []
    answers = iter(["Go with the third one [[D-20230901-001]].", "Stewarding the car park [[D-20230901-001]]."])

    def gen(req: ChatRequest) -> str:
        prompts.append(req.prompt)
        return next(answers)

    out = chat_with_memory(
        ws,
        "Should I sign up for counting the float or stewarding the car park?",
        recall_fn=_hits,
        generator=gen,
        computed_facts=True,
        on_invalid="reject",
    )
    assert len(prompts) == 2 and "must name exactly one of these options" in prompts[1]
    assert "Stewarding the car park" in out.answer
