# ruff: noqa: E501  (tables of natural-language test sentences)
"""Every input any verifier used to break fact_slots, as regression tables.

Merge rule (slots_differ): a merge may be proposed only for the SAME statement
after a meaning-preserving normalisation (case, NFC, spacing, trailing full
stop, unambiguous abbreviations, canonical clock times). Differently worded
look-alikes are never merge candidates; they appear in find_similar_pairs.

Contradiction rule (changed_fact): strip when/zone values and correction words;
the rest must be identical; the value sets must differ or the negation flip.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from mind_mem.fact_slots import changed_fact, slots_differ
from mind_mem.granularity_align import find_merge_candidates, find_similar_pairs

NEVER_MERGE = [
    # case, abbreviations and time formats can change meaning: never merged
    ("Call with Ilona on Tue at 10 am.", "Call with Ilona on Tuesday at 10am."),
    ("meeting with ilona tuesday", "meeting with Ilona tuesday"),
    ("The call isn't on Tuesday.", "the call is not on Tuesday"),
    ("Standup at 10 a.m.", "Standup at 10am"),
    ("Ship the samples to US", "Ship the samples to us"),
    ("IT owns the laptop rollout", "it owns the laptop rollout"),
    ("Ask May about the budget", "Ask may about the budget"),
    ("Polish the release notes", "polish the release notes"),
    ("Use Jun's laptop", "Use June's laptop"),
    ("Review the loan APR", "Review the loan April"),
    ("Dinner with Bob at 7:00", "Dinner with Bob at 7am"),
    ("Hire 3 PM for Q4", "Hire 15:00 for Q4"),
    ("Ilona call Tue 10am Arizona time.", "Contractor call Tue 10am Pacific time."),
    ("Weekly planning call with Ilona on Tuesday at 10am PT.", "Weekly planning call with Ilona on Tuesday at 10am ET."),
    ("Call with Ilona on Tuesday at 10am in Pacific.", "Call with Ilona on Tuesday at 10am in Eastern."),
    ("Call with José on Tuesday at 10am.", "Call with Joséphine on Tuesday at 10am."),
    ("Call with José on Tuesday at 10am.", "Call with Joséphine on Tuesday at 10am."),
    ("Call with Ángel on Tuesday at 10am.", "Call with Óscar on Tuesday at 10am."),
    ("Weekly planning call with 李明 on Tuesday at 10am.", "Weekly planning call with 王芳 on Tuesday at 10am."),
    ("Lunch with Jan on Friday at noon.", "Lunch with May on Friday at noon."),
    ("Rent payment to Ilona is due on the 5th of each month.", "Rent payment to Ilona is due on the 15th of each month."),
    ("Budget review at 10am in 2025.", "Budget review at 10am in 2026."),
    ("Standup today at 10am.", "Standup tomorrow at 10am."),
    ("Standup at 10 in the morning.", "Standup at 10 in the evening."),
    ("Standup at 10h.", "Standup at 14h."),
    ("Standup at 1000 hours.", "Standup at 1400 hours."),
    ("Standup at 9.30.", "Standup at 9.45."),
    ("Standup at 0930.", "Standup at 0945."),
    ("Standup at half past nine.", "Standup at quarter to ten."),
    ("Ilona: Tuesday. 10am: Marcus.", "Ilona: 10am. Tuesday: Marcus."),
    ("Ilona: Tuesday\n10am: Marcus", "Ilona: 10am\nTuesday: Marcus"),
    ("Ilona Tuesday, PT Marcus", "Ilona PT, Tuesday Marcus"),
    ("Ilona PT, 10am Marcus ET", "Ilona 10am, PT Marcus ET"),
    ("Budget item 1. 5 engineers approved", "Budget item 1.5 engineers approved"),
    ("Decision 1.\n5 engineers join Ilona", "Decision 1.5 engineers join Ilona"),
    ("Rooms 1, 5 booked", "Rooms 1,5 booked"),
    ("Marcus ordered 10 May shipments", "Marcus ordered May 10 shipments"),
    ("Ship 5 May orders", "Ship May 5 orders"),
    ("Office hours Tue-Fri.", "Office hours Tue, Fri."),
    ("Ilona calls Marcus every Monday.", "Marcus calls Ilona every Monday."),
    ("Azariel must never share the investor deck outside the board.", "Azariel must share the investor deck outside the board."),
    ("Azariel must share the deck.", "Azariel may share the deck."),
    ("Meet at 10:30.", "Meet at 10:45."),
    ("Meet at 10am UTC+2.", "Meet at 10am UTC+3."),
    ("call with ilona tue 10am arizona", "call with marcus tue 10am arizona"),
    ("weekly call with the contractor on tuesday at 10am pacific", "weekly call with the designer on tuesday at 10am pacific"),
    ("the client pays the vendor on friday", "the vendor pays the client on friday"),
    ("standup on mon at 10am", "standup on wed at 10am"),
    ("deploy freeze starts sat at 10am", "deploy freeze starts sun at 10am"),
    ("report due jan 5", "report due mar 5"),
    ("Report due Jan 5", "Report due Mar 5"),
    ("call at 10am ist", "call at 10am hst"),
    ("call at 10am london time", "call at 10am tokyo time"),
    ("call at 10am akst", "call at 10am nzst"),
    ("the subscription costs $5 per month", "the subscription costs €5 per month"),
    ("set the offset to -5 hours", "set the offset to +5 hours"),
    ("alert when latency > 500 ms", "alert when latency < 500 ms"),
    ("the run is 5 km long", "the run is 5 mi long"),
    ("send invoices to billing@example.com every month", "send invoices to billing@example.org every month"),
    ("docs live at https://example.com/alpha for the team", "docs live at https://example.com/beta for the team"),
    ("call set_timeout before reconnecting the socket", "call set_interval before reconnecting the socket"),
    ("launch phase ii of the migration on tuesday", "launch phase iii of the migration on tuesday"),
    ("the eleventh release ships on tuesday", "the twelfth release ships on tuesday"),
    ("on the twentieth of the month rent is due", "on the thirtieth of the month rent is due"),
    ("order a dozen chairs for the office", "order a couple chairs for the office"),
    ("enable telemetry in production builds", "disable telemetry in production builds"),
    ("the board approved the budget on tuesday", "the board rejected the budget on tuesday"),
    ("run migrations before the deploy", "run migrations after the deploy"),
    ("increase the cache size for recall", "decrease the cache size for recall"),
    ("feature flag recall_v2 is set to true", "feature flag recall_v2 is set to false"),
    ("build status for main is ✅ today", "build status for main is ❌ today"),
    ('the answer from legal was "yes" today', 'the answer from legal was "maybe" today'),
    ("use ~~sqlite~~ postgres for the index", "use sqlite ~~postgres~~ for the index"),
    ("users are allowed to delete blocks", "users are forbidden to delete blocks"),
    ("you need to sign the release", "you needn't sign the release"),
    ("call at 10am pacific", "call at 10am atlantic"),
    ("use one half of the budget", "use one third of the budget"),
    ("discount is ten percent", "discount is ten permille"),
    ("meet at 10 in the am", "meet at 10 in the pm"),
    ("office hours mon-wed", "office hours mon-fri"),
    ("office hours mon/wed", "office hours wed/mon"),
    ("call with ilona tue", "call with il​ona tue"),
    ("Call with Ilona on Tuesday", "Call with Ilоna on Tuesday"),
    ("We use Postgres for the index.", "The index uses Postgres."),
    ("Ilona sat at 10am for the exam.", "Ilona sat for the exam at 10am."),
]

EXACT_DUPLICATES = [
    ("Use  Postgres   for the index.", "Use Postgres for the index"),
    ("Call with Ilona on Tuesday at 10am.", "Call with Ilona on Tuesday at 10am"),
    ("Cafe\u0301 opens at 9am", "Café opens at 9am"),  # NFD vs NFC
]

LOOKALIKES = [
    (
        "Weekly planning call with Ilona on Tuesday at 10am Arizona time.",
        "Weekly planning call with Ilona, Tuesday 10am Arizona time, recurring.",
    ),
    ("fixed-point arithmetic ensures determinism across substrates", "use fixed-point arithmetic for cross-substrate determinism"),
    ("the release workflow publishes wheels to PyPI on tag push", "release workflow publishes the wheels on a tag push to PyPI"),
]

CONTRADICTS = [
    (
        "Azariel's weekly planning call with Ilona is on Thursday at 10am Arizona time.",
        "Azariel's weekly planning call with Ilona is on Tuesday at 10am Arizona time.",
    ),
    (
        "Weekly planning call with Ilona on Tuesday at 11am Arizona time.",
        "Weekly planning call with Ilona on Tuesday at 10am Arizona time.",
    ),
    ("Ilona's planning call is on Thursdays at 10am.", "Ilona's planning call is on Tuesdays at 10am."),
    (
        "Azariel's weekly planning call with Ilona moved to Thursday at 10am Arizona time; it is no longer on Tuesday.",
        "Azariel's weekly planning call with Ilona is on Tuesday at 10am Arizona time.",
    ),
    ("The planning call is now at 11am.", "The planning call is at 10am."),
    ("Board review on 2026-10-22.", "Board review on 2026-10-15."),
    ("Standup at 10:30.", "Standup at 9:30."),
    ("ilona call on thursday at 10am", "ilona call on tuesday at 10am"),
    ("The standup moved from Tuesday to Thursday", "The standup is on Tuesday"),
    ("The standup is now on Thursdays", "The standup is on Tuesdays"),
    ("The standup is no longer at 10am, it is 11am", "The standup is at 10am"),
    ("The standup was rescheduled for Thursday", "The standup is on Tuesday"),
    ("The standup has been postponed to Thursday", "The standup is on Tuesday"),
    ("The standup got pushed to Thursday", "The standup is on Tuesday"),
    ("The standup is on Wed", "The standup is on Mon"),
    ("The standup is on Sat", "The standup is on Sun"),
    ("Report due 2026-10-09", "Report due 2026-10-08"),
    ("Report due Oct 9", "Report due 2026-10-08"),
    ("Report due 10/09/2026", "Report due 10/08/2026"),
    ("Report due 9 October", "Report due 8 October"),
    ("Report due October 9th", "Report due October 8th"),
    ("Report due Jan 9", "Report due Mar 9"),
    ("Standup at 11:00", "Standup at 10:00"),
    ("Standup at 11 am", "Standup at 10 a.m."),
    ("Ilona's call is Thursday 10am Pacific, not Tuesday", "Ilona's call is Tuesday 10am Pacific"),
    ("Correction: Ilona's call is on Thursday", "Ilona's call is on Tuesday"),
    ("Actually, Ilona's call is on Thursday", "Ilona's call is on Tuesday"),
    ("Ilona's call switched to Thursday", "Ilona's call is on Tuesday"),
    ("Ilona's call is on Thursday instead of Tuesday", "Ilona's call is on Tuesday"),
    ("Ilona's call: Thursday", "Ilona's call: Tuesday"),
    ("Ilona's call is at 10am Arizona time", "Ilona's call is at 10am Pacific time"),
    ("The standup is on Thursday from now on", "The standup is on Tuesday"),
    ("Starting next week the standup is on Thursday", "The standup is on Tuesday"),
    ("Rent payment to Ilona is due on the 15th of each month.", "Rent payment to Ilona is due on the 5th of each month."),
    ("The call is not on Tuesday", "The call is on Tuesday"),
    ("Alice no longer works on Tuesdays", "Alice works on Tuesdays"),
    ("The call has been moved up to Thursday", "The call is on Tuesday"),
    ("The call's on Thursday", "The call is on Tuesday"),
    ("Rent is due on the 7th", "Rent is due on the 5th"),
    ("The meeting was rescheduled to Thursday", "The meeting is scheduled for Tuesday"),
]

NOT_CONTRADICTS = [
    (
        "Azariel's weekly planning call with the engineering contractor is on Tuesday at 10am Pacific time.",
        "Azariel's weekly planning call with Ilona is on Thursday at 10am Arizona time.",
    ),
    ("Ilona's dentist visit is on Friday, the day after the planning call.", "Ilona's planning call is on Thursday."),
    ("Dentist visit on Friday, after the planning call.", "Planning call on Tuesday."),
    ("The daily lunch is at noon.", "The daily standup is at 9am."),
    ("Azariel's weekly dentist visit is on Friday.", "Azariel's weekly call is on Tuesday."),
    ("Ilona's return flight on Thursday.", "Ilona's flight on Tuesday."),
    ("Ilona's final exam on May 12.", "Ilona's exam on May 5."),
    ("Planning call prep at 9am.", "Planning call at 10am."),
    ("Call with Ilona's lawyer on Thursday.", "Call with Ilona on Tuesday."),
    ("Dentist appointment for son on Thursday.", "Dentist appointment on Tuesday."),
    ("Lunch with Marcus on Friday.", "Lunch with Ilona on Thursday."),
    ("Bug 5678 was fixed on Tuesday", "Bug 1234 was fixed on Tuesday"),
    ("PR 513 merged on Thursday", "PR 512 merged on Tuesday"),
    ("Invoice 1002 is due on Friday", "Invoice 1001 is due on Tuesday"),
    ("standup with team two at 10am", "standup with team one at 10am"),
    ("Room 102 is booked at 10am", "Room 101 is booked at 10am"),
    ("Flight UA 90 departs at 10am", "Flight UA 80 departs at 10am"),
    ("The 2025 budget review is in March", "The 2024 budget review is in March"),
    ("Meet at 10:00am", "Meet at 10am"),
    ("Release 4.1 shipped on Tuesday", "Release 4.0 shipped on Tuesday"),
    ("Q3 review on Friday", "Q2 review on Friday"),
    ("Ilona call was on Tuesday", "Ilona call is on Tuesday"),
    ("The 12th release ships on Tuesday", "The 11th release ships on Tuesday"),
    (
        "Weekly planning call with Ilona: Thursday, 10am Arizona time.",
        "Azariel's weekly planning call with Ilona is on Thursday at 10am Arizona time.",
    ),
]

CRASH_INPUTS = [
    "Office on ²nd floor",
    "Room ³rd",
    "Meet ¹st May",
    "x ①a",
    "٣ apples",
    "\x00\x01",
    "",
    "   ",
    "...",
    "\ud800",
    "\U0010ffff",
    "á́́",
]


@pytest.mark.parametrize(("a", "b"), NEVER_MERGE)
def test_never_merge(a, b):
    assert slots_differ(a, b) and slots_differ(b, a)
    assert find_merge_candidates([{"_id": "A", "content": a}, {"_id": "B", "content": b}], min_similarity=0.0) == []


@pytest.mark.parametrize("field", ["Date", "speaker", "Scope", "tags"])
def test_any_differing_content_field_blocks_merge(field):
    a = {"_id": "A", "content": "Standup with Bob", field: "x1"}
    b = {"_id": "B", "content": "Standup with Bob", field: "x2"}
    assert find_merge_candidates([a, b], min_similarity=0.0) == []


def test_planner_projection_carries_frontmatter_fields():
    from mind_mem.mcp.tools.consolidation import _granularity_block

    a = _granularity_block("A", "active", "t", {"Statement": "Standup with Bob", "Date": "2026-01-01"})
    b = _granularity_block("B", "active", "t", {"Statement": "Standup with Bob", "Date": "2026-03-01"})
    assert find_merge_candidates([a, b], min_similarity=0.0) == []


def test_tag_only_entity_blocks_merge():
    blocks = [
        {"_id": "A", "content": "weekly planning call tuesday 10am", "tags": "ilona"},
        {"_id": "B", "content": "weekly planning call tuesday 10am", "tags": "marcus"},
    ]
    assert find_merge_candidates(blocks, min_similarity=0.0) == []


@pytest.mark.parametrize(("a", "b"), EXACT_DUPLICATES)
def test_exact_duplicates_merge(a, b):
    assert not slots_differ(a, b)
    assert len(find_merge_candidates([{"_id": "A", "content": a}, {"_id": "B", "content": b}], min_similarity=0.0)) == 1


@pytest.mark.parametrize(("a", "b"), LOOKALIKES)
def test_lookalikes_are_similar_pairs_not_merges(a, b):
    blocks = [{"_id": "A", "content": a}, {"_id": "B", "content": b}]
    assert find_merge_candidates(blocks, min_similarity=0.0) == []
    assert [(p.id_a, p.id_b) for p in find_similar_pairs(blocks, min_similarity=0.0)] == [("A", "B")]


@pytest.mark.parametrize(("new", "old"), CONTRADICTS)
def test_changed_fact_is_flagged(new, old):
    assert changed_fact(new, old)


@pytest.mark.parametrize(("new", "old"), NOT_CONTRADICTS)
def test_different_statement_is_not_a_contradiction(new, old):
    assert not changed_fact(new, old)


@pytest.mark.parametrize("text", CRASH_INPUTS)
def test_never_raises(text):
    slots_differ(text, "Office on 2nd floor")
    changed_fact(text, "Office on 2nd floor")
    find_merge_candidates([{"_id": "A", "content": text}, {"_id": "B", "content": text}], min_similarity=0.0)


_QUADRATIC = [
    (" ".join("abc"[i % 3] for i in range(1999)), " ".join("ab"[i % 2] for i in range(1999))),
    ("q " + " ".join(["x"] * 1998) + " z", "r " + " ".join(["x", "y"] * 999) + " w"),
    ("start " + " ".join(["alpha"] * 1998) + " end", "begin " + " ".join(["alpha", "beta"] * 999) + " finish"),
    (" " * 300_000 + "x", " " * 300_000 + "y"),
    ("1" * 300_000, "1" * 300_001),
    (":" * 300_000, ":" * 300_001),
    ("1:" * 150_000, "1:" * 150_001),
    ("Tue 10am PT, " * 25_000, "Tue 10am PT, " * 25_000 + " moved"),
]


def _compare_seconds(a: str, b: str) -> float:
    start = time.perf_counter()
    slots_differ(a, b)
    changed_fact(a, b)
    find_merge_candidates([{"_id": "A", "excerpt": a}, {"_id": "B", "excerpt": b}], min_similarity=0.0)
    return time.perf_counter() - start


@pytest.mark.parametrize(("a", "b"), _QUADRATIC)
def test_linear_time(a, b):
    # Linearity is the property: doubling the input must not much more than double the time.
    # A wall-clock budget alone fails on a loaded host; the ratio does not. The absolute cap
    # still catches a pathological constant.
    half = _compare_seconds(a[: len(a) // 2], b[: len(b) // 2])
    full = _compare_seconds(a, b)
    assert full < 30.0
    assert full <= 3.0 * half + 0.25, f"doubling the input took {full:.2f}s vs {half:.2f}s: super-linear"


def test_end_to_end_contradiction_check(tmp_path: Path):
    from mind_mem.contradiction_detector import check_proposal_contradictions

    os.makedirs(tmp_path / "decisions")
    blocks = [
        ("D-20261001-001", "Bug 1234 in the recall index was fixed on Tuesday by the platform team."),
        ("D-20261001-002", "The weekly standup with the platform team is on Tuesday at 10am Pacific time."),
        ("D-20261001-003", "Ilona's weekly planning call is on Mon at 10am Arizona time."),
        ("D-20261001-004", "Invoice 1001 from the hosting vendor is due on Tuesday."),
    ]
    body = "# DECISIONS\n\n" + "\n".join(
        f"[{bid}]\nDate: 2026-10-01\nStatus: active\nScope: global\nStatement: {st}\nRationale: seed.\nSupersedes: none\nTags: test\nSources: test\n"
        for bid, st in blocks
    )
    (tmp_path / "decisions" / "DECISIONS.md").write_text(body, encoding="utf-8")

    def flagged(statement: str) -> list[str]:
        rep = check_proposal_contradictions(str(tmp_path), {"Statement": statement})
        return sorted(c["block_id"] for c in rep["conflicts"] if c["conflict_type"] == "contradiction")

    assert flagged("Bug 5678 in the recall index was fixed on Tuesday by the platform team.") == []
    assert flagged("Invoice 1002 from the hosting vendor is due on Friday.") == []
    assert flagged("The weekly standup with the platform team was rescheduled for Thursday at 10am Pacific time.") == ["D-20261001-002"]
    assert flagged("Ilona's weekly planning call is on Wed at 10am Arizona time.") == ["D-20261001-003"]


def test_halb_with_an_impossible_hour_is_not_a_clock():
    """'halb 99' has no clock reading; it must never put None into the token stream.

    Positive control: 'halb zehn' still reads as 9:30, so the guard does not hide the clock path.
    """
    from mind_mem.fact_slots import _compare_tokens

    tokens = _compare_tokens("Das Meeting ist um halb 99")
    assert None not in tokens
    assert all(isinstance(t, str) for t in tokens)
    assert any(t.startswith("t09:30") for t in _compare_tokens("Das Meeting ist um halb zehn"))
    assert isinstance(changed_fact("Das Meeting ist um halb 99", "Das Meeting ist um halb 98"), bool)
