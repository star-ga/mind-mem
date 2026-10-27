# Copyright 2026 STARGA, Inc.
"""Context rules for changed_fact and the merge identity (blind-review findings B1-B4).

Each case is a phrasing a reviewer used to break an earlier version: names that
look like months or weekdays, places that look like zones, identifiers that look
like dates, and corrections written the many ways people write them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.fact_slots import canonical_tokens, changed_fact, fact_slots, slots_differ
from mind_mem.granularity_align import find_merge_candidates, find_similar_pairs, merge_blocks, scan_pairs

# (new, old): different people, places, IDs or events, or an additive change. Not a correction.
NOT_CORRECTIONS = [
    ("Email Jan about the invoice", "Email June about the invoice"),
    ("Ask May about the budget on Tuesday", "Ask April about the budget on Tuesday"),
    ("Lunch with Sun at noon", "Lunch with Wed at noon"),
    ("Alice flies to Arizona at 10am", "Alice flies to Hawaii at 10am"),
    ("Bob is moving to Alaska at 9am", "Bob is moving to Hawaii at 9am"),
    ("Invoice 2026-10-08 is paid", "Invoice 2026-10-09 is paid"),
    ("Ticket 10/8 is closed", "Ticket 10/9 is closed"),
    ("Release 2026-10-08 build is green", "Release 2026-10-09 build is green"),
    ("Ilona's call is on Tuesday", "Ilona's call is on Tuesday and Thursday"),
    ("The call is on Tuesday and Thursday", "The call is on Tuesday"),
    ("The standup is on Tuesday and Thursday.", "The standup is on Tuesday."),
    ("Standup is on Tuesday", "Standup was on Tuesday"),
    ("The March report is late", "The May report is late"),
    ("Book the August conference room", "Book the June conference room"),
    ("Alice was born on Tuesday", "Bob was born on Thursday"),
    ("Meet Alice at 10am Pacific time", "Meet Alice at 10am Pacific"),
    ("Use the 3rd floor room on Tuesday", "Use the 5th floor room on Tuesday"),
    ("Pay the vendor on the 5th floor", "Pay the vendor on the 3rd floor"),
    ("Hire 3 PM interns", "Hire 5 PM interns"),
    ("The standup is at 10 Downing Street", "The standup is at 11 Downing Street"),
    ("The price is 1.50", "The price is 2.50"),
    ("Q3 review on Friday", "Q2 review on Friday"),
    ("Our office is at 10.30 Main Street", "Our office is at 9.30 Main Street"),
    ("Invoice INV-10% is due", "Invoice INV-12% is due"),
    ("The meeting is at noon", "The meeting is at 12pm"),
    ("The meeting is at 12:00", "The meeting is at noon"),
    ("See section 1.2.3", "See section 1.2.4"),
    ("Ticket 2026.10.09 is open", "Ticket 2026.10.10 is open"),
    ("Release 10.8 is out.", "Release 10.9 is out."),
    ("Version 2026.10 released.", "Version 2026.11 released."),
    ("The Tuesday team ships on Friday.", "The Thursday team ships on Friday."),
    ("The Monday group meets at noon", "The Friday group meets at noon"),
    ("Ilona's Tuesday class is at 6pm.", "Ilona's Thursday class is at 6pm."),
    ("Alice pays Bob on Tuesday.", "Bob pays Alice on Thursday."),
    ("Ilona's in-laws arrive from Arizona time to time.", "Ilona's in-laws arrive from Alaska time to time."),
    ("Send the report to June.", "Send the report to April."),
    ("Send the weekly digest to May.", "Send the weekly digest to August."),
    ("Assign the ticket to Jan", "Assign the ticket to Mar"),
    ("The review for Q4 is on Thursday.", "The review for Q3 is on Tuesday."),
    ("Review of April numbers is on Thursday.", "Review of March numbers is on Tuesday."),
    ("See John 3:17 for the quote.", "See John 3:16 for the quote."),
    ("Ilona lives at 14, Elm Street.", "Ilona lives at 12, Elm Street."),
    ("Ilona's desk is at 12 in building B.", "Ilona's desk is at 10 in building B."),
    ("Offsite of 2026 is in Denver.", "Offsite of 2025 is in Denver."),
    ("The ratio is 16:10.", "The ratio is 16:12."),
    ("Ilona's score was 21:15.", "Ilona's score was 21:17."),
    ("Ilona's locker is at 10 and Bob's at 12.", "Ilona's locker is at 11 and Bob's at 12."),
    ("Take the 6am Pacific Surfliner", "Take the 6am Central Surfliner"),
    ("Ilona owns INC-2026-10-09.", "Ilona owns INC-2026-10-08."),
    ("Ilona's employee id is 2026-10-09", "Ilona's employee id is 2026-10-08"),
    ("See doc 10/9 for details.", "See doc 10/8 for details."),
    ("Run job 10/8", "Run job 10/9"),
    ("The Thursday Murder Club is on the list.", "The Tuesday Murder Club is on the list."),
    ("The Monday Club meets", "The Friday Club meets"),
    ("Ilona joins Project Tuesday", "Ilona joins Project Thursday"),
    ("Ilona works at April Networks.", "Ilona works at March Networks."),
    ("Meeting in room 10:30 annex", "Meeting in room 11:30 annex"),
    ("Ilona is at 1700 Pennsylvania Ave.", "Ilona is at 1600 Pennsylvania Ave."),
    ("Ilona lives at 1300 Main St.", "Ilona lives at 1200 Main St."),
    ("Deliver to 1700 Amphitheatre Pkwy.", "Deliver to 1600 Amphitheatre Pkwy."),
    (
        "The weekly call with the contractor is on Tuesday at 10am Pacific time.",
        "The weekly call with Ilona is on Tuesday at 10am Arizona time.",
    ),
]

# (new, old): the same fact with a changed time, day, date, zone or polarity.
CORRECTIONS = [
    ("The weekly call with Ilona is on Thursday at 10am Arizona time.", "The weekly call with Ilona is on Tuesday at 10am Arizona time."),
    (
        "The weekly call with Ilona moved to Thursday at 10am Arizona time.",
        "The weekly call with Ilona is on Tuesday at 10am Arizona time.",
    ),
    ("Moved the call from Tuesday to Thursday", "The call is on Tuesday"),
    ("The call is on Thursday rather than Tuesday", "The call is on Tuesday"),
    ("The call is on Thursday instead of Tuesday", "The call is on Tuesday"),
    ("The call is not on Tuesday", "The call is on Tuesday"),
    ("Standup is at 11", "Standup is at 10"),
    ("Standup is at 11 o'clock", "Standup is at 10 o'clock"),
    ("The conference is in 2027", "The conference is in 2026"),
    ("Launch is next Tuesday", "Launch is this Tuesday"),
    ("Launch slipped to Q4", "Launch is in Q3"),
    ("The call is at half past ten", "The call is at ten"),
    ("Standup is at 10.30am", "Standup is at 10am"),
    ("Standup is at 10h30", "Standup is at 10h00"),
    ("Standup moved to 1400", "Standup is at 1000"),
    ("Standup is at 14:00 UTC", "Standup is at 14:00 CET"),
    ("Ilona's call is on Thursday", "Ilona's call is Tuesday"),
    ("Ilona's call is Thursday, not Tuesday", "Ilona's call is on Tuesday"),
    ("Ilona call Thu 10am AZ", "Ilona call Tue 10am AZ"),
    ("Ilona's call is on Thursday.", "Call with Ilona is on Tuesday."),
    ("The call with Ilona moved to Thursday", "Ilona's call is on Tuesday"),
    ("Ilona: call Thursday", "Ilona: call Tuesday"),
    ("We'll meet on Thursday", "We will meet on Tuesday"),
    ("We'll meet the auditors on Thursday.", "We will meet the auditors on Tuesday."),
    ("The deadline is October 9", "The deadline is October 8"),
    ("The deadline is 9 Oct 2026", "The deadline is 8 Oct 2026"),
    ("The deadline is Oct. 9", "The deadline is Oct. 8"),
    ("Rent is due on the 7th of each month", "Rent is due on the 5th of each month"),
    ("Deploys are allowed on Fridays", "Deploys are not allowed on Fridays"),
    ("Do not deploy on Friday", "Deploy on Friday"),
    ("Do not deploy to production on Friday.", "Deploy to production on Friday."),
    ("Alice does not work on Tuesdays", "Alice works on Tuesdays"),
    ("The gym closes at 9pm", "The gym closes at 10pm"),
    ("The gym closes at 21:00", "The gym closes at 22:00"),
    ("The gym closes at 9 PM", "The gym closes at 10 PM"),
    ("The office opens at 8:30 Eastern", "The office opens at 8:30 Central"),
    ("Ilona is in Arizona time", "Ilona is in Pacific time"),
    ("Alice is in Pacific time", "Alice is in Eastern time"),
    ("Ilona's timezone is Arizona", "Ilona's timezone is Pacific"),
    ("Ilona's time zone: MST", "Ilona's time zone: PST"),
    ("Ilona lives in UTC-7", "Ilona lives in UTC-8"),
    ("The 1:1 is Tuesdays at 10", "The 1:1 is Thursdays at 10"),
    ("Standup: Tue→Thu", "Standup: Tue"),
    ("Standup: Tue -> Thu", "Standup: Tue"),
    ("TUESDAY standup", "THURSDAY standup"),
    ("Weekly call with Ilona at 10am (Pacific).", "Weekly call with Ilona at 10am (Arizona)."),
    ("Weekly call with Ilona at 10am - Pacific.", "Weekly call with Ilona at 10am - Arizona."),
    ("Weekly call with Ilona at 10am/Pacific.", "Weekly call with Ilona at 10am/Arizona."),
    ("Weekly call with Ilona at Pacific 10am.", "Weekly call with Ilona at Arizona 10am."),
    ("Weekly call with Ilona at 10am, Pacific.", "Weekly call with Ilona at 10am, Arizona."),
    ("Weekly call with Ilona at 10am [Pacific].", "Weekly call with Ilona at 10am [Arizona]."),
    ("Meet at 1600 tomorrow.", "Meet at 1700 tomorrow."),
    ("Deadline is 2026-10-09T17:00.", "Deadline is 2026-10-08T17:00."),
    ("Ilona does not attend the Tuesday sync.", "Ilona attends the Tuesday sync."),
    ("The standup is on Tue and Thu.", "The standup is on Mon and Wed."),
    ("The backup job is disabled.", "The backup job is enabled."),
    ("Ilona's call is Thu 10/9", "Ilona's call is Tue 10/7"),
    ("Ilona is not on the call anymore.", "Ilona is on the call."),
    ("The call with Ilona is cancelled.", "The call with Ilona is on Tuesday."),
    ("The review moved to March", "The review is in February"),
    ("The meeting is at the end of May", "The meeting is at the end of June"),
    ("Ilona prefers the 9:00 train.", "Ilona prefers the 8:00 train."),
    ("Ilona joined in 2020.", "Ilona joined in 2019."),
    ("Office hours are 10-11am", "Office hours are 9-10am"),
    ("Office hours are 10 - 11am", "Office hours are 9 - 10am"),
    ("Office hours are 10–11am", "Office hours are 9–10am"),
    ("Office hours are 10:00-11:00", "Office hours are 9:00-10:00"),
    ("Office hours are 14-16h.", "Office hours are 13-15h."),
    ("Standup at １０am", "Standup at ９am"),
    ("Standup at ٩am", "Standup at ٨am"),
    ("Standup at 10.30.", "Standup at 9.30."),
    ("Ilona is in CET.", "Ilona is in PST."),
    ("Ilona is in Europe/Berlin.", "Ilona is in America/Phoenix."),
    ("Ilona has no availability on Friday.", "Ilona has availability on Friday."),
    ("The call is Thursday (was Tuesday).", "The call is on Tuesday."),
    ("Retro is every other Friday.", "Retro is every Friday."),
    ("Retro is every Friday.", "Retro is every Monday."),
    ("Payroll runs on the first Monday.", "Payroll runs on the second Monday."),
    ("The sync is biweekly.", "The sync is weekly."),
    ("The deadline is 9.10.2026.", "The deadline is 8.10.2026."),
    ("The deadline is 2026.12.31.", "The deadline is 2026.12.30."),
    ("The deadline is end of Q3.", "The deadline is end of Q4."),
    ("Ilona is off 9-10 Oct.", "Ilona is off 8-9 Oct."),
    ("Ilona is off from 9 to 10 Oct.", "Ilona is off from 8 to 9 Oct."),
    ("Ilona is off on 9 octobre.", "Ilona is off on 8 octobre."),
    ("会议在星期四。", "会议在星期二。"),
    ("会议在10:00。", "会议在9:00。"),
    ("The review takes 2 hours.", "The review takes 1 hour."),
    ("The release is in 3 days.", "The release is in 5 days."),
    ("The release is in two weeks.", "The release is in one week."),
    ("Lunch costs $12.", "Lunch costs $10."),
    ("The budget is 12%.", "The budget is 10%."),
    ("Prod runs v1.3.", "Prod runs v1.2."),
    ("Ilona's rate is €90/h.", "Ilona's rate is €80/h."),
    ("Launch is on the 9th of October", "Launch is on the 9th of November"),
    ("Launch is on the ninth of October", "Launch is on the tenth of October"),
    ("Call at 10am UTC+5:30", "Call at 10am UTC+1"),
    ("Growth was -5%", "Growth was 5%"),
    ("Growth was −5%", "Growth was 5%"),
    ("The fee is -$50", "The fee is $50"),
    ("The license fee is 12 USD.", "The license fee is 15 USD."),
    ("The fee is 10€", "The fee is 12€"),
    ("The fee is EUR 10", "The fee is EUR 12"),
    ("The fee is 500円", "The fee is 600円"),
    ("The budget is $5k", "The budget is $6k"),
    ("Growth was 5 percent", "Growth was 6 percent"),
    ("The freezer is at -18 degrees", "The freezer is at -20 degrees"),
    ("The freezer is at -18°C", "The freezer is at -20°C"),
    ("Use Python 3.12.", "Use Python 3.11."),
    ("Prod runs 1.2.4", "Prod runs 1.2.3"),
    ("Prod runs version 1.2.4", "Prod runs version 1.2.3"),
    ("Prod runs v1.2.3-rc2", "Prod runs v1.2.3-rc1"),
    ("Office hours are 9-5", "Office hours are 8-4"),
    ("Ilona is in America/New_York.", "Ilona is in America/Chicago."),
    ("Termin am 9. Oktober", "Termin am 10. Oktober"),
    ("Termin um 10 Uhr", "Termin um 11 Uhr"),
    ("Standup at 10：30", "Standup at 10：45"),
    ("会議は月曜日です", "会議は火曜日です"),
    ("会議は10月9日です", "会議は10月10日です"),
    ("会議は午後3時です", "会議は午後4時です"),
    ("회의는 월요일입니다", "회의는 화요일입니다"),
    ("Встреча в понедельник", "Встреча во вторник"),
    ("Встреча 9 октября", "Встреча 10 октября"),
    ("Spotkanie w poniedziałek", "Spotkanie we wtorek"),
    ("Vergadering op maandag", "Vergadering op dinsdag"),
    ("Möte på måndag", "Möte på tisdag"),
    ("Toplantı pazartesi", "Toplantı salı"),
    ("The standup is on _Thursday_.", "The standup is on _Tuesday_."),
]


@pytest.mark.parametrize("new,old", NOT_CORRECTIONS)
def test_not_a_correction(new: str, old: str) -> None:
    assert changed_fact(new, old) is False
    assert changed_fact(old, new) is False


@pytest.mark.parametrize("new,old", CORRECTIONS)
def test_correction_flagged(new: str, old: str) -> None:
    assert changed_fact(new, old) is True


@pytest.mark.parametrize("new,old", CORRECTIONS)
def test_correction_against_itself_is_not_flagged(new: str, old: str) -> None:
    # Positive control: the same text never contradicts itself.
    assert changed_fact(new, new) is False
    assert changed_fact(old, old) is False


# ----------------------------------------------------------------------------- merge normal form


@pytest.mark.parametrize(
    "a,b",
    [
        ("Call set_timeout before reconnect.", "Call set _timeout before reconnect."),
        ("Run deploy with --force on prod.", "Run deploy with - -force on prod."),
        ("Logs live in /var/log/app today.", "Logs live in /var/ log/app today."),
        ("Send invoices to ilona@acme.com each month.", "Send invoices to ilona @ acme . com each month."),
        ("The decimal separator is .", "The decimal separator is"),
    ],
)
def test_spacing_around_punctuation_is_content(a: str, b: str) -> None:
    assert slots_differ(a, b)


def test_amount_of_whitespace_and_attached_full_stop_do_not_matter() -> None:
    assert canonical_tokens("Weekly  call\non Tuesday.") == canonical_tokens("Weekly call on Tuesday")


# ----------------------------------------------------------------------------- merge identity


def _blk(bid: str, **fields: object) -> dict:
    return {"_id": bid, "content": "Weekly planning call at 10am.", "Status": "active", **fields}


@pytest.mark.parametrize(
    "field,a,b",
    [
        ("Participants", ["Ilona"], ["the contractor"]),
        ("Attendees", ["Ilona (Arizona)"], ["contractor (Pacific)"]),
        ("When", ["2026-10-06 10am MST"], ["2026-10-08 10am PST"]),
        ("ConstraintSignatures", [{"object": "10am Arizona"}], [{"object": "10am Pacific"}]),
        ("Rationale", "Ilona asked (Arizona time).", "The contractor asked (Pacific time)."),
    ],
)
def test_list_and_mapping_fields_separate_blocks(field: str, a: object, b: object) -> None:
    blocks = [_blk("D-1", **{field: a}), _blk("D-2", **{field: b})]
    assert find_merge_candidates(blocks) == []
    assert [(p.id_a, p.id_b) for p in find_similar_pairs(blocks, min_similarity=0.0)] == [("D-1", "D-2")]
    # Positive control: equal values in the same field DO merge.
    same = [_blk("D-1", **{field: a}), _blk("D-2", **{field: a})]
    assert len(find_merge_candidates(same)) == 1


def test_nested_fields_projection_is_compared_too() -> None:
    a = {"_id": "D-1", "content": "Weekly call.", "fields": {"Participants": ["Ilona"]}}
    b = {"_id": "D-2", "content": "Weekly call.", "fields": {"Participants": ["the contractor"]}}
    assert find_merge_candidates([a, b]) == []


@pytest.mark.parametrize("key", [7, ("a",), 1.5])
def test_non_string_keys_do_not_crash(key: object) -> None:
    a = {"_id": "A", "content": "call ilona tuesday", key: "x"}
    b = {"_id": "B", "content": "call ilona tuesday", key: "x"}
    c = {"_id": "C", "content": "call ilona tuesday", "fields": {key: "y", "Date": "d"}}
    assert len(find_merge_candidates([a, b])) == 1
    candidates, pairs = scan_pairs([a, b, c], min_similarity=0.0)
    assert len(candidates) == 1 and len(pairs) == 2


def test_non_mapping_entries_are_skipped() -> None:
    a = {"_id": "A", "content": "call ilona tuesday"}
    assert len(find_merge_candidates([a, None, dict(a, _id="B"), "text"])) == 1  # type: ignore[list-item]


@pytest.mark.parametrize("status", ["superseded", "archived", "retracted", "Superseded"])
def test_only_active_blocks_merge(status: str) -> None:
    blocks = [_blk("D-1"), _blk("D-2", Status=status)]
    assert find_merge_candidates(blocks) == []
    assert len(find_merge_candidates([_blk("D-1"), _blk("D-2")])) == 1


def test_merge_keeps_both_blocks_provenance() -> None:
    a = _blk("D-1", Sources="email from Ilona 2026-09-01", fields={"Sources": ["email A"]})
    b = _blk("D-2", Sources="email from contractor 2026-09-30", fields={"Sources": ["email B"]})
    (cand,) = find_merge_candidates([a, b])
    merged = merge_blocks(cand.block_a, cand.block_b, strategy=cand.suggested_strategy)
    assert merged["Sources"] == "email from Ilona 2026-09-01; email from contractor 2026-09-30"
    assert merged["fields"]["Sources"] == ["email A", "email B"]


def test_one_pass_returns_both_lists() -> None:
    blocks = [_blk("D-1"), _blk("D-2"), _blk("D-3", Participants=["Ilona"])]
    candidates, pairs = scan_pairs(blocks, min_similarity=0.0)
    assert [(c.block_a["_id"], c.block_b["_id"]) for c in candidates] == [("D-1", "D-2")]
    assert sorted((p.id_a, p.id_b) for p in pairs) == [("D-1", "D-3"), ("D-2", "D-3")]


# ----------------------------------------------------------------------------- end to end


def _workspace(tmp_path: Path, body: str) -> Path:
    (tmp_path / "decisions").mkdir()
    (tmp_path / "mind-mem.json").write_text(
        json.dumps({"version": "5.0.1", "v4": {"granularity_align": {"enabled": True, "min_similarity": 0.5}}}), encoding="utf-8"
    )
    (tmp_path / "decisions" / "DECISIONS.md").write_text("# DECISIONS\n\n" + body, encoding="utf-8")
    return tmp_path


def _decision(bid: str, extra: str = "", status: str = "active") -> str:
    return (
        f"[{bid}]\nDate: 2026-10-01\nStatus: {status}\nScope: global\nStatement: Weekly planning call at 10am.\n"
        f"{extra}Rationale: seed.\nSupersedes: none\nTags: test\nSources: test\n\n"
    )


def _plan(ws: Path) -> dict:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.consolidation import plan_consolidation
    from mind_mem.sqlite_index import build_index

    build_index(str(ws))
    with use_workspace(str(ws)):
        return json.loads(plan_consolidation())["granularity_align"]


def test_plan_consolidation_list_participants_not_merged(tmp_path: Path) -> None:
    ws = _workspace(
        tmp_path,
        _decision("D-20261001-001", "Participants:\n- Ilona\n") + _decision("D-20261001-002", "Participants:\n- the contractor\n"),
    )
    g = _plan(ws)
    assert g["scanned_blocks"] == 2  # positive control: both blocks were read
    assert g["candidates"] == []
    assert [(p["id_a"], p["id_b"]) for p in g["similar_pairs"]] == [("D-20261001-001", "D-20261001-002")]


def test_plan_consolidation_exact_duplicates_still_merge(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, _decision("D-20261001-001") + _decision("D-20261001-002"))
    g = _plan(ws)
    assert [(c["id_a"], c["id_b"]) for c in g["candidates"]] == [("D-20261001-001", "D-20261001-002")]


def test_plan_consolidation_superseded_copy_not_merged(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, _decision("D-20261001-001", status="superseded") + _decision("D-20261001-002"))
    g = _plan(ws)
    assert g["scanned_blocks"] == 2
    assert g["candidates"] == []


def test_build_index_accepts_list_statement(tmp_path: Path) -> None:
    from mind_mem.sqlite_index import build_index

    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: active\n"
        "Statement:\n- first item of a list statement\n- second item\nContext:\n- a\nTags: t\n",
        encoding="utf-8",
    )
    result = build_index(str(tmp_path))
    assert result["blocks_indexed"] == 1


def test_contradiction_detector_end_to_end(tmp_path: Path) -> None:
    from mind_mem.contradiction_detector import check_proposal_contradictions

    old = "The weekly call with Ilona is on Tuesday at 10am Arizona time."
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        f"# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        f"Statement: {old}\nRationale: seed.\nSupersedes: none\nTags: test\nSources: test\n",
        encoding="utf-8",
    )

    def kinds(statement: str) -> list[str]:
        return [c["conflict_type"] for c in check_proposal_contradictions(str(tmp_path), {"Statement": statement})["conflicts"]]

    assert "contradiction" in kinds("The weekly call with Ilona is on Thursday at 10am Arizona time.")
    assert "contradiction" not in kinds("Alice flies to Arizona at 10am on Tuesday with Ilona weekly.")
    assert "contradiction" not in kinds("The weekly call with the contractor is on Tuesday at 10am Pacific time.")


# ----------------------------------------------------------------------------- round 3


def test_plan_consolidation_list_statements_not_merged(tmp_path: Path) -> None:
    body = (
        "[D-20261001-001]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement:\n- Weekly call with Ilona Tue 10am Arizona time\nTags: test\n\n"
        "[D-20261001-002]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement:\n- Weekly call with the contractor Thu 3pm Pacific time\nTags: test\n\n"
    )
    g = _plan(_workspace(tmp_path, body))
    assert g["scanned_blocks"] == 2
    assert g["candidates"] == []


@pytest.mark.parametrize(
    "extra_a,extra_b",
    [
        ("speaker: Ilona\n", "speaker: Bob\n"),
        ("Meta:\n  tz: Arizona\n", "Meta:\n  tz: Pacific\n"),
    ],
)
def test_plan_consolidation_unparsed_lines_block_merge(tmp_path: Path, extra_a: str, extra_b: str) -> None:
    g = _plan(_workspace(tmp_path, _decision("D-20261001-001", extra_a) + _decision("D-20261001-002", extra_b)))
    assert g["scanned_blocks"] == 2
    assert g["candidates"] == []
    # Positive control: the same unparsed line on both sides still merges.
    same = tmp_path / "same"
    same.mkdir()
    g2 = _plan(_workspace(same, _decision("D-20261001-001", extra_a) + _decision("D-20261001-002", extra_a)))
    assert len(g2["candidates"]) == 1


def test_parser_keeps_unrecognised_lines() -> None:
    from mind_mem.block_parser import parse_blocks

    (block,) = parse_blocks("[D-20261001-001]\nStatement: x\nspeaker: Bob\nMeta:\n  tz: Pacific\n## Heading\n")
    # A '#' line at end of file has no next section to belong to: it stays with the block.
    assert block["_unparsed"] == ["speaker: Bob", "  tz: Pacific", "## Heading"]


@pytest.mark.parametrize(
    "new,old",
    [
        ("Enable the weekly report for Ilona on Tuesday.", "Disable the weekly report for Bob on Tuesday."),
        ("Ilona will start the weekly call on Tuesday at 10am.", "Bob will stop the weekly call on Tuesday at 10am."),
    ],
)
def test_reversal_words_need_the_same_subject(tmp_path: Path, new: str, old: str) -> None:
    from mind_mem.contradiction_detector import _classify_conflict

    assert _classify_conflict(new, old, 0.8) != "contradiction"
    # Positive control: the same subject with the reversal IS a contradiction.
    reversed_same = new.replace("Enable", "Disable").replace("start", "stop")
    assert _classify_conflict(new, reversed_same, 0.8) == "contradiction"


def test_superseded_block_is_not_a_contradiction_candidate(tmp_path: Path) -> None:
    from mind_mem.contradiction_detector import check_proposal_contradictions

    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: superseded\nScope: global\n"
        "Statement: The weekly call with Ilona is on Tuesday at 10am.\nTags: test\n\n"
        "[D-20261001-002]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement: The weekly call with Ilona is on Wednesday at 10am.\nTags: test\n",
        encoding="utf-8",
    )
    report = check_proposal_contradictions(str(tmp_path), {"Statement": "The weekly call with Ilona is on Thursday at 10am."})
    ids = {c["block_id"] for c in report["conflicts"] if c["conflict_type"] == "contradiction"}
    assert ids == {"D-20261001-002"}  # positive control: the active block is still flagged


# ----------------------------------------------------------------------------- round 4


@pytest.mark.parametrize(
    "extra_a,extra_b",
    [
        ("Statement: Tue 10am Arizona time\n", "Statement: Tue 10am Arizona time\n"),
    ],
)
def test_repeated_statement_key_keeps_the_overwritten_value(tmp_path: Path, extra_a: str, extra_b: str) -> None:
    body = (
        "[D-20261001-001]\nDate: 2026-10-01\nStatus: active\nScope: global\nStatement: Weekly call with Ilona\n"
        + extra_a
        + "Tags: test\n\n[D-20261001-002]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement: Weekly call with the contractor\n" + extra_b + "Tags: test\n\n"
    )
    g = _plan(_workspace(tmp_path, body))
    assert g["scanned_blocks"] == 2
    assert g["candidates"] == []


@pytest.mark.parametrize(
    "extra_a,extra_b",
    [
        ("#ilona\n", "#contractor\n"),
        ("## Attendee: Ilona\n", "## Attendee: Bob\n"),
        ("Statement2: x\n  - with Ilona\n", "Statement2: x\n  - with Bob\n"),
    ],
)
def test_lines_the_parser_used_to_drop_block_merge(tmp_path: Path, extra_a: str, extra_b: str) -> None:
    g = _plan(_workspace(tmp_path, _decision("D-20261001-001", extra_a) + _decision("D-20261001-002", extra_b)))
    assert g["scanned_blocks"] == 2
    assert g["candidates"] == []


def test_parser_capture_rules() -> None:
    from mind_mem.block_parser import parse_blocks

    (block,) = parse_blocks("[D-1]\nStatement: first\nStatement: second\nTags: t\n")
    assert block["Statement"] == "second" and block["_unparsed"] == ["Statement: first"]
    a, b = parse_blocks("[D-1]\nStatement: S\n#tag\nRationale: r\n\n## Next section\n\n[D-2]\nStatement: x\n## trailing\n")
    assert a["_unparsed"] == ["#tag"]  # a heading after the block's content is the next section's
    assert b["_unparsed"] == ["## trailing"]  # end of file: kept with the last block (fail closed)
    (sig,) = parse_blocks("[D-1]\nStatement: x\nConstraintSignatures:\n- id: CS-1\n  scope:\n    time:\n      start: 2026-01-01\n")
    assert "_unparsed" not in sig  # a well-formed key-only nested line is structure
    assert sig["ConstraintSignatures"][0]["scope"]["time"] == {"start": "2026-01-01"}


def test_supersedes_link_removes_a_block_from_contradiction_candidates(tmp_path: Path) -> None:
    from mind_mem.contradiction_detector import check_proposal_contradictions

    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement: The weekly call with Ilona is on Tuesday at 10am.\nTags: test\n\n"
        "[D-20261001-002]\nDate: 2026-10-02\nStatus: active\nScope: global\nSupersedes: D-20261001-001\n"
        "Statement: The weekly call with Ilona is on Wednesday at 10am.\nTags: test\n",
        encoding="utf-8",
    )
    report = check_proposal_contradictions(str(tmp_path), {"Statement": "The weekly call with Ilona is on Thursday at 10am."})
    ids = {c["block_id"] for c in report["conflicts"] if c["conflict_type"] == "contradiction"}
    assert ids == {"D-20261001-002"}


def test_contradiction_report_never_echoes_withheld_text(tmp_path: Path) -> None:
    from mind_mem.contradiction_detector import check_proposal_contradictions

    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: quarantined\nScope: global\n"
        "Statement: The weekly call with Ilona is on Tuesday at 10am SECRET-QUARANTINED.\nTags: test\n\n"
        "[D-20261001-002]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement: The weekly call with Ilona is on Wednesday at 10am.\nTags: test\n",
        encoding="utf-8",
    )
    report = check_proposal_contradictions(str(tmp_path), {"Statement": "The weekly call with Ilona is on Thursday at 10am."})
    text = json.dumps(report)
    assert "D-20261001-002" in text  # positive control: the active block is compared
    assert "SECRET-QUARANTINED" not in text and "D-20261001-001" not in text


def test_to_month_text_stays_linear() -> None:
    import time

    def seconds(n: int) -> float:
        start = time.perf_counter()
        changed_fact("to May " * n, "to May " * n + "x")
        return time.perf_counter() - start

    # Scaling, not a wall-clock budget, which fails on a loaded host. Was quadratic: ~570 s at 40,000.
    half = seconds(20000)
    full = seconds(40000)
    assert full < 30.0
    assert full <= 3.0 * half + 0.25, f"doubling the input took {full:.2f}s vs {half:.2f}s: super-linear"


def test_planner_never_keeps_a_block_another_block_supersedes(tmp_path: Path) -> None:
    body = (
        _decision("D-20261001-001") + "[D-20261001-002]\nDate: 2026-10-02\nStatus: active\nScope: global\nSupersedes: D-20261001-001\n"
        "Statement: Weekly planning call at 11am.\nTags: test\n\n" + _decision("D-20261001-003")
    )
    g = _plan(_workspace(tmp_path, body))
    assert g["scanned_blocks"] == 3
    for cand in g["candidates"]:
        assert "D-20261001-001" not in (cand["id_a"], cand["id_b"])


def test_parser_round5_shapes() -> None:
    from mind_mem.block_parser import parse_blocks

    (a,) = parse_blocks("[D-1]\nParticipants:\n- Ilona\nParticipants:\n- the contractor\n")
    assert a["_unparsed"] == ["Participants: Ilona"]
    (b,) = parse_blocks("[D-1]\nStatement: s\n--- with Ilona\nTags: t\n")
    assert b["_unparsed"] == ["--- with Ilona"] and b["Tags"] == "t"
    assert len(parse_blocks("[D-1]\nStatement: a\n---\n[D-2]\nStatement: b\n")) == 2


def test_long_statement_is_compared_in_full(tmp_path: Path) -> None:
    from mind_mem.contradiction_detector import check_proposal_contradictions

    tail = " ".join(f"Context sentence number {k} about the planning process." for k in range(15))
    old = "The weekly call with Ilona is on Tuesday at 10am. " + tail
    new = "The weekly call with Ilona is on Thursday at 10am. " + tail
    assert len(old) > 600
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        f"# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: active\nScope: global\nStatement: {old}\nTags: test\n",
        encoding="utf-8",
    )
    report = check_proposal_contradictions(str(tmp_path), {"Statement": new})
    assert report["has_contradictions"]


@pytest.mark.parametrize("field", ["SupersededBy", "Superseded_by"])
def test_block_marked_superseded_never_merges(field: str) -> None:
    a = _blk("D-1", **{field: "D-3"})
    b = _blk("D-2", **{field: "D-3"})
    assert find_merge_candidates([a, b]) == []
    assert len(find_merge_candidates([_blk("D-1"), _blk("D-2")])) == 1  # positive control


@pytest.mark.parametrize("ref", ["d-1", "D-1", "1001"])
def test_supersede_reference_matching_is_case_insensitive(ref: str) -> None:
    a = _blk("D-1" if ref != "1001" else "1001")
    b = _blk("D-2" if ref != "1001" else "1002")
    c = {"_id": "D-9", "content": "Something else entirely.", "Status": "active", "Supersedes": ref}
    assert find_merge_candidates([a, b, c]) == []


# ----------------------------------------------------------------------------- round 7 battery

_BATTERY = json.loads((Path(__file__).parent / "fixtures" / "fact_slots_changed_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _BATTERY, ids=[c["id"] for c in _BATTERY])
def test_changed_fact_battery(case: dict) -> None:
    # Independent blind-review battery: currencies, signs, ranges, units, versions, ids,
    # noon/midnight, and dates/times in 12 languages. Expectations follow the operator's rulings.
    assert changed_fact(case["new"], case["old"]) is case["changed"]


@pytest.mark.parametrize(
    "new,old",
    [
        ("Deploy 1.2.3 - Alice owns it", "Deploy 1.2.3 - Bob owns it"),
        ("mind-mem 5.0.4 fixes the recall bug.", "mind-mem 5.0.3 fixes the recall bug."),
        ("Django 4.2.2 is the LTS release.", "Django 4.2.1 is the LTS release."),
        ("Bob works at Pazar on Monday.", "Bob works at Mars on Monday."),
        ("Call May on Thursday.", "Call Mai on Thursday."),
        ("The backup runs at 00:00.", "The backup runs at midnight."),
        ("The budget is $5,000.", "The budget is $5000."),
        ("The deadline is 31.12.2026", "The deadline is 2026-12-31"),
    ],
)
def test_round7_not_changes(new: str, old: str) -> None:
    assert changed_fact(new, old) is False


def test_supersedes_with_trailing_punctuation(tmp_path: Path) -> None:
    for ref in ("D-1.", "D-1, D-9."):
        blocks = [_blk("D-1"), _blk("D-2"), {"_id": "D-3", "content": "Other.", "Status": "active", "Supersedes": ref}]
        assert find_merge_candidates(blocks) == []


@pytest.mark.parametrize(
    "new,old",
    [
        ("The job runs 2.5 hours.", "The job runs 2.50 hours."),
        ("Set the room to 21C.", "Set the room to 21 °C."),
        ("Mötet är i mars", "Mötet är i mars."),
        ("The range is 10–20 units.", "The range is 10-20 units."),
        ("I am April's manager.", "I am April's manager."),
        ("Budget is 10–20 USD.", "Budget is 10-20 USD."),
    ],
)
def test_round8_not_changes(new: str, old: str) -> None:
    assert changed_fact(new, old) is False


@pytest.mark.parametrize(
    "new,old",
    [
        ("The job runs 2.5 hours.", "The job runs 3.5 hours."),
        ("Set the room to 21C.", "Set the room to 23C."),
        ("Mötet är i mars", "Mötet är i maj"),
        ("Budget is 10–20 USD.", "Budget is 10-25 USD."),
        ("Budget is $10-20.", "Budget is $10-25."),
        ("The range is 10 units.", "The range is 12 units."),  # a count with its noun is a stated value
    ],
)
def test_round8_changes(new: str, old: str) -> None:
    assert changed_fact(new, old) is True


@pytest.mark.parametrize(
    "new,old",
    [
        ("La réunion est le mardi à 14h.", "La réunion est le mardi à 15h."),
        ("Das Meeting ist um 14h.", "Das Meeting ist um 16h."),
        ("The grant is 40 000 EUR.", "The grant is 45 000 EUR."),
        ("The disk is 500 GB.", "The disk is 1 TB."),
        ("Box weighs 2.5 kg.", "Box weighs 3.5 kg."),
        ("Timeout is 200 ms.", "Timeout is 250 ms."),
        ("The team has 7 engineers.", "The team has 9 engineers."),
        ("The team is 7 people.", "The team is 9 people."),
        ("The demo is Thu at 4pm ET.", "The demo is Fri at 4pm ET."),
        ("Backup at 02:00 UTC", "Backup at 2pm UTC"),
        ("La riunione è martedì.", "La riunione è giovedì."),
        ("A reunião é na terça-feira.", "A reunião é na quinta-feira."),
        ("La cita es el 15 de marzo.", "La cita es el 18 de marzo."),
        ("Аренда 500 000 рублей.", "Аренда 600 000 рублей."),
        ("Kira 15.000 TL.", "Kira 17.500 TL."),
        ("家賃は50万円です。", "家賃は60万円です。"),
        ("월세는 500만원입니다.", "월세는 600만원입니다."),
        ("The timezone for reports is UTC.", "The timezone for reports is Pacific."),
    ],
)
def test_round9_changes(new: str, old: str) -> None:
    assert changed_fact(new, old) is True


@pytest.mark.parametrize(
    "new,old",
    [
        ("The grant is 40 000 EUR.", "The grant is 40,000 EUR."),
        ("Le loyer est de 1 200 €.", "Le loyer est de 1200 €."),
        ("Backup at 2am UTC", "Backup at 02:00 UTC"),
        ("Sync at 10:00", "Sync at 10am"),
        ("Sync at 10:00", "Sync at 10pm"),
        ("Alice has 3 cats.", "Alice has 2 dogs."),
        ("Route 66 runs east", "Route 101 runs east"),
        ("The fee is €5.000.", "The fee is €5."),
    ],
)
def test_round9_not_changes(new: str, old: str) -> None:
    assert changed_fact(new, old) is False


@pytest.mark.parametrize(
    "text",
    [
        "1.5 " * 16_000,
        "node 1.2 and python 3.11 and go 1.22 and rust 1.80 and java 21.0 " * 500,
        "The window is 14h and 30min long " * 2_000,
    ],
    ids=["decimals", "versions", "durations"],  # pytest puts the id in an env var, capped at 32,767 chars on Windows
)
def test_context_checks_stay_linear(text: str) -> None:
    import time

    def seconds(t: str) -> float:
        start = time.perf_counter()
        changed_fact(t, t + " x")
        return time.perf_counter() - start

    half = seconds(text[: len(text) // 2])
    full = seconds(text)
    assert full < 30.0
    assert full <= 3.0 * half + 0.25, f"doubling the input took {full:.2f}s vs {half:.2f}s: super-linear"


@pytest.mark.parametrize("superseder", [{"_id": "D-9", "Supersedes": "C"}, {"_id": "D-9", "Supersedes": "C", "content": "x"}])
def test_superseder_without_text_still_blocks_the_merge(superseder: dict) -> None:
    blocks = [
        {"_id": "A", "content": "The standup is on Tuesday at 10am.", "Status": "active"},
        {"_id": "C", "content": "The standup is on Tuesday at 10am.", "Status": "active"},
        superseder,
    ]
    assert find_merge_candidates([b for b in blocks if b["_id"] != "D-9"]) != []  # positive control
    assert find_merge_candidates(blocks) == []


@pytest.mark.parametrize(
    "new,old,changed",
    [
        ("Rate limit is 100 req/min.", "Rate limit is 100 req/s.", True),
        ("CI uses Node 22.", "CI uses Node 20.", True),
        ("The SDK requires Java 21 or later.", "The SDK requires Java 17 or later.", True),
        ("Use Room 14 today", "Use Room 12 today", False),
    ],
)
def test_round9_rates_and_product_versions(new: str, old: str, changed: bool) -> None:
    assert changed_fact(new, old) is changed


@pytest.mark.parametrize(
    "old,new",
    [
        ("Meet Sat at 10am", "Meet Sun at 10am"),
        ("Deploys happen Tue at 10am", "Deploys happen Thu at 10am"),
        ("The review is Mon at 2pm and Wed at 4pm", "The review is Mon at 2pm and Thu at 4pm"),
        ("Møtet er på tirsdag", "Møtet er på torsdag"),
        ("Retention is 48h for logs", "Retention is 72h for logs"),
        ("Latency is 120ms", "Latency is 150ms"),
        ("Disk is 500GB", "Disk is 750GB"),
        ("Bandwidth is 100 Mbps", "Bandwidth is 1 Gbps"),
        ("We send 10 messages per minute", "We send 20 messages per minute"),
        ("We allow 60 calls per hour", "We allow 60 calls per day"),
        ("Budget is 1.250.000 ₽", "Budget is 1.300.000 ₽"),
        ("Rent is 1 200,50 €", "Rent is 2 200,50 €"),
        ("Cena je 500 Kč", "Cena je 600 Kč"),
        ("Budget is $5-10k", "Budget is $5-12k"),
        ("O prazo é 30 de setembro", "O prazo é 30 de outubro"),
        ("La riunione è il 3 marzo alle 10", "La riunione è il 3 marzo alle 11"),
        ("会议在下午3点", "会议在下午4点"),
        ("Kokous alkaa klo 9", "Kokous alkaa klo 10"),
        ("The team has 7 engineers and 2 designers", "The team has 7 engineers and 3 designers"),
        ("인원은 12명입니다", "인원은 15명입니다"),
        ("Schůzka je ve čtvrtek", "Schůzka je v pátek"),
        ("Η συνάντηση είναι την Τρίτη.", "Η συνάντηση είναι την Πέμπτη."),
    ],
)
def test_round10_changes(old: str, new: str) -> None:
    assert changed_fact(new, old) is True


@pytest.mark.parametrize(
    "old,new",
    [
        ("Lunch with Sun at noon", "Lunch with Wed at noon"),
        ("予算は50万円です", "予算は500000円です"),
        ("予算は50万円です", "予算は500,000円です"),
        ("Kosten: 2.500 €", "Kosten: 2.500,00 €"),
        ("Use the 3D printer in the lab.", "Use the 2D printer in the lab."),
        ("Take bus 5d to work.", "Take bus 6d to work."),
    ],
)
def test_round10_not_changes(old: str, new: str) -> None:
    assert changed_fact(new, old) is False


@pytest.mark.parametrize("digits", [30, 308, 309, 400, 5000])
def test_cjk_ten_thousands_never_overflow(digits: int) -> None:
    text = "予算は" + "9" * digits + "万円です。"
    fact_slots(text)
    changed_fact(text, text.replace("予算", "費用"))


@pytest.mark.parametrize(
    "old,new",
    [
        ("EOD note: ship review is Tue.", "EOD note: ship review is Thu."),
        ("Meet Tues.", "Meet Thurs."),
        ("We pushed the review from Mon to Wed.", "The review is on Mon."),
        ("The price changed from $19 to $29.", "The price is $19."),
        ("The stipend went from $2k to $3k.", "The stipend is $2k."),
        ("The budget is 5 grand.", "The budget is 8 grand."),
        ("The discount is thirteen percent.", "The discount is fourteen percent."),
        ("The run is 5 km.", "The run is 8 km."),
        ("The run is 5 km.", "The run is 5 mi."),
        ("Timeout is 30s.", "Timeout is 45s."),
        ("Set the thermostat to 72 F.", "Set the thermostat to 68 F."),
        ("Setpoint is 21C.", "setpoint is 23C"),
        ("We pinned Postgres to 15.", "We pinned Postgres to 16."),
        ("Install version 3.1.0.", "Install version 3.2.0."),
        ("We run 1.3.2.", "We now run 1.4.0, not 1.3.2."),
        ("We run 1.2.4 in prod.", "we now run 1.2.5 in prod"),
        ("The SLA is 100-200 ms.", "The SLA is 100-300 ms."),
        ("The temperature range is 68-72°F.", "The temperature range is 65-72°F."),
        ("Headcount band is 10-12 people.", "Headcount band is 10-15 people."),
        ("The approved range is 3-5 days.", "The approved range is 3-8 days."),
        ("Minutes: approved 10 contractors.", "Minutes: approved 14 contractors."),
        ("We have thirteen engineers.", "We have fourteen engineers."),
        ("We have twenty engineers.", "We have twenty-one engineers."),
    ],
)
def test_round11_changes(old: str, new: str) -> None:
    assert changed_fact(new, old) is True


@pytest.mark.parametrize(
    "old,new",
    [
        ("The call is at 10am PT.", "The call is at 10am Pacific."),
        ("The call is at 10am ET.", "The call is at 10am Eastern."),
        ("The demo is Thu at 4pm ET.", "The demo is Thursday at 4:00 PM Eastern."),
        ("The report is due on 10/09/2026.", "The report is due on 2026-10-09."),
        ("Ship target is 2026-04-01.", "Ship target is 1 April 2026."),
        ("The budget is $5,000.", "The budget is $5k."),
        ("The budget is $2,500,000.", "The budget is $2.5m."),
        ("Travel cap is 3000 dollars.", "Travel cap is $3k."),
        ("The discount is 10%.", "The discount is ten percent."),
        ("Locker 5-10 belongs to Ilona", "Locker 5-12 belongs to Ilona"),
        ("Run 12 failed", "Run 14 failed"),
    ],
)
def test_round11_not_changes(old: str, new: str) -> None:
    assert changed_fact(new, old) is False
