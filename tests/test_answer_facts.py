"""Computed evidence facts: resolved event dates, deltas, neighbours and choice options.

Fixtures reproduce the arithmetic shapes the answerer got wrong (weeks across months, a relative date
in an older block, before/after neighbours) with their own wording."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from mind_mem.answer_facts import choice_options, computed_facts, event_date, names_an_option, parse_date, timeline_facts


@dataclass
class Ev:
    block_id: str
    date: str
    excerpt: str


def test_parse_date_forms():
    assert parse_date("on 25 February 2024.") == date(2024, 2, 25)
    assert parse_date("on February 25, 2024") == date(2024, 2, 25)
    assert parse_date("on 2024-02-25") == date(2024, 2, 25)
    assert parse_date("on 9 Jun 2024") == date(2024, 6, 9)
    assert parse_date("no date here") is None


def test_relative_phrases_resolve_against_the_block_date():
    assert event_date("I planted the hedge yesterday.", date(2023, 8, 13))[0] == date(2023, 8, 12)
    assert event_date("Finished it two days ago.", date(2024, 3, 28))[0] == date(2024, 3, 26)
    assert event_date("I did it today.", date(2024, 3, 28))[0] == date(2024, 3, 28)
    assert event_date("I went last Friday.", date(2024, 6, 1))[0] == date(2024, 5, 31)  # 2024-06-01 is a Saturday
    assert event_date("Explicit wins: on 3 May 2024, yesterday.", date(2024, 6, 1))[0] == date(2024, 5, 3)


def _lines(question, rows):
    return "\n".join(timeline_facts(question, [(b, parse_date(d), t) for b, d, t in rows]))


def test_weeks_ago_across_months_is_floor_of_days_over_seven():
    text = _lines(
        "Today is 9 Jun 2024. How many weeks ago was collecting the kitten?",
        [
            ("T-20240226-045", "2024-02-26", "I collected the kitten on 25 February 2024."),
            ("D-20240301-002", "2024-03-01", "The vet is open on Mondays."),
        ],
    )
    assert "[[T-20240226-045]] was 105 days (weeks = days // 7 = 15) ago" in text


def test_yesterday_in_an_older_block_then_weeks_to_today():
    text = _lines(
        "Today is 2023-12-09. How many weeks ago was laying the patio?", [("D-20230813-042", "2023-08-13", "I laid the patio yesterday.")]
    )
    assert "- 2023-08-12 (yesterday as of the block date) [[D-20230813-042]]" in text
    assert "119 days (weeks = days // 7 = 17)" in text


def test_current_date_block_supplies_today():
    text = _lines(
        "How many days ago was the bake-off loaf?",
        [
            ("PER-20240328-025", "2024-03-28", "I baked the bake-off loaf yesterday."),
            ("T-20240722-018", "2024-07-22", "Current date: 2024-07-22."),
        ],
    )
    assert "Today: 2024-07-22." in text
    assert "[[PER-20240328-025]] was 117 days" in text
    assert "[[T-20240722-018]] was" not in text  # the date block is not an event


def test_between_two_named_events():
    text = _lines(
        "How many days passed between opening the savings pot and the first five hundred?",
        [
            ("D-20230930-035", "2023-09-30", "I opened the savings pot on 27 September 2023."),
            ("D-20231102-044", "2023-11-02", "The savings pot hit the first five hundred today."),
        ],
    )
    assert "- [[D-20230930-035]] -> [[D-20231102-044]]: 36 days (5 weeks)" in text


def test_neighbours_are_adjacent_timeline_lines_not_guessed_anchors():
    rows = [
        ("A-20240301-001", "2024-03-01", "I joined the brass band on 18 March 2024."),
        ("B-20240410-002", "2024-04-10", "I played at the spring fete on 7 April 2024."),
        ("C-20240415-003", "2024-04-15", "I had the trumpet audition today."),
    ]
    text = _lines("What was the last thing I did before the trumpet audition?", rows)
    lines = text.splitlines()
    assert lines.index(next(line for line in lines if "[[B-20240410-002]]" in line)) + 1 == lines.index(
        next(line for line in lines if "[[C-20240415-003]]" in line)
    )
    assert "Right before" not in text


def test_all_pairwise_gaps_no_guessed_pair():
    rows = [
        ("A-1", "2024-03-01", "I bought the new phone on 1 March 2024."),
        ("B-2", "2024-03-10", "I bought a new phone case on 10 March 2024."),
        ("C-3", "2024-03-20", "I ran the dry-run cutover on 20 March 2024."),
    ]
    text = _lines("How many days passed between the new phone and the new phone case?", rows)
    assert "- [[A-1]] -> [[B-2]]: 9 days (1 weeks)" in text and "- [[A-1]] -> [[C-3]]: 19 days (2 weeks)" in text
    assert "Between" not in text


def test_no_temporal_cue_no_lines():
    assert _lines("Which dentist do I use?", [("D-20240101-001", "2024-01-01", "I use Dr Lane.")]) == ""


def test_choice_options_are_verbatim():
    assert choice_options(
        "Which slot should I book: the 9 am room with a projector, the 2 pm room with a projector or the 11 am room without one?"
    ) == ["the 9 am room with a projector", "the 2 pm room with a projector", "the 11 am room without one"]
    assert choice_options("Is the used bike on instalments or the used bike at full price today the right call?") == [
        "the used bike on instalments",
        "the used bike at full price today",
    ]
    assert choice_options("Should I sign up for running the raffle or marshalling the race?") == [
        "running the raffle",
        "marshalling the race",
    ]
    assert choice_options("Is taking the dog in the van or leaving him with my sister the better plan?") == [
        "taking the dog in the van",
        "leaving him with my sister",
    ]
    assert choice_options("What is my rent?") == []


def test_names_an_option():
    opts = ["the 9 am room", "the 2 pm room"]
    assert names_an_option("Book the 2 pm room [[D-1]].", opts)
    assert not names_an_option("Book the 10 am room.", opts)


def test_computed_facts_block_and_empty_case():
    ev = [Ev("D-20230813-042", "2023-08-13", "I laid the patio yesterday.")]
    block = computed_facts("Today is 2023-12-09. How many weeks ago was laying the patio?", ev)
    assert block.startswith("Computed from the evidence")
    assert computed_facts("What is my rent?", ev) == ""


def test_weeks_use_floor_not_rounding():
    text = _lines(
        "Today is 2024-06-14. How many weeks ago was collecting the kitten?",
        [("T-20240226-045", "2024-02-26", "I collected the kitten on 25 February 2024.")],
    )
    assert "110 days (weeks = days // 7 = 15)" in text


def test_bridge_name_links_two_blocks():
    from mind_mem.answer_facts import bridge_facts

    rows = [
        ("D-1", date(2023, 10, 29), "The clutch was replaced at Harwick Garage."),
        ("D-2", date(2024, 5, 8), "Work done at Harwick Garage is guaranteed for 18 months."),
        ("D-3", date(2023, 8, 25), "Work done at Moss Road Motors is guaranteed for 6 months."),
    ]
    lines = bridge_facts("For how long is the clutch covered?", rows)
    assert lines and "- Harwick Garage: [[D-1]], [[D-2]]" in lines
    assert not any("Moss Road" in line for line in lines)


def test_no_bridge_when_name_is_in_the_question():
    from mind_mem.answer_facts import bridge_facts

    rows = [("D-1", None, "I met Ana at the market."), ("D-2", None, "Ana moved to Leeds.")]
    assert bridge_facts("Where did Ana move?", rows) == []


def test_pair_and_neighbour_lines_only_for_their_cues():
    rows = [
        ("A-1", "2024-03-01", "I joined the brass band on 18 March 2024."),
        ("B-2", "2024-04-10", "I played at the spring fete on 7 April 2024."),
    ]
    text = _lines("Today is 2024-05-01. How many days ago was the spring fete?", rows)
    assert "Between" not in text


def test_long_evidence_keeps_only_worked_out_dates():
    rows = [(f"D-{i}", "2024-01-01", f"user: chat line number {i} about nothing") for i in range(30)]
    rows.append(("D-x", "2024-01-05", "user: I planted the tulips yesterday."))
    text = _lines("Today is 2024-02-01. How many days ago did I plant the tulips?", rows)
    assert "[[D-x]]" in text and "[[D-3]]" not in text


def test_no_name_links_for_long_evidence():
    ev = [Ev(f"D-{i}", "2024-01-01", f"assistant: Additional Tips for Android users, item {i}.") for i in range(20)]
    assert "linked by the same name" not in computed_facts("Which phone do I use?", ev)


def test_names_skip_common_capitalised_words_after_a_colon_and_calendar_words():
    """10-07 gate #159: "Assistant: Last month's ..." made "Last" a name, and "Thursday" linked unrelated blocks."""
    from mind_mem.answer_facts import _names

    text = "Assistant: Last month's book club pick was The Quiet Ferry. See you Thursday in March."
    names = _names(text)
    assert "Last" not in names
    assert "Thursday" not in names
    assert "March" not in names
    assert "The Quiet Ferry" in names or "Quiet Ferry" in names  # a real title still counts


def test_between_question_with_both_events_present_keeps_the_gap_table():
    from datetime import date

    from mind_mem.answer_facts import timeline_facts

    items = [
        ("A", date(2023, 8, 18), "I submitted the dissertation on 15 August 2023."),
        ("B", date(2023, 9, 5), "I had the viva yesterday."),
    ]
    lines = timeline_facts("How long was it, in days, from the dissertation submission to the viva?", items)
    assert any(line.startswith("Days between events") for line in lines)
    assert any("20 days" in line for line in lines)


def test_bridge_needs_a_block_not_already_about_the_question():
    from mind_mem.answer_facts import bridge_facts

    redundant = [("A", None, "Last month's book club pick was The Quiet Ferry."), ("B", None, "The book club loved The Quiet Ferry.")]
    assert bridge_facts("What is the book club reading this month?", redundant) == []
    real = [
        ("A", None, "The timing belt was fitted at Tolland Motors."),
        ("B", None, "Work done at Tolland Motors is guaranteed for 24 months."),
    ]
    assert any("Tolland Motors" in line for line in bridge_facts("For how long is the timing belt covered?", real))
