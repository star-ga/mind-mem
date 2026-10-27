"""Deterministic facts computed from the evidence before the answerer sees it.

Date arithmetic and option bookkeeping are code, not generation: a 4B answerer that has to turn
"yesterday" in a block dated 2023-08-13 into 2023-08-12 and then count weeks to 2023-12-09 gets it
wrong often enough to matter; the same computation in ``datetime`` is exact and reproducible.

* :func:`timeline_facts` resolves every evidence item to an absolute EVENT date (an explicit date in
  the text, else a relative phrase -- today, yesterday, N days ago, last <weekday> -- resolved against
  the block date, else the block date), lists the events oldest first, and precomputes the deltas the
  question asks about: elapsed time to "today" (days, weeks = days // 7, calendar months), the gap
  between the two events the question names, and the events right before / after the named one.
* :func:`choice_options` extracts the options a choice question offers, verbatim.
* :func:`computed_facts` renders both as one block for the prompt ('' when there is nothing to add).

Pure functions; no clock, no I/O, no logging. Nothing here changes a prompt unless a caller opts in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Sequence

__all__ = ["bridge_facts", "Event", "event_date", "question_today", "timeline_facts", "choice_options", "computed_facts", "names_an_option"]

_MONTHS = {
    m: i
    for i, m in enumerate(
        ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"), 1
    )
}
_MON = {k[:3]: v for k, v in _MONTHS.items()}
_MONTH_RE = (
    r"(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?"
)
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_RE + r",?\s+(\d{4})\b", re.IGNORECASE)
_MDY = re.compile(r"\b" + _MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.IGNORECASE)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_AGO = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago\b", re.IGNORECASE)
_LAST_WD = re.compile(r"\blast\s+(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)
_TODAY_Q = re.compile(r"\btoday is\s+(?:[A-Za-z]+,\s+)?([^.?!]+?\d{4}|\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_CURRENT = re.compile(r"\bcurrent date:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_TEMPORAL_CUES = re.compile(
    r"\b(ago|between|before|after|first|earlier|later|when|how long|how many (?:days|weeks|months)|date|since|order)\b", re.IGNORECASE
)
MAX_PAIR_EVENTS = 8
SMALL_EVIDENCE = 12  # up to this many evidence items, every item is a timeline event
MAX_TIMELINE = 40
MAX_LONG_EVENTS = 12
_ELAPSED_CUES = re.compile(r"\bago\b|\bsince\b|how long|\btoday\b.*\b(days|weeks|months)\b", re.IGNORECASE)
_PAIR_LABELS = re.compile(r"(?:\bbetween|\bfrom|\bfirst,)\s+(.+?)\s+(?:and|to|or)\s+(.+?)\s*\?", re.IGNORECASE)
_ORDER_LABEL = re.compile(r"\b(?:before|after)\s+(.+?)\s*\?", re.IGNORECASE)


def _best_pair(l1: str, l2: str, events: list["Event"]) -> tuple["Event | None", "Event | None"]:
    """The two distinct events that best match the two labels jointly; (None, None) when ambiguous."""
    i = _confident(l1, events)
    j = _confident(l2, events)
    if i is None or j is None or i == j:
        return None, None
    return events[i], events[j]


def _confident(label: str, events: list["Event"]) -> int | None:
    """Index of the event that matches *label* clearly: it covers at least 60% of the label's terms and
    beats every other event by at least one term. None otherwise (no fact beats a wrong fact)."""
    t = _terms(label)
    if not t:
        return None
    scores = sorted(((len(t & _terms(e.text)), i) for i, e in enumerate(events)), reverse=True)
    top, idx = scores[0]
    runner = scores[1][0] if len(scores) > 1 else 0
    return idx if top >= max(1, 0.6 * len(t)) and top - runner >= 1 else None


def _best(label: str, events: list["Event"]) -> "Event | None":
    """The event whose text shares the most terms with *label*; None when absent or tied."""
    t = _terms(label)
    scores = sorted(((len(t & _terms(e.text)), i) for i, e in enumerate(events)), reverse=True)
    if not scores or scores[0][0] == 0 or (len(scores) > 1 and scores[1][0] == scores[0][0]):
        return None
    return events[scores[0][1]]


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of to in on at for and or is was were did do does i my me you your what when where who which how "
    "many much long days weeks months ago between before after first last next thing things passed it "
    "today from with this that be".split()
)


@dataclass(frozen=True)
class Event:
    block_id: str
    day: date
    text: str
    how: str  # "explicit" | phrase used | "block date"


def _mk(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(text: str) -> date | None:
    """The first explicit calendar date in *text* (ISO, '25 February 2024', 'Feb 25, 2024')."""
    found: list[tuple[int, date]] = []
    for m in _ISO.finditer(text):
        d = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.append((m.start(), d))
    for m in _DMY.finditer(text):
        d = _mk(int(m.group(3)), _MON[m.group(2).lower()[:3]], int(m.group(1)))
        if d:
            found.append((m.start(), d))
    for m in _MDY.finditer(text):
        d = _mk(int(m.group(3)), _MON[m.group(1).lower()[:3]], int(m.group(2)))
        if d:
            found.append((m.start(), d))
    return min(found)[1] if found else None


def event_date(text: str, block_day: date | None) -> tuple[date | None, str]:
    """The date the text's event happened, and how it was resolved."""
    explicit = parse_date(text)
    if explicit:
        return explicit, "explicit"
    if block_day is None:
        return None, ""
    low = text.lower()
    m = _AGO.search(low)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]
        return block_day - timedelta(days=n), m.group(0)
    m = _LAST_WD.search(low)
    if m:
        wd = _WEEKDAYS.index(m.group(1).lower())
        return block_day - timedelta(days=(block_day.weekday() - wd) % 7 or 7), m.group(0)
    if re.search(r"\byesterday\b", low):
        return block_day - timedelta(days=1), "yesterday"
    if re.search(r"\b(today|tonight|this morning|this evening)\b", low):
        return block_day, "today"
    return block_day, "block date"


def question_today(question: str, items: Sequence[tuple[str, date | None, str]]) -> tuple[date | None, str]:
    """'Today' for the question: 'Today is <date>' in the question, else a 'Current date:' evidence block."""
    m = _TODAY_Q.search(question)
    if m:
        d = parse_date(m.group(1))
        if d:
            return d, ""
    for block_id, _day, text in items:
        m = _CURRENT.search(text)
        if m:
            d = parse_date(m.group(1))
            if d:
                return d, block_id
    return None, ""


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w[:5]


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def _months(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + b.month - a.month - (1 if b.day < a.day else 0)


def _span(days: int) -> str:
    return f"{days} days (weeks = days // 7 = {days // 7})"


def timeline_facts(question: str, items: Sequence[tuple[str, date | None, str]]) -> list[str]:
    """Computed temporal lines for *question* over evidence *items* (block_id, block date, text)."""
    if not _TEMPORAL_CUES.search(question):
        return []
    today, today_id = question_today(question, items)
    events: list[Event] = []
    for block_id, day, text in items:
        if block_id == today_id:
            continue
        d, how = event_date(text, day)
        if d is not None:
            events.append(Event(block_id, d, text, how))
    if len(items) > SMALL_EVIDENCE:
        # Long evidence (whole sessions): every turn carries its block date, so listing them all would
        # repeat the evidence. Keep the events whose date had to be worked out of the text.
        q = _terms(question)
        events = [e for e in events if e.how != "block date" and q & _terms(e.text)][:MAX_LONG_EVENTS]
    if not events:
        return []
    events.sort(key=lambda e: (e.day, e.block_id))
    events = events[:MAX_TIMELINE]
    lines = ["Timeline (event dates resolved from the evidence, oldest first):"]
    for e in events:
        note = "" if e.how in ("explicit", "block date") else f" ({e.how} as of the block date)"
        lines.append(f"- {e.day.isoformat()}{note} [[{e.block_id}]] {e.text}")
    if today is not None and _ELAPSED_CUES.search(question):
        lines.append(f"Today: {today.isoformat()}.")
        for e in events:
            if e.day <= today:
                days = (today - e.day).days
                lines.append(f"- [[{e.block_id}]] was {_span(days)} ago, {_months(e.day, today)} calendar months ago.")
    pair = _PAIR_LABELS.search(question)
    if pair and 2 <= len(events) <= MAX_PAIR_EVENTS:
        # Every pairwise gap, so the answer is a lookup; which two events the question means is left to
        # the reader (a lexical guess picks decoys like "dry-run cutover" for "the cutover").
        lines.append("Days between events (later minus earlier):")
        for i, a in enumerate(events):
            for b in events[i + 1 :]:
                gap = (b.day - a.day).days
                lines.append(f"- [[{a.block_id}]] -> [[{b.block_id}]]: {gap} days ({gap // 7} weeks)")
    return lines


_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|&|of|de|la))*\s+[A-Z][a-z]+|[A-Z][a-z]+(?:'s)?)\b")
# A sentence also starts after a colon or an opening quote ("Assistant: Last month's ..." is not a name).
_SENT_START = re.compile(r"(?:^|[.!?:]\s+|[\"\u201c]\s*)([A-Z][a-z]+)")
# Capitalised words that are never the name of a person, place or thing in a block (calendar words and
# common sentence openers); linking two blocks through them only adds decoys.
_NOT_NAMES = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
    "august september october november december last this next today yesterday tomorrow the a an i my our "
    "your we you assistant user".split()
)


def _names(text: str) -> set[str]:
    starts = {m.group(1) for m in _SENT_START.finditer(text)}
    out = set()
    for m in _NAME.finditer(text):
        n = m.group(1).removesuffix("'s")
        words = n.split()
        while words and words[0].lower() in _NOT_NAMES:  # "The Quiet Ferry" keeps "Quiet Ferry"
            words = words[1:]
        if not words or all(w.lower() in _NOT_NAMES for w in words):
            continue
        n = " ".join(words)
        if len(words) > 1 or n not in starts:
            out.add(n)
    return out


def bridge_facts(question: str, items: Sequence[tuple[str, date | None, str]]) -> list[str]:
    """Two-step links: a name the question does NOT mention that joins a block about the question to
    another block ("X was fitted at Name" + "work at Name is guaranteed for N")."""
    q = _terms(question)
    qnames = {n.lower() for n in _names(question)}
    about = [b for b, _d, t in items if len(q & _terms(t)) >= 1]
    by_name: dict[str, list[str]] = {}
    for b, _d, t in items:
        for n in _names(t):
            if n.lower() not in qnames:
                by_name.setdefault(n, []).append(b)
    lines = []
    for name, ids in sorted(by_name.items()):
        ids = list(dict.fromkeys(ids))
        # A bridge brings in a block that is NOT already about the question; linking two blocks that are both
        # about it adds nothing and nudges an answer (10-07 gate #159).
        if len(ids) >= 2 and any(i in about for i in ids) and any(i not in about for i in ids):
            lines.append(f"- {name}: " + ", ".join(f"[[{i}]]" for i in ids))
    return (["Blocks linked by the same name (combine them to answer):"] + lines) if lines else []


_CHOICE_START = re.compile(
    r"^\s*(?:which\b[^:?]*:\s*|should i\s+(?:sign up for|book|choose|pick|take|go with|go for|do|buy|get)?\s*|is\s+)", re.IGNORECASE
)
_CHOICE_END = re.compile(
    r"\s+(?:the better (?:plan|option|choice|call)|the right (?:call|choice|option)|better)\s*\?\s*$|\s*\?\s*$", re.IGNORECASE
)


def choice_options(question: str) -> list[str]:
    """Options offered by a choice question ('A, B or C'), verbatim; [] when it is not one."""
    q = re.sub(r"^today is [^.]+\.\s*", "", question.strip(), flags=re.IGNORECASE)
    if " or " not in q or not _CHOICE_START.match(q):
        return []
    body = _CHOICE_END.sub("", _CHOICE_START.sub("", q, count=1))
    parts = re.split(r",\s*|\s+or\s+", body)
    opts = [p.strip(" .") for p in parts if p.strip(" .")]
    return opts if 2 <= len(opts) <= 6 else []


def names_an_option(answer: str, options: Sequence[str]) -> bool:
    low = answer.lower()
    return any(o.lower() in low for o in options)


def computed_facts(question: str, evidence: Sequence[Any]) -> str:
    """The computed block for a prompt ('' when nothing applies). *evidence*: items with block_id,
    excerpt and an ISO-ish date (EvidenceItem)."""
    items: list[tuple[str, date | None, str]] = []
    for it in evidence:
        raw = str(getattr(it, "date", "") or "")[:10]
        items.append((str(getattr(it, "block_id", "")), parse_date(raw), str(getattr(it, "excerpt", ""))))
    lines = timeline_facts(question, items)
    if len(items) <= SMALL_EVIDENCE:  # in long evidence, shared capitalised words are mostly noise
        lines += bridge_facts(question, items)
    opts = choice_options(question)
    if opts:
        lines.append("Options (answer with exactly one of these, verbatim): " + " | ".join(opts))
    if not lines:
        return ""
    return "Computed from the evidence (use these values; do not recompute):\n" + "\n".join(lines)
