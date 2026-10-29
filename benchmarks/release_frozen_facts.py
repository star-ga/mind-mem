"""Historical rendering arithmetic, retained only for archive reconstruction."""

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
    "(january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\\.?"
)
_ISO = re.compile("\\b(\\d{4})-(\\d{2})-(\\d{2})\\b")
_DMY = re.compile("\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+" + _MONTH_RE + ",?\\s+(\\d{4})\\b", re.IGNORECASE)
_MDY = re.compile("\\b" + _MONTH_RE + "\\s+(\\d{1,2})(?:st|nd|rd|th)?,?\\s+(\\d{4})\\b", re.IGNORECASE)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_COUNT = "(\\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
_AGO = re.compile("\\b" + _COUNT + "\\s+days?\\s+ago\\b", re.IGNORECASE)
_LAST_WD = re.compile("\\blast\\s+(" + "|".join(_WEEKDAYS) + ")\\b", re.IGNORECASE)
_YEARS = re.compile("\\b(?:(last|next)\\s+year|" + _COUNT + "\\s+years?\\s+(ago)|in\\s+" + _COUNT + "\\s+years?)\\b", re.IGNORECASE)
_MONTHS_REL = re.compile("\\b(?:(last|next)\\s+month|" + _COUNT + "\\s+months?\\s+(ago)|in\\s+" + _COUNT + "\\s+months?)\\b", re.IGNORECASE)
_WEEKS_REL = re.compile("\\b(?:(last|next)\\s+week|" + _COUNT + "\\s+weeks?\\s+(ago)|in\\s+" + _COUNT + "\\s+weeks?)\\b", re.IGNORECASE)
_WEEK_BEFORE = re.compile("\\bthe\\s+week\\s+(before|after)\\b", re.IGNORECASE)
_LAST_WEEKEND = re.compile("\\blast\\s+weekend\\b", re.IGNORECASE)
_TODAY_Q = re.compile("\\btoday is\\s+(?:[A-Za-z]+,\\s+)?([^.?!]+?\\d{4}|\\d{4}-\\d{2}-\\d{2})", re.IGNORECASE)
_CURRENT = re.compile("\\bcurrent date:\\s*(\\d{4}-\\d{2}-\\d{2})", re.IGNORECASE)
_TEMPORAL_CUES = re.compile(
    "\\b(ago|between|before|after|first|earlier|later|when|how long|how many (?:days|weeks|months)|date|since|order)\\b", re.IGNORECASE
)
MAX_PAIR_EVENTS = 8
SMALL_EVIDENCE = 12
MAX_TIMELINE = 40
MAX_LONG_EVENTS = 12
_UNIT_COUNT_Q = re.compile("\\bhow many\\s+(?:days|weeks)\\b", re.IGNORECASE)
_INCLUSIVE_Q = re.compile("\\binclusive\\b|\\bboth ends\\b|\\bcounting\\b", re.IGNORECASE)
_ELAPSED_CUES = re.compile(
    "\\bago\\b|\\bsince\\b|how long|\\bpassed\\b|\\bhas it been\\b|\\btoday\\b.*\\b(days|weeks|months)\\b", re.IGNORECASE
)
_PAIR_LABELS = re.compile("(?:\\bbetween|\\bfrom|\\bfirst,)\\s+(.+?)\\s+(?:and|to|or)\\s+(.+?)\\s*\\?", re.IGNORECASE)
_ORDER_LABEL = re.compile("\\b(?:before|after)\\s+(.+?)\\s*\\?", re.IGNORECASE)


def _best_pair(l1: str, l2: str, events: list["Event"]) -> tuple["Event | None", "Event | None"]:
    i = _confident(l1, events)
    j = _confident(l2, events)
    if i is None or j is None or i == j:
        return (None, None)
    return (events[i], events[j])


def _confident(label: str, events: list["Event"]) -> int | None:
    t = _terms(label)
    if not t:
        return None
    scores = sorted(((len(t & _terms(e.text)), i) for i, e in enumerate(events)), reverse=True)
    top, idx = scores[0]
    runner = scores[1][0] if len(scores) > 1 else 0
    return idx if top >= max(1, 0.6 * len(t)) and top - runner >= 1 else None


def _best(label: str, events: list["Event"]) -> "Event | None":
    t = _terms(label)
    scores = sorted(((len(t & _terms(e.text)), i) for i, e in enumerate(events)), reverse=True)
    if not scores or scores[0][0] == 0 or (len(scores) > 1 and scores[1][0] == scores[0][0]):
        return None
    return events[scores[0][1]]


_WORD = re.compile("[a-z0-9]+")
_STOP = frozenset(
    "the a an of to in on at for and or is was were did do does i my me you your what when where who which how many much long "
    "days weeks months ago between before after first last next thing things passed it today from with this that be".split()
)


@dataclass(frozen=True)
class Event:
    block_id: str
    day: date
    text: str
    how: str
    granularity: str = "day"
    block_day: date | None = None

    def label(self) -> str:
        if self.granularity == "year":
            return str(self.day.year)
        if self.granularity == "month":
            return f"{_MONTH_NAMES[self.day.month]} {self.day.year}"
        return self.day.isoformat()


_MONTH_NAMES = {i: m.capitalize() for m, i in _MONTHS.items()}


def _mk(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(text: str) -> date | None:
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


def _count(word: str) -> int:
    w = word.lower()
    return int(w) if w.isdigit() else _NUM.get(w, 1)


def _signed(m: "re.Match[str]") -> int:
    if m.group(1):
        return -1 if m.group(1).lower() == "last" else 1
    if m.group(2):
        return -_count(m.group(2))
    return _count(m.group(4))


def _month_shift(d: date, n: int) -> date:
    i = d.year * 12 + d.month - 1 + n
    return date(i // 12, i % 12 + 1, 1)


def event_date(text: str, block_day: date | None) -> tuple[date | None, str, str]:
    explicit = parse_date(text)
    if explicit:
        return (explicit, "explicit", "day")
    if block_day is None:
        return (None, "", "day")
    low = text.lower()
    m = _AGO.search(low)
    if m:
        return (block_day - timedelta(days=_count(m.group(1))), m.group(0), "day")
    m = _LAST_WD.search(low)
    if m:
        wd = _WEEKDAYS.index(m.group(1).lower())
        return (block_day - timedelta(days=(block_day.weekday() - wd) % 7 or 7), m.group(0), "day")
    if re.search("\\byesterday\\b", low):
        return (block_day - timedelta(days=1), "yesterday", "day")
    if re.search("\\b(today|tonight|this morning|this evening)\\b", low):
        return (block_day, "today", "day")
    m = _YEARS.search(low)
    if m:
        return (date(block_day.year + _signed(m), 1, 1), m.group(0), "year")
    m = _MONTHS_REL.search(low)
    if m:
        return (_month_shift(block_day, _signed(m)), m.group(0), "month")
    m = _WEEKS_REL.search(low)
    if m:
        return (block_day + timedelta(days=7 * _signed(m)), m.group(0), "day")
    m = _WEEK_BEFORE.search(low)
    if m:
        return (block_day + timedelta(days=-7 if m.group(1).lower() == "before" else 7), m.group(0), "day")
    if _LAST_WEEKEND.search(low):
        return (block_day - timedelta(days=(block_day.weekday() - 5) % 7 or 7), "last weekend", "day")
    return (block_day, "block date", "day")


def _week_anchor(how: str, block_day: date | None) -> str:
    if block_day is None:
        return ""
    m = _WEEK_BEFORE.fullmatch(how)
    if m:
        return f"the week {m.group(1).lower()} {block_day.isoformat()}"
    m = _WEEKS_REL.fullmatch(how)
    if m and abs(_signed(m)) == 1:
        return f"the week {('before' if _signed(m) < 0 else 'after')} {block_day.isoformat()}"
    return ""


def question_today(question: str, items: Sequence[tuple[str, date | None, str]]) -> tuple[date | None, str]:
    m = _TODAY_Q.search(question)
    if m:
        d = parse_date(m.group(1))
        if d:
            return (d, "")
    for block_id, _day, text in items:
        m = _CURRENT.search(text)
        if m:
            d = parse_date(m.group(1))
            if d:
                return (d, block_id)
    return (None, "")


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


def _weeks(days: int) -> str:
    w, r = divmod(days, 7)
    return f"{w} weeks exactly" if r == 0 else f"{w} weeks {r} day{('s' if r != 1 else '')}"


def _span(days: int) -> str:
    return f"{days} days (weeks = days // 7 = {days // 7})"


def _unit_only(question: str) -> bool:
    return bool(_UNIT_COUNT_Q.search(question)) and (not _INCLUSIVE_Q.search(question))


def _breakdown(days: int, question: str = "") -> str:
    head = f"{days} days = {_weeks(days)}"
    if _INCLUSIVE_Q.search(question):
        return f"{head}; {days + 1} days counting both ends (inclusive)"
    return head


def timeline_facts(question: str, items: Sequence[tuple[str, date | None, str]]) -> list[str]:
    if not _TEMPORAL_CUES.search(question):
        return []
    today, today_id = question_today(question, items)
    events: list[Event] = []
    for block_id, day, text in items:
        if block_id == today_id:
            continue
        d, how, gran = event_date(text, day)
        if d is not None:
            events.append(Event(block_id, d, text, how, gran, day))
    if len(items) > SMALL_EVIDENCE:
        q = _terms(question)
        events = [e for e in events if e.how != "block date" and q & _terms(e.text)][:MAX_LONG_EVENTS]
    if not events:
        return []
    events.sort(key=lambda e: (e.day, e.block_id))
    events = events[:MAX_TIMELINE]
    lines = ["Timeline (event dates resolved from the evidence, oldest first):"]
    for e in events:
        if e.how in ("explicit", "block date"):
            note = ""
        elif e.granularity != "day":
            note = f''' ({e.granularity} only; "{e.how}" as of {(e.block_day.isoformat() if e.block_day else "the block date")})'''
        else:
            anchor = _week_anchor(e.how, e.block_day)
            note = f" ({e.how} as of the block date{('; ' + anchor if anchor else '')})"
        lines.append(f"- {e.label()}{note} [[{e.block_id}]] {e.text}")
    if today is not None and _ELAPSED_CUES.search(question):
        lines.append(f"Today: {today.isoformat()}.")
        for e in events:
            if e.day <= today and e.granularity == "day":
                days = (today - e.day).days
                if _unit_only(question):
                    lines.append(f"- [[{e.block_id}]] was {_breakdown(days, question)} ago, {_months(e.day, today)} calendar months ago.")
                else:
                    lines.append(
                        f"- [[{e.block_id}]] was {_span(days)} ago, {_months(e.day, today)} calendar months ago; "
                        f"{_breakdown(days, question)}."
                    )
    pair = _PAIR_LABELS.search(question)
    dated = [e for e in events if e.granularity == "day"]
    if pair and 2 <= len(dated) <= MAX_PAIR_EVENTS:
        lines.append("Days between events (later minus earlier):")
        for i, a in enumerate(dated):
            for b in dated[i + 1 :]:
                d = (b.day - a.day).days
                if _unit_only(question):
                    lines.append(f"- [[{a.block_id}]] -> [[{b.block_id}]]: {_breakdown(d, question)}")
                else:
                    lines.append(f"- [[{a.block_id}]] -> [[{b.block_id}]]: {d} days ({d // 7} weeks); {_breakdown(d, question)}")
    return lines


_NAME = re.compile("\\b([A-Z][a-z]+(?:\\s+(?:[A-Z][a-z]+|&|of|de|la))*\\s+[A-Z][a-z]+|[A-Z][a-z]+(?:'s)?)\\b")
_SENT_START = re.compile('(?:^|[.!?:]\\s+|[\\"\\u201c]\\s*)([A-Z][a-z]+)')
_NOT_NAMES = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday january february march april may june july august "
    "september october november december last this next today yesterday tomorrow the a an i my our your we you assistant user".split()
)


def _names(text: str) -> set[str]:
    starts = {m.group(1) for m in _SENT_START.finditer(text)}
    out = set()
    for m in _NAME.finditer(text):
        n = m.group(1).removesuffix("'s")
        words = n.split()
        while words and words[0].lower() in _NOT_NAMES:
            words = words[1:]
        if not words or all((w.lower() in _NOT_NAMES for w in words)):
            continue
        n = " ".join(words)
        if len(words) > 1 or n not in starts:
            out.add(n)
    return out


def bridge_facts(question: str, items: Sequence[tuple[str, date | None, str]]) -> list[str]:
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
        if len(ids) >= 2 and any((i in about for i in ids)) and any((i not in about for i in ids)):
            lines.append(f"- {name}: " + ", ".join((f"[[{i}]]" for i in ids)))
    return ["Blocks linked by the same name (combine them to answer):"] + lines if lines else []


_CHOICE_START = re.compile(
    "^\\s*(?:which\\b[^:?]*:\\s*|should i\\s+(?:sign up for|book|choose|pick|take|go with|go for|do|buy|get)?\\s*|is\\s+)", re.IGNORECASE
)
_CHOICE_END = re.compile(
    "\\s+(?:the better (?:plan|option|choice|call)|the right (?:call|choice|option)|better)\\s*\\?\\s*$|\\s*\\?\\s*$", re.IGNORECASE
)


def choice_options(question: str) -> list[str]:
    q = re.sub("^today is [^.]+\\.\\s*", "", question.strip(), flags=re.IGNORECASE)
    if " or " not in q or not _CHOICE_START.match(q):
        return []
    body = _CHOICE_END.sub("", _CHOICE_START.sub("", q, count=1))
    parts = re.split(",\\s*|\\s+or\\s+", body)
    opts = [p.strip(" .") for p in parts if p.strip(" .")]
    return opts if 2 <= len(opts) <= 6 else []


def names_an_option(answer: str, options: Sequence[str]) -> bool:
    low = answer.lower()
    return any((o.lower() in low for o in options))


def computed_facts(question: str, evidence: Sequence[Any]) -> str:
    items: list[tuple[str, date | None, str]] = []
    for it in evidence:
        stamp = str(getattr(it, "date", "") or "")
        day = parse_date(stamp[:10]) or parse_date(stamp)
        items.append((str(getattr(it, "block_id", "")), day, str(getattr(it, "excerpt", ""))))
    lines = timeline_facts(question, items)
    if len(items) <= SMALL_EVIDENCE:
        lines += bridge_facts(question, items)
    opts = choice_options(question)
    if opts:
        lines.append("Options (answer with exactly one of these, verbatim): " + " | ".join(opts))
    if not lines:
        return ""
    return "Computed from the evidence (use these values; do not recompute):\n" + "\n".join(lines)
