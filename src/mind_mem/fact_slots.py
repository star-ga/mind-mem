# Copyright 2026 STARGA, Inc.
"""Fail-closed fact comparison for memory merges and contradictions.

Two decisions read this module, built so that words alone can decide them safely:

* ``slots_differ(a, b)`` / ``canonical_tokens`` — may two memories be MERGED?
  Only when they are the same statement up to Unicode NFC, whitespace and a
  trailing full stop. Case, spelling, abbreviations and number/time formats
  must match exactly: "US" vs "us", "Jun" vs "June", "7:00" vs "7am" can each
  mean different things, and a wrongly proposed merge loses a memory.
* ``changed_fact(new, old)`` — is ``new`` a correction of ``old``? Both texts
  are normalised for comparison (case, unambiguous abbreviations, canonical
  clock times, contractions), the when/zone values and correction words are
  removed IN CONTEXT, and what remains (people, event, identifiers, places,
  quantities) must be identical; then the value sets must differ or the
  negation flip. The context rules keep different people, places, teams and
  IDs from being flagged.

Both decisions are linear in the text length. Only ASCII 0-9 count as digits.
Deterministic, no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

__all__ = ["FactSlots", "canonical_tokens", "changed_fact", "fact_slots", "slots_differ"]

_ASCII_DIGITS = frozenset("0123456789")
# Kept inside a token when an ASCII digit is directly on both sides: "1.5", "1,000", "10:30", "2026-10-08", "10/8".
_DIGIT_JOINERS = frozenset(".,:-/")


def _is_digits(tok: str) -> bool:
    return tok.isascii() and tok.isdigit()


def _spaced_tokens(text: str) -> list[tuple[str, bool]]:
    """(token, preceded_by_whitespace) pairs: word runs (letters, marks, ASCII digits) and single non-space characters."""
    text = unicodedata.normalize("NFC", text)
    out: list[tuple[str, bool]] = []
    cur: list[str] = []
    cur_space = False
    space = False
    n = len(text)
    for i, ch in enumerate(text):
        is_word = ch in _ASCII_DIGITS or unicodedata.category(ch)[0] in ("L", "M")
        if not is_word and ch in _DIGIT_JOINERS and cur and cur[-1] in _ASCII_DIGITS and i + 1 < n and text[i + 1] in _ASCII_DIGITS:
            is_word = True
        if is_word:
            if not cur:
                cur_space = space
            cur.append(ch)
            space = False
            continue
        if cur:
            out.append(("".join(cur), cur_space))
            cur = []
        if ch.isspace():
            space = True
        else:
            out.append((ch, space))
            space = False
    if cur:
        out.append(("".join(cur), cur_space))
    return out


def _raw_tokens(text: str) -> list[str]:
    """Word runs (letters, marks, ASCII digits) and single non-space characters, case preserved."""
    return [tok for tok, _ in _spaced_tokens(text)]


def canonical_tokens(text: str) -> tuple[str, ...]:
    """The MERGE normal form: NFC, and the amount of whitespace does not matter. Nothing else.

    WHETHER whitespace separates two tokens is kept, so "set_timeout" and
    "set _timeout", or "/var/log" and "/var/ log", stay different. A final full
    stop is dropped only when it is attached to a word ("on Tuesday." ==
    "on Tuesday"); a free-standing "." is content ("the separator is .").
    """
    pairs = _spaced_tokens(text)
    if len(pairs) >= 2 and pairs[-1][0] in (".", "。") and not pairs[-1][1] and pairs[-2][0] not in (".", "。"):
        pairs = pairs[:-1]
    return tuple((" " + tok) if (i and space) else tok for i, (tok, space) in enumerate(pairs))


def slots_differ(a: str, b: str) -> bool:
    """True unless the two texts are the same statement (see :func:`canonical_tokens`)."""
    return canonical_tokens(a) != canonical_tokens(b)


# ----------------------------------------------------------------------------- changed_fact

_EQUIV = {
    "tue": "tuesday",
    "tues": "tuesday",
    "tuesdays": "tuesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "thursdays": "thursday",
    "fri": "friday",
    "fridays": "friday",
    "mon": "monday",
    "mondays": "monday",
    "wed": "wednesday",
    "wednesdays": "wednesday",
    "sat": "saturday",
    "saturdays": "saturday",
    "sun": "sunday",
    "sundays": "sunday",
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sept": "september",
    "sep": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}
#: Abbreviations that are also ordinary words or names ("Sun", "Wed", "Jan",
#: "Mar"): a when-value only in date context. Full weekday names and their
#: plurals are always values.
_WEEKDAY_ABBR = frozenset({"tue", "tues", "thu", "thur", "thurs", "fri", "mon", "wed", "sat", "sun"})
_WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
_MONTHS = frozenset(
    {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
)
_MONTH_ABBR = frozenset(k for k, v in _EQUIV.items() if v in _MONTHS)
#: Zone abbreviations: a value next to a clock or in a time-zone phrase.
_ZONE_ABBR = frozenset("utc gmt pt pst pdt et est edt ct cst cdt mt mst mdt hst akst akdt ist cet cest bst jst aest aedt az".split())
#: Zone NAMES are also places ("flies to Arizona"): a value only right after a
#: clock ("10am Pacific"), before "time" ("Arizona time"), or in a time-zone
#: phrase ("timezone is Arizona").
_ZONE_NAMES = frozenset("pacific eastern central mountain arizona hawaii alaska".split())
_ZONE_CONTEXT = frozenset({"time", "tz", "timezone", "zone"})
#: Spellings of one US zone ("PT" == "Pacific" == "PST"/"PDT" as people write them).
_ZONE_CANON = {
    **{z: "pacific" for z in ("pt", "pst", "pdt", "pacific")},
    **{z: "eastern" for z in ("et", "est", "edt", "eastern")},
    **{z: "central" for z in ("ct", "cst", "cdt", "central")},
    **{z: "mountain" for z in ("mt", "mst", "mdt", "mountain")},
    **{z: "arizona" for z in ("az", "arizona")},
}
_RELATIVE = frozenset("today tomorrow yesterday tonight morning afternoon evening weekend".split())
_PERIOD_AFTER = frozenset({"next", "this", "last", "every", "following"})
_PERIODS = frozenset({"week", "month", "year", "weekend"})
#: Words before a month or abbreviated weekday that make it a date ("in May",
#: "by Oct 9", "Standup: Tue") rather than a name ("Ask May", "the May report").
_DATE_PREV = frozenset(
    (
        "from on at in by until till through since before after during early mid late each every next this last following : → , "
        "в во на w we um im ab bis op på le à en el la di de il"
    ).split()
)
#: "to" makes a month a date only in a scheduling sentence ("moved the review
#: to March"); otherwise it is usually a recipient ("Send the report to June").
#: "of" counts only after a span word ("end of May"), never in a label
#: ("Review of April numbers").
_SCHEDULING = frozenset(
    "moved move moves moving rescheduled reschedule postponed delayed pushed slipped shifted bumped "
    "due deadline scheduled starts start begins ends launch launches".split()
)
_MOVE_WORDS = frozenset(
    "moved move moving changed change switched rescheduled postponed delayed pushed shifted bumped slipped "
    "went goes go increased decreased raised lowered rose fell grew dropped cut reduced extended shortened upped".split()
)
_SPAN_OF = frozenset({"end", "start", "beginning", "middle", "mid", "rest", "first", "last"})
_QUARTER_PREV = frozenset("in by to until till before after during through from since".split())
#: Nouns whose following date-shaped token is an identifier ("Invoice 2026-10-08", "Ticket 10/8").
_ID_NOUNS = frozenset(
    "invoice ticket order build release ref id case po issue pr bug incident "
    "receipt account room flight item sku batch run job commit tag # no doc document page section chapter "
    "employee badge serial code file row line inc ticket req section clause article paragraph figure table step".split()
)
#: Words that make a following weekday a name ("Project Tuesday", "Operation Monday").
_LABEL_PREV = frozenset({"project", "operation", "codename", "club", "team"})
_ID_GLUE = frozenset({"-", "_", "#", ".", "/"})
_ID_SKIP = frozenset({"is", "was", ":", "=", "no", "number", "#"})
#: Identifier nouns that are also verbs: an identifier only at the start or after a determiner ("Run 12", "the run 12").
_VERB_OR_ID = frozenset({"run", "build", "order", "release", "tag", "commit"})
_ID_DETERMINERS = frozenset("the a an this that each every last next our your their ci test job nightly per #".split())
_YEAR_PREV = frozenset("in since by until till from to before after during".split())
# Correction/function words dropped from both cores anywhere.
_CORRECTION = frozenset(
    "moved move moving now instead rescheduled reschedule scheduled changed change switched postponed delayed pushed "
    "slipped slip slips shifted bumped up back got get it is was are were am has have had been be will would do does did "
    "of and but correction actually update updated again starting as the a an with rather than anymore "
    "went goes increased decreased raised lowered rose fell grew dropped reduced extended shortened upped "
    "we they i you he she us them our their my your".split()
)
# Prepositions dropped only when a when-value follows ("to Thursday" yes, "to Denver" no).
_VALUE_PREPOSITIONS = frozenset(
    {"to", "from", "for", "on", "at", "in", "by", "until", "till", "through"}
    | {"в", "во", "на", "w", "we", "am", "um", "le", "el", "la", "di", "de", "op", "på", "i", "en", "à", "a", "の", "에"}
    | {"v", "ve", "na", "em", "no", "alle", "klo", "kl", "την", "τη", "στις", "στην"}
)
_NEGATION = frozenset({"not", "never", "cancelled", "canceled", "cancel"})
#: State words whose opposite is a negation: "disabled" == "not enabled".
_ANTONYM = {
    "disabled": "enabled",
    "disable": "enable",
    "disallowed": "allowed",
    "inactive": "active",
    "unlocked": "locked",
    "unavailable": "available",
    "unsupported": "supported",
}
_CONTRACTION_BASE = {
    "isn": "is",
    "aren": "are",
    "wasn": "was",
    "weren": "were",
    "don": "do",
    "doesn": "does",
    "didn": "did",
    "won": "will",
    "can": "can",
    "ca": "can",
    "couldn": "could",
    "shouldn": "should",
    "wouldn": "would",
    "hasn": "has",
    "haven": "have",
    "hadn": "had",
    "mustn": "must",
    "needn": "need",
}
_CLITICS = {"ll": "will", "re": "are", "ve": "have", "m": "am", "d": "would"}
_MERIDIEM_FORMS = (
    (("a", ".", "m", "."), "am"),
    (("p", ".", "m", "."), "pm"),
    (("a", ".", "m"), "am"),
    (("p", ".", "m"), "pm"),
    (("am",), "am"),
    (("pm",), "pm"),
)
_NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
_NUMBER_WORDS.update({w: 13 + i for i, w in enumerate("thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())})
_TENS = {w: 20 + 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_NUMBER_WORDS.update(_TENS)
_COMPOUND_NUMBER = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(one|two|three|four|five|six|seven|eight|nine)\b", re.IGNORECASE
)
#: Words that may follow a bare hour ("at 10 on Tuesday", "at ten sharp"); any
#: other word means the number is not a time ("at 10 Downing Street").
_CLOCK_FOLLOWERS = frozenset("on in at for every each daily sharp and or tomorrow today tonight then until till before after".split())
_BARE_HOUR_PREV = frozenset({"at", "until", "till", "alle", "klo", "kl", "um", "om", "às"})
#: "The locker is at 10", "desk at 12": a number after "at" that locates a thing is not an hour.
_PLACE_NOUNS = frozenset(
    "locker desk room seat gate table platform booth slot stand berth bay dock pier lane row aisle "
    "cabin bed office suite unit apartment apt floor level stall pump terminal".split()
)
#: Ratios, scores and references written with a colon are not clock times.
_NOT_TIME_NOUNS = frozenset("ratio score odds verse chapter version aspect result tally".split())
_HHMM_PREV = frozenset({"at", "to", "until", "till", "by", "from"})
_ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}(t\d{1,2}:\d{2}(:\d{2})?z?)?$")
_SLASH_DATE = re.compile(r"^\d{1,4}/\d{1,2}(/\d{1,4})?$")
_CLOCK = re.compile(r"^(\d{1,2})(?:[:.](\d{2}))?(am|pm)?$")
_H_CLOCK = re.compile(r"^(\d{1,2})h(\d{2})$")
_CANON_CLOCK = re.compile(r"^t\d{2}:\d{2}a?$")
_ORDINAL_DAY = re.compile(r"^(\d{1,2})(st|nd|rd|th)$")
_OFFSET_ZONE = re.compile(r"^(utc|gmt)[+-]\d{1,2}(:\d{2})?$")
_QUARTER = re.compile(r"^q[1-4]$")
_DOT_DATE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")  # 9.10.2026
_RANGE = re.compile(r"^(\d{1,2})(?::(\d{2}))?-(\d{1,2})(?::(\d{2}))?(am|pm|h)?$")  # 10-11am, 9:30-9:45, 14-16h
_DASHES = ("-", "–", "—")
#: Nouns after a weekday that make it a label ("The Tuesday team").
_LABEL_NEXT = frozenset("team crew group squad club cohort committee choir band league gang".split())
#: Recurrence words are when-values ("weekly" vs "biweekly").
_RECURRENCE = frozenset("daily weekly biweekly fortnightly monthly bimonthly quarterly yearly annually hourly".split())
_ORDINAL_WORDS = frozenset("first second third fourth fifth".split())
_ORDINAL_DAY_WORDS = {
    w: i + 1
    for i, w in enumerate(
        "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth "
        "sixteenth seventeenth eighteenth nineteenth twentieth".split()
    )
}
_OLD_VALUE_PREV = frozenset({"formerly", "previously", "originally"})
_LABEL_NEXT_EXTRA = frozenset("class course lecture seminar workshop practice".split())
_DURATION_UNITS = {
    **{u: "min" for u in ("min", "mins", "minute", "minutes")},
    **{u: "h" for u in ("h", "hr", "hrs", "hour", "hours")},
    **{u: "d" for u in ("day", "days")},
    **{u: "w" for u in ("week", "weeks", "wk", "wks")},
    **{u: "mo" for u in ("month", "months")},
    **{u: "y" for u in ("year", "years", "yr", "yrs")},
}
_CURRENCY = frozenset("$ € £ ¥ ₹ ₽ ₩ 円 元".split())
_CURRENCY_CODES = frozenset("usd eur gbp jpy cny rmb inr rub chf cad aud nzd sek nok dkk pln krw brl mxn".split())
_MAGNITUDE = re.compile(r"^(\d+(?:[.,]\d+)?)(k|m|bn|b)$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_PERCENT_WORDS = frozenset({"percent", "pct", "per cent"})
_SIGNS = ("-", "−", "+")
_VERSION = re.compile(r"^v\d+(\.\d+)*$")
_DOT_ISO = re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}$")
_DAY_RANGE = re.compile(r"^\d{1,2}-\d{1,2}$")
_IANA_AREAS = frozenset("africa america antarctica asia atlantic australia europe indian pacific etc".split())
#: Month and weekday names in other common languages, mapped to English for comparison.
_FOREIGN = {
    # French
    "janvier": "january",
    "février": "february",
    "fevrier": "february",
    "mars": "march",
    "avril": "april",
    "mai": "may",
    "juin": "june",
    "juillet": "july",
    "août": "august",
    "aout": "august",
    "septembre": "september",
    "octobre": "october",
    "novembre": "november",
    "décembre": "december",
    "decembre": "december",
    "lundi": "monday",
    "mardi": "tuesday",
    "mercredi": "wednesday",
    "jeudi": "thursday",
    "vendredi": "friday",
    "samedi": "saturday",
    "dimanche": "sunday",
    # Spanish
    "enero": "january",
    "febrero": "february",
    "marzo": "march",
    "abril": "april",
    "mayo": "may",
    "junio": "june",
    "julio": "july",
    "agosto": "august",
    "septiembre": "september",
    "setiembre": "september",
    "octubre": "october",
    "noviembre": "november",
    "diciembre": "december",
    "lunes": "monday",
    "martes": "tuesday",
    "miércoles": "wednesday",
    "miercoles": "wednesday",
    "jueves": "thursday",
    "viernes": "friday",
    "sábado": "saturday",
    "sabado": "saturday",
    "domingo": "sunday",
    # Portuguese months
    "marco": "march",
    "maio": "may",
    "setembro": "september",
    "novembro": "november",
    # Norwegian / Danish
    "mandag": "monday",
    "tirsdag": "tuesday",
    "lørdag": "saturday",
    "søndag": "sunday",
    # Finnish (with the -na "on" ending)
    "maanantai": "monday",
    "maanantaina": "monday",
    "tiistai": "tuesday",
    "tiistaina": "tuesday",
    "keskiviikko": "wednesday",
    "keskiviikkona": "wednesday",
    "torstai": "thursday",
    "torstaina": "thursday",
    "perjantai": "friday",
    "perjantaina": "friday",
    "lauantai": "saturday",
    "lauantaina": "saturday",
    "sunnuntai": "sunday",
    "sunnuntaina": "sunday",
    # Czech
    "pondělí": "monday",
    "úterý": "tuesday",
    "středa": "wednesday",
    "středu": "wednesday",
    "čtvrtek": "thursday",
    "pátek": "friday",
    "sobotu": "saturday",
    "neděle": "sunday",
    "neděli": "sunday",
    # Greek ("τρίτη" is also "third": ambiguous, needs a date cue)
    "δευτέρα": "monday",
    "τρίτη": "tuesday",
    "τετάρτη": "wednesday",
    "πέμπτη": "thursday",
    "παρασκευή": "friday",
    "σάββατο": "saturday",
    "κυριακή": "sunday",
    # Italian
    "lunedì": "monday",
    "lunedi": "monday",
    "martedì": "tuesday",
    "martedi": "tuesday",
    "mercoledì": "wednesday",
    "mercoledi": "wednesday",
    "giovedì": "thursday",
    "giovedi": "thursday",
    "venerdì": "friday",
    "venerdi": "friday",
    "sabato": "saturday",
    "domenica": "sunday",
    # German
    "januar": "january",
    "februar": "february",
    "märz": "march",
    "maerz": "march",
    "juni": "june",
    "juli": "july",
    "oktober": "october",
    "dezember": "december",
    "montag": "monday",
    "dienstag": "tuesday",
    "mittwoch": "wednesday",
    "donnerstag": "thursday",
    "freitag": "friday",
    "samstag": "saturday",
    "sonntag": "sunday",
    # Italian / Portuguese (distinct spellings only)
    "gennaio": "january",
    "febbraio": "february",
    "aprile": "april",
    "maggio": "may",
    "giugno": "june",
    "luglio": "july",
    "settembre": "september",
    "ottobre": "october",
    "dicembre": "december",
    "janeiro": "january",
    "fevereiro": "february",
    "março": "march",
    "junho": "june",
    "julho": "july",
    "outubro": "october",
    "dezembro": "december",
}
_FOREIGN.update(
    {
        # Dutch
        "maandag": "monday",
        "dinsdag": "tuesday",
        "woensdag": "wednesday",
        "donderdag": "thursday",
        "vrijdag": "friday",
        "zaterdag": "saturday",
        "zondag": "sunday",
        "januari": "january",
        "februari": "february",
        "maart": "march",
        "mei": "may",
        "augustus": "august",
        # Swedish
        "måndag": "monday",
        "tisdag": "tuesday",
        "onsdag": "wednesday",
        "torsdag": "thursday",
        "fredag": "friday",
        "lördag": "saturday",
        "söndag": "sunday",
        "maj": "may",
        "augusti": "august",
        # Polish (incl. accusative forms used after "w")
        "poniedziałek": "monday",
        "wtorek": "tuesday",
        "środa": "wednesday",
        "środę": "wednesday",
        "czwartek": "thursday",
        "piątek": "friday",
        "sobota": "saturday",
        "sobotę": "saturday",
        "niedziela": "sunday",
        "niedzielę": "sunday",
        # Turkish
        "pazartesi": "monday",
        "salı": "tuesday",
        "çarşamba": "wednesday",
        "perşembe": "thursday",
        "cuma": "friday",
        "cumartesi": "saturday",
        "pazar": "sunday",
        # Russian (incl. accusative forms used after "в")
        "понедельник": "monday",
        "вторник": "tuesday",
        "среда": "wednesday",
        "среду": "wednesday",
        "четверг": "thursday",
        "пятница": "friday",
        "пятницу": "friday",
        "суббота": "saturday",
        "субботу": "saturday",
        "воскресенье": "sunday",
        "январь": "january",
        "января": "january",
        "февраль": "february",
        "февраля": "february",
        "март": "march",
        "марта": "march",
        "апрель": "april",
        "апреля": "april",
        "май": "may",
        "мая": "may",
        "июнь": "june",
        "июня": "june",
        "июль": "july",
        "июля": "july",
        "август": "august",
        "августа": "august",
        "сентябрь": "september",
        "сентября": "september",
        "октябрь": "october",
        "октября": "october",
        "ноябрь": "november",
        "ноября": "november",
        "декабрь": "december",
        "декабря": "december",
    }
)
#: Foreign date words that are also common names or words ("Mars", "Mai", "Pazar" = market):
#: a date only with a day number beside them or a date preposition before them.
_FOREIGN_AMBIGUOUS = frozenset({"mars", "mai", "maj", "pazar", "cuma", "mayo", "marzo", "juni", "juli", "august", "april", "mart", "τρίτη"})
_FOREIGN.update(
    {
        # German adverbial / Swedish definite weekday forms
        "montags": "monday",
        "dienstags": "tuesday",
        "mittwochs": "wednesday",
        "donnerstags": "thursday",
        "freitags": "friday",
        "samstags": "saturday",
        "sonntags": "sunday",
        "måndagen": "monday",
        "tisdagen": "tuesday",
        "onsdagen": "wednesday",
        "torsdagen": "thursday",
        "fredagen": "friday",
        "lördagen": "saturday",
        "söndagen": "sunday",
        # Turkish months
        "ocak": "january",
        "şubat": "february",
        "mart": "march",
        "nisan": "april",
        "mayıs": "may",
        "haziran": "june",
        "temmuz": "july",
        "ağustos": "august",
        "eylül": "september",
        "ekim": "october",
        "kasım": "november",
        "aralık": "december",
        # Polish months: nominative, genitive ("9 października"), locative ("w maju")
        "styczeń": "january",
        "stycznia": "january",
        "styczniu": "january",
        "luty": "february",
        "lutego": "february",
        "lutym": "february",
        "marzec": "march",
        "marca": "march",
        "marcu": "march",
        "kwiecień": "april",
        "kwietnia": "april",
        "kwietniu": "april",
        "maja": "may",
        "maju": "may",
        "czerwiec": "june",
        "czerwca": "june",
        "czerwcu": "june",
        "lipiec": "july",
        "lipca": "july",
        "lipcu": "july",
        "sierpień": "august",
        "sierpnia": "august",
        "sierpniu": "august",
        "wrzesień": "september",
        "września": "september",
        "wrześniu": "september",
        "październik": "october",
        "października": "october",
        "październiku": "october",
        "listopad": "november",
        "listopada": "november",
        "listopadzie": "november",
        "grudzień": "december",
        "grudnia": "december",
        "grudniu": "december",
    }
)
_DATE_NEXT_CUES = frozenset({"günü", "gün", "dag", "dagen", "tag"})  # Turkish/Swedish/German "day" after a weekday
_HOUR_WORDS = frozenset({"uur", "uhr", "часов", "часа", "час", "o'clock"})
_KLOCKAN = frozenset({"klockan", "kl"})
_GERMAN_NUMBERS = {w: i + 1 for i, w in enumerate("eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf".split())}
_CURRENCY_WORDS = {
    "dollar": "$",
    "dollars": "$",
    "bucks": "$",
    "usd": "$",
    "euro": "€",
    "euros": "€",
    "eur": "€",
    "pound": "£",
    "pounds": "£",
    "gbp": "£",
    "yen": "¥",
    "jpy": "¥",
    "rupee": "₹",
    "rupees": "₹",
    "rs": "₹",
    "inr": "₹",
    "kr": "kr",
    "₽": "₽",
    "원": "₩",
    "₩": "₩",
    "francs": "fr",
    "franc": "fr",
    "fr": "fr",
    "kč": "kč",
    "czk": "kč",
    "euroa": "€",
    "euron": "€",
    "euroja": "€",
    "sek": "kr",
    "nok": "kr",
    "dkk": "kr",
    "zł": "zł",
    "zl": "zł",
    "pln": "zł",
    "tl": "₺",
    "lira": "₺",
    "₺": "₺",
    "руб": "₽",
    "рубль": "₽",
    "рубля": "₽",
    "рублей": "₽",
    "won": "₩",
}
_UNIT_GLUED = re.compile(r"^(-?\d+(?:\.\d+)?)(min|mins|h|hr|hrs|d|wk|c|f)$")
_MEASURE_GLUED = re.compile(
    r"^(-?\d+(?:\.\d+)?)(ms|us|ns|kb|mb|gb|tb|pb|kib|mib|gib|tib|kg|mg|km|cm|mm|mbps|gbps|kbps|ghz|mhz|khz|kw|kwh|mph|kph|ml)$"
)
_CJK_WEEKDAY = re.compile(r"(?:星期|礼拜|週|周)([一二三四五六日天])")
_JA_WEEKDAY = re.compile(r"([月火水木金土日])曜(?:日)?")
_JA_DAYS = {"月": "monday", "火": "tuesday", "水": "wednesday", "木": "thursday", "金": "friday", "土": "saturday", "日": "sunday"}
_KO_WEEKDAY = re.compile(r"([월화수목금토일])요일")
_KO_DAYS = {"월": "monday", "화": "tuesday", "수": "wednesday", "목": "thursday", "금": "friday", "토": "saturday", "일": "sunday"}
_CJK_MONTH_DAY = re.compile(r"(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]")
_JA_TIME = re.compile(r"(午前|午後)?\s*(\d{1,2})\s*時(?:\s*(\d{1,2})\s*分)?")
_KO_TIME = re.compile(r"(오전|오후)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?")
_MONTH_NAMES = ("january february march april may june july august september october november december").split()
_CJK_DAYS = {
    "一": "monday",
    "二": "tuesday",
    "三": "wednesday",
    "四": "thursday",
    "五": "friday",
    "六": "saturday",
    "日": "sunday",
    "天": "sunday",
}
_APOSTROPHES = ("'", "’")
_SPECIAL_PUNCT = frozenset({"'", "’", "-", "=", "+", "−", "→", "⇒", "⟶"})
_ARROWS = ("→", "⇒", "⟶")


def _canon_clock(hour: int, minute: int | None, meridiem: str | None, *, twenty_four: bool = False) -> str | None:
    m = minute or 0
    if m > 59:
        return None
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        return f"t{hour % 12 + (12 if meridiem == 'pm' else 0):02d}:{m:02d}"
    if hour == 24 and m == 0:
        hour = 0  # "24:00" == midnight
    if hour > 23:
        return None
    # A bare "7:00" or "at 7" does not say am or pm: it stays distinct from both ("a" = ambiguous).
    # "12:00" reads as noon (midnight is written 0:00/00:00 or "midnight").
    return f"t{hour:02d}:{m:02d}" if (twenty_four or hour >= 12 or hour == 0) else f"t{hour:02d}:{m:02d}a"


def _time_word(tok: str) -> bool:
    return tok in _ZONE_ABBR or tok in _ZONE_NAMES or tok in _RELATIVE or _EQUIV.get(tok, tok) in _WEEKDAYS


def _follower_ok(raw: list[str], j: int) -> bool:
    """True when what follows a bare number lets it be an hour (end, punctuation, a time word).

    "at 14, Elm Street" and "at 12 in building B" are addresses: a comma or
    "in" counts only when a time word follows ("at 10, tomorrow", "at 10 in
    the morning").
    """
    n = len(raw)
    if j >= n:
        return True
    nxt = raw[j]
    if nxt in (",", "in"):
        k = j + 1
        while k < n and raw[k] in ("the", "a"):
            k += 1
        if k >= n:
            return True
        after = raw[k]
        if not after.isalpha():
            return nxt == ","
        return _time_word(after) or after in _CLOCK_FOLLOWERS
    if not nxt.isalpha():
        return True
    return nxt in _CLOCK_FOLLOWERS or _time_word(nxt)


def _plural_noun(tok: str) -> bool:
    """A counted noun after "3 PM" ("3 PM interns" is a head count, not a time)."""
    return tok.isalpha() and len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and tok not in _ZONE_ABBR


def _compare_tokens(text: str) -> list[str]:
    return _compare_tokens_cased(text)[0]


def _signed_tokens(text: str) -> list[str]:
    """Raw tokens with a minus sign joined to the number it touches when the sign itself
    stands free ("was -5%" -> "-5"); a sign inside an identifier ("INV-10%") stays apart."""
    pairs = _spaced_tokens(text)
    out: list[str] = []
    k = 0
    while k < len(pairs):
        tok, space = pairs[k]
        if tok in ("-", "−") and k + 1 < len(pairs) and not pairs[k + 1][1] and pairs[k + 1][0][:1].isdigit() and (k == 0 or space):
            out.append("-" + pairs[k + 1][0])
            k += 2
            continue
        out.append(tok)
        k += 1
    return out


def _ascii_digits(text: str) -> str:
    """Map every Unicode decimal digit (full-width, Arabic-Indic, ...) to ASCII for comparison only."""
    if text.isascii():
        return text
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() and not c.isascii() else c for c in text)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xAC00 <= o <= 0xD7AF


_ZH_MERIDIEM = {"下午": "pm", "晚上": "pm", "上午": "am", "早上": "am"}
_ZH_TIME = re.compile(r"(上午|下午|晚上|早上)?\s*(\d{1,2})\s*点(?:\s*(\d{1,2})\s*分)?")
_CJK_MAN = re.compile(r"(\d+(?:\.\d+)?)\s*(万|萬|만|億|억)\s*(円|원|元)?")
_CJK_SCALE = {"万": 10_000, "萬": 10_000, "만": 10_000, "億": 100_000_000, "억": 100_000_000}
_CJK_CURRENCY = {"円": "円", "원": "원", "元": "元"}


def _cjk_man_amount(m: re.Match[str]) -> str:
    """'50万円' -> ' 500000 ¥ ': a counted ten-thousand is an ordinary number."""
    digits = m.group(1)
    if len(digits) > 40:
        return m.group(0)  # absurdly long: leave as written rather than invent a value
    amount = Decimal(digits) * _CJK_SCALE[m.group(2)]
    amount = amount.to_integral_value() if amount == amount.to_integral_value() else amount.normalize()
    cur = _CJK_CURRENCY.get(m.group(3) or "", "")
    return f" {amount} {cur} " if cur else f" {amount} "


def _cjk_spaced(text: str) -> str:
    """CJK has no spaces between words: weekday names become English words and every
    other ideograph its own token, so the core and values compare as in English."""
    if text.isascii():
        return text
    text = _CJK_MONTH_DAY.sub(
        lambda m: f" {_MONTH_NAMES[int(m.group(1)) - 1] if 1 <= int(m.group(1)) <= 12 else m.group(1)} {m.group(2)} ", text
    )
    text = _KO_TIME.sub(
        lambda m: f" {m.group(2)}:{(m.group(3) or '0').zfill(2)}{'pm' if m.group(1) == '오후' else 'am' if m.group(1) == '오전' else ''} ",
        text,
    )
    text = _JA_TIME.sub(
        lambda m: f" {m.group(2)}:{(m.group(3) or '0').zfill(2)}{'pm' if m.group(1) == '午後' else 'am' if m.group(1) == '午前' else ''} ",
        text,
    )
    text = _CJK_MAN.sub(_cjk_man_amount, text)
    text = _ZH_TIME.sub(lambda m: f" {m.group(2)}:{(m.group(3) or '0').zfill(2)}{_ZH_MERIDIEM.get(m.group(1) or '', '')} ", text)
    text = _JA_WEEKDAY.sub(lambda m: f" {_JA_DAYS[m.group(1)]} ", text)
    text = _KO_WEEKDAY.sub(lambda m: f" {_KO_DAYS[m.group(1)]} ", text)
    text = _CJK_WEEKDAY.sub(lambda m: f" {_CJK_DAYS[m.group(1)]} ", text)
    return "".join(f" {c} " if _is_cjk(c) else c for c in text)


_HOURS_WORDS = frozenset("hours hour open opens opening shift shifts".split())


def _range_clocks(tok: str, hours_text: bool = False) -> tuple[str, str] | None:
    """'10-11am' / '9:30-9:45' -> two canonical clocks; the start inherits the end's am/pm.

    A bare range ('9-5') is a time range only in a sentence about hours ("Office hours are 9-5").
    """
    m = _RANGE.match(tok)
    if not m or not (m.group(5) or m.group(2) or m.group(4) or hours_text):
        return None
    h1, m1, h2, m2, mer = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), m.group(5)
    if mer == "h":  # "14-16h": 24-hour clock
        a, b = (
            _canon_clock(h1, int(m1) if m1 else 0, None, twenty_four=True),
            _canon_clock(h2, int(m2) if m2 else 0, None, twenty_four=True),
        )
        return (a, b) if a and b else None
    mer1 = mer
    if mer and h1 % 12 > h2 % 12:
        mer1 = "am" if mer == "pm" else "pm"  # "11-1pm" starts at 11am
    a = _canon_clock(h1, int(m1) if m1 else None, mer1, twenty_four=not mer and m1 is not None)
    b = _canon_clock(h2, int(m2) if m2 else None, mer, twenty_four=not mer and m2 is not None)
    return (a, b) if a and b else None


_SPACED_THOUSANDS = re.compile(r"(?<![\d.,])(\d{1,3})((?:[ \u00a0\u202f]\d{3})+)((?:,\d{1,2})?)(?![.,]?\d)")
_PT_DAYS = {"segunda": "monday", "terça": "tuesday", "terca": "tuesday", "quarta": "wednesday", "quinta": "thursday", "sexta": "friday"}
_PT_WEEKDAY = re.compile(r"\b(segunda|terça|terca|quarta|quinta|sexta)-feira\b", re.IGNORECASE)


def _compare_tokens_cased(text: str) -> tuple[list[str], list[bool]]:
    """Normal form for changed_fact: case-folded; contractions, clocks, arrows and offsets resolved.

    Abbreviated month and weekday names are kept as written here: whether one
    is a date or a name is decided in context by :func:`_value_at`.
    """
    text = re.sub(r"(?<=\d)[–—](?=\d)", "-", text)  # "5–10%" == "5-10%"
    text = _SPACED_THOUSANDS.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)) + m.group(3), text)  # "40 000" == "40000"
    text = _PT_WEEKDAY.sub(lambda m: _PT_DAYS[m.group(1).casefold()], text)  # "terça-feira" == "tuesday"
    text = _COMPOUND_NUMBER.sub(lambda m: str(_TENS[m.group(1).casefold()] + _NUMBER_WORDS[m.group(2).casefold()]), text)
    orig = _signed_tokens(_cjk_spaced(_ascii_digits(unicodedata.normalize("NFKC", text))))
    raw = [t.casefold() for t in orig]
    out: list[str] = []
    titled: list[bool] = []  # was the source token Title-case ("Networks", "Club")?
    i, n = 0, len(raw)
    hours_text = any(t in _HOURS_WORDS for t in raw)
    while i < n:
        mark = len(out)
        tok = raw[i]
        if len(tok) == 1 and not tok.isalnum() and tok not in _SPECIAL_PUNCT:
            out.append(tok)  # plain punctuation: nothing below can apply (fast path)
            titled.append(False)
            i += 1
            continue
        prev = out[-1] if out else ""
        if tok in _CONTRACTION_BASE and i + 2 < n and raw[i + 1] in _APOSTROPHES and raw[i + 2] == "t":
            out.extend((_CONTRACTION_BASE[tok], "not"))
            i += 3
            titled.extend([False] * (len(out) - mark))
            continue
        if tok == "cannot":
            out.extend(("can", "not"))
            i += 1
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _APOSTROPHES and i + 1 < n and raw[i + 1] == "s":
            i += 2  # "call's" == "call is"; a possessive "Ilona's" drops the same way on both sides
            titled.extend([False] * (len(out) - mark))
            continue
        if (
            tok in _APOSTROPHES
            and i + 1 < n
            and out
            and (out[-1].startswith("~") or out[-1] in _WEEKDAYS or out[-1] in _MONTHS)
            and raw[i + 1].isalpha()
            and len(raw[i + 1]) <= 4
        ):
            i += 2  # Turkish case suffix on a date word: "Salı'ya", "Ekim'de"
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _APOSTROPHES and i + 1 < n and raw[i + 1] in _CLITICS and out and out[-1].isalpha():
            out.append(_CLITICS[raw[i + 1]])  # "We'll" == "We will"
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in ("-", "=") and i + 1 < n and raw[i + 1] == ">":
            out.append("→")
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _ARROWS:
            out.append("→")
            i += 1
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in ("utc", "gmt") and i + 2 < n and raw[i + 1] in ("+", "-", "−") and re.match(r"^\d{1,2}(:\d{2})?$", raw[i + 2]):
            hh, _, mm = raw[i + 2].partition(":")
            out.append(f"{tok}{'+' if raw[i + 1] == '+' else '-'}{int(hh)}" + (f":{mm}" if mm and mm != "00" else ""))
            i += 3
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _MONTH_ABBR and i + 1 < n and raw[i + 1] == ".":
            out.append(tok)  # "Oct. 9" == "Oct 9"
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if tok == "12" and i + 1 < n and raw[i + 1] in ("noon", "midday", "midnight"):
            i += 1  # "12 noon" == "noon"
            titled.extend([False] * (len(out) - mark))
            continue
        if (
            tok in ("noon", "midday", "midnight")
            and orig[i].istitle()
            and i + 1 < n
            and orig[i + 1].istitle()
            and raw[i + 1] not in _WEEKDAYS
            and _EQUIV.get(raw[i + 1], raw[i + 1]) not in _WEEKDAYS
            and raw[i + 1] not in _MONTHS
        ):
            out.append(tok)  # "Midnight Oil": a name
            i += 1
            titled.extend([True] * (len(out) - mark))
            continue
        if _is_digits(tok) and i + 1 < n and raw[i + 1] in _HOUR_WORDS and int(tok) < 24:
            out.append(_canon_clock(int(tok), 0, None, twenty_four=True) or tok)  # "10 uur", "10 часов"
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _KLOCKAN and i + 1 < n and re.match(r"^\d{1,2}([.:]\d{2})?$", raw[i + 1]):
            hh, _, mm = raw[i + 1].replace(".", ":").partition(":")
            out.append(_canon_clock(int(hh), int(mm) if mm else 0, None, twenty_four=True) or raw[i + 1])  # "klockan 10"
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if tok == "halb" and i + 1 < n and (raw[i + 1] in _GERMAN_NUMBERS or _is_digits(raw[i + 1])):
            h = _GERMAN_NUMBERS.get(raw[i + 1]) or int(raw[i + 1])
            clock = _canon_clock((h - 1) or 12, 30, None)  # German "halb zehn" == 9:30
            if clock is not None:  # "halb 99" is no clock: never put None in the token stream
                out.append(clock)
                i += 2
                titled.extend([False] * (len(out) - mark))
                continue
        if tok in ("noon", "midday"):
            out.append("t12:00")  # "noon" == "12pm" == "12:00"
            i += 1
            titled.extend([False] * (len(out) - mark))
            continue
        if tok == "midnight":
            out.append("t00:00")
            i += 1
            titled.extend([False] * (len(out) - mark))
            continue
        if (
            _is_digits(tok)
            and i + 2 < n
            and raw[i + 1] == "."
            and (raw[i + 2] in _FOREIGN or raw[i + 2] in _MONTH_NAMES or raw[i + 2] in _MONTH_ABBR)
        ):
            out.append(tok)  # German "9. Oktober" == "9 Oktober"
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if _is_digits(tok) and i + 1 < n and raw[i + 1] == "uhr" and int(tok) < 24:
            out.append(_canon_clock(int(tok), 0, None, twenty_four=True) or tok)  # "um 10 Uhr"
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _FOREIGN:
            out.append(f"~{_FOREIGN[tok]}|{tok}")  # foreign date word: judged in context by _value_at
            i += 1
            titled.extend([orig[i - 1].istitle()] * (len(out) - mark))
            continue
        if tok in _IANA_AREAS and i + 2 < n and raw[i + 1] == "/" and raw[i + 2].isalpha():
            j = i + 3
            parts = [raw[i + 2]]
            while j + 1 < n and raw[j] == "_" and raw[j + 1].isalpha():  # "America/New_York"
                parts.append(raw[j + 1])
                j += 2
            out.append(f"tz:{tok}/{'_'.join(parts)}")  # "Europe/Berlin"
            i = j
            titled.extend([False] * (len(out) - mark))
            continue
        rng = _range_clocks(tok, hours_text)
        if rng is not None:
            out.extend(rng)
            i += 1
            titled.extend([False] * (len(out) - mark))
            continue
        if _is_digits(tok) and i + 2 < n and raw[i + 1] in _DASHES:
            end_m = _CLOCK.match(raw[i + 2])
            if end_m and end_m.group(3):  # "10 - 11am", "10–11am": the start takes the end's am/pm
                mer = end_m.group(3)
                h1, h2 = int(tok), int(end_m.group(1))
                mer1 = mer if h1 % 12 <= h2 % 12 else ("am" if mer == "pm" else "pm")
                start = _canon_clock(h1, None, mer1)
                if start is not None:
                    out.append(start)
                    i += 1
                    titled.extend([False] * (len(out) - mark))
                    continue
        clock, used = _clock_at(raw, i, prev)
        if clock is not None:
            out.append(clock)
            i += used
            titled.extend([False] * (len(out) - mark))
            continue
        out.append(tok)
        i += 1
        titled.extend([orig[i - 1].istitle() and not orig[i - 1].isupper()] * (len(out) - mark))
        continue
    return out, titled


def _clock_at(raw: list[str], i: int, prev: str) -> tuple[str | None, int]:
    """A canonical clock starting at raw[i] and the number of tokens it uses, or (None, 0)."""
    tok = raw[i]
    n = len(raw)
    # "half past ten", "quarter to 11"
    if tok in ("half", "quarter") and i + 2 < n and raw[i + 1] in ("past", "to"):
        hour = _NUMBER_WORDS.get(raw[i + 2], int(raw[i + 2]) if _is_digits(raw[i + 2]) else None)
        if hour is not None and 1 <= hour <= 12:
            minute = 30 if tok == "half" else 15
            if raw[i + 1] == "to":
                if tok == "half":
                    return None, 0
                hour, minute = (hour - 1) or 12, 45
            return _canon_clock(hour, minute, None), 3
    m = _H_CLOCK.match(tok)
    if m:
        return _canon_clock(int(m.group(1)), int(m.group(2)), None, twenty_four=True), 1
    hour_word = _NUMBER_WORDS.get(tok)
    if hour_word is not None and 1 <= hour_word <= 12:
        if i + 3 < n + 1 and raw[i + 1 : i + 4] == ["o", "'", "clock"]:
            return _canon_clock(hour_word, None, None), 4
        if prev in _BARE_HOUR_PREV and _follower_ok(raw, i + 1):
            return _canon_clock(hour_word, None, None), 1
        return None, 0
    if prev in _ID_NOUNS:
        return None, 0  # "room 10:30", "ticket 1400": identifiers, not times
    if not any(c in _ASCII_DIGITS for c in tok) and tok not in _NUMBER_WORDS and tok not in ("half", "quarter"):
        return None, 0
    back = [raw[k] for k in range(max(0, i - 5), i)]
    if any(t in _NOT_TIME_NOUNS for t in back[-3:]):
        return None, 0  # "the ratio is 16:10", "score was 21:15"
    if prev in _BARE_HOUR_PREV and _is_digits(tok) and any(t in _PLACE_NOUNS for t in back):
        return None, 0  # "Ilona's locker is at 10": a bare number locating a thing, not an hour
    if _is_digits(tok) and len(tok) == 4 and prev in _HHMM_PREV:
        value = int(tok)
        is_year = 1900 <= value <= 2099 and prev != "at"
        hours = i + 1 < n and raw[i + 1] in ("hours", "hrs")
        # "at 1600 Pennsylvania Ave" is an address: a word after the number rules out a time.
        if not is_year and value // 100 < 24 and value % 100 < 60 and (hours or _follower_ok(raw, i + 1)):
            used = 2 if i + 1 < n and raw[i + 1] in ("hours", "hrs") else 1
            return _canon_clock(value // 100, value % 100, None, twenty_four=True), used
    m = _CLOCK.match(tok)
    if not m:
        return None, 0
    hour, minute_s, mer = int(m.group(1)), m.group(2), m.group(3)
    j = i + 1
    if mer is None:
        for form, mer_form in _MERIDIEM_FORMS:
            if tuple(raw[j : j + len(form)]) == form:
                mer, j = mer_form, j + len(form)
                break
    if minute_s is not None and "." in tok and mer is None and (prev not in _BARE_HOUR_PREV or not _follower_ok(raw, i + 1)):
        return None, 0  # "1.50" is a number, "at 10.30 Main Street" an address; "10.30am", "at 10.30" are times
    if raw[j : j + 3] == ["o", "'", "clock"] and minute_s is None and mer is None:
        return _canon_clock(hour, None, None), j + 3 - i
    if mer is not None:
        if minute_s is None and j < n and _plural_noun(raw[j]):
            return None, 0  # "Hire 3 PM interns"
        return _canon_clock(hour, int(minute_s) if minute_s else None, mer), j - i
    if minute_s is not None:
        hour_s = m.group(1)
        return _canon_clock(hour, int(minute_s), None, twenty_four=len(hour_s) == 2 and hour_s[0] == "0"), 1
    if prev in _BARE_HOUR_PREV and _follower_ok(raw, i + 1):
        return _canon_clock(hour, None, None), 1  # "The standup is at 10"
    return None, 0


@dataclass(frozen=True)
class FactSlots:
    """A statement split into its when/zone values and everything else (the core)."""

    core: tuple[str, ...]
    values: frozenset[str]
    negations: int
    names: tuple[str, ...] = ()  # capitalised words in order (who did what to whom)


def _is_clock(tok: str) -> bool:
    return bool(_CANON_CLOCK.match(tok))


def _near_clock(toks: list[str], i: int, span: int = 2) -> bool:
    return any(_is_clock(toks[k]) for k in range(max(0, i - span), min(len(toks), i + span + 1)) if k != i)


def _zone_phrase_before(toks: list[str], i: int) -> bool:
    return any(toks[k] in _ZONE_CONTEXT for k in range(max(0, i - 5), i))  # "timezone for reports is UTC"


#: Words before a bare weekday abbreviation that make it a date ("review is Tue", "Meet Sat").
_ABBR_DAY_PREV = frozenset("is are was were be meet meeting meets see until till by due every on moved moves".split())

#: Words before "<Day> at <clock>" that make the day a person's name ("Lunch with Sun at noon", "Ask Wed at 5").
_DAY_AT_NOT_PREV = frozenset("with ask tell call thank thanks dear hi hello cc ping email text".split())


def _date_context(toks: list[str], i: int, scheduling: bool | None = None) -> bool:
    """Is the month/abbreviated weekday at *i* used as a date?"""
    prev = toks[i - 1] if i > 0 else ""
    nxt = toks[i + 1] if i + 1 < len(toks) else ""

    def dayish(t: str) -> bool:
        return _is_digits(t) or bool(_ORDINAL_DAY.match(t)) or _is_clock(t) or bool(_SLASH_DATE.match(t) or _ISO_DATE.match(t))

    if prev in _DATE_PREV or dayish(prev) or dayish(nxt) or nxt == "→":
        return True
    if nxt in ("at", "@", "um", "à") and i + 2 < len(toks) and _is_clock(toks[i + 2]) and prev not in _DAY_AT_NOT_PREV:
        return True  # "The demo is Thu at 4pm"; not "Lunch with Sun at noon"
    if prev == "to":
        return scheduling if scheduling is not None else any(t in _SCHEDULING for t in toks)
    if prev == "of":
        before = toks[i - 2] if i >= 2 else ""
        dayish_before = bool(_ORDINAL_DAY.match(before)) or _is_digits(before) or before in _ORDINAL_DAY_WORDS
        return before in _SPAN_OF or dayish_before  # "end of May", "the 9th of October"
    if prev in ("and", "or", "&", "/") and i >= 2:
        before = toks[i - 2]  # "Tue and Thu": a list continues the date it follows
        return before in _WEEKDAYS or before in _WEEKDAY_ABBR or before in _MONTHS or before in _MONTH_ABBR
    return False


def _id_context(toks: list[str], i: int) -> bool:
    """Is the date-shaped token at *i* part of an identifier ("INC-2026-10-09", "employee id is 2026-10-08")?"""
    prev = toks[i - 1] if i > 0 else ""
    if prev in _ID_GLUE and i >= 2 and toks[i - 2].isalnum() and not _is_digits(toks[i - 2]):
        return True
    k = i - 1
    while k >= 0 and toks[k] in _ID_SKIP and i - k <= 3:
        k -= 1
    if k >= 0 and toks[k] in _VERB_OR_ID and not (k == 0 or toks[k - 1] in _ID_DETERMINERS):
        return False  # "We run 1.3.2": the verb, not "run 12"
    return k >= 0 and (toks[k] in _ID_NOUNS or prev in _ID_NOUNS)


def _named(toks: list[str], titled: list[bool], i: int) -> bool:
    """Is the day/month word at *i* part of a proper name ("March Networks", "The Thursday Murder Club", "Project Tuesday")?"""
    prev = toks[i - 1] if i > 0 else ""
    if prev in _LABEL_PREV:
        return True
    j = i + 1
    if j < len(toks) and titled[j] and toks[j].isalpha():
        nxt = toks[j]
        date_word = nxt in _WEEKDAYS or nxt in _MONTHS or nxt in _ZONE_NAMES or nxt in _ZONE_ABBR or nxt in _RELATIVE or nxt in _EQUIV
        return not date_word and nxt not in _ZONE_CONTEXT
    return False


def _zone_name_by_clock(toks: list[str], i: int) -> bool:
    """A zone name whose nearest word on either side, skipping punctuation, is a clock ("10am (Pacific)", "Pacific 10am")."""
    for step in (-1, 1):
        k = i + step
        while 0 <= k < len(toks) and len(toks[k]) == 1 and not toks[k].isalnum():
            k += step
        if 0 <= k < len(toks) and _is_clock(toks[k]):
            return True
    return False


#: Verbs that put a version in object position ("Prod runs 1.2.4", "Use Python 3.12", "upgrade to 2.0").
_VERSION_VERBS = frozenset(
    "run runs running ran use uses using used deploy deploys deployed deploying require requires required pin pins pinned "
    "install installs installed upgrade upgrades upgraded downgrade downgraded target targets targeting".split()
)


#: Words that make a glued "30s" a duration in seconds rather than a decade.
_SECONDS_CUES = frozenset(
    "timeout ttl latency delay interval every after wait waits takes took lasts expiry expires retry backoff poll".split()
)

#: Words before a glued "14h" that make it a clock, not a duration ("à 14h", "um 14h", "at 14h").
_H_CLOCK_PREV = frozenset("at à a las um om alle às kl klo kello ore vers dès gegen o".split())

_TEMP_CUES = frozenset(
    "thermostat temperature temp degrees heat heating cool cooling freezer fridge oven warm cold celsius fahrenheit "
    "room ac hvac indoor outdoor weather forecast setpoint set-point heat".split()
)
_DURATION_CUES = frozenset(
    "lasts last takes take took duration long for every runs run within after in about "
    "retention retain retained ttl expiry expires expire window timeout interval sla delay period cooldown".split()
)


def _glued_unit_cue(toks: list[str], i: int, unit: str) -> bool:
    """A glued unit ("21C", "30min", "1h") counts only where the sentence talks about that quantity."""
    window = toks[max(0, i - _CLAUSE_WINDOW) : i + _CLAUSE_WINDOW + 1]
    if unit in ("c", "f"):
        return any(t in _TEMP_CUES for t in window)
    if unit in ("min", "mins"):
        return True  # "30min" is never a label
    prev = toks[i - 1] if i > 0 else ""
    num = re.match(r"^\d+(?:\.\d+)?", toks[i])
    if unit == "h" and num is not None and float(num.group(0)) > 24:
        return True  # "48h" is never a clock; "3D printer", "bus 5d" stay labels
    cue_words = ("lasts", "takes", "duration", "long", "retention", "ttl", "timeout", "interval", "window", "expiry")
    return prev in _DURATION_CUES or any(t in cue_words for t in window)


#: Units of measure: the number and its unit are the value ("500 GB" -> "q:500gb").
_MEASURE_UNITS = {
    **{u: "b" for u in ("b", "byte", "bytes")},
    **{u: u for u in ("kb", "mb", "gb", "tb", "pb", "kib", "mib", "gib", "tib")},
    **{u: "kg" for u in ("kg", "kgs", "kilogram", "kilograms", "kilo", "kilos")},
    **{u: "g" for u in ("g", "gram", "grams")},
    **{u: "mg" for u in ("mg", "milligram", "milligrams")},
    **{u: "lb" for u in ("lb", "lbs")},
    **{u: "oz" for u in ("oz", "ounce", "ounces")},
    **{u: "km" for u in ("km", "kilometer", "kilometers", "kilometre", "kilometres")},
    **{u: "m" for u in ("m", "meter", "meters", "metre", "metres")},
    **{u: "cm" for u in ("cm", "centimeter", "centimeters")},
    **{u: "mm" for u in ("mm", "millimeter", "millimeters")},
    **{u: "mi" for u in ("mi", "mile", "miles")},
    **{u: "ft" for u in ("ft", "foot", "feet")},
    **{u: "yd" for u in ("yd", "yds", "yard", "yards")},
    **{u: "in" for u in ("inch", "inches")},
    **{u: u for u in ("mph", "kph", "kmh", "rpm", "rps", "qps", "fps", "hz", "khz", "mhz", "ghz", "w", "kw", "mw", "kwh", "v")},
    **{u: u for u in ("kbps", "mbps", "gbps", "tbps")},
    **{u: "s" for u in ("s", "sec", "secs", "second", "seconds")},
    **{u: "ms" for u in ("ms", "millisecond", "milliseconds")},
    **{u: "us" for u in ("us", "µs", "microsecond", "microseconds")},
    **{u: "ns" for u in ("ns", "nanosecond", "nanoseconds")},
    **{u: "l" for u in ("l", "liter", "liters", "litre", "litres")},
    **{u: "ml" for u in ("ml", "milliliter", "milliliters")},
    **{u: "gal" for u in ("gal", "gallon", "gallons")},
}
_RATE_NOUNS = {
    **{u: "req" for u in ("req", "reqs", "request", "requests")},
    **{u: "call" for u in ("call", "calls")},
    **{u: "op" for u in ("op", "ops")},
    **{u: "msg" for u in ("msg", "msgs", "message", "messages")},
    **{u: "event" for u in ("event", "events")},
}
_RATE_PER = {
    **{u: "s" for u in ("s", "sec", "second")},
    **{u: "min" for u in ("min", "minute")},
    **{u: "h" for u in ("h", "hr", "hour")},
    **{u: "d" for u in ("d", "day")},
    **{u: "w" for u in ("week", "wk")},
    **{u: "mo" for u in ("month",)},
    **{u: "y" for u in ("year", "yr")},
}
_CJK_COUNTERS = frozenset("명 人 名 个 個 件 台 位".split())
_COUNT_NOUNS = frozenset("people persons staff children men women employees".split())
#: Words that introduce a count ("has 7 engineers", "a team of 12 people"); a number after a
#: name or label ("Route 66 runs", "UA 10.30 departs") is an identifier, not a count.
_COUNT_PREV = frozenset(
    "has have had having of with needs need needed hire hires hired hiring are is was were be "
    "about around approximately only just exactly employs employ employed total totals to from by "
    "approved approve approves added add adds onboarded hired hire budgeted staffed".split()
)


def _counted_noun(tok: str) -> bool:
    """A plural noun after a number ("7 engineers", "2 cores", "50 people"): a count of it."""
    return tok in _COUNT_NOUNS or (_plural_noun(tok) and tok not in _MEASURE_UNITS and tok not in _DURATION_UNITS)


def _numval(t: str) -> str:
    """'ten' -> '10', '5,000' -> '5000'."""
    return str(_NUMBER_WORDS[t]) if t in _NUMBER_WORDS else _canon_num(t)


_MAGNITUDE_SCALE = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "bn": 1_000_000_000}


def _scaled(num: str, mag: str) -> str:
    """'5', 'k' -> '5000'; '2.5', 'm' -> '2500000' (exact)."""
    value = Decimal(num.replace(",", ".")) * _MAGNITUDE_SCALE[mag]
    return str(value.to_integral_value()) if value == value.to_integral_value() else str(value.normalize())


def _year_like(t: str) -> bool:
    return _is_digits(t) and len(t) == 4 and 1900 <= int(t) <= 2099


def _canon_num(t: str, cur: str = "", *, postfix: bool = False) -> str:
    """'5,000' == '5000', '5.00' == '5', '5,5' (decimal comma) == '5.5'.

    With a currency, two or more dotted groups are thousands ("1.200.000 ₽"); a single
    group is thousands only when the currency follows ("15.000 TL"), so "€5.000" and
    "$1.500" stay decimals.
    """
    neg = t.startswith("-")
    body = t.lstrip("-")
    if cur and re.match(r"^\d{1,3}(\.\d{3})+,\d{1,2}$", body):
        body = body.replace(".", "").replace(",", ".")  # "2.500,00 €"
    elif cur and re.match(r"^\d{1,3}(\.\d{3})+$", body) and (body.count(".") >= 2 or postfix):
        body = body.replace(".", "")
    elif re.match(r"^\d{1,3}(,\d{3})+(\.\d+)?$", body):
        body = body.replace(",", "")
    elif re.match(r"^\d+,\d{1,2}$", body):
        body = body.replace(",", ".")
    if "." in body:
        body = body.rstrip("0").rstrip(".") or "0"
    return ("-" if neg else "") + body


def _plain(tok: str) -> str:
    """A token with the foreign-word marker removed ("~october|octobre" -> "october")."""
    return tok[1:].partition("|")[0] if tok.startswith("~") else tok


#: How far context checks look around a token. Bounded so every per-token check is O(1)
#: and a statement is scanned in linear time however long it is.
_CLAUSE_WINDOW = 24


def _version_object(toks: list[str], i: int) -> bool:
    """Is the version at *i* a stated value rather than the subject's name?

    True after a usage verb anywhere earlier in the same clause ("We run the billing
    service on version 2.3.1", "upgraded the cluster to 1.29.3"), or after a copula
    that states a version ("The server version is 1.2.4", "Prod is on 1.2.4").
    """
    k = i - 1
    while k >= 0 and i - k <= _CLAUSE_WINDOW and toks[k] not in (".", ";", "!", "?", ":"):
        if toks[k] in _VERSION_VERBS:
            return True
        k -= 1
    prev = toks[i - 1] if i > 0 else ""
    before = toks[i - 2] if i > 1 else ""
    if prev in ("is", "was", "now") and "version" in toks[max(0, i - 4) : i]:
        return True  # "The server version is 1.2.4"
    return prev == "on" and before in ("is", "are", "was", "runs", "stays")  # "Prod is on 1.2.4"


_NUMBER_TOKEN = re.compile(r"^-?\d+(?:[.,]\d+)*$")
_DECIMAL_TOKEN = re.compile(r"^\d+\.\d+$")
_NUMBER_RANGE = re.compile(r"^(\d+(?:[.,]\d+)?)-(\d+(?:[.,]\d+)?)([km])?$")
_DAY_DOT_MONTH = re.compile(r"^\d{1,2}\.\d{1,2}$")
_HOUR_H = re.compile(r"^(\d{1,2})h$")
_PRERELEASE = re.compile(r"^(rc|beta|alpha|pre|dev|post|b|a)\d*$")


def _value_at(toks: list[str], i: int, titled: list[bool] | None = None, scheduling: bool | None = None) -> str | None:
    """The when/zone value at position *i*, judged in context, or None."""
    titled = titled if titled is not None else [False] * len(toks)
    t = toks[i]
    if t.startswith("~"):  # a foreign month/weekday word: "~english|source"
        word, _, src = t[1:].partition("|")
        prev = _plain(toks[i - 1]) if i > 0 else ""
        nxt = _plain(toks[i + 1]) if i + 1 < len(toks) else ""
        dayish = _is_digits(prev) or _is_digits(nxt) or bool(_ORDINAL_DAY.match(prev or ""))
        if _named(toks, titled, i):
            return None
        german_cue = prev in ("am", "im", "um", "ab", "bis", "vom", "zum", "seit", "i", "på", "w", "we", "в", "во", "op", "την", "τη")
        german_cue = german_cue or (prev in ("de", "del") and i >= 2 and _is_digits(_plain(toks[i - 2])))  # "15 de marzo"
        if src in _FOREIGN_AMBIGUOUS:  # "Mars", "Mai", "Pazar" are names or words too
            cue = dayish or german_cue or nxt in _DATE_NEXT_CUES or (_date_context(toks, i, scheduling) and not titled[i])
            return word if cue else None
        return word
    if len(t) == 1 and not t.isalnum():
        return None  # punctuation and symbols are never values (fast path)
    n = len(toks)
    prev = _plain(toks[i - 1]) if i > 0 else ""
    nxt = _plain(toks[i + 1]) if i + 1 < n else ""
    if _is_clock(t) and t.endswith("a") and i >= 2 and titled[i - 1] and prev.isalpha() and not _time_word(prev):
        return None  # "John 3:16", "Genesis 1:10": a reference after a name, not a time
    if _is_clock(t) or t in _RELATIVE or _OFFSET_ZONE.match(t) or t.startswith("tz:"):
        return t
    if _QUARTER.match(t):
        span = prev == "of" and i >= 2 and toks[i - 2] in _SPAN_OF  # "end of Q3"
        return t if (prev in _QUARTER_PREV or span) else None  # "slipped to Q4" yes, "the review for Q3" no
    if (t in _WEEKDAYS or t in _MONTHS or t in _EQUIV) and _named(toks, titled, i):
        return None
    if (t in _WEEKDAYS or t in _WEEKDAY_ABBR) and (
        (nxt in _LABEL_NEXT and prev in ("", "the", "a", "our", "their"))
        or (nxt in _LABEL_NEXT_EXTRA and (prev in ("", "the", "a", "our", "their") or (i > 0 and titled[i - 1])))
    ):
        return None  # "The Tuesday team", "Ilona's Tuesday class": a label
    if t in _RECURRENCE:
        return t
    if _DOT_ISO.match(t):
        y, mth, d = t.split(".")
        return None if _id_context(toks, i) else f"d{int(y)}-{int(mth)}-{int(d)}"  # "2026.12.31"
    if _DOT_DATE.match(t):
        if _id_context(toks, i):
            return None
        d, mth, y = t.split(".")
        return f"d{y}-{int(mth)}-{int(d)}"  # "31.12.2026" == "2026-12-31"
    if _id_context(toks, i) and (_SEMVER.match(t) or _DOT_ISO.match(t) or _VERSION.match(t)):
        return None  # "section 1.2.3", "Ticket 2026.10.09": identifiers
    verb_first = i >= 2 and toks[i - 2] in _VERSION_VERBS
    if (
        (_VERSION.match(t) or _SEMVER.match(t) or _DECIMAL_TOKEN.match(t))
        and prev in ("version", "release", "v", "build")
        and i <= 2
        and not verb_first
    ):
        return None  # "Release 10.8 is out": the version names the subject, not a changed value
    unit_next = nxt in _DURATION_UNITS or nxt in _CURRENCY_WORDS or nxt in ("%", "°", "degrees", "percent")
    money_near = nxt in _CURRENCY or nxt in _CURRENCY_CODES or nxt in _CURRENCY_WORDS or prev in _CURRENCY or prev in _CURRENCY_CODES
    unit_next = unit_next or money_near
    if (_VERSION.match(t) or _SEMVER.match(t)) and not money_near and not _version_object(toks, i):
        return None  # "mind-mem 5.0.4 fixes ...", "The 1.2.5 release": the version names the subject
    if (_VERSION.match(t) or _SEMVER.match(t)) and not money_near:
        suffix = ""
        if i + 2 < n and toks[i + 1] == "-" and _PRERELEASE.match(toks[i + 2]):
            suffix = "-" + toks[i + 2]  # "v1.2.3-rc1"
        return f"v:{t.lstrip('v')}{suffix}"  # "v1.3", "1.2.3"
    if _DECIMAL_TOKEN.match(t) and i > 0 and not unit_next and _version_object(toks, i):
        return f"v:{t}"  # "Use Python 3.12", "runs version 2.1"
    if t[:1].isdigit() and _id_context(toks, i) and nxt not in _MEASURE_UNITS and nxt not in _DURATION_UNITS:
        return None  # "INV-10%": part of an identifier; "The run is 5 km" keeps its distance
    sign = "-" if (prev in ("-", "−") and (i < 2 or not toks[i - 2].isalnum())) else ""
    num = bool(_NUMBER_TOKEN.match(t)) or t in _NUMBER_WORDS
    hour_h = _HOUR_H.match(t)
    if hour_h and int(hour_h.group(1)) <= 23 and (prev in _H_CLOCK_PREV or _EQUIV.get(prev, prev) in _WEEKDAYS):
        return _canon_clock(int(hour_h.group(1)), 0, None, twenty_four=True)  # "à 14h", "um 14h", "mardi 14h"
    secs = re.match(r"^(\d+(?:\.\d+)?)s$", t)
    if secs and any(w in _SECONDS_CUES for w in toks[max(0, i - _CLAUSE_WINDOW) : i + 3]):
        return f"q:{_canon_num(secs.group(1))}s"  # "Timeout is 30s" (not "the 30s")
    measured = _MEASURE_GLUED.match(t)
    if measured and not _id_context(toks, i):
        return f"q:{_canon_num(measured.group(1))}{_MEASURE_UNITS.get(measured.group(2), measured.group(2))}"  # "120ms", "500GB"
    glued = _UNIT_GLUED.match(t)
    if glued and not _glued_unit_cue(toks, i, glued.group(2)):
        glued = None  # "seat 12C", "bus 5d", "3D printer": a label, not a unit
    if glued:
        unit = glued.group(2)
        if unit in ("c", "f"):
            return f"deg:{glued.group(1)}{unit}"  # "21C"
        return f"q:{glued.group(1)}{_DURATION_UNITS.get(unit, unit)}"  # "30min", "1h"
    if _DAY_DOT_MONTH.match(t) and nxt == "." and prev in ("am", "bis", "ab", "vom", "zum", "den", "le", "el"):
        d, mth = t.split(".")
        return f"d--{int(mth)}-{int(d)}"  # German "am 9.10."
    if num and nxt in _CURRENCY_WORDS:
        return f"m:{_CURRENCY_WORDS[nxt]}{_canon_num(t, _CURRENCY_WORDS[nxt], postfix=True)}"  # "5000 dollars", "100 kr"
    if num and prev in _CURRENCY_WORDS:
        return f"m:{_CURRENCY_WORDS[prev]}{_canon_num(t, _CURRENCY_WORDS[prev])}"  # "Rs 500"
    if num and nxt in ("to", "-", "–") and i + 3 < n and _NUMBER_TOKEN.match(toks[i + 2]):
        unit = toks[i + 3]
        if unit in ("degrees", "°", "%") or unit in _DURATION_UNITS or unit in _CURRENCY_WORDS:
            return f"r:{t}:{unit}"  # the lower bound of "-5 to 5 degrees"
    mag = _MAGNITUDE.match(t)
    if mag and (prev in _CURRENCY or prev in _CURRENCY_CODES):
        return f"m:{prev}{_scaled(mag.group(1), mag.group(2))}"  # "$5k" == "$5,000"
    if num and nxt == "grand":
        return f"m:${_scaled(_numval(t), 'k')}"  # "5 grand" == "$5,000"
    if num and (nxt in _CURRENCY or nxt in _CURRENCY_CODES):
        neg = "-" if (prev in ("-", "−")) else ""
        return f"m:{nxt}{neg}{_canon_num(t, nxt, postfix=True)}"  # "12 USD", "10€", "500円"
    if num and prev in _CURRENCY_CODES:
        return f"m:{prev}{_canon_num(t, prev)}"  # "EUR 10"
    if num and prev in _CURRENCY and i >= 2 and toks[i - 2] in ("-", "−"):
        return f"m:{prev}-{_canon_num(t)}"  # "-$50"
    if num and (nxt in _PERCENT_WORDS or (nxt == "per" and i + 2 < n and toks[i + 2] == "cent")):
        return f"p:{sign}{_numval(t)}"  # "ten percent" == "10%"
    if num and nxt in ("°", "degrees", "°c", "°f", "deg"):
        scale = toks[i + 2] if nxt == "°" and i + 2 < n and toks[i + 2] in ("c", "f") else ""
        return f"deg:{sign}{_canon_num(t)}{scale}"  # "21°C" != "21°F"
    if num and nxt in _DURATION_UNITS:
        return f"q:{_NUMBER_WORDS.get(t, _canon_num(t))}{_DURATION_UNITS[nxt]}"  # "2 hours", "in 3 days", "two weeks"
    if num and prev in _CURRENCY:
        return f"m:{prev}{_canon_num(t, prev)}"  # "$12", "€90"
    if num and nxt.isalpha() and i + 3 < n and toks[i + 2] == "per" and toks[i + 3] in _RATE_PER and not _id_context(toks, i):
        return f"q:{_canon_num(t)}{_RATE_NOUNS.get(nxt, _stem(nxt))}/{_RATE_PER[toks[i + 3]]}"  # "10 messages per minute"
    if num and nxt in _RATE_NOUNS and i + 3 < n and toks[i + 2] == "/" and toks[i + 3] in _RATE_PER:
        return f"q:{_canon_num(t)}{_RATE_NOUNS[nxt]}/{_RATE_PER[toks[i + 3]]}"  # "100 req/s" != "100 req/min"
    if num and nxt in _MEASURE_UNITS and prev not in _ID_NOUNS:
        return f"q:{_numval(t)}{_MEASURE_UNITS[nxt]}"  # "500 GB", "2.5 kg", "200 ms", "5 km"
    if num and nxt in ("f", "c") and any(w in _TEMP_CUES for w in toks[max(0, i - _CLAUSE_WINDOW) : i + 3]):
        return f"deg:{sign}{_canon_num(t)}{nxt}"  # "72 F"
    if num and nxt in _CJK_COUNTERS:
        return f"c:{_canon_num(t)}{nxt}"  # "12명", "1.5万人"
    count_and = prev == "and" and i >= 3 and _counted_noun(toks[i - 2]) and _NUMBER_TOKEN.match(toks[i - 3]) is not None
    if num and (prev in _COUNT_PREV or count_and) and _counted_noun(nxt) and not _id_context(toks, i) and not _year_like(t):
        return f"c:{_NUMBER_WORDS.get(t, _canon_num(t))}"  # "7 engineers": the count; the noun stays in the core
    if num and nxt == "%":
        return f"p:{sign}{_canon_num(t)}"  # "12%", "-5%"
    rng = _NUMBER_RANGE.match(t)
    if rng and not rng.group(3):
        lo, hi = _canon_num(rng.group(1)), _canon_num(rng.group(2))
        if nxt in _MEASURE_UNITS:
            return f"q:{lo}-{hi}{_MEASURE_UNITS[nxt]}"
        if nxt in _DURATION_UNITS:
            return f"q:{lo}-{hi}{_DURATION_UNITS[nxt]}"
        if nxt in ("°", "degrees"):
            scale = toks[i + 2] if nxt == "°" and i + 2 < n and toks[i + 2] in ("c", "f") else ""
            return f"deg:{lo}-{hi}{scale}"
        if _counted_noun(nxt) and prev in _COUNT_PREV:
            return f"q:{lo}-{hi}{_stem(nxt)}"  # "band is 10-12 people", not "Locker 5-10 belongs"
    if rng:
        cur = (
            _CURRENCY_WORDS.get(nxt)
            or (nxt if (nxt in _CURRENCY or nxt in _CURRENCY_CODES) else "")
            or (prev if (prev in _CURRENCY or prev in _CURRENCY_CODES) else "")
            or _CURRENCY_WORDS.get(prev, "")
        )
        if cur:
            return f"m:{cur}{_canon_num(rng.group(1))}-{_canon_num(rng.group(2))}{rng.group(3) or ''}"  # "10-20 USD", "$10-20"
    if _DAY_RANGE.match(t) and nxt == "%":
        return f"p:{t}"  # "5-10%"
    if _DAY_RANGE.match(t) and (prev in _MONTHS or prev in _MONTH_ABBR or nxt in _MONTHS or nxt in _MONTH_ABBR):
        return "r" + t  # "9-10 Oct"
    product_before = (i > 0 and titled[i - 1] and toks[i - 1].isalpha()) or (
        prev == "to" and i >= 2 and titled[i - 2] and toks[i - 2].isalpha()
    )
    if _is_digits(t) and len(t) <= 3 and product_before and _version_object(toks, i):
        return f"v:{t}"  # "CI uses Node 20", "requires Java 17": a major version named after the product
    if t in _WEEKDAYS or (t in _EQUIV and _EQUIV[t] in _WEEKDAYS and t not in _WEEKDAY_ABBR):
        day = _EQUIV.get(t, t)
        if prev in ("next", "this", "last", "following") or prev in _ORDINAL_WORDS or prev == "other":
            return f"{prev}-{day}"  # "next Tuesday", "first Monday", "every other Tuesday"
        return day
    if t in _WEEKDAY_ABBR:
        return _EQUIV[t] if (_date_context(toks, i, scheduling) or prev in _ABBR_DAY_PREV) else None
    if t in _MONTHS or t in _MONTH_ABBR:
        return _EQUIV.get(t, t) if _date_context(toks, i, scheduling) else None
    if _ISO_DATE.match(t) and "t" not in t:
        y, mth, d = t.split("-")
        return None if _id_context(toks, i) else f"d{int(y)}-{int(mth)}-{int(d)}"
    if _DOT_ISO.match(t):
        y, mth, d = t.split(".")
        return None if _id_context(toks, i) else f"d{int(y)}-{int(mth)}-{int(d)}"
    slash = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", t)
    if slash and not _id_context(toks, i):
        first, second, year = int(slash.group(1)), int(slash.group(2)), int(slash.group(3))
        month, mday = (second, first) if first > 12 else (first, second)  # month-first unless the first part cannot be a month
        if 1 <= month <= 12 and 1 <= mday <= 31:
            return f"d{year}-{month}-{mday}"
    if _ISO_DATE.match(t) or _SLASH_DATE.match(t) or _DOT_DATE.match(t):
        return None if _id_context(toks, i) else "d" + t
    if t in _ZONE_ABBR:
        in_zone = prev == "in" and len(t) >= 3  # "Ilona is in CET"
        return _ZONE_CANON.get(t, t) if (_near_clock(toks, i) or nxt in _ZONE_CONTEXT or _zone_phrase_before(toks, i) or in_zone) else None
    if t in _ZONE_NAMES or t in _ZONE_ABBR:
        if nxt == "time" and i + 3 < n and toks[i + 2] == "to" and toks[i + 3] == "time":
            return None  # "from Arizona time to time": the idiom, not a zone
    if t in _ZONE_NAMES:
        if i + 1 < n and titled[i + 1] and nxt.isalpha() and not _time_word(nxt) and nxt not in _ZONE_CONTEXT:
            return None  # "6am Pacific Surfliner": part of a proper name
        return _ZONE_CANON.get(t, t) if (_zone_name_by_clock(toks, i) or nxt in _ZONE_CONTEXT or _zone_phrase_before(toks, i)) else None
    if t in _PERIODS and prev in _PERIOD_AFTER:
        return prev + "-" + t
    if _is_digits(t) and len(t) == 4 and 1900 <= int(t) <= 2099 and (prev in _YEAR_PREV or prev in _MONTHS or prev in _MONTH_ABBR):
        return "y" + t
    od = _ORDINAL_DAY.match(t)
    month_near = prev in _MONTHS or prev in _MONTH_ABBR or nxt in _MONTHS or nxt in _MONTH_ABBR
    if nxt in ("de", "del") and i + 2 < n:
        month_near = month_near or _plain(toks[i + 2]) in _MONTHS  # "15 de marzo"
    if _is_digits(t) and len(t) <= 2 and nxt in ("to", "through", "until", "till") + _DASHES and i + 3 < n:
        month_near = month_near or (_is_digits(toks[i + 2]) and (toks[i + 3] in _MONTHS or toks[i + 3] in _MONTH_ABBR))
    if (_is_digits(t) or od) and len(t) <= 4 and month_near:
        return "n" + (od.group(1) if od else t)
    if od and prev == "the" and (nxt in ("", "of", "day") or len(nxt) == 1 and not nxt.isalnum()):
        return "n" + od.group(1)
    if t in _ORDINAL_DAY_WORDS and prev == "the" and nxt == "of" and i + 2 < n and (toks[i + 2] in _MONTHS or toks[i + 2] in _MONTH_ABBR):
        return f"n{_ORDINAL_DAY_WORDS[t]}"  # "the ninth of October"
    return None


def _stem(tok: str) -> str:
    """Light plural/3rd-person fold for core comparison ("works" == "work", "deploys" == "deploy")."""
    return tok[:-1] if tok.isalpha() and len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def fact_slots(text: str) -> FactSlots:
    toks, titled = _compare_tokens_cased(text)
    n = len(toks)
    scheduling = any(t in _SCHEDULING for t in toks)  # once per text: keeps the scan linear
    found = [_value_at(toks, i, titled, scheduling) for i in range(n)]
    is_value = [v is not None for v in found]
    for i, v in enumerate(found):
        if v is not None and i > 0 and (toks[i] in _PERIODS or v.startswith(toks[i - 1] + "-")):
            is_value[i - 1] = True  # "next" in "next week" / "next Tuesday" is part of the value
        if v is not None and v.startswith("q:") and i + 1 < n:
            is_value[i + 1] = True  # the unit in "2 hours"
            if "/" in v and i + 3 < n:
                is_value[i + 2] = is_value[i + 3] = True  # "req / s"
        if v is not None and v.startswith("p:") and i + 1 < n:
            is_value[i + 1] = True  # the % sign / "percent"
        if v is not None and v.startswith("deg:") and i + 1 < n:
            is_value[i + 1] = True
            if toks[i + 1] == "°" and i + 2 < n and toks[i + 2] in ("c", "f"):
                is_value[i + 2] = True
        if v is not None and v.startswith("m:"):
            for k in (i - 1, i + 1):
                if 0 <= k < n and toks[k] in _CURRENCY_WORDS:
                    is_value[k] = True
        if v is not None and v.startswith("m:"):
            for k in (i - 2, i - 1, i + 1):
                if 0 <= k < n and (toks[k] in _CURRENCY or toks[k] in _CURRENCY_CODES or toks[k] in ("-", "−")):
                    is_value[k] = True  # currency symbol/code and sign
        if v is not None and (v.startswith("p:-") or v.startswith("deg:-")) and i > 0:
            is_value[i - 1] = True  # the minus sign
        if v is not None and v.startswith("v:") and "-" in v and i + 2 < n:
            is_value[i + 1] = is_value[i + 2] = True  # "-rc1"

    def prev_word(i: int) -> int:
        k = i - 1
        while k >= 0 and (toks[k] in ("the", "a", "an", "on", "at", "in") or toks[k] in _CURRENCY or toks[k] in _CURRENCY_CODES):
            k -= 1
        return k

    moving = any(t in _SCHEDULING or t in _MOVE_WORDS for t in toks)
    values: set[str] = set()
    rejected: list[tuple[int, str]] = []
    for i, v in enumerate(found):
        if v is None:
            continue
        k = prev_word(i)
        if k >= 0 and toks[k] == "from" and moving:
            continue  # "moved from Tuesday to Thursday": Tuesday is the old value ("off from 9 to 10 Oct" is a range)
        if k >= 1 and (toks[k], toks[k - 1]) in (("of", "instead"), ("than", "rather")):
            continue  # "Thursday instead of Tuesday": Tuesday is the old value
        if k >= 0 and (toks[k] in _OLD_VALUE_PREV or (toks[k] == "was" and "(" in toks[max(0, k - 2) : k])):
            continue  # "Thursday (was Tuesday)", "formerly Tuesday": the old value
        if i + 1 < n and toks[i + 1] == "→":
            continue  # "Tue → Thu": the value before the arrow is the old one
        if k >= 0 and toks[k] in _NEGATION:
            rejected.append((k, v))
            continue
        values.add(v)
    skipped_negation: set[int] = set()
    if values:
        # "Thursday, not Tuesday": a contrast names the rejected value; it is not a negated fact.
        skipped_negation = {k for k, _ in rejected}
    else:
        # "The call is not on Tuesday": the negation applies to the fact itself.
        values.update(v for _, v in rejected)
    core: list[str] = []
    negations = 0
    for i, t in enumerate(toks):
        if is_value[i] or t == "→":
            continue
        if t in _NEGATION or (t == "no" and i + 1 < n and toks[i + 1].isalpha()):
            if i not in skipped_negation:
                negations += 1
            continue
        if t == "longer" and i > 0 and toks[i - 1] == "no":
            continue
        if (t == "from" and i + 1 < n and toks[i + 1] == "now") or (t == "on" and i > 0 and toks[i - 1] == "now"):
            continue  # "from now on"
        if t in _CORRECTION:
            continue
        if t in _ANTONYM:
            negations += 1
            core.append(_stem(_ANTONYM[t]))
            continue
        if t in _ZONE_CONTEXT and i > 0 and is_value[i - 1] and (toks[i - 1] in _ZONE_ABBR or toks[i - 1] in _ZONE_NAMES):
            continue  # "Arizona time"
        if t in _ZONE_CONTEXT and any(is_value[k] for k in range(i + 1, min(n, i + 4))):
            continue  # "timezone is Arizona"
        if t in _VALUE_PREPOSITIONS:
            k = i + 1
            while k < n and toks[k] in ("the", "a", "an"):
                k += 1
            if k < n and is_value[k]:
                continue  # "to Thursday", "from Tuesday", "on the 5th"
        if len(t) == 1 and unicodedata.category(t)[0] == "P":
            continue  # punctuation; symbols ($, €, +, <, emoji) stay in the core
        core.append(_stem(t))
    names = tuple(
        t for t, cap in zip(toks, titled) if cap and t.isalpha() and t not in _WEEKDAYS and t not in _MONTHS and t not in _ZONE_NAMES
    )
    return FactSlots(core=tuple(sorted(core)), values=frozenset(_full_dates(values)), negations=negations, names=names)


def _full_dates(values: set[str]) -> set[str]:
    """One month + one day + one year ("1 April 2026") is the same value as the ISO date."""
    months = [v for v in values if v in _MONTHS]
    days = [v for v in values if v.startswith("n") and v[1:].isdigit()]
    years = [v for v in values if v.startswith("y") and v[1:].isdigit()]
    if len(months) == 1 and len(days) == 1 and len(years) == 1:
        iso = f"d{years[0][1:]}-{_MONTH_NAMES.index(months[0]) + 1}-{int(days[0][1:])}"
        return (values - {months[0], days[0], years[0]}) | {iso}
    return values


def _resolve_ambiguous(slots: FactSlots, other: frozenset[str]) -> FactSlots:
    """'t10:00a' (a bare "10:00") becomes 't10:00' or 't22:00' when the other side states one of them."""
    resolved = set()
    for v in slots.values:
        if _CANON_CLOCK.match(v) and v.endswith("a"):
            hour, minute = int(v[1:3]), v[4:6]
            v = next((c for c in (f"t{hour:02d}:{minute}", f"t{hour + 12:02d}:{minute}") if c in other), v)
        resolved.add(v)
    return FactSlots(core=slots.core, values=frozenset(resolved), negations=slots.negations, names=slots.names)


def changed_fact(new: str, old: str) -> bool:
    """True when ``new`` corrects ``old``.

    The cores (everything but when/zone values and correction words) must hold
    the same words, in any order ("Ilona's call" == "call with Ilona"). Then
    either the value sets differ, without one merely adding to the other
    ("Tuesday" -> "Tuesday and Thursday" is a refinement), or the negation flips.
    """
    sn, so = fact_slots(new), fact_slots(old)
    sn, so = _resolve_ambiguous(sn, so.values), _resolve_ambiguous(so, sn.values)
    if not sn.core or sn.core != so.core:
        return False
    names_new, names_old = list(sn.names), list(so.names)
    if len(names_new) >= 2 and sorted(names_new) == sorted(names_old) and names_new != names_old:
        return False  # the same names in swapped roles are two different facts
    if sn.values and so.values and sn.values != so.values:
        if not (sn.values < so.values or so.values < sn.values):
            return True
    return (sn.negations % 2) != (so.negations % 2)
