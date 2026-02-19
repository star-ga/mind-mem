#!/usr/bin/env python3
"""mind-mem Entity & Fact Extractor (Regex NER-lite). Zero external deps.

Decomposes raw conversation observations into atomic fact cards:
- FACT: identity, attribute, state facts
- EVENT: actions, occurrences with optional dates
- PREFERENCE: likes, dislikes, interests, hobbies
- RELATION: interpersonal connections
- NEGATION: things explicitly denied or not done

Each card is independently retrievable by BM25 and links back to its
source observation via source_id. Cards are short (10-20 words) so
BM25 keyword matching is precise — a 15-word fact card scores much
higher than a 200-word conversation block for single-hop queries.

Usage:
    from extractor import extract_facts

    cards = extract_facts(
        text="[Caroline] I went to a LGBTQ support group yesterday",
        speaker="Caroline",
        date="2023-05-07",
        source_id="DIA-D1-3",
    )
    # => [{"type": "EVENT", "content": "Caroline went to a LGBTQ support group",
    #       "date": "2023-05-07", "source_id": "DIA-D1-3", "confidence": 0.8}]
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Fact extraction patterns
# ---------------------------------------------------------------------------

# Identity: "I am/I'm a/an X", "I identify as X", "as a X", "journey as a X"
_IDENTITY_RE = re.compile(
    r"\b(?:i(?:'m| am| was| have been))\s+(?:a |an )?(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)
# Secondary identity: "as a X woman/man/person/professional"
_IDENTITY_AS_RE = re.compile(
    r"\bas\s+(?:a |an )?(\w+(?:\s+\w+)?(?:\s+(?:woman|man|person|professional|"
    r"artist|engineer|student|mother|father|parent)))\b",
    re.IGNORECASE,
)

# First-word filter for identity matches (emotional states, auxiliaries, generics)
_IDENTITY_FILTER_WORDS = frozenset({
    "so", "too", "also", "just", "here", "there", "ok", "okay", "really",
    "happy", "thankful", "grateful", "excited", "proud", "glad", "thrilled",
    "sad", "sorry", "worried", "nervous", "anxious", "scared",
    "lucky", "fortunate", "blessed", "ready", "stoked", "sure", "certain",
    "going", "about", "trying", "hoping", "having", "getting", "giving",
    "putting", "looking", "thinking", "still", "definitely", "totally",
    "not", "off", "to", "doing", "feeling", "being", "working",
})

# Possession/attribute: "my X is Y", "I have X"
_ATTRIBUTE_RE = re.compile(
    r"\bmy\s+(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Preference: "I love/like/enjoy/prefer X", "my favorite X is Y"
_PREFERENCE_RE = re.compile(
    r"\b(?:i\s+(?:love|like|enjoy|prefer|adore|really like|"
    r"am (?:into|fond of|passionate about)))\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)
_FAVORITE_RE = re.compile(
    r"\b(?:my\s+)?(?:favorite|favourite)\s+(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Dislike: "I hate/dislike/don't like X"
_DISLIKE_RE = re.compile(
    r"\b(?:i\s+(?:hate|dislike|(?:don'?t|do not)\s+(?:like|enjoy|care for)))\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Event: "I went/visited/attended/started/moved/ran/did X"
_EVENT_RE = re.compile(
    r"\bi\s+(went to|visited|attended|started|moved to|ran|"
    r"finished|completed|joined|signed up for|enrolled in|"
    r"graduated from|bought|got|received|found|made|created|"
    r"launched|published|wrote|painted|cooked|built|organized|"
    r"hosted|planned|celebrated|traveled to|drove to|flew to|"
    r"ran a|participated in|volunteered at|worked on|worked at|"
    r"took a|took up|picked up|tried|began|discovered|adopted|"
    r"researched|explored|donated to|read|watched|saw|heard|"
    r"applied to|passed|contacted|chose|collected|"
    r"performed|sang|played|composed|recorded|practiced)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Update/change events: "I changed/switched/stopped/quit/resumed X"
_UPDATE_RE = re.compile(
    r"\bi\s+(changed|switched to|switched from|stopped|quit|gave up|"
    r"resumed|restarted|went back to|returned to|moved from|"
    r"upgraded to|downgraded|replaced|swapped|transitioned to|"
    r"decided to|decided against|changed my mind about|"
    r"no longer|dropped|cancelled|canceled|postponed)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Gerund event: "Researching X", "Painting X", "Working on X" (at sentence start)
_GERUND_RE = re.compile(
    r"^(researching|painting|working on|studying|learning|practicing|"
    r"building|creating|exploring|organizing|planning|collecting|"
    r"training for|preparing for|volunteering at)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Relation: "X is my Y", "I met X", "I work with X"
_RELATION_RE = re.compile(
    r"\b(\w+)\s+is\s+my\s+(\w+(?:\s+\w+)?)\b",
    re.IGNORECASE,
)
_MET_RE = re.compile(
    r"\bi\s+(?:met|know|work with|live with|dated|married)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Negation: "I never/didn't/haven't/don't X"
_NEGATION_RE = re.compile(
    r"\bi\s+(?:never|didn'?t|haven'?t|don'?t|do not|did not|have not|"
    r"am not|was not|wasn'?t|can'?t|cannot|won'?t|will not)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Plan/intention: "I want to/plan to/hope to/will X"
_PLAN_RE = re.compile(
    r"\bi\s+(?:want to|plan to|hope to|intend to|am going to|will|"
    r"am planning to|am hoping to|dream of|aspire to|aim to)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# State/feeling: "I feel/am feeling/am X (adjective)"
_STATE_RE = re.compile(
    r"\bi(?:'m| am)\s+(?:feeling\s+)?(\w+)\b.*?(?:about|because|since|that)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# --- D.3 patterns: Possessives, habitual preferences, third-person references ---

# Possessive relation: "Tim's brother", "my wife's doctor", "her sister"
_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][a-z]+)'s\s+(\w+(?:\s+\w+)?)\b",
)

# Habitual preference: "I usually/often/always X", "I prefer X"
_HABITUAL_RE = re.compile(
    r"\bi\s+(?:usually|often|always|typically|generally|normally|prefer to|tend to)\s+(.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Third-person fact: "He/She is a X", "He/She works at X" (uses last speaker)
_THIRD_PERSON_RE = re.compile(
    r"\b(?:he|she)\s+(is\s+(?:a |an )?.+?|works?\s+(?:at|for|as)\s+.+?"
    r"|lives?\s+(?:in|at|near)\s+.+?|plays?\s+.+?|loves?\s+.+?|likes?\s+.+?)(?:\.|,|!|\?|$)",
    re.IGNORECASE,
)

# Date extraction from surrounding text
_DATE_MENTION_RE = re.compile(
    r"\b(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}|"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2},?\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2}|"
    r"yesterday|today|last\s+(?:week|month|year|sunday|monday|tuesday|"
    r"wednesday|thursday|friday|saturday)|"
    r"this\s+(?:morning|afternoon|evening|weekend)|"
    r"(?:a\s+)?(?:few\s+)?(?:days?|weeks?|months?|years?)\s+ago)\b",
    re.IGNORECASE,
)


def _clean_content(text: str) -> str:
    """Clean extracted content: strip, truncate, remove trailing fragments."""
    text = text.strip()
    # Remove trailing conjunctions/prepositions
    text = re.sub(r"\s+(?:and|but|or|so|because|since|which|that|who)$", "", text, flags=re.IGNORECASE)
    # Truncate to reasonable length
    if len(text) > 120:
        text = text[:120].rsplit(" ", 1)[0]
    return text


_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

# D.3: Temporal normalization — "March 2023" → "2023-03", "October 15, 2023" → "2023-10"
_MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"(?:\s+\d{1,2},?)?\s+(\d{4})\b",
    re.IGNORECASE,
)
_YEAR_MONTH_RE = re.compile(
    r"\b(\d{4})\s*[-/]\s*(\d{1,2})\b",
)


def _extract_date_from_text(text: str) -> Optional[str]:
    """Try to find a date mention in the text. Normalizes to YYYY-MM when possible."""
    # Try YYYY-MM-DD first
    m = _YEAR_MONTH_RE.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}"

    # Try "Month YYYY" pattern
    m = _MONTH_YEAR_RE.search(text)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower(), "")
        year = m.group(2)
        if month:
            return f"{year}-{month}"

    # Fallback to raw date mention
    m = _DATE_MENTION_RE.search(text)
    return m.group(0) if m else None


def extract_facts(
    text: str,
    speaker: str = "",
    date: str = "",
    source_id: str = "",
) -> List[Dict]:
    """Extract atomic fact cards from a single observation/turn.

    Args:
        text: Raw text (may include "[speaker] ..." prefix).
        speaker: Speaker name (if known from metadata).
        date: Observation date string.
        source_id: Source block ID for linking (e.g., "DIA-D1-3").

    Returns list of fact card dicts with keys:
        type, content, speaker, date, source_id, confidence
    """
    # Strip speaker prefix if present: "[Caroline] text..." -> text
    speaker_match = re.match(r"^\[([^\]]+)\]\s*", text)
    if speaker_match:
        if not speaker:
            speaker = speaker_match.group(1)
        text = text[speaker_match.end():]

    cards = []
    prefix = f"{speaker} " if speaker else ""

    # Extract date mentioned in text
    mentioned_date = _extract_date_from_text(text)

    # --- Identity facts ---
    for m in _IDENTITY_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 2:
            # Filter out emotional states, auxiliaries, and generics
            first_word = content.lower().split()[0] if content else ""
            if first_word not in _IDENTITY_FILTER_WORDS and len(content.split()) >= 2:
                cards.append({
                    "type": "FACT",
                    "content": f"{prefix}is {content}",
                    "speaker": speaker,
                    "date": date,
                    "source_id": source_id,
                    "confidence": 0.85,
                })

    # Secondary identity: "as a transgender woman", "as a counselor"
    for m in _IDENTITY_AS_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 2:
            cards.append({
                "type": "FACT",
                "content": f"{prefix}is a {content}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.9,
            })

    # --- Attribute facts ---
    for m in _ATTRIBUTE_RE.finditer(text):
        attr = _clean_content(m.group(1))
        val = _clean_content(m.group(2))
        if attr and val and len(val) > 1:
            cards.append({
                "type": "FACT",
                "content": f"{prefix}{attr} is {val}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.8,
            })

    # --- Preferences ---
    for m in _PREFERENCE_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 2:
            cards.append({
                "type": "PREFERENCE",
                "content": f"{prefix}likes {content}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.8,
            })

    for m in _FAVORITE_RE.finditer(text):
        category = _clean_content(m.group(1))
        value = _clean_content(m.group(2))
        if category and value:
            cards.append({
                "type": "PREFERENCE",
                "content": f"{prefix}favorite {category} is {value}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.9,
            })

    # --- Dislikes ---
    for m in _DISLIKE_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 2:
            cards.append({
                "type": "PREFERENCE",
                "content": f"{prefix}dislikes {content}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.8,
            })

    # --- Events ---
    for m in _EVENT_RE.finditer(text):
        verb = m.group(1).strip()
        obj = _clean_content(m.group(2))
        if obj and len(obj) > 2:
            # Truncate at first conjunction/dash for cleaner cards
            obj = re.split(r"\s+[-–—]\s+|\s+and\s+(?:it|I|he|she|they|we)\b", obj, maxsplit=1)[0].strip()
            event_date = mentioned_date or date
            cards.append({
                "type": "EVENT",
                "content": f"{prefix}{verb} {obj}",
                "speaker": speaker,
                "date": event_date,
                "source_id": source_id,
                "confidence": 0.8,
            })

    # --- Gerund events: "Researching adoption agencies" ---
    for m in _GERUND_RE.finditer(text):
        verb = m.group(1).strip()
        obj = _clean_content(m.group(2))
        if obj and len(obj) > 2:
            obj = re.split(r"\s+[-–—]\s+|\s+and\s+(?:it|I|he|she|they|we)\b", obj, maxsplit=1)[0].strip()
            cards.append({
                "type": "EVENT",
                "content": f"{prefix}{verb} {obj}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.75,
            })

    # --- Relations ---
    for m in _RELATION_RE.finditer(text):
        person = m.group(1).strip()
        relation = _clean_content(m.group(2))
        if person and relation and person.lower() not in ("it", "this", "that", "there"):
            cards.append({
                "type": "RELATION",
                "content": f"{person} is {prefix}{relation}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.75,
            })

    for m in _MET_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 1:
            cards.append({
                "type": "RELATION",
                "content": f"{prefix}met {content}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.7,
            })

    # --- Negations ---
    for m in _NEGATION_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 2:
            cards.append({
                "type": "NEGATION",
                "content": f"{prefix}never {content}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.85,
            })

    # --- Plans/intentions ---
    for m in _PLAN_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 2:
            cards.append({
                "type": "PLAN",
                "content": f"{prefix}plans to {content}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.7,
            })

    # --- Update/change events ---
    for m in _UPDATE_RE.finditer(text):
        verb = m.group(1).strip()
        obj = _clean_content(m.group(2))
        if obj and len(obj) > 2:
            obj = re.split(r"\s+[-–—]\s+|\s+and\s+(?:it|I|he|she|they|we)\b", obj, maxsplit=1)[0].strip()
            event_date = mentioned_date or date
            cards.append({
                "type": "EVENT",
                "content": f"{prefix}{verb} {obj}",
                "speaker": speaker,
                "date": event_date,
                "source_id": source_id,
                "confidence": 0.8,
            })

    # --- Activity lists: "running, reading, or playing my violin" ---
    # Match comma/or/and-separated gerund phrases (handles ", or" and ", and")
    list_match = re.search(
        r"((?:\w+ing(?:\s+\w+){0,3}(?:,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)){2,}\w+ing(?:\s+\w+){0,3})",
        text, re.IGNORECASE,
    )
    if list_match:
        raw = list_match.group(1)
        activities = re.split(r",\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+", raw)
        for activity in activities:
            activity = activity.strip().rstrip(".")
            if activity and len(activity) > 2 and re.match(r"\w+ing", activity):
                cards.append({
                    "type": "PREFERENCE",
                    "content": f"{prefix}enjoys {activity}",
                    "speaker": speaker,
                    "date": date,
                    "source_id": source_id,
                    "confidence": 0.7,
                })

    # --- D.3: Possessive relations ("Tim's brother", "John's doctor") ---
    for m in _POSSESSIVE_RE.finditer(text):
        owner = m.group(1)
        thing = _clean_content(m.group(2))
        if thing and len(thing) > 1 and owner.lower() not in ("i", "it", "this", "that"):
            # Relation-like: "brother", "sister", "wife", "doctor", "friend"
            relation_words = {"brother", "sister", "wife", "husband", "mother", "father",
                              "mom", "dad", "son", "daughter", "friend", "boyfriend",
                              "girlfriend", "partner", "doctor", "boss", "coach",
                              "teacher", "neighbor", "roommate", "uncle", "aunt",
                              "cousin", "grandma", "grandpa", "grandmother", "grandfather"}
            first_word = thing.split()[0].lower()
            if first_word in relation_words:
                cards.append({
                    "type": "RELATION",
                    "content": f"{prefix}{owner}'s {thing}",
                    "speaker": speaker,
                    "date": date,
                    "source_id": source_id,
                    "confidence": 0.75,
                })
            else:
                # Possessive fact: "Tim's car", "John's injury"
                cards.append({
                    "type": "FACT",
                    "content": f"{prefix}{owner}'s {thing}",
                    "speaker": speaker,
                    "date": date,
                    "source_id": source_id,
                    "confidence": 0.7,
                })

    # --- D.3: Habitual preferences ("I usually...", "I often...", "I prefer to...") ---
    for m in _HABITUAL_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 2:
            cards.append({
                "type": "PREFERENCE",
                "content": f"{prefix}usually {content}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.75,
            })

    # --- D.3: Third-person references ("He is a doctor", "She works at Google") ---
    # Only extract if we have a speaker context (from previous turn)
    for m in _THIRD_PERSON_RE.finditer(text):
        content = _clean_content(m.group(1))
        if content and len(content) > 2:
            # These get speaker from context (coreference resolved in extract_from_conversation)
            cards.append({
                "type": "FACT",
                "content": f"{prefix}{content}",
                "speaker": speaker,
                "date": date,
                "source_id": source_id,
                "confidence": 0.65,
            })

    # Deduplicate by content (keep highest confidence)
    seen = {}
    for card in cards:
        key = card["content"].lower()
        if key not in seen or card["confidence"] > seen[key]["confidence"]:
            seen[key] = card

    return list(seen.values())


def format_as_blocks(
    cards: List[Dict],
    id_prefix: str = "FACT",
    counter_start: int = 1,
) -> str:
    """Format extracted fact cards as mind-mem blocks for DECISIONS.md.

    Returns markdown text with one block per card:
        [FACT-001]
        Statement: Caroline went to a LGBTQ support group
        Date: 2023-05-07
        Status: active
        Tags: EVENT, Caroline
        Sources: DIA-D1-3
    """
    lines = []
    counter = counter_start

    for card in cards:
        block_id = f"{id_prefix}-{counter:03d}"
        content = card["content"]
        card_type = card.get("type", "FACT")
        date = card.get("date", "")
        source = card.get("source_id", "")
        speaker = card.get("speaker", "")

        # Semantic labels: BM25 vocabulary bridge for generic queries.
        # Embedded in Statement (3.0x weight) instead of Context (0.5x)
        # so "What is Caroline's identity?" strongly matches fact cards.
        _SEM_LABELS = {
            "FACT": "identity description who is",
            "EVENT": "activity event did went action",
            "PREFERENCE": "preference interest hobby likes enjoys",
            "RELATION": "relationship connection knows",
            "NEGATION": "never not does not",
            "PLAN": "goal intention plan wants future",
        }
        sem = _SEM_LABELS.get(card_type, "")
        prefix = f"({sem}) " if sem else ""

        # Idempotency guard: don't double-prepend if content already has a label
        if prefix and content.startswith("(") and ") " in content[:60]:
            prefix = ""

        lines.append(f"[{block_id}]")
        lines.append(f"Statement: {prefix}{content}")
        if date:
            lines.append(f"Date: {date}")
        lines.append("Status: active")

        # Carry DiaID so fact cards map back to evidence in evaluation
        dia_id = card.get("dia_id", "")
        if dia_id:
            lines.append(f"DiaID: {dia_id}")

        tags = [card_type]
        if speaker:
            tags.append(speaker)
        lines.append(f"Tags: {', '.join(tags)}")

        if source:
            lines.append(f"Sources: {source}")

        lines.append("")
        counter += 1

    return "\n".join(lines)


def extract_from_conversation(
    turns: list,
    speaker_a: str = "",
    speaker_b: str = "",
    session_date: str = "",
) -> List[Dict]:
    """Extract facts from a list of LoCoMo conversation turns.

    Args:
        turns: List of {"speaker": ..., "dia_id": ..., "text": ...}
        speaker_a: Name of speaker A.
        speaker_b: Name of speaker B.
        session_date: Date string for this session.

    Returns all extracted fact cards across all turns.
    """
    speaker_map = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
    }
    other_speaker = {"speaker_a": speaker_b, "speaker_b": speaker_a}
    all_cards = []

    for i, turn in enumerate(turns):
        speaker_key = turn.get("speaker", "")
        speaker_name = speaker_map.get(speaker_key, speaker_key)
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        source_id = f"DIA-{dia_id.replace(':', '-')}" if dia_id else ""

        cards = extract_facts(
            text=text,
            speaker=speaker_name,
            date=session_date,
            source_id=source_id,
        )

        # D.3: Coreference-lite — if the original text uses "he/she" at key
        # positions, the extracted card wrongly attributes to the current speaker.
        # Fix by swapping the speaker prefix to the other speaker's name.
        coref_name = other_speaker.get(speaker_key, "")
        coref_pat = r"\b(?:he|she)\s+(?:is|was|has|had|does|did|went|works?|lives?|plays?|loves?|likes?)\b"
        if coref_name and re.search(coref_pat, text, re.IGNORECASE):
            sp_prefix = f"{speaker_name} "
            coref_prefix = f"{coref_name} "
            for card in cards:
                content = card.get("content", "")
                # Only swap if card content starts with current speaker prefix
                # and the original text had a third-person reference
                if content.startswith(sp_prefix) and card.get("confidence", 1) <= 0.65:
                    card["content"] = coref_prefix + content[len(sp_prefix):]

        # Carry original dia_id so evaluator can map fact cards to evidence
        for card in cards:
            card["dia_id"] = dia_id
        all_cards.extend(cards)

    return all_cards
