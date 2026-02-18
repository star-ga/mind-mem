#!/usr/bin/env python3
"""mind-mem Recall Engine (BM25 + TF-IDF + Graph + Stemming). Zero external deps.

Default recall backend: BM25 scoring with Porter stemming, stopword filtering,
query expansion, field boosts, recency weighting, and optional graph-based
neighbor boosting via cross-reference traversal.

For semantic recall (embeddings), see RecallBackend interface below.
Optional vector backends (Qdrant/Pinecone) can be plugged in via config.

Usage:
    python3 scripts/recall.py --query "authentication" --workspace "."
    python3 scripts/recall.py --query "auth" --workspace "." --json --limit 5
    python3 scripts/recall.py --query "deadline" --active-only
    python3 scripts/recall.py --query "database" --graph --workspace .
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_parser import parse_file, get_active
from observability import get_logger, metrics

_log = get_logger("recall")


# ---------------------------------------------------------------------------
# RecallBackend interface — plug in vector/semantic backends here
# ---------------------------------------------------------------------------

class RecallBackend(ABC):
    """Interface for recall backends. Default: BM25Backend (below).

    To add a vector backend:
    1. Implement this interface in recall_vector.py
    2. Set recall.backend = "vector" in mind-mem.json
    3. recall.py will load it dynamically, falling back to BM25 on error.
    """

    @abstractmethod
    def search(self, workspace, query, limit=10, active_only=False):
        """Return list of {_id, type, score, excerpt, file, line, status}."""
        ...

    @abstractmethod
    def index(self, workspace):
        """(Re)build index from workspace files."""
        ...


# Fields to index for search (in priority order)
SEARCH_FIELDS = [
    "Statement", "Title", "Summary", "Description", "Context",
    "Rationale", "Tags", "Keywords", "Name", "Purpose",
    "RootCause", "Fix", "Prevention", "ProposedFix",
    "Sources",  # Provenance links emitted by fact extractor (e.g. "DIA-D1-3")
]

# Fields from ConstraintSignatures
SIG_FIELDS = ["subject", "predicate", "object", "domain"]

# Files to scan
CORPUS_FILES = {
    "decisions": "decisions/DECISIONS.md",
    "tasks": "tasks/TASKS.md",
    "projects": "entities/projects.md",
    "people": "entities/people.md",
    "tools": "entities/tools.md",
    "incidents": "entities/incidents.md",
    "contradictions": "intelligence/CONTRADICTIONS.md",
    "drift": "intelligence/DRIFT.md",
    "signals": "intelligence/SIGNALS.md",
}

# BM25 parameters
BM25_K1 = 1.2   # Term frequency saturation
BM25_B = 0.75   # Document length normalization


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "and", "but", "or",
    "nor", "not", "so", "yet", "for", "of", "to", "in", "on", "at", "by",
    "with", "from", "as", "into", "about", "between", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "while", "that", "this",
    "it", "its", "we", "they", "them", "their", "he", "she", "his", "her",
})


# ---------------------------------------------------------------------------
# Irregular verb lemmatization — maps irregular past tenses to base form
# ---------------------------------------------------------------------------

_IRREGULAR_LEMMA = {
    # go
    "went": "go", "gone": "go",
    # get
    "got": "get", "gotten": "get",
    # meet
    "met": "meet",
    # eat
    "ate": "eat", "eaten": "eat",
    # give
    "gave": "give", "given": "give",
    # take
    "took": "take", "taken": "take",
    # come
    "came": "come",
    # see
    "saw": "see", "seen": "see",
    # know
    "knew": "know", "known": "know",
    # think
    "thought": "think",
    # buy
    "bought": "buy",
    # bring
    "brought": "bring",
    # drive
    "drove": "drive", "driven": "drive",
    # write
    "wrote": "write", "written": "write",
    # break
    "broke": "break", "broken": "break",
    # speak
    "spoke": "speak", "spoken": "speak",
    # sing
    "sang": "sing", "sung": "sing",
    # run
    "ran": "run",
    # begin
    "began": "begin", "begun": "begin",
    # fall
    "fell": "fall", "fallen": "fall",
    # tell
    "told": "tell",
    # find
    "found": "find",
    # make
    "made": "make",
    # say
    "said": "say",
    # lose
    "lost": "lose",
    # win
    "won": "win",
    # sit
    "sat": "sit",
    # stand
    "stood": "stand",
    # leave
    "left": "leave",
    # build
    "built": "build",
    # send
    "sent": "send",
    # spend
    "spent": "spend",
    # catch
    "caught": "catch",
    # teach
    "taught": "teach",
    # feel
    "felt": "feel",
    # keep
    "kept": "keep",
    # sleep
    "slept": "sleep",
    # pay
    "paid": "pay",
    # lend
    "lent": "lend",
    # choose
    "chose": "choose", "chosen": "choose",
    # grow
    "grew": "grow", "grown": "grow",
    # throw
    "threw": "throw", "thrown": "throw",
    # fly
    "flew": "fly", "flown": "fly",
    # draw
    "drew": "draw", "drawn": "draw",
    # wear
    "wore": "wear", "worn": "wear",
    # ride
    "rode": "ride", "ridden": "ride",
    # hide
    "hid": "hide", "hidden": "hide",
    # hang
    "hung": "hang",
    # hold
    "held": "hold",
    # lead
    "led": "lead",
    # fight
    "fought": "fight",
    # sell
    "sold": "sell",
    # swim
    "swam": "swim", "swum": "swim",
    # drink
    "drank": "drink", "drunk": "drink",
}


# ---------------------------------------------------------------------------
# Porter Stemmer (simplified, zero-dependency)
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    """Simplified Porter stemmer — handles common English suffixes.

    Not a full Porter implementation, but covers the most impactful rules
    for recall quality: -ing, -ed, -tion, -ies, -ment, -ness, -ous, -ize.
    Also normalizes irregular past tenses via lemma table.
    """
    # Step 0: irregular verb lemmatization
    word = _IRREGULAR_LEMMA.get(word, word)

    if len(word) <= 3:
        return word

    # Step 1: Plurals and past participles
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ness"):
        word = word[:-4]
    elif word.endswith("ment") and len(word) > 5:
        word = word[:-4]
    elif word.endswith("tion"):
        word = word[:-4] + "t"
    elif word.endswith("sion"):
        word = word[:-4] + "s"
    elif word.endswith("ized"):
        word = word[:-1]
    elif word.endswith("izing"):
        word = word[:-3] + "e"
    elif word.endswith("ize"):
        word = word  # keep as-is
    elif word.endswith("ating"):
        word = word[:-3] + "e"
    elif word.endswith("ation"):
        word = word[:-5] + "ate"
    elif word.endswith("ously"):
        word = word[:-5] + "ous"
    elif word.endswith("ous") and len(word) > 5:
        word = word  # keep as-is
    elif word.endswith("ful"):
        word = word[:-3]
    elif word.endswith("ally"):
        word = word[:-4] + "al"
    elif word.endswith("ably"):
        word = word[:-4] + "able"
    elif word.endswith("ibly"):
        word = word[:-4] + "ible"
    elif word.endswith("able") and len(word) > 5:
        word = word[:-4]
    elif word.endswith("ible") and len(word) > 5:
        word = word[:-4]
    elif word.endswith("ing") and len(word) > 4:
        word = word[:-3]
        # Restore trailing 'e': computing -> comput -> compute
        if word.endswith(("at", "iz", "bl")):
            word += "e"
    elif word.endswith("ated") and len(word) > 5:
        word = word[:-1]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if word.endswith(("at", "iz", "bl")):
            word += "e"
    elif word.endswith("ly") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("er") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("est") and len(word) > 4:
        word = word[:-3]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]

    return word


def tokenize(text: str) -> list[str]:
    """Split text into lowercase stemmed tokens, filtering stopwords."""
    return [_stem(t) for t in re.findall(r"[a-z0-9_]+", text.lower())
            if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Query Expansion — domain-aware synonyms
# ---------------------------------------------------------------------------

_QUERY_EXPANSIONS = {
    # Auth & identity
    "auth": ["authentication", "login", "oauth", "jwt", "session"],
    "authentication": ["auth", "login", "oauth", "jwt"],
    "login": ["auth", "signin", "authentication", "password"],
    "password": ["credential", "login", "secret", "auth"],

    # Data layer
    "db": ["database", "postgresql", "mysql", "sqlite", "sql"],
    "database": ["db", "postgresql", "mysql", "sqlite", "sql"],
    "sql": ["database", "query", "postgresql", "mysql"],
    "cache": ["redis", "memcached", "caching", "ttl"],

    # API & networking
    "api": ["endpoint", "rest", "graphql", "route", "handler"],
    "endpoint": ["api", "route", "url", "path", "handler"],
    "request": ["http", "api", "call", "fetch"],

    # DevOps & deployment
    "deploy": ["deployment", "ci", "cd", "pipeline", "release"],
    "deployment": ["deploy", "ci", "cd", "pipeline", "release"],
    "docker": ["container", "image", "kubernetes", "pod"],
    "kubernetes": ["k8s", "pod", "container", "docker", "cluster"],

    # Quality & issues
    "bug": ["error", "issue", "defect", "fix", "regression"],
    "error": ["bug", "exception", "failure", "crash"],
    "fix": ["bug", "patch", "repair", "resolve"],
    "test": ["testing", "pytest", "unittest", "spec", "coverage"],
    "testing": ["test", "pytest", "assertion", "mock"],

    # Security
    "security": ["vulnerability", "auth", "encryption", "xss", "injection"],
    "vulnerability": ["security", "cve", "exploit", "risk"],

    # Performance
    "perf": ["performance", "latency", "throughput", "optimization"],
    "performance": ["perf", "latency", "throughput", "optimization", "speed"],
    "slow": ["performance", "latency", "bottleneck", "optimization"],
    "fast": ["performance", "speed", "quick", "optimization"],

    # Configuration & infrastructure
    "config": ["configuration", "settings", "env", "environment"],
    "infra": ["infrastructure", "server", "cloud", "devops"],
    "server": ["backend", "service", "host", "instance"],

    # Frontend & UI
    "ui": ["interface", "frontend", "component", "view", "page"],
    "frontend": ["ui", "client", "browser", "react", "component"],
    "css": ["style", "stylesheet", "tailwind", "design"],
    "component": ["widget", "element", "ui", "module"],

    # Common conversational terms (helps bridge LoCoMo's casual language)
    "talk": ["discuss", "mention", "conversation", "chat", "say"],
    "discuss": ["talk", "mention", "conversation", "said"],
    "mention": ["talk", "discuss", "said", "refer"],
    "said": ["mention", "talk", "told", "stated"],
    "told": ["said", "mention", "informed", "stated"],
    "want": ["wish", "prefer", "desire", "plan"],
    "like": ["prefer", "enjoy", "love", "favorite"],
    "prefer": ["like", "want", "favorite", "choose"],
    "plan": ["intend", "schedule", "goal", "strategy"],
    "work": ["job", "career", "employ", "occupation"],
    "job": ["work", "career", "employ", "position"],
    "live": ["reside", "home", "house", "location", "stay"],
    "home": ["live", "house", "reside", "apartment"],
    "movie": ["film", "cinema", "watch", "show"],
    "film": ["movie", "cinema", "watch"],
    "book": ["read", "novel", "author", "story"],
    "food": ["eat", "restaurant", "cuisine", "meal", "cook"],
    "eat": ["food", "restaurant", "meal", "dinner", "lunch"],
    "travel": ["trip", "visit", "vacation", "journey"],
    "trip": ["travel", "visit", "vacation", "journey"],

    # Personal life & relationships
    "hobby": ["enjoy", "interest", "activity", "pastime", "practice"],
    "hobbi": ["enjoy", "interest", "activity", "pastime"],  # stemmed form
    "activity": ["hobby", "sport", "exercise", "practice", "interest"],
    "partner": ["girlfriend", "boyfriend", "wife", "husband", "spouse"],
    "spouse": ["wife", "husband", "partner", "married"],
    "married": ["wedding", "marriage", "engaged", "wife", "husband"],
    "wedding": ["married", "marriage", "bride", "ceremony"],
    "mom": ["mother", "mama", "parent"],
    "mother": ["mom", "mama", "parent"],
    "dad": ["father", "papa", "parent"],
    "father": ["dad", "papa", "parent"],
    "sibling": ["brother", "sister"],
    "parent": ["mother", "father", "mom", "dad"],
    "family": ["mother", "father", "brother", "sister", "parent"],
    "child": ["son", "daughter", "kid", "baby"],
    "children": ["son", "daughter", "kid", "baby"],

    # Food & cooking
    "meal": ["food", "cook", "dinner", "lunch", "breakfast", "recipe"],
    "snack": ["food", "candy", "chips", "munch", "junk"],
    "cook": ["recipe", "food", "meal", "kitchen", "bake", "prepare"],
    "bake": ["cook", "recipe", "cookie", "cake", "bread"],
    "cookie": ["bake", "chocolate", "dessert"],
    "restaurant": ["food", "dine", "dining"],
    "diet": ["food", "healthy", "meal", "nutrition"],
    "healthy": ["diet", "nutrition", "exercise", "wellness"],

    # Travel & places
    "country": ["nation", "abroad", "overseas", "visit"],
    "abroad": ["overseas", "country", "foreign", "international"],
    "visit": ["go", "travel", "trip", "tour"],
    "vacation": ["trip", "travel", "holiday", "getaway"],
    "roadtrip": ["drive", "trip", "travel", "car"],
    "city": ["town", "place", "location", "area"],
    "state": ["region", "province", "area"],

    # Health & body
    "injury": ["hurt", "pain", "twist", "broken", "sprain"],
    "injuri": ["hurt", "pain", "twist", "broken", "sprain"],  # stemmed
    "hurt": ["injury", "pain", "injure", "ache"],
    "doctor": ["physician", "medical", "appointment", "health"],
    "appointment": ["doctor", "visit", "medical", "schedule"],
    "health": ["medical", "doctor", "wellness", "condition"],
    "digestive": ["stomach", "gastric", "gut", "intestinal"],
    "digest": ["stomach", "gastric", "gut", "intestinal"],  # stemmed

    # Objects & gifts
    "gift": ["present", "receive", "gave", "surprise"],
    "present": ["gift", "receive", "gave"],

    # Actions & suggestions
    "suggest": ["recommend", "propose", "advise", "idea"],
    "recommend": ["suggest", "propose", "advise", "try"],
    "resume": ["restart", "continue", "begin", "start"],

    # Entertainment
    "game": ["play", "gaming", "video"],
    "novel": ["book", "read", "story", "fiction"],
    "outdoor": ["outside", "nature", "hiking", "camping"],
    "exercise": ["workout", "fitness", "gym", "training", "sport"],

    # Work & education
    "internship": ["intern", "company", "firm"],
    "intern": ["internship", "company", "firm"],
    "project": ["plan", "develop", "build"],
    "career": ["job", "profession", "occupation"],
    "occupation": ["job", "career", "profession"],
    "profession": ["job", "career", "occupation"],
}


_MONTH_NUM = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12",
}

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def expand_months(query: str, tokens: list[str]) -> list[str]:
    """Add numeric month tokens when query mentions month names.

    "March 2023" → adds "03", "2023-03" to token list.
    Enables BM25 matching against date-normalized blocks (e.g. "Date: 2023-03").
    """
    query_lower = query.lower()
    augmented = list(tokens)
    seen = set(tokens)

    months_found = []
    for name, num in _MONTH_NUM.items():
        if name in query_lower:
            months_found.append(num)
            if num not in seen:
                augmented.append(num)
                seen.add(num)

    years_found = _YEAR_RE.findall(query)
    for year in years_found:
        for month_num in months_found:
            combo = f"{year}-{month_num}"
            if combo not in seen:
                augmented.append(combo)
                seen.add(combo)

    return augmented


# Pre-compute stemmed-form lookup so aggressively stemmed query tokens
# still find their expansion entries (e.g. "partn" → partner's expansions)
_EXPANSION_BY_STEM: dict[str, list[str]] = {}
for _key, _syns in _QUERY_EXPANSIONS.items():
    _sk = _stem(_key)
    if _sk not in _EXPANSION_BY_STEM:
        _EXPANSION_BY_STEM[_sk] = list(_syns)
    else:
        _EXPANSION_BY_STEM[_sk].extend(_syns)
    # Also keep the unstemmed key if different
    if _key != _sk and _key not in _EXPANSION_BY_STEM:
        _EXPANSION_BY_STEM[_key] = list(_syns)


def expand_query(tokens: list[str], max_expansions: int = 8,
                  mode: str = "full") -> list[str]:
    """Expand query tokens with domain synonyms. Returns expanded token list.

    mode="full": apply all semantic synonyms (default).
    mode="morph_only": skip semantic synonym expansion entirely.
        Lemma normalization and month expansion happen BEFORE this function,
        so morph_only still benefits from those normalizations.
    """
    if mode == "morph_only":
        return list(tokens)
    expanded = list(tokens)
    added = set(tokens)
    for token in tokens:
        # Look up in both the original dict and the stemmed-key index
        syns = _QUERY_EXPANSIONS.get(token) or _EXPANSION_BY_STEM.get(token) or []
        for synonym in syns:
            stemmed = _stem(synonym)
            if stemmed not in added and len(expanded) < len(tokens) + max_expansions:
                expanded.append(stemmed)
                added.add(stemmed)
    return expanded


def _rm3_language_model(doc_tokens, collection_freq, total_tokens, mu=2000.0):
    """JM-smoothed document language model.
    P(w|D) = (tf(w,D) + mu * P(w|C)) / (|D| + mu)
    where P(w|C) = cf(w) / total_tokens
    """
    tf = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    probs = {}
    vocab = set(doc_tokens) | set(collection_freq.keys())
    for w in vocab:
        p_collection = collection_freq.get(w, 0) / max(total_tokens, 1)
        p_doc = (tf.get(w, 0) + mu * p_collection) / (doc_len + mu)
        probs[w] = p_doc
    return probs


def rm3_expand(query_tokens, top_docs, collection_freq, total_tokens,
               alpha=0.6, fb_terms=10, fb_docs=5, min_idf=1.0,
               doc_freq=None, N=1):
    """RM3 pseudo-relevance feedback query expansion.

    Estimates P(w|R) from top-K docs using JM-smoothed language models.
    Selects top fb_terms expansion terms by P(w|R) not in original query.
    Interpolates: P'(w|Q) = alpha * P(w|Q) + (1-alpha) * P(w|R)

    Args:
        query_tokens: list of stemmed query tokens
        top_docs: list of (doc_tokens, score) tuples from initial retrieval
        collection_freq: dict mapping token -> collection frequency
        total_tokens: total tokens in collection
        alpha: interpolation weight (1.0 = original query only)
        fb_terms: number of expansion terms to add
        fb_docs: number of feedback documents to use
        min_idf: minimum IDF for expansion terms
        doc_freq: dict mapping token -> document frequency (for IDF filtering)
        N: total number of documents in collection

    Returns:
        dict mapping token -> weight (original tokens + expansion tokens)
    """
    if not top_docs or alpha >= 1.0:
        return {t: 1.0 for t in query_tokens}

    # Use top fb_docs
    feedback = top_docs[:fb_docs]

    # Compute P(w|R) = avg P(w|D) across feedback docs
    relevance_model = Counter()
    for doc_tokens, _score in feedback:
        lm = _rm3_language_model(doc_tokens, collection_freq, total_tokens)
        for w, p in lm.items():
            relevance_model[w] += p

    # Normalize
    total_p = sum(relevance_model.values())
    if total_p > 0:
        for w in relevance_model:
            relevance_model[w] /= total_p

    # Filter: remove original query tokens, apply IDF filter
    query_set = set(query_tokens)
    expansion_candidates = {}
    for w, p in relevance_model.items():
        if w in query_set:
            continue
        # IDF filter
        if doc_freq and N > 0 and min_idf > 0:
            df_val = doc_freq.get(w, 0)
            if df_val > 0:
                idf = math.log(N / df_val)
                if idf < min_idf:
                    continue
        expansion_candidates[w] = p

    # Select top fb_terms
    expansion = sorted(expansion_candidates.items(), key=lambda x: x[1], reverse=True)[:fb_terms]

    # Interpolate: P'(w|Q) = alpha * P(w|Q) + (1-alpha) * P(w|R)
    result = {}
    for t in query_tokens:
        result[t] = alpha * 1.0 + (1 - alpha) * relevance_model.get(t, 0)
    for t, p in expansion:
        result[t] = (1 - alpha) * p

    return result


def extract_text(block: dict) -> str:
    """Extract searchable text from a block."""
    parts = []
    # Include block ID so users can search by ID
    bid = block.get("_id", "")
    if bid:
        parts.append(bid)
    for field in SEARCH_FIELDS:
        val = block.get(field, "")
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            parts.extend(str(v) for v in val)

    # Extract from ConstraintSignatures
    for sig in block.get("ConstraintSignatures", []):
        for sf in SIG_FIELDS:
            val = sig.get(sf, "")
            if isinstance(val, str):
                parts.append(val)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# BM25F — Field-weighted term extraction
# ---------------------------------------------------------------------------

# Weight multipliers per field for BM25F scoring.
# Higher weight = terms in this field contribute more to the score.
FIELD_WEIGHTS = {
    "Statement": 3.0, "Title": 2.5, "Name": 2.0,
    "Summary": 1.5, "Description": 1.2, "Purpose": 1.2,
    "RootCause": 1.2, "Fix": 1.2, "Prevention": 1.0,
    "Tags": 0.8, "Keywords": 0.8,
    "Context": 0.5, "Rationale": 0.5, "ProposedFix": 0.5,
    "History": 0.3,
}


def extract_field_tokens(block: dict) -> dict[str, list[str]]:
    """Extract and tokenize per-field for BM25F scoring.

    Returns {field_name: [stemmed_tokens]} for each non-empty field.
    Also includes _id tokens under a synthetic '_id' key.
    """
    field_tokens = {}

    bid = block.get("_id", "")
    if bid:
        field_tokens["_id"] = tokenize(bid)

    for field in SEARCH_FIELDS:
        val = block.get(field, "")
        if isinstance(val, str) and val:
            tokens = tokenize(val)
            if tokens:
                field_tokens[field] = tokens
        elif isinstance(val, list):
            combined = " ".join(str(v) for v in val)
            tokens = tokenize(combined)
            if tokens:
                field_tokens[field] = tokens

    # ConstraintSignature fields
    sig_parts = []
    for sig in block.get("ConstraintSignatures", []):
        for sf in SIG_FIELDS:
            val = sig.get(sf, "")
            if isinstance(val, str):
                sig_parts.append(val)
    if sig_parts:
        tokens = tokenize(" ".join(sig_parts))
        if tokens:
            field_tokens["_sig"] = tokens

    return field_tokens


# ---------------------------------------------------------------------------
# Bigram phrase matching
# ---------------------------------------------------------------------------

def get_bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Generate set of adjacent token pairs (bigrams)."""
    return set(zip(tokens, tokens[1:]))


# ---------------------------------------------------------------------------
# Overlapping chunk indexing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Skeptical Mode Detection — distractor-prone query identification
# ---------------------------------------------------------------------------

_SKEPTICAL_TRIGGERS_RE = re.compile(
    r"\b(favorite|favourite|least|best|worst|most|one of|"
    r"as mentioned on|mentioned on \w+|according to|"
    r"on \d{1,2}(?:st|nd|rd|th)?\s+\w+)\b",
    re.IGNORECASE,
)


def is_skeptical_query(query: str) -> bool:
    """Detect if a query is distractor-prone and needs skeptical retrieval.

    Triggers: superlatives, embedded date references, very low lexical specificity.
    """
    if _SKEPTICAL_TRIGGERS_RE.search(query):
        return True
    # Very short query with broad terms = low specificity
    words = query.split()
    if len(words) <= 5:
        specific = [w for w in words if len(w) > 4 and w.lower() not in
                     {"what", "which", "where", "about", "their", "there", "these", "those"}]
        if len(specific) <= 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Query Type Detection — category-specific retrieval tuning
# ---------------------------------------------------------------------------

# Temporal signal words/patterns
_TEMPORAL_PATTERNS = re.compile(
    r"\b(when|before|after|during|while|since|until|first|last|latest|earliest"
    r"|recent|previous|next|ago|year|month|week|day|date|time|period"
    r"|january|february|march|april|may|june|july|august|september"
    r"|october|november|december|spring|summer|fall|winter|autumn"
    r"|morning|evening|night|weekend|holiday|birthday|anniversary"
    r"|how long|how often|what time|what date|what year|what month"
    r"|started|ended|began|finished|happened|occurred|changed"
    r"|earlier|later|prior|subsequent|meanwhile|eventually)\b",
    re.IGNORECASE,
)

# Adversarial negation patterns
_ADVERSARIAL_PATTERNS = re.compile(
    r"\b(never|not|no one|nobody|nothing|nowhere|neither|nor"
    r"|didn.t|doesn.t|don.t|wasn.t|weren.t|isn.t|aren.t|hasn.t"
    r"|haven.t|hadn.t|won.t|wouldn.t|can.t|couldn.t|shouldn.t"
    r"|false|incorrect|wrong|untrue|lie|fake|fabricat"
    r"|no interest|no desire|no intention|no plan"
    r"|contrary|opposite|instead|rather than|unlike|different from)\b",
    re.IGNORECASE,
)

# Broader verification-intent patterns — queries expecting yes/no confirmation.
# These need precision over recall, so semantic synonym expansion is suppressed.
_VERIFICATION_INTENT_RE = re.compile(
    r"(?:"
    r"^(?:did|was|were|is|has|have|does|do)\s+"   # Yes/no question openers
    r"|(?:ever|actually|really)\b"                  # Certainty qualifiers
    r"|(?:is it true|is that true|is this true)"    # Truth verification
    r"|(?:no mention|not mention|never mention)"    # Absence checks
    r"|(?:deny|denied|denial)\b"                    # Denial patterns
    r")",
    re.IGNORECASE,
)

# Multi-hop indicators (questions requiring info from multiple sources)
_MULTIHOP_PATTERNS = re.compile(
    r"\b(both|and also|as well as|in addition|together|combined"
    r"|relationship|connection|between|compare|comparison|versus|vs|both"
    r"|how many|how much|total|count|sum|all|every|each"
    r"|who else|what else|where else|which other"
    r"|same|similar|different|shared|common|overlap"
    r"|because|caused|resulted|led to|due to|reason|why did)\b",
    re.IGNORECASE,
)


def detect_query_type(query: str) -> str:
    """Classify query into temporal/adversarial/multi-hop/single-hop.

    Returns one of: 'temporal', 'adversarial', 'multi-hop', 'single-hop'.
    Uses pattern-based heuristics with scoring to handle ambiguous queries.
    """
    query_lower = query.lower()

    temporal_hits = len(_TEMPORAL_PATTERNS.findall(query_lower))
    adversarial_hits = len(_ADVERSARIAL_PATTERNS.findall(query_lower))
    multihop_hits = len(_MULTIHOP_PATTERNS.findall(query_lower))

    # Question mark count can indicate complexity
    qmarks = query.count("?")

    # Score each category
    scores = {
        "temporal": temporal_hits * 2,
        "adversarial": adversarial_hits * 2.5,
        "multi-hop": multihop_hits * 1.5 + (1 if qmarks > 1 else 0),
        "single-hop": 1,  # default baseline
    }

    # Strong negation at start is a very strong adversarial signal
    if re.match(r"^(did|was|were|is|has|have|does|do)\b.+\b(not|never|n.t)\b", query_lower):
        scores["adversarial"] += 3
    # "Did X ever" is also adversarial (expects yes/no about something that may not have happened)
    if re.search(r"\bever\b", query_lower):
        scores["adversarial"] += 2
    # Broader verification intent (yes/no confirmation queries)
    if _VERIFICATION_INTENT_RE.search(query_lower):
        scores["adversarial"] += 1.5

    # Long queries with conjunctions are more likely multi-hop
    word_count = len(query.split())
    if word_count > 15 and multihop_hits > 0:
        scores["multi-hop"] += 2

    # Pick the highest-scoring category
    best = max(scores, key=scores.get)

    # Only override single-hop if signal is strong enough
    if best == "single-hop" or scores[best] < 1.5:
        # Distinguish single-hop from open-domain:
        # Open-domain = short query with no specific temporal/adversarial/multi-hop signal
        # and broad topic words (what, who, describe, tell me about)
        if word_count <= 10 and re.search(
            r"\b(what|who|describe|tell me|identity|about|background)\b",
            query_lower,
        ):
            return "open-domain"
        return "single-hop"

    return best


# Category-specific retrieval parameters
_QUERY_TYPE_PARAMS = {
    "temporal": {
        "recency_weight": 0.6,     # Higher recency matters more
        "date_boost": 2.0,         # Boost blocks with dates
        "expand_query": True,      # Keep expansions
        "extra_limit_factor": 1.5, # Retrieve more candidates
    },
    "adversarial": {
        "recency_weight": 0.3,     # Standard — adversarial needs same broad recall
        "date_boost": 1.0,         # No date boost
        "expand_query": "morph_only",  # Lemma + months only, no semantic synonyms
        "extra_limit_factor": 1.0, # Standard — handled at answerer level, not retrieval
    },
    "multi-hop": {
        "recency_weight": 0.3,     # Standard
        "date_boost": 1.0,
        "expand_query": True,
        "extra_limit_factor": 3.0, # Need more blocks to find all hops
        "graph_boost_override": True,  # Force graph traversal
    },
    "single-hop": {
        "recency_weight": 0.3,     # Standard
        "date_boost": 1.0,
        "expand_query": True,
        "extra_limit_factor": 1.0,
    },
    "open-domain": {
        "recency_weight": 0.2,     # Less recency bias for broad questions
        "date_boost": 1.0,
        "expand_query": True,
        "extra_limit_factor": 2.0, # Retrieve more for diversity
    },
}


def chunk_text(text: str, chunk_size: int = 3, overlap: int = 1) -> list[str]:
    """Split text into overlapping sentence chunks.

    Args:
        text: Full text to chunk.
        chunk_size: Sentences per chunk.
        overlap: Overlapping sentences between chunks.

    Returns list of chunk strings. Returns [text] if <=chunk_size sentences.
    """
    # Simple sentence splitting on .!? followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(sentences):
            break
    return chunks if chunks else [text]


def get_excerpt(block: dict, max_len: int = 300) -> str:
    """Get a short excerpt from a block."""
    for field in ("Statement", "Title", "Summary", "Description", "Name", "Context"):
        val = block.get(field, "")
        if isinstance(val, str) and val:
            return val[:max_len]
    return block.get("_id", "?")


def _parse_speaker_from_tags(tags_str: str) -> str:
    """Extract speaker name from Tags field. Tags format: 'FACT, Caroline'."""
    if not tags_str:
        return ""
    parts = [t.strip() for t in tags_str.split(",")]
    # First tag is the card type (FACT/EVENT/etc), rest are speaker/metadata
    for p in parts[1:]:
        if p and p[0].isupper() and p not in (
            "FACT", "EVENT", "PREFERENCE", "RELATION", "NEGATION", "PLAN",
        ):
            return p
    return ""


def get_block_type(block_id: str) -> str:
    """Infer block type from ID prefix."""
    prefixes = {
        "D-": "decision", "T-": "task", "PRJ-": "project",
        "PER-": "person", "TOOL-": "tool", "INC-": "incident",
        "C-": "contradiction", "DREF-": "drift", "SIG-": "signal",
        "P-": "proposal", "I-": "impact",
    }
    for prefix, btype in prefixes.items():
        if block_id.startswith(prefix):
            return btype
    return "unknown"


def date_score(block: dict) -> float:
    """Boost recent blocks. Returns 0.0-1.0."""
    date_str = block.get("Date", "")
    if not date_str:
        return 0.5
    try:
        from datetime import datetime
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        now = datetime.now()
        days_old = (now - d).days
        if days_old <= 0:
            return 1.0
        return max(0.1, 1.0 - (days_old / 365))
    except (ValueError, TypeError):
        return 0.5


# ---------------------------------------------------------------------------
# Graph-based recall — cross-reference neighbor boosting
# ---------------------------------------------------------------------------

# Regex matching any block ID pattern (D-..., T-..., PRJ-..., DIA-..., FACT-..., etc.)
_BLOCK_ID_RE = re.compile(
    r"\b(D-\d{8}-\d{3}|T-\d{8}-\d{3}|PRJ-\d{3}|PER-\d{3}|TOOL-\d{3}"
    r"|INC-\d{3}|C-\d{8}-\d{3}|SIG-\d{8}-\d{3}|P-\d{8}-\d{3}"
    r"|DIA-[A-Za-z0-9]+-\d+"       # LoCoMo dialog turns: DIA-D1-3
    r"|FACT-\d{3,}"                 # Extracted fact cards: FACT-001
    r")\b"
)

# Graph neighbor boost factor: a neighbor gets this fraction of the referencing block's score
GRAPH_BOOST_FACTOR = 0.4


def build_xref_graph(all_blocks: list[dict]) -> dict[str, set[str]]:
    """Build bidirectional adjacency graph from cross-references.

    Scans every block's text fields for mentions of other block IDs.
    Returns {block_id: set(neighbor_ids)} with edges in both directions.
    """
    block_ids = {b.get("_id") for b in all_blocks if b.get("_id")}
    graph = {bid: set() for bid in block_ids}

    # Fields to scan for cross-references
    xref_fields = SEARCH_FIELDS + [
        "Supersedes", "SupersededBy", "AlignsWith", "Dependencies",
        "Next", "Sources", "Evidence", "Rollback", "History",
    ]

    for block in all_blocks:
        bid = block.get("_id")
        if not bid:
            continue

        # Collect all text from the block
        texts = []
        for field in xref_fields:
            val = block.get(field, "")
            if isinstance(val, str):
                texts.append(val)
            elif isinstance(val, list):
                texts.extend(str(v) for v in val)

        # Also scan ConstraintSignature scope.projects
        for sig in block.get("ConstraintSignatures", []):
            scope = sig.get("scope", {})
            if isinstance(scope, dict):
                for v in scope.values():
                    if isinstance(v, list):
                        texts.extend(str(x) for x in v)
                    elif isinstance(v, str):
                        texts.append(v)

        # Find all referenced block IDs
        full_text = " ".join(texts)
        for match in _BLOCK_ID_RE.finditer(full_text):
            ref_id = match.group(1)
            if ref_id != bid and ref_id in block_ids:
                graph[bid].add(ref_id)
                graph[ref_id].add(bid)  # bidirectional

    return graph


# ---------------------------------------------------------------------------
# v7: Deterministic Reranker — wider retrieve + feature-based re-scoring
# ---------------------------------------------------------------------------

# Regex for detecting time-intent in queries
_TIME_INTENT_RE = re.compile(
    r"\b(what month|what day|when|what date|what year|what week|how long ago"
    r"|which month|which year|which day|what time"
    r"|as mentioned on|mentioned on|on\s+(?:january|february|march|april|may|june"
    r"|july|august|september|october|november|december)"
    r"|on\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june"
    r"|july|august|september|october|november|december)"
    r"|in\s+(?:january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s+\d{4}"
    r"|on\s+\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)

# Month and day tokens for time-overlap scoring
_MONTH_TOKENS = frozenset({
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
})
_DAY_TOKENS = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
})
_TEMPORAL_CONTENT_TOKENS = _MONTH_TOKENS | _DAY_TOKENS | frozenset({
    "yesterday", "tomorrow", "tonight", "weekend",
    "spring", "summer", "fall", "winter", "autumn",
    "holiday", "birthday", "anniversary", "christmas",
    "planned", "planning", "going", "trip", "visit", "travel", "vacation",
})

# Reranker feature weights (tuned for LoCoMo coverage)
# v1.0.2: Reduced speaker dominance (was 0.40), increased entity/phrase overlap
# to avoid "right speaker, wrong topic" over-preference.
_RERANK_W_ENTITY = 0.30
_RERANK_W_TIME = 0.15
_RERANK_W_BIGRAM = 0.15
_RERANK_W_RECENCY = 0.10
_RERANK_W_SPEAKER = 0.25
_RERANK_W_SPEAKER_MISMATCH = -0.10


def _extract_entities(text: str) -> set[str]:
    """Extract likely entity tokens: capitalized words + multi-word proper nouns."""
    entities = set()
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        entities.add(m.group(0).lower())
    # Also grab individual capitalized tokens
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
        entities.add(m.group(0).lower())
    return entities


def _extract_bigram_phrases(text: str) -> set[str]:
    """Extract 2+ word proper nouns / quoted phrases for exact matching."""
    phrases = set()
    # Quoted phrases
    for m in re.finditer(r'"([^"]{3,})"', text):
        phrases.add(m.group(1).lower())
    # Multi-word proper nouns (2-4 capitalized words in sequence)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text):
        phrases.add(m.group(0).lower())
    return phrases


def _extract_speaker_names(query: str, all_results: list[dict]) -> set[str]:
    """Find speaker names mentioned in the query by cross-referencing known speakers."""
    known_speakers = set()
    for r in all_results:
        sp = r.get("speaker", "")
        if sp:
            known_speakers.add(sp.lower())
    query_lower = query.lower()
    mentioned = set()
    for sp in known_speakers:
        if sp in query_lower:
            mentioned.add(sp)
    return mentioned


def rerank_hits(
    query: str,
    hits: list[dict],
    debug: bool = False,
) -> list[dict]:
    """Deterministic reranker: rescores BM25 hits with entity/time/phrase/recency/speaker features.

    Each feature is in [0,1]. Final score = bm25_score + weighted sum of features.
    The top-1 BM25 hit is always preserved in the final set (anchor rule).

    Args:
        query: The original search query.
        hits: BM25-scored results (must have 'score', 'excerpt', 'speaker', optionally 'DiaID').
        debug: If True, attach '_rerank_features' dict to each hit.

    Returns:
        Hits with updated scores, sorted descending.
    """
    if not hits:
        return hits

    # Preserve BM25 top-1 anchor
    bm25_top1_id = hits[0]["_id"] if hits else None

    # --- Extract query signals ---
    q_entities = _extract_entities(query)
    q_phrases = _extract_bigram_phrases(query)
    q_lower = query.lower()
    has_time_intent = bool(_TIME_INTENT_RE.search(q_lower))
    q_mentioned_speakers = _extract_speaker_names(query, hits)

    # Plan-related boosting: if query mentions plan/going/trip, boost those verbs
    plan_intent = bool(re.search(r"\b(plan|going|trip|visit|travel|vacation)\b", q_lower))

    # Find max DiaID for recency normalization
    max_dia = 0
    for h in hits:
        dia = h.get("DiaID", "")
        if dia:
            try:
                # DiaID format varies: could be numeric or "D123" etc.
                dia_num = int(re.sub(r"[^0-9]", "", dia) or "0")
                max_dia = max(max_dia, dia_num)
            except (ValueError, TypeError):
                pass
    if max_dia == 0:
        # Fallback: use position in list
        max_dia = len(hits)

    for idx, h in enumerate(hits):
        bm25_score = h["score"]
        excerpt = h.get("excerpt", "")
        excerpt_lower = excerpt.lower()
        tags_str = h.get("tags", "")
        speaker = h.get("speaker", "").lower()

        # (a) Entity overlap
        h_entities = _extract_entities(excerpt)
        # Also extract from tags
        if tags_str:
            h_entities |= _extract_entities(tags_str)
        if q_entities:
            entity_overlap = len(q_entities & h_entities) / max(1, len(q_entities))
        else:
            entity_overlap = 0.0

        # (b) Time overlap
        time_overlap = 0.0
        if has_time_intent:
            # Check if hit contains temporal content tokens
            hit_tokens = set(re.findall(r"[a-z]+", excerpt_lower))
            temporal_matches = hit_tokens & _TEMPORAL_CONTENT_TOKENS
            if temporal_matches:
                time_overlap = min(1.0, len(temporal_matches) / 2.0)
            # Also check for date patterns (YYYY-MM-DD, "Month Day", etc.)
            if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", excerpt):
                time_overlap = max(time_overlap, 0.5)
            if re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b", excerpt, re.IGNORECASE):
                time_overlap = max(time_overlap, 0.8)
            # Plan-intent boost
            if plan_intent and re.search(r"\b(plan|going|trip|visit|travel|vacation)\b", excerpt_lower):
                time_overlap = max(time_overlap, 0.6)

        # (c) Bigram phrase bonus
        bigram_bonus = 0.0
        if q_phrases:
            for phrase in q_phrases:
                if phrase in excerpt_lower:
                    bigram_bonus = 1.0
                    break

        # (d) Recency bonus (later turns preferred)
        recency_bonus = 0.5  # default mid-range
        dia = h.get("DiaID", "")
        if dia:
            try:
                dia_num = int(re.sub(r"[^0-9]", "", dia) or "0")
                if max_dia > 0:
                    recency_bonus = dia_num / max_dia
            except (ValueError, TypeError):
                pass
        else:
            # Fallback: use line number if available
            line = h.get("line", 0)
            if line > 0:
                recency_bonus = min(1.0, line / 1000.0)

        # (e) Speaker boost / mismatch penalty
        speaker_bonus = 0.0
        if q_mentioned_speakers:
            if speaker and speaker in q_mentioned_speakers:
                speaker_bonus = 1.0
            elif speaker and speaker not in q_mentioned_speakers:
                # Block belongs to a different speaker than the one asked about
                speaker_bonus = _RERANK_W_SPEAKER_MISMATCH / max(abs(_RERANK_W_SPEAKER), 0.01)
            # No speaker tag → neutral (0.0)

        # --- Combine ---
        feature_sum = (
            _RERANK_W_ENTITY * entity_overlap
            + _RERANK_W_TIME * time_overlap
            + _RERANK_W_BIGRAM * bigram_bonus
            + _RERANK_W_RECENCY * recency_bonus
            + _RERANK_W_SPEAKER * speaker_bonus
        )
        h["score"] = round(bm25_score + feature_sum, 4)

        if debug:
            h["_rerank_features"] = {
                "entity_overlap": round(entity_overlap, 3),
                "time_overlap": round(time_overlap, 3),
                "bigram_bonus": round(bigram_bonus, 3),
                "recency_bonus": round(recency_bonus, 3),
                "speaker_bonus": round(speaker_bonus, 3),
                "feature_sum": round(feature_sum, 4),
                "bm25_original": round(bm25_score, 4),
            }

    # Sort by reranked score
    hits.sort(key=lambda r: r["score"], reverse=True)

    # Anchor rule: ensure BM25 top-1 is in final set
    if bm25_top1_id:
        if bm25_top1_id not in {h["_id"] for h in hits[:10]}:
            # Find it and swap into position
            for i, h in enumerate(hits):
                if h["_id"] == bm25_top1_id:
                    # Insert at position 1 (keep reranked #1, add anchor at #2)
                    anchor = hits.pop(i)
                    hits.insert(min(1, len(hits)), anchor)
                    break

    return hits


# ---------------------------------------------------------------------------
# v8: Context Packing — post-retrieval augmentation rules
# ---------------------------------------------------------------------------

# Question-turn cue patterns (triggers adjacency expansion)
_QUESTION_CUE_RE = re.compile(
    r"(?:\?\s*$|any tips|do you|what should|how do|how about|what's your|"
    r"have you|can you|could you|would you|tell me|what kind|you think)",
    re.IGNORECASE,
)

# Multi-entity / plural question patterns (triggers diversity enforcement)
_MULTI_ENTITY_RE = re.compile(
    r"[A-Z][a-z]+\s+and\s+[A-Z][a-z]+",
)
_PLURAL_CUE_RE = re.compile(
    r"\b(ways|scares|reasons|things|hobbies|activities|events|gifts|tips|"
    r"strategies|memories|experiences|goals|plans|problems|both|each|"
    r"meals|snacks|books|games|songs|friends|times|items)\b",
    re.IGNORECASE,
)

# Words unlikely to be speaker names (filter for multi-entity detection)
_NOT_NAMES = frozenset({
    "what", "which", "when", "where", "who", "how", "the", "did", "does",
    "will", "has", "had", "was", "were", "are", "can", "could", "would",
    "should", "may", "might", "shall",
})

# Pronouns that signal coreference (Rule 3 trigger)
_PRONOUN_RE = re.compile(r"\b(it|this|that|these|those)\b", re.IGNORECASE)


def _parse_dia_id(dia: str) -> tuple[str, int] | None:
    """Parse 'D{session}:{turn}' → (session_str, turn_int)."""
    m = re.match(r"D(\d+):(\d+)", dia)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _block_to_result(block: dict, score: float = 0.0) -> dict:
    """Convert a raw parsed block to a result dict."""
    tags_str = block.get("Tags", "")
    return {
        "_id": block.get("_id", "?"),
        "type": get_block_type(block.get("_id", "")),
        "score": round(score, 4),
        "excerpt": get_excerpt(block),
        "speaker": _parse_speaker_from_tags(tags_str),
        "tags": tags_str,
        "file": block.get("_source_file", "?"),
        "line": block.get("_line", 0),
        "status": block.get("Status", ""),
        "DiaID": block.get("DiaID", ""),
    }


def context_pack(
    query: str,
    top_results: list[dict],
    all_blocks: list[dict],
    wider_pool: list[dict],
    limit: int = 10,
) -> list[dict]:
    """Post-retrieval context packing. Augments top-K with deterministic rules.

    Rules:
    1. Dialog adjacency — if a hit is a question turn, add next 1-2 answer turns.
    2. Multi-entity diversity — for plural/multi-speaker queries, ensure distinct
       speakers and DiaIDs in the context.
    3. Pronoun-target rescue — if top hits use pronouns but query has a concrete
       noun, pull ±3 neighbor turns to recover the explicit mention.

    Args:
        query: Original search query.
        top_results: Reranked top-K results.
        all_blocks: All parsed blocks (for neighbor lookup).
        wider_pool: Larger deduped candidate pool (for diversity fallback).
        limit: Original requested limit.

    Returns:
        Augmented result list. May exceed limit by a few context-pack blocks.
    """
    if not top_results or not all_blocks:
        return top_results

    # Build DiaID → block lookup (dialog turns only, not fact cards)
    dia_lookup: dict[str, dict] = {}
    for block in all_blocks:
        dia = block.get("DiaID", "")
        bid = block.get("_id", "")
        if dia and bid.startswith("DIA-"):
            dia_lookup[dia] = block

    augmented = list(top_results)
    existing_ids = {r.get("_id", "") for r in augmented}
    existing_dias = {r.get("DiaID", "") for r in augmented}

    adjacency_added = 0
    diversity_forced = 0
    pronoun_rescue = 0

    # --- Rule 1: Dialog adjacency expansion ---
    for r in list(augmented):
        excerpt = r.get("excerpt", "")
        dia = r.get("DiaID", "")
        bid = r.get("_id", "")
        if not dia:
            continue
        # Only expand dialog turns, not fact cards
        if not bid.startswith("DIA-"):
            continue

        is_question = excerpt.rstrip().endswith("?") or bool(_QUESTION_CUE_RE.search(excerpt))
        if not is_question:
            continue

        parsed = _parse_dia_id(dia)
        if not parsed:
            continue
        session, turn_num = parsed

        for offset in [1, 2]:
            next_dia = f"D{session}:{turn_num + offset}"
            if next_dia in existing_dias:
                continue
            block = dia_lookup.get(next_dia)
            if not block:
                break  # end of session segment

            result = _block_to_result(block, score=r["score"] * 0.8)
            result["via_adjacency"] = True
            augmented.append(result)
            existing_dias.add(next_dia)
            existing_ids.add(result["_id"])
            adjacency_added += 1

    # --- Rule 2: Multi-entity diversity enforcement ---
    is_multi_entity = bool(_MULTI_ENTITY_RE.search(query))
    is_plural = bool(_PLURAL_CUE_RE.search(query))

    if is_multi_entity or is_plural:
        # Extract names from query
        query_names = set()
        for m_name in re.finditer(r"\b([A-Z][a-z]{2,})\b", query):
            name = m_name.group(1)
            if name.lower() not in _NOT_NAMES:
                query_names.add(name.lower())

        # Current speaker coverage
        current_speakers = {
            r.get("speaker", "").lower()
            for r in augmented if r.get("speaker")
        }
        # Current session coverage
        current_sessions = set()
        for r in augmented:
            parsed = _parse_dia_id(r.get("DiaID", ""))
            if parsed:
                current_sessions.add(parsed[0])

        # Check if diversity is needed
        missing_speakers = query_names - current_speakers if query_names else set()
        needs_diversity = bool(missing_speakers) or len(current_sessions) < 2

        if needs_diversity:
            max_extra = 5
            added_this_rule = 0
            for r in wider_pool:
                if added_this_rule >= max_extra:
                    break
                rid = r.get("_id", "")
                dia = r.get("DiaID", "")
                if rid in existing_ids:
                    continue

                sp = r.get("speaker", "").lower()
                parsed = _parse_dia_id(dia)
                new_session = parsed[0] if parsed else ""

                adds_speaker = sp in missing_speakers
                adds_session = new_session and new_session not in current_sessions

                if adds_speaker or adds_session:
                    r_copy = dict(r)
                    r_copy["via_diversity"] = True
                    augmented.append(r_copy)
                    existing_ids.add(rid)
                    if dia:
                        existing_dias.add(dia)
                    if sp:
                        current_speakers.add(sp)
                        missing_speakers.discard(sp)
                    if new_session:
                        current_sessions.add(new_session)
                    diversity_forced += 1
                    added_this_rule += 1

    # --- Rule 3: Pronoun-target rescue ---
    # Extract salient nouns from query (concrete nouns, not common words)
    query_nouns = set()
    for tok in re.findall(r"[a-z]+", query.lower()):
        if tok not in _STOPWORDS and len(tok) > 3 and tok not in _NOT_NAMES:
            query_nouns.add(tok)

    if query_nouns:
        # Check if top hits use pronouns without the salient nouns
        for r in list(augmented[:5]):
            excerpt = r.get("excerpt", "")
            excerpt_lower = excerpt.lower()
            dia = r.get("DiaID", "")
            bid = r.get("_id", "")

            if not dia or not bid.startswith("DIA-"):
                continue

            has_pronoun = bool(_PRONOUN_RE.search(excerpt_lower))
            # Check if any query noun is missing from this hit
            nouns_missing = {n for n in query_nouns if n not in excerpt_lower}

            if has_pronoun and len(nouns_missing) >= len(query_nouns) * 0.5:
                parsed = _parse_dia_id(dia)
                if not parsed:
                    continue
                session, turn_num = parsed

                # Search ±3 turns for explicit noun mentions
                for offset in range(-3, 4):
                    if offset == 0:
                        continue
                    neighbor_dia = f"D{session}:{turn_num + offset}"
                    if neighbor_dia in existing_dias:
                        continue
                    block = dia_lookup.get(neighbor_dia)
                    if not block:
                        continue
                    block_text = get_excerpt(block).lower()
                    # Check if this neighbor contains any missing noun
                    if any(n in block_text for n in nouns_missing):
                        result = _block_to_result(block, score=r["score"] * 0.6)
                        result["via_pronoun_rescue"] = True
                        augmented.append(result)
                        existing_dias.add(neighbor_dia)
                        existing_ids.add(result["_id"])
                        pronoun_rescue += 1
                        if pronoun_rescue >= 3:
                            break
            if pronoun_rescue >= 3:
                break

    if adjacency_added or diversity_forced or pronoun_rescue:
        _log.info("context_pack",
                  adjacency_added=adjacency_added,
                  diversity_forced=diversity_forced,
                  pronoun_rescue=pronoun_rescue)
        metrics.inc("pack_adjacency", adjacency_added)
        metrics.inc("pack_diversity", diversity_forced)
        metrics.inc("pack_pronoun_rescue", pronoun_rescue)

    return augmented


def recall(workspace: str, query: str, limit: int = 10, active_only: bool = False, graph_boost: bool = False, agent_id: str | None = None, retrieve_wide_k: int = 200, rerank: bool = True, rerank_debug: bool = False) -> list[dict]:
    """Search across all memory files using BM25 scoring. Returns ranked results.

    Args:
        workspace: Workspace root path.
        query: Search query.
        limit: Max results to return (final top-k after reranking).
        active_only: Only return blocks with active status.
        graph_boost: Enable cross-reference neighbor boosting.
        agent_id: Optional agent ID for namespace ACL filtering.
        retrieve_wide_k: Number of candidates to retrieve before reranking.
        rerank: Enable deterministic reranking (v7).
        rerank_debug: Log reranker feature breakdowns.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # Detect query type for category-specific tuning
    query_type = detect_query_type(query)
    qparams = _QUERY_TYPE_PARAMS.get(query_type, _QUERY_TYPE_PARAMS["single-hop"])

    # Month normalization: inject numeric month tokens for date matching
    query_tokens = expand_months(query, query_tokens)

    # Skeptical mode: for distractor-prone queries, keep morph-only tokens
    # separate to penalize expansion-only matches later.
    skeptical = is_skeptical_query(query) and query_type in ("adversarial", "single-hop")
    # morph_tokens preserved for future skeptical-mode penalty scoring

    # Query expansion: add domain synonyms
    # adversarial/verification queries use morph_only (no semantic synonyms)
    expand_mode = qparams.get("expand_query", True)
    if expand_mode:
        mode = expand_mode if isinstance(expand_mode, str) else "full"
        # In skeptical mode, force morph_only to suppress semantic drift
        if skeptical:
            mode = "morph_only"
        query_tokens = expand_query(query_tokens, mode=mode)

    # Force graph boost for multi-hop queries
    if qparams.get("graph_boost_override", False):
        graph_boost = True

    # Adjust effective limit for retrieval (retrieve more candidates, trim later)
    limit = int(limit * qparams.get("extra_limit_factor", 1.0))

    # Namespace ACL: resolve accessible paths if agent_id is provided
    ns_manager = None
    if agent_id:
        try:
            from namespaces import NamespaceManager
            ns_manager = NamespaceManager(workspace, agent_id=agent_id)
        except ImportError:
            pass

    # Load all blocks with source file tracking
    all_blocks = []
    for label, rel_path in CORPUS_FILES.items():
        # ACL check: skip files the agent cannot read
        if ns_manager and not ns_manager.can_read(rel_path):
            continue

        path = os.path.join(workspace, rel_path)
        if not os.path.isfile(path):
            continue
        try:
            blocks = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if active_only:
            blocks = get_active(blocks)
        for b in blocks:
            b["_source_file"] = rel_path
            b["_source_label"] = label
            all_blocks.append(b)

    # If agent has namespace, also search agent-private corpus files
    if ns_manager and agent_id:
        agent_ns = f"agents/{agent_id}"
        for label, rel_path in CORPUS_FILES.items():
            ns_path = os.path.join(agent_ns, rel_path)
            full_path = os.path.join(workspace, ns_path)
            if not os.path.isfile(full_path):
                continue
            if not ns_manager.can_read(ns_path):
                continue
            try:
                blocks = parse_file(full_path)
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            if active_only:
                blocks = get_active(blocks)
            for b in blocks:
                b["_source_file"] = ns_path
                b["_source_label"] = f"{label}@{agent_id}"
                all_blocks.append(b)

    if not all_blocks:
        return []

    # --- BM25F: per-field tokenization + flat token list for IDF ---
    doc_field_tokens = []   # [{field: [tokens]}] per block
    doc_flat_tokens = []    # [[all_tokens]] per block (for IDF + bigrams)
    for block in all_blocks:
        ft = extract_field_tokens(block)
        doc_field_tokens.append(ft)
        flat = []
        for tokens in ft.values():
            flat.extend(tokens)
        doc_flat_tokens.append(flat)

    # Document frequency + average weighted doc length
    df = Counter()
    total_wdl = 0.0
    for i, ft in enumerate(doc_field_tokens):
        seen = set()
        wdl = 0.0
        for field, tokens in ft.items():
            w = FIELD_WEIGHTS.get(field, 1.0)
            wdl += len(tokens) * w
            for t in tokens:
                seen.add(t)
        for t in seen:
            df[t] += 1
        total_wdl += wdl

    N = len(all_blocks)
    avg_wdl = total_wdl / N if N > 0 else 1.0

    # Pre-compute query bigrams for phrase matching
    query_bigrams = get_bigrams(query_tokens)

    results = []

    for i, block in enumerate(all_blocks):
        ft = doc_field_tokens[i]
        flat = doc_flat_tokens[i]
        if not flat:
            continue

        # --- BM25F: field-weighted term frequency ---
        # Compute weighted TF across all fields
        weighted_tf = Counter()
        wdl = 0.0
        for field, tokens in ft.items():
            w = FIELD_WEIGHTS.get(field, 1.0)
            wdl += len(tokens) * w
            for t in tokens:
                weighted_tf[t] += w

        score = 0.0
        for qt in query_tokens:
            if qt in weighted_tf:
                wtf = weighted_tf[qt]
                idf = math.log((N - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1)
                numerator = wtf * (BM25_K1 + 1)
                denominator = wtf + BM25_K1 * (1 - BM25_B + BM25_B * wdl / avg_wdl)
                score += idf * numerator / denominator

        if score <= 0:
            continue

        # --- Bigram phrase matching boost ---
        if query_bigrams:
            doc_bigrams = get_bigrams(flat)
            phrase_matches = len(query_bigrams & doc_bigrams)
            if phrase_matches > 0:
                score *= (1.0 + 0.25 * phrase_matches)

        # --- Chunking boost: score best chunk separately, blend ---
        # For long blocks, check if a chunk scores much higher
        statement = block.get("Statement", "") or block.get("Title", "") or ""
        if len(statement) > 200:
            chunks = chunk_text(statement)
            if len(chunks) > 1:
                best_chunk_score = 0.0
                for chunk in chunks:
                    ctokens = tokenize(chunk)
                    ctf = Counter(ctokens)
                    cdl = len(ctokens)
                    cs = 0.0
                    for qt in query_tokens:
                        if qt in ctf:
                            freq = ctf[qt]
                            idf = math.log((N - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1)
                            cs += idf * freq * (BM25_K1 + 1) / (freq + BM25_K1 * (1 - BM25_B + BM25_B * cdl / max(avg_wdl, 1)))
                    best_chunk_score = max(best_chunk_score, cs)
                # Blend: take the better of full-block or best-chunk score
                if best_chunk_score > score:
                    score = 0.6 * best_chunk_score + 0.4 * score

        # --- Boost factors (query-type-aware) ---
        recency = date_score(block)
        rw = qparams.get("recency_weight", 0.3)
        score *= (1.0 - rw + rw * recency)

        # Temporal queries: boost blocks that contain dates
        date_boost = qparams.get("date_boost", 1.0)
        if date_boost > 1.0 and block.get("Date", ""):
            score *= date_boost

        status = block.get("Status", "")
        if status == "active":
            score *= 1.2
        elif status in ("todo", "doing"):
            score *= 1.1

        priority = block.get("Priority", "")
        if priority in ("P0", "P1"):
            score *= 1.1

        # Build rich result payload with speaker + display text
        raw_excerpt = get_excerpt(block)
        tags_str = block.get("Tags", "")
        speaker = _parse_speaker_from_tags(tags_str)

        result = {
            "_id": block.get("_id", "?"),
            "type": get_block_type(block.get("_id", "")),
            "score": round(score, 4),
            "excerpt": raw_excerpt,
            "speaker": speaker,
            "tags": tags_str,
            "file": block.get("_source_file", "?"),
            "line": block.get("_line", 0),
            "status": status,
        }
        # Pass through DiaID for benchmark evidence matching
        if block.get("DiaID"):
            result["DiaID"] = block["DiaID"]
        results.append(result)

    # Graph-based neighbor boosting: 2-hop traversal for multi-hop recall
    if graph_boost and results:
        xref_graph = build_xref_graph(all_blocks)
        score_by_id = {r["_id"]: r["score"] for r in results}
        block_by_id = {b.get("_id"): b for b in all_blocks if b.get("_id")}

        neighbor_scores = {}

        # Multi-hop traversal with progressive decay.
        # For multi-hop queries, extend to 3-hop with stronger propagation.
        hop_decays = [GRAPH_BOOST_FACTOR, GRAPH_BOOST_FACTOR * 0.5]
        if query_type == "multi-hop":
            hop_decays.append(GRAPH_BOOST_FACTOR * 0.25)  # 3rd hop at 0.1

        for hop, decay in enumerate(hop_decays):
            # On first hop, seed from BM25 results; on later hops, seed from
            # newly discovered neighbors
            seeds = results if hop == 0 else [
                {"_id": nid, "score": ns}
                for nid, ns in neighbor_scores.items()
                if nid not in score_by_id
            ]
            for r in seeds:
                rid = r["_id"]
                for neighbor_id in xref_graph.get(rid, set()):
                    boost = r["score"] * decay
                    if neighbor_id not in score_by_id:
                        neighbor_scores[neighbor_id] = (
                            neighbor_scores.get(neighbor_id, 0) + boost
                        )
                    else:
                        neighbor_scores[neighbor_id] = (
                            neighbor_scores.get(neighbor_id, 0) + boost * 0.5
                        )

        # Apply boosts to existing results
        for r in results:
            if r["_id"] in neighbor_scores:
                r["score"] = round(r["score"] + neighbor_scores[r["_id"]], 4)
                r["via_graph"] = True

        # Add new neighbors discovered via graph
        for nid, nscore in neighbor_scores.items():
            if nid not in score_by_id and nid in block_by_id:
                nb = block_by_id[nid]
                nb_tags = nb.get("Tags", "")
                results.append({
                    "_id": nid,
                    "type": get_block_type(nid),
                    "score": round(nscore, 4),
                    "excerpt": get_excerpt(nb),
                    "speaker": _parse_speaker_from_tags(nb_tags),
                    "tags": nb_tags,
                    "file": nb.get("_source_file", "?"),
                    "line": nb.get("_line", 0),
                    "status": nb.get("Status", ""),
                    "via_graph": True,
                })

    # --- RM3 Dynamic Query Expansion ---
    # When enabled via config, use RM3 (Relevance Model 3) instead of the
    # simpler PRF heuristic below.  RM3 estimates a relevance language model
    # from top-K feedback docs, then interpolates expansion terms with the
    # original query.  Skipped for adversarial queries (static expansions only).
    is_adversarial = query_type == "adversarial"
    _rm3_used = False

    config_path = os.path.join(workspace, "mind-mem.json")
    rm3_config = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path) as _f:
                _cfg = json.load(_f)
            rm3_config = _cfg.get("recall", {}).get("rm3", {})
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    if rm3_config.get("enabled", False) and not is_adversarial and results:
        results.sort(key=lambda r: r["score"], reverse=True)

        # Build collection frequency from all flat doc tokens
        collection_freq = Counter()
        total_collection_tokens = 0
        for flat_toks in doc_flat_tokens:
            for t in flat_toks:
                collection_freq[t] += 1
                total_collection_tokens += 1

        # Prepare top docs as (doc_tokens, score) tuples
        result_id_to_idx = {}
        for i, block in enumerate(all_blocks):
            result_id_to_idx[block.get("_id", "")] = i

        top_doc_tokens = []
        for r in results[:rm3_config.get("fb_docs", 5)]:
            idx = result_id_to_idx.get(r["_id"])
            if idx is not None:
                top_doc_tokens.append((doc_flat_tokens[idx], r["score"]))

        expanded_weights = rm3_expand(
            query_tokens, top_doc_tokens,
            collection_freq, total_collection_tokens,
            alpha=rm3_config.get("alpha", 0.6),
            fb_terms=rm3_config.get("fb_terms", 10),
            fb_docs=rm3_config.get("fb_docs", 5),
            min_idf=rm3_config.get("min_idf", 1.0),
            doc_freq={t: c for t, c in df.items()},
            N=N,
        )

        # Re-score all blocks using RM3-expanded weighted query
        expansion_terms_rm3 = [t for t in expanded_weights if t not in set(query_tokens)]
        if expansion_terms_rm3:
            _rm3_used = True
            rm3_weight = 0.4
            for i, block in enumerate(all_blocks):
                ft = doc_field_tokens[i]
                flat = doc_flat_tokens[i]
                if not flat:
                    continue

                weighted_tf_rm3 = Counter()
                wdl = 0.0
                for field, tokens in ft.items():
                    w = FIELD_WEIGHTS.get(field, 1.0)
                    wdl += len(tokens) * w
                    for t in tokens:
                        weighted_tf_rm3[t] += w

                rm3_score = 0.0
                for et in expansion_terms_rm3:
                    if et in weighted_tf_rm3:
                        wtf = weighted_tf_rm3[et]
                        idf = math.log((N - df.get(et, 0) + 0.5) / (df.get(et, 0) + 0.5) + 1)
                        numerator = wtf * (BM25_K1 + 1)
                        denominator = wtf + BM25_K1 * (1 - BM25_B + BM25_B * wdl / avg_wdl)
                        rm3_score += idf * numerator / denominator * expanded_weights[et]

                if rm3_score > 0:
                    bid = block.get("_id", "?")
                    found = False
                    for r in results:
                        if r["_id"] == bid:
                            r["score"] = round(r["score"] + rm3_score * rm3_weight, 4)
                            found = True
                            break
                    if not found:
                        result = {
                            "_id": bid,
                            "type": get_block_type(bid),
                            "score": round(rm3_score * rm3_weight, 4),
                            "excerpt": get_excerpt(block),
                            "file": block.get("_source_file", "?"),
                            "line": block.get("_line", 0),
                            "status": block.get("Status", ""),
                        }
                        if block.get("DiaID"):
                            result["DiaID"] = block["DiaID"]
                        results.append(result)

            _log.info("rm3_expansion", expansion_terms=expansion_terms_rm3[:5],
                      alpha=rm3_config.get("alpha", 0.6))

    # --- Pseudo-Relevance Feedback (PRF) ---
    # For single-hop, open-domain, and multi-hop queries, bridge the lexical
    # gap by extracting expansion terms from top-5 initial results and
    # re-scoring.  Multi-hop benefits from PRF because scattered facts often
    # use different vocabulary than the query.
    # Skipped when RM3 was used (they serve the same purpose).
    if not _rm3_used and query_type in ("single-hop", "open-domain", "multi-hop") and results:
        results.sort(key=lambda r: r["score"], reverse=True)
        prf_top = results[:5]

        # Extract expansion terms: high-TF tokens from top-5 statements,
        # excluding query tokens and very common terms (low IDF).
        prf_terms = Counter()
        for r in prf_top:
            # Tokenize the excerpt (which is the Statement/Description)
            prf_tokens = tokenize(r.get("excerpt", ""))
            for t in prf_tokens:
                if t not in query_tokens and len(t) > 2:
                    prf_terms[t] += 1

        # Keep terms that appear in 2+ of top-5 docs (co-occurring = relevant)
        # and have moderate IDF (not too common, not too rare)
        expansion_terms = []
        for term, count in prf_terms.most_common(15):
            if count >= 2 and df.get(term, 0) < N * 0.3:
                expansion_terms.append(term)
            if len(expansion_terms) >= 8:
                break

        if expansion_terms:
            # Re-score all blocks with expanded query (original + PRF terms).
            # Multi-hop uses lower weight to avoid drifting away from the query.
            prf_weight = 0.25 if query_type == "multi-hop" else 0.4
            for i, block in enumerate(all_blocks):
                ft = doc_field_tokens[i]
                flat = doc_flat_tokens[i]
                if not flat:
                    continue

                # Compute weighted TF
                weighted_tf_prf = Counter()
                wdl = 0.0
                for field, tokens in ft.items():
                    w = FIELD_WEIGHTS.get(field, 1.0)
                    wdl += len(tokens) * w
                    for t in tokens:
                        weighted_tf_prf[t] += w

                # Score only expansion terms
                prf_score = 0.0
                for et in expansion_terms:
                    if et in weighted_tf_prf:
                        wtf = weighted_tf_prf[et]
                        idf = math.log((N - df.get(et, 0) + 0.5) / (df.get(et, 0) + 0.5) + 1)
                        numerator = wtf * (BM25_K1 + 1)
                        denominator = wtf + BM25_K1 * (1 - BM25_B + BM25_B * wdl / avg_wdl)
                        prf_score += idf * numerator / denominator

                if prf_score > 0:
                    bid = block.get("_id", "?")
                    # Find existing result and boost, or add new result
                    found = False
                    for r in results:
                        if r["_id"] == bid:
                            r["score"] = round(r["score"] + prf_score * prf_weight, 4)
                            found = True
                            break
                    if not found:
                        result = {
                            "_id": bid,
                            "type": get_block_type(bid),
                            "score": round(prf_score * prf_weight, 4),
                            "excerpt": get_excerpt(block),
                            "file": block.get("_source_file", "?"),
                            "line": block.get("_line", 0),
                            "status": block.get("Status", ""),
                        }
                        if block.get("DiaID"):
                            result["DiaID"] = block["DiaID"]
                        results.append(result)

    # --- Chain-of-retrieval for multi-hop queries ---
    # Emulates iterative search deterministically:
    # 1. Take top-N from first pass
    # 2. Extract bridge terms (capitalized entities, rare shared tokens)
    # 3. Re-score all blocks using bridge terms
    # 4. Merge new hits into results
    if query_type == "multi-hop" and results:
        results.sort(key=lambda r: r["score"], reverse=True)
        hop1_top = results[:10]
        # hop1_ids reserved for future deduplication in multi-hop bridging

        # Extract bridge terms: capitalized entities from top-10 that aren't in query
        query_lower_set = set(re.findall(r"[a-z]+", query.lower()))
        bridge_terms = Counter()
        for r in hop1_top:
            excerpt = r.get("excerpt", "")
            # Capitalized entities
            for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", excerpt):
                term = m.group(1).lower()
                if term not in query_lower_set and term not in _STOPWORDS:
                    bridge_terms[term] += 1
            # Rare content tokens (appear in 2+ of top-10)
            for tok in tokenize(excerpt):
                if tok not in set(query_tokens) and len(tok) > 3:
                    bridge_terms[tok] += 0.5

        # Keep bridge terms that appear in 2+ top-10 results
        bridge_tokens = [t for t, c in bridge_terms.most_common(12) if c >= 2]

        if bridge_tokens:
            # Second retrieval pass using bridge terms
            for i, block in enumerate(all_blocks):
                bid = block.get("_id", "?")
                if bid in {r["_id"] for r in results}:
                    continue  # already in results

                ft = doc_field_tokens[i]
                flat = doc_flat_tokens[i]
                if not flat:
                    continue

                weighted_tf_br = Counter()
                wdl = 0.0
                for field, tokens in ft.items():
                    w = FIELD_WEIGHTS.get(field, 1.0)
                    wdl += len(tokens) * w
                    for t in tokens:
                        weighted_tf_br[t] += w

                bridge_score = 0.0
                for bt in bridge_tokens:
                    if bt in weighted_tf_br:
                        wtf = weighted_tf_br[bt]
                        idf = math.log((N - df.get(bt, 0) + 0.5) / (df.get(bt, 0) + 0.5) + 1)
                        numerator = wtf * (BM25_K1 + 1)
                        denominator = wtf + BM25_K1 * (1 - BM25_B + BM25_B * wdl / avg_wdl)
                        bridge_score += idf * numerator / denominator

                if bridge_score > 0:
                    # Also check original query overlap — bridge-only hits with
                    # zero original query overlap are likely noise
                    orig_overlap = sum(1 for qt in query_tokens if qt in weighted_tf_br)
                    if orig_overlap > 0:
                        # Blend: 0.3 * bridge_score (second hop is supplementary)
                        tags_str = block.get("Tags", "")
                        result = {
                            "_id": bid,
                            "type": get_block_type(bid),
                            "score": round(bridge_score * 0.3, 4),
                            "excerpt": get_excerpt(block),
                            "speaker": _parse_speaker_from_tags(tags_str),
                            "tags": tags_str,
                            "file": block.get("_source_file", "?"),
                            "line": block.get("_line", 0),
                            "status": block.get("Status", ""),
                            "via_chain": True,
                        }
                        if block.get("DiaID"):
                            result["DiaID"] = block["DiaID"]
                        results.append(result)

            _log.info("chain_of_retrieval", bridge_terms=bridge_tokens[:5],
                      new_hits=sum(1 for r in results if r.get("via_chain")))

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    # --- v7: Two-stage pipeline — wide BM25 retrieve → dedup → rerank → top-k ---
    # Stage 1: Take wide candidate set (retrieve_wide_k)
    wide_k = max(retrieve_wide_k, limit)  # never retrieve fewer than final limit
    wide_candidates = results[:wide_k]

    # Deduplicate by (file, line) stable key — prevents near-duplicate slots
    seen_keys = set()
    deduped = []
    for r in wide_candidates:
        # Primary dedup: file+line
        stable_key = (r.get("file", ""), r.get("line", 0))
        if stable_key != ("", 0) and stable_key in seen_keys:
            continue
        if stable_key != ("", 0):
            seen_keys.add(stable_key)

        # Secondary dedup: DiaID — compound key (DiaID, id_prefix) so one FACT
        # and one DIA can coexist for the same dialog turn.
        dia = r.get("DiaID", "")
        if dia:
            rid = r.get("_id", "")
            prefix = "FACT" if rid.startswith("FACT-") else "DIA" if rid.startswith("DIA-") else rid[:4]
            dia_key = (dia, prefix)
            if dia_key in seen_keys:
                continue
            seen_keys.add(dia_key)

        deduped.append(r)

    # Stage 2: Deterministic rerank (v7)
    if rerank and len(deduped) > limit:
        deduped = rerank_hits(query, deduped, debug=rerank_debug)

    top = deduped[:limit]

    # Stage 3: Context packing — augment top-K with adjacency/diversity/rescue
    top = context_pack(query, top, all_blocks, deduped, limit)

    _log.info("query_complete", query=query, query_type=query_type,
              blocks_searched=N, wide_k=wide_k, reranked=rerank,
              results=len(top),
              top_score=top[0]["score"] if top else 0)
    metrics.inc("recall_queries")
    metrics.inc("recall_results", len(top))
    return top


def _load_backend(workspace: str) -> str:
    """Load recall backend from config. Falls back to BM25 scan.

    Supported backends:
        "scan" / "tfidf" — in-memory BM25 scan (default, O(corpus))
        "sqlite"          — SQLite FTS5 index (O(log N))
        "vector"          — vector embedding backend (requires recall_vector)
    """
    config_path = os.path.join(workspace, "mind-mem.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            backend = cfg.get("recall", {}).get("backend", "scan")
            if backend == "sqlite":
                return "sqlite"
            if backend == "vector":
                try:
                    from recall_vector import VectorBackend
                    return VectorBackend(cfg.get("recall", {}))
                except ImportError:
                    pass  # fall through to scan
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    return None  # use built-in BM25 scan


def main():
    parser = argparse.ArgumentParser(description="mind-mem Recall Engine (BM25 + Graph)")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--workspace", "-w", default=".", help="Workspace path")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    parser.add_argument("--active-only", action="store_true", help="Only search active blocks")
    parser.add_argument("--graph", action="store_true", help="Enable graph-based neighbor boosting")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--retrieve-wide-k", type=int, default=200,
                        help="Candidates to retrieve before reranking (default 200)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable v7 deterministic reranking (use pure BM25)")
    parser.add_argument("--rerank-debug", action="store_true",
                        help="Show reranker feature breakdowns in JSON output")
    parser.add_argument("--backend", choices=["scan", "sqlite", "auto"],
                        default="auto",
                        help="Recall backend: scan (O(corpus)), sqlite (O(log N)), auto (config)")
    args = parser.parse_args()

    # Resolve backend: CLI flag > config > default scan
    backend = args.backend
    if backend == "auto":
        cfg_backend = _load_backend(args.workspace)
        if cfg_backend == "sqlite":
            backend = "sqlite"
        elif cfg_backend is not None:
            # Vector or other custom backend
            try:
                results = cfg_backend.search(
                    args.workspace, args.query, args.limit, args.active_only
                )
            except (OSError, ValueError, TypeError) as e:
                print(f"recall: backend error ({e}), falling back to scan", file=sys.stderr)
                backend = "scan"
            else:
                backend = None  # already have results
        else:
            backend = "scan"

    if backend == "sqlite":
        from sqlite_index import query_index
        results = query_index(
            args.workspace, args.query, limit=args.limit,
            active_only=args.active_only, graph_boost=args.graph,
            retrieve_wide_k=args.retrieve_wide_k,
            rerank=not args.no_rerank, rerank_debug=args.rerank_debug,
        )
    elif backend == "scan":
        results = recall(args.workspace, args.query, args.limit, args.active_only,
                         args.graph, retrieve_wide_k=args.retrieve_wide_k,
                         rerank=not args.no_rerank, rerank_debug=args.rerank_debug)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print("No results found.")
        else:
            for r in results:
                graph_tag = " [graph]" if r.get("via_graph") else ""
                rerank_tag = ""
                if args.rerank_debug and "_rerank_features" in r:
                    feats = r["_rerank_features"]
                    rerank_tag = f" [rerank: ent={feats['entity_overlap']:.2f} time={feats['time_overlap']:.2f} bi={feats['bigram_bonus']:.2f} rec={feats['recency_bonus']:.2f} spk={feats['speaker_bonus']:.2f}]"
                print(f"[{r['score']:.3f}] {r['_id']} ({r['type']}{graph_tag}) — {r['excerpt'][:80]}{rerank_tag}")
                print(f"        {r['file']}:{r['line']}")


if __name__ == "__main__":
    main()
