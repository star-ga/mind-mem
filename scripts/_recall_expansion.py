"""Recall engine query expansion — domain synonyms, month normalization, RM3."""

from __future__ import annotations

import math
import re
from collections import Counter

from _recall_tokenization import _stem

__all__ = [
    "_QUERY_EXPANSIONS", "expand_months", "expand_query",
    "_rm3_language_model", "rm3_expand",
]


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

    "March 2023" -> adds "03", "2023-03" to token list.
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
# still find their expansion entries (e.g. "partn" -> partner's expansions)
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
