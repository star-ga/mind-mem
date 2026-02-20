"""Recall engine constants — search fields, BM25 params, regex patterns, limits."""

from __future__ import annotations

import re

__all__ = [
    "SEARCH_FIELDS", "SIG_FIELDS", "CORPUS_FILES",
    "BM25_K1", "BM25_B",
    "FIELD_WEIGHTS",
    "_STOPWORDS", "_IRREGULAR_LEMMA",
    "GRAPH_BOOST_FACTOR", "_BLOCK_ID_RE",
    "MAX_BLOCKS_PER_QUERY", "MAX_GRAPH_NEIGHBORS_PER_HOP", "MAX_RERANK_CANDIDATES",
    "_VALID_RECALL_KEYS",
]

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

# Cap neighbors per hop to prevent blowup on dense graphs (recommendation)
MAX_GRAPH_NEIGHBORS_PER_HOP = 50

# Maximum rerank candidates to prevent latency spikes (#9)
MAX_RERANK_CANDIDATES = 200

# Maximum blocks to process in a single recall query (#15)
MAX_BLOCKS_PER_QUERY = 50000

_VALID_RECALL_KEYS = frozenset({
    "backend", "limit", "rm3", "cross_encoder", "graph_boost",
    "pinecone_api_key", "pinecone_index", "pinecone_namespace",
    "qdrant_url", "qdrant_collection", "embedding_model",
    "retrieve_wide_k", "rerank", "active_only",
})
