"""Recall engine tokenization — Porter stemmer and tokenizer."""

from __future__ import annotations

import functools
import re

from ._recall_constants import _IRREGULAR_LEMMA, _STOPWORDS

__all__ = ["_stem", "tokenize"]


# ---------------------------------------------------------------------------
# Porter Stemmer (simplified, zero-dependency)
# ---------------------------------------------------------------------------


#: Memo size for :func:`_stem`.
#:
#: PROFILED 2026-09-11, after the corpus-parse cache removed the previous 81% of recall
#: time: `_stem` was 124,203 calls per 15 recalls, driving 2,328,980 `str.endswith` calls
#: (~19 per call, which is what a suffix-rule stemmer does). Corpus words repeat heavily,
#: so nearly all of that was recomputation.
#:
#: BOUNDED on purpose. An unbounded memo over the tokens of an arbitrarily large corpus is
#: a memory leak with a plausible-sounding justification, and a corpus has many distinct
#: words. 16384 covers a large vocabulary at a few hundred KB.
#:
#: Safe ONLY because `_stem` is pure -- one str in, one str out, no clock, no I/O, no
#: module state. `tests/test_stem_memoisation.py` pins that purity over the AST, because if
#: a later edit makes the stemmer context-dependent the memo will serve its first answer
#: forever, silently, for every word in every corpus.
_STEM_CACHE_SIZE = 16384


@functools.lru_cache(maxsize=_STEM_CACHE_SIZE)
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
        pass  # keep as-is
    elif word.endswith("ating"):
        word = word[:-3] + "e"
    elif word.endswith("ation"):
        word = word[:-5] + "ate"
    elif word.endswith("ously"):
        word = word[:-5] + "ous"
    elif word.endswith("ous") and len(word) > 5:
        pass  # keep as-is
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
    return [_stem(t) for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOPWORDS and len(t) > 1]
