"""Regression tests for over-broad synonym entries in _QUERY_EXPANSIONS.

These synonyms caused false-positive BM25F score inflation on unrelated
sessions: "stay" (from "live") matched office/workplace sessions; "show"
and "watch" (from "movie") matched unrelated tokens; "house"/"live" (from
"home") were too semantically broad.  The entries were narrowed so only
clear paraphrases remain.
"""

from __future__ import annotations

from mind_mem._recall_expansion import _QUERY_EXPANSIONS


def _synonyms_of(word: str) -> frozenset[str]:
    return frozenset(_QUERY_EXPANSIONS.get(word, []))


class TestLiveSynonyms:
    """'live' should not expand to place/activity words that appear in
    unrelated contexts (office, hotel, sports)."""

    def test_stay_removed_from_live(self) -> None:
        assert "stay" not in _synonyms_of("live"), (
            "'stay' in live-synonyms causes false-positive boosting of sessions about staying late, staying at a hotel, etc."
        )

    def test_house_removed_from_live(self) -> None:
        assert "house" not in _synonyms_of("live"), (
            "'house' is too broad — triggers on house music, house rules, restaurant-house-wine, etc."
        )

    def test_home_removed_from_live(self) -> None:
        assert "home" not in _synonyms_of("live"), "'home' is circular and too broad in compound nouns."

    def test_location_removed_from_live(self) -> None:
        # "location" is a generic term that appears in many non-residence contexts
        assert "location" not in _synonyms_of("live"), "'location' matches filming locations, GPS location queries, etc."

    def test_reside_kept_for_live(self) -> None:
        """'reside' is a genuine paraphrase — must remain."""
        assert "reside" in _synonyms_of("live"), (
            "'reside' is an unambiguous paraphrase of 'live (somewhere)' and must stay in the expansion table."
        )


class TestHomeSynonyms:
    """'home' should not expand to words that cause cross-document leakage."""

    def test_live_removed_from_home(self) -> None:
        assert "live" not in _synonyms_of("home"), "Circular 'home'↔'live' expansion amplifies noise across sessions."

    def test_house_removed_from_home(self) -> None:
        assert "house" not in _synonyms_of("home"), "'house' in home-synonyms fires on house music, The White House, etc."


class TestMovieSynonyms:
    """'movie' should only expand to close paraphrases, not generic verbs."""

    def test_show_removed_from_movie(self) -> None:
        assert "show" not in _synonyms_of("movie"), (
            "'show' matches TV show, show me, show-and-tell, etc., causing massive false-positive boosting."
        )

    def test_watch_removed_from_movie(self) -> None:
        assert "watch" not in _synonyms_of("movie"), "'watch' matches watch (timepiece), watch out, neighbourhood watch, etc."

    def test_film_kept_for_movie(self) -> None:
        """'film' is an unambiguous paraphrase — must remain."""
        assert "film" in _synonyms_of("movie"), "'film' is a direct synonym of 'movie' and must stay."

    def test_cinema_kept_for_movie(self) -> None:
        assert "cinema" in _synonyms_of("movie"), "'cinema' is contextually close and must stay."
