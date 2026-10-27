"""RRF fusion keeps every leg's metadata, not only the dict it picked.

The fused item is a copy of ONE leg's dict (the freshest by ``Date``). The lexical leg carries
``speaker`` / ``DiaID`` / ``tags`` and no ``Date``; the dense leg carries ``Date`` and none of those.
Picking the dense dict therefore erased the speaker and dialogue id of every fused hit, and a
downstream evidence packer rendered the block with no speaker and no source id.
"""

from __future__ import annotations

from mind_mem.hybrid_recall import rrf_fuse


def _legs():
    bm25 = [
        {
            "_id": "DIA-D1-9",
            "excerpt": "[Ana] I want to study counseling.",
            "speaker": "Ana",
            "DiaID": "D1:9",
            "tags": "session-1, Ana",
            "score": 7.5,
        }
    ]
    vec = [{"_id": "DIA-D1-9", "excerpt": "[Ana] I want to study counseling.", "Date": "1:56 pm on 8 May, 2023", "score": 0.61}]
    return bm25, vec


def test_fused_hit_keeps_fields_from_both_legs():
    bm25, vec = _legs()
    (hit,) = rrf_fuse([bm25, vec], [1.0, 1.0], source_names=["bm25", "vector"])
    assert hit["speaker"] == "Ana"
    assert hit["DiaID"] == "D1:9"
    assert hit["tags"] == "session-1, Ana"
    assert hit["Date"] == "1:56 pm on 8 May, 2023"


def test_selected_dict_still_wins_on_conflicting_fields():
    bm25 = [{"_id": "B-1", "Date": "2024-01-01", "Statement": "old", "score": 1.0}]
    vec = [{"_id": "B-1", "Date": "2024-06-01", "Statement": "new", "score": 0.5}]
    (hit,) = rrf_fuse([bm25, vec], [1.0, 1.0])
    assert hit["Statement"] == "new"  # Audit R-1: the freshest dict is the primary
    assert hit["Date"] == "2024-06-01"


def test_inputs_are_not_mutated():
    bm25, vec = _legs()
    rrf_fuse([bm25, vec], [1.0, 1.0])
    assert "Date" not in bm25[0] and "speaker" not in vec[0]
