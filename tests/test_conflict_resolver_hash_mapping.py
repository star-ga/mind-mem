"""Audit trail must print the WINNER's hash next to Winner, not block_a's.

Regression: hash_a/hash_b are positional (block_a/block_b); the writer
emitted Winner:(hash_a)/Loser:(hash_b) unconditionally, so when the winner
was block_b the hashes were swapped, breaking the tamper-evidence check.
"""

from __future__ import annotations

from pathlib import Path

from mind_mem.conflict_resolver import ResolutionStrategy, generate_resolution_proposals


def test_winner_is_block_b_prints_winner_hash(tmp_path: Path) -> None:
    resolutions = [
        {
            "strategy": ResolutionStrategy.TIMESTAMP,
            "confidence": "high",
            "contradiction_id": "C-1",
            "block_a": "D-A",
            "block_b": "D-B",
            "hash_a": "AAAA-blockA",
            "hash_b": "BBBB-blockB",
            "winner_id": "D-B",  # winner is block_b
            "loser_id": "D-A",
            "rationale": "B is newer",
        }
    ]
    n = generate_resolution_proposals(str(tmp_path), resolutions)
    assert n == 1
    text = (Path(tmp_path) / "intelligence" / "proposed" / "RESOLUTIONS_PROPOSED.md").read_text()
    assert "Winner: D-B (hash: BBBB-blockB)" in text  # winner's own hash
    assert "Loser: D-A (hash: AAAA-blockA)" in text
