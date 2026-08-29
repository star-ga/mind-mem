"""RA.0 — one tier axis, and the other ladders deleted rather than abstracted.

The Group-R audit found three block-lifecycle tier ladders in the tree
(``tiered_memory.py``, ``memory_tiers.py``, ``v4/tier_memory.py``), a fourth
surface duplicating one of them (``tier_recall.py``), and none of them wired to
recall. The ruling was explicit: **collapse to one and delete the other two —
do not abstract over them.** An abstraction over three ladders is still three
ladders, and a dashboard cannot render one axis while three exist.

The survivor is :class:`mind_mem.memory_tiers.MemoryTier`. It is the only
ladder reachable from a real entry point (``mcp.server`` and ``mm_cli`` both
reach it through ``compaction.run_promotion_cycle``); the other two were
reachable from nothing but their own tests.

These are ratchets, not one-shot cleanups: each asserts a *frozen set*, so a
fourth ladder fails the build the moment it is written rather than being
discovered by the next audit.
"""

from __future__ import annotations

import ast
import pathlib

import mind_mem

SRC = pathlib.Path(mind_mem.__file__).parent

#: Enum classes in ``src/mind_mem`` whose name carries "Tier", and the axis
#: each one names. Two survive, and they are **different questions**:
#:
#: * ``IngestTier`` — provenance: which door a write arrived through. Closed
#:   by construction (``AdmissionReceipt.tier`` has no default), governance,
#:   not a ladder — nothing is ever promoted along it.
#: * ``MemoryTier`` — the ONE lifecycle ladder: WORKING → SHARED → LONG_TERM
#:   → VERIFIED.
#:
#: Anything else is the fourth tier system the roadmap forbids.
TIER_VOCABULARIES = frozenset({("enums", "IngestTier"), ("memory_tiers", "MemoryTier")})

#: Ladders deleted by RA.0, with the vocabulary each one duplicated.
DELETED_LADDERS = {
    "tiered_memory": "Tier: WORKING/EPISODIC/SEMANTIC/PROCEDURAL + retrieval_boost",
    "tier_recall": "a hard copy of MemoryTier's ordinals as score multipliers",
    "v4/tier_memory": "RecallTier: HOT/WARM/COLD + block_recall_tier",
}


def _tier_classes() -> set[tuple[str, str]]:
    """Every ``class *Tier*(...)`` enum defined under ``src/mind_mem``."""
    found: set[tuple[str, str]] = set()
    scanned = 0
    for path in sorted(SRC.rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = path.relative_to(SRC).with_suffix("").as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or "Tier" not in node.name:
                continue
            bases = {ast.unparse(b).rsplit(".", 1)[-1] for b in node.bases}
            if bases & {"Enum", "IntEnum", "StrEnum"}:
                found.add((module, node.name))
    # Vacuity guard: a walker that parsed nothing would "pass" this file
    # end to end. Assert it actually read the tree and found the survivor.
    assert scanned > 100, f"tier scan visited only {scanned} modules — walker is broken"
    assert ("memory_tiers", "MemoryTier") in found, "scan did not find the surviving ladder"
    return found


def test_ra0_exactly_one_lifecycle_tier_ladder_exists() -> None:
    """A frozen set, so a fourth ladder fails the build rather than the audit."""
    assert _tier_classes() == TIER_VOCABULARIES


def test_ra0_the_deleted_ladders_are_deleted_not_abstracted() -> None:
    """Deleted means *absent*. A shim re-exporting them would still be a ladder."""
    for module, what in DELETED_LADDERS.items():
        path = SRC.joinpath(*module.split("/")).with_suffix(".py")
        assert not path.exists(), f"{module}.py still present ({what}) — RA.0 says delete, not abstract"


def test_ra0_nothing_imports_a_deleted_ladder() -> None:
    """Including lazily: a function-local import is still an import."""
    offenders: dict[str, set[str]] = {}
    names = {m.replace("/", ".") for m in DELETED_LADDERS}
    for path in sorted(SRC.rglob("*.py")):
        module = path.relative_to(SRC).with_suffix("").as_posix().replace("/", ".")
        text = path.read_text(encoding="utf-8")
        hits = {n for n in names if n.rsplit(".", 1)[-1] in text}
        if hits:
            offenders[module] = hits
    assert not offenders, f"deleted ladders still named in {offenders}"


def test_ra0_one_ladder_means_one_tier_store() -> None:
    """``block_recall_tier`` was the second ladder's table; it has no writer left.

    Leaving readers of a table nothing creates is not a collapse — it is a
    permanently-degraded branch that reads like a feature.
    """
    readers = [p.relative_to(SRC).as_posix() for p in sorted(SRC.rglob("*.py")) if "block_recall_tier" in p.read_text(encoding="utf-8")]
    assert readers == [], f"block_recall_tier has no writer but is still read by {readers}"

    owners = {p.relative_to(SRC).as_posix() for p in sorted(SRC.rglob("*.py")) if "block_tier_meta" in p.read_text(encoding="utf-8")}
    assert owners == {"memory_tiers.py"}, f"the surviving ladder's table is touched outside it: {owners}"
