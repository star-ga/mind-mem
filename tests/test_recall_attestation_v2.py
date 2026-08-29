"""Acceptance gate for the ``RECALL_ATTEST_v2`` preimage.

The v1 record already bound the served ids in rank order (the determinism
seam), so the collision it was accused of — two runs of equal cardinality
serving *disjoint* sets — was already closed when this file was written.
T8/T9 below are therefore **locks**, not repros: they pin a property that
must survive the layout change rather than demonstrate a live defect.

Two holes did survive into this bump, and both are the same shape — a run
input, or a record field, that the fingerprint does not cover:

1. **The query was not an input.** A fingerprint of a pure function has to
   bind every argument. Two different questions answered with the same
   ranked list produced one identical hash, so the record asserted a
   reproducibility it could not deliver: replaying it did not tell you what
   was asked.
2. **``schema`` was an unbound sibling.** The preimage's domain separator
   was the module constant, never the record's own field, so anyone holding
   an attestation could relabel it — to a future tag, or *down* to an
   earlier one — and it stayed internally consistent. A version field that
   does not change the hash version-stamps nothing.

The bump closes both, and closes them the only way that is honest: in the
preimage, under a new tag. A sibling field is unbound by definition, and
editing v1's layout under v1's name would leave two incompatible things
answering to one version string.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
from typing import Any

import pytest

import mind_mem
from mind_mem.hybrid_recall import _as_results
from mind_mem.recall import recall
from mind_mem.recall_attestation import (
    GENESIS_ANCHOR,
    RECALL_ATTEST_TAG,
    RecallAttestation,
    build_recall_attestation,
    derive_recall_attestation,
    verify_recall_attestation,
)
from mind_mem.recall_digests import query_hash, served_set_digest

#: The tag this bump replaces. It is spelled out **only here** — the
#: source tree must not contain it at all (see T11), because a literal a
#: producer can reach is a downgrade target.
V1_TAG = "RECALL_ATTEST_v1"

INSTANT = "2026-08-29"
QUERY = "pineapple protocol rollout"

_COMMON: dict[str, Any] = {
    "legs_ran": ("bm25",),
    "legs_degraded": (),
    "config_hash": "CFG",
    "degraded": None,
    "index_anchor": GENESIS_ANCHOR,
    "scoring_instant": INSTANT,
}


def _att(**over: Any) -> RecallAttestation:
    """Build an attestation over the shared fixture, overriding some fields."""
    kwargs: dict[str, Any] = {**_COMMON, "query": QUERY, "result_count": 2, **over}
    return build_recall_attestation(**kwargs)


# ---------------------------------------------------------------------------
# T8 / T9 — the served answer is bound, set AND order
# ---------------------------------------------------------------------------


def test_t8_disjoint_served_sets_of_equal_cardinality_differ() -> None:
    """Two runs that share a cardinality and share no block are two answers.

    A LOCK, not a repro: ``results_digest`` bound this before the bump.
    It is here because the digest's *encoding* changes in v2 (one owner,
    length-prefixed, its own domain tag), and an encoding swap is exactly
    the kind of edit that could quietly reintroduce the collision.
    """
    left = _att(served_ids=("D-1", "D-2"))
    right = _att(served_ids=("D-3", "D-4"))

    assert left.result_count == right.result_count, "fixture is not equal-cardinality"
    assert set(("D-1", "D-2")).isdisjoint(("D-3", "D-4")), "fixture is not disjoint"
    assert left.attestation_hash != right.attestation_hash


def test_t9_the_same_ids_in_a_different_rank_order_differ() -> None:
    """Top-1 vs top-5 is a different answer, so rank order is semantic."""
    forward = _att(served_ids=("D-1", "D-2"))
    reverse = _att(served_ids=("D-2", "D-1"))
    assert forward.attestation_hash != reverse.attestation_hash
    assert forward.results_digest != reverse.results_digest


def test_the_query_is_an_input_and_the_fingerprint_binds_it() -> None:
    """THE LIVE HOLE. Same ranked answer, two different questions.

    Under v1 these collided: the preimage carried the answer and every
    other input except the one the caller actually typed.
    """
    asked = _att(served_ids=("D-1", "D-2"), query="what did we decide about latency")
    other = _att(served_ids=("D-1", "D-2"), query="who owns the ingest gate")

    assert asked.results_digest == other.results_digest, "fixture must hold the answer fixed"
    assert asked.attestation_hash != other.attestation_hash


def test_the_query_binding_is_a_digest_not_the_text() -> None:
    """The record commits to the question without restating it.

    The envelope already carries the query verbatim; the attestation is a
    fingerprint, and a fingerprint that embeds its input is a copy.
    """
    att = _att(served_ids=("D-1",), query="a very private question")
    assert att.query_hash == query_hash("a very private question")
    assert "private" not in json.dumps(att.to_dict())


def test_the_query_hash_is_tamper_evident() -> None:
    att = _att(served_ids=("D-1",))
    assert att.is_internally_consistent()
    assert not dataclasses.replace(att, query_hash="0" * 64).is_internally_consistent()


# ---------------------------------------------------------------------------
# served_set_digest — one canonical encoding, one owner
# ---------------------------------------------------------------------------


def test_served_set_digest_matches_the_specified_encoding_byte_for_byte() -> None:
    """The encoding is the contract, so it is asserted against a literal.

    Recomputing it with the module's own helpers would only prove the
    function agrees with itself.
    """
    ids = ("D-1", "D-22")
    expected = hashlib.sha256(
        b"MM_SERVED_v1\x00" + struct.pack(">I", 2) + struct.pack(">I", 3) + b"D-1" + struct.pack(">I", 4) + b"D-22"
    ).hexdigest()
    assert served_set_digest(ids) == expected


@pytest.mark.parametrize(
    "left,right",
    [
        (("AB", "C"), ("A", "BC")),
        (("ABC",), ("AB", "C")),
        ((), ("",)),
        (("", ""), ("",)),
    ],
)
def test_length_prefixing_makes_concatenation_unambiguous(left: tuple, right: tuple) -> None:
    """Without the per-id length there is no boundary, and these collide."""
    assert served_set_digest(left) != served_set_digest(right)


def test_the_served_digest_is_domain_separated_from_the_query_digest() -> None:
    """One string, two roles, two tags — so the digests can never be swapped."""
    assert served_set_digest(("x",)) != query_hash("x")


def test_the_record_has_exactly_one_served_encoding() -> None:
    """``results_digest`` IS ``served_set_digest`` — no second spelling."""
    ids = ("D-9", "D-8", "D-7")
    assert _att(served_ids=ids, result_count=3).results_digest == served_set_digest(ids)


# ---------------------------------------------------------------------------
# T11 — the v1 tag is gone, and verify() will not take it back
# ---------------------------------------------------------------------------


def test_t11_the_emitted_tag_is_v2() -> None:
    assert RECALL_ATTEST_TAG == "RECALL_ATTEST_v2"
    assert _att(served_ids=("D-1",)).to_dict()["schema"] == "RECALL_ATTEST_v2"


def test_t11_the_v1_literal_appears_nowhere_in_the_source_tree() -> None:
    """A tag a producer can reach is a downgrade target, so none is shipped."""
    root = pathlib.Path(mind_mem.__file__).parent
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py") if V1_TAG in p.read_text(encoding="utf-8")]
    assert not offenders, f"the retired tag survives in {offenders}"


def test_t11_verify_rejects_a_v1_tagged_value() -> None:
    """No dual-tag support: accepting both is the downgrade this closes."""
    good = _att(served_ids=("D-1",))
    assert verify_recall_attestation(good.to_dict()) is True

    downgraded = dict(good.to_dict())
    downgraded["schema"] = V1_TAG
    assert verify_recall_attestation(downgraded) is False


def test_t11_the_schema_field_is_bound_not_a_sibling() -> None:
    """Relabelling the record must break it — that is what a version is for."""
    good = _att(served_ids=("D-1",))
    for tag in (V1_TAG, "RECALL_ATTEST_v3", ""):
        relabelled = dataclasses.replace(good, schema=tag)
        assert not relabelled.is_internally_consistent(), f"{tag!r} left the hash intact"


def test_t11_the_parse_boundary_refuses_a_foreign_tag_by_name() -> None:
    d = dict(_att(served_ids=("D-1",)).to_dict())
    d["schema"] = V1_TAG
    with pytest.raises(ValueError, match="schema"):
        RecallAttestation.from_dict(d)


def test_verify_rejects_a_tampered_record_and_survives_hostile_input() -> None:
    """A verifier meets malformed input by construction; it must not raise."""
    good = _att(served_ids=("D-1",))
    assert verify_recall_attestation(good) is True
    assert verify_recall_attestation(dataclasses.replace(good, result_count=99)) is False
    for hostile in ({}, {"schema": RECALL_ATTEST_TAG}, {"schema": 7}, []):
        assert verify_recall_attestation(hostile) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T10 — determinism across processes
# ---------------------------------------------------------------------------


_CHILD = """
import json, sys
from mind_mem.recall_attestation import build_recall_attestation
att = build_recall_attestation(
    legs_ran=("vector", "bm25"), legs_degraded=(), config_hash="CFG",
    degraded={"leg": "vector", "reason": "deadline_exceeded"},
    index_anchor="0" * 64, result_count=3, served_ids=("D-3", "D-1", "D-2"),
    query="pineapple protocol rollout", scoring_instant="2026-08-29",
)
sys.stdout.write(json.dumps(att.to_dict(), sort_keys=True))
"""


def test_t10_two_processes_with_different_hash_seeds_agree_byte_for_byte() -> None:
    """Same corpus, config, instant and query in two interpreters -> one record.

    The seeds are forced apart so a dependence on set-iteration order —
    the classic way a "deterministic" digest drifts — cannot hide behind
    two identically-seeded runs.
    """
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(mind_mem.__file__).parent.parent)}
    out = [
        subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True,
            text=True,
            timeout=120,
            env={**env, "PYTHONHASHSEED": seed},
            check=True,
        ).stdout
        for seed in ("0", "524287")
    ]
    assert out[0] == out[1]
    assert json.loads(out[0])["schema"] == RECALL_ATTEST_TAG


# ---------------------------------------------------------------------------
# T12 — the structural rail: the scoring path may not read a ledger
# ---------------------------------------------------------------------------

#: Modules that persist per-run retrieval outcomes, or anchor them. A read
#: from one of these into scoring is credibility feeding back on itself.
LEDGER_MODULES = frozenset({"retrieval_graph", "ledger_anchor", "usage_meter", "served_ledger"})

#: Every ledger symbol ``_recall_core`` is allowed to name, and why. WRITE
#: entries record an outcome after the ranking is fixed and cannot feed it.
#: READ entries pull *prior-run* state into the current ranking — a live
#: breach of the rail, pre-existing and out of this bump's scope, pinned
#: here so a third one cannot arrive unnoticed.
LEDGER_WRITES = frozenset({"log_retrieval", "record_hard_negatives"})
LEDGER_READS = frozenset({"get_hard_negative_ids", "propagate_scores"})
LEDGER_PURE = frozenset({"feedback_quality_credit", "recall_sufficiency"})


def _eager_imports(module: str) -> set[str]:
    """First-party modules *module* pulls in at import time (not lazily)."""
    root = pathlib.Path(mind_mem.__file__).parent
    path = root.joinpath(*module.split(".")).with_suffix(".py")
    if not path.is_file():
        path = root.joinpath(*module.split("."), "__init__.py")
    if not path.is_file():
        return set()
    package = module.split(".")[:-1]
    found: set[str] = set()
    stack: list[tuple[Any, bool]] = [(ast.parse(path.read_text(encoding="utf-8")), False)]
    while stack:
        node, lazy = stack.pop()
        for child in ast.iter_child_nodes(node):
            nested = lazy or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if not nested and isinstance(child, ast.ImportFrom) and child.level and child.module:
                found.add(".".join([*package, child.module]))
            elif not nested and isinstance(child, (ast.Import, ast.ImportFrom)):
                name = getattr(child, "module", None) or ""
                names = [name] if name else [a.name for a in getattr(child, "names", [])]
                found.update(n[len("mind_mem.") :] for n in names if n.startswith("mind_mem."))
            stack.append((child, nested))
    return found


def _eager_closure(start: str) -> dict[str, str]:
    """Transitive eager-import closure of *start* -> the edge that reached it."""
    reached = {start: start}
    queue = collections.deque([start])
    while queue:
        current = queue.popleft()
        for nxt in sorted(_eager_imports(current)):
            if nxt not in reached:
                reached[nxt] = current
                queue.append(nxt)
    return reached


def test_t12_the_attestation_has_no_import_path_to_any_ledger() -> None:
    """Transitive, not depth-1: a two-hop edge is still an edge.

    The attestation is the one artifact that must be a pure function of the
    run it describes. A module that cannot reach a ledger cannot read one,
    on any path, including those no test exercises.
    """
    reached = _eager_closure("recall_attestation")
    offenders = {m: reached[m] for m in reached if m in LEDGER_MODULES}
    assert not offenders, f"recall_attestation reaches a ledger via {offenders}"


def test_t12_the_canonical_encodings_are_a_leaf() -> None:
    """``recall_digests`` must import nothing first-party — that is the split.

    The served-set ledger will need these exact bytes. If the encodings
    lived in the attestation module, the ledger would have to import it to
    get them, putting a ledger one hop from the scoring path. A leaf both
    can depend on removes the direction entirely.
    """
    assert _eager_imports("recall_digests") == set()
    assert _eager_closure("recall_digests") == {"recall_digests": "recall_digests"}


def test_t12_the_scoring_path_cannot_reach_the_served_set_ledger() -> None:
    """The rail, stated as the one edge RA.1 must never create.

    Not closure-emptiness: ``_recall_core`` already reaches ``retrieval_graph``
    and pulls prior-run state out of it, a live breach this bump does not own
    (see the ratchet below). What is enforceable — and what the served-set
    ledger must never join — is that the ledger RA.1 adds stays unreachable.
    Frequency-of-serving is derivable from any served-set ledger, and that was
    accepted; it is harmless only while it cannot flow backward into scoring.

    Transitive, so a two-hop edge counts. Guarded against vacuity by a
    known-true positive: if the walker cannot see the ledger edge that DOES
    exist, its silence about the one that must not is worth nothing.
    """
    reached = _eager_closure("_recall_core")
    assert "retrieval_graph" in reached, "walker found no ledger edge at all — it is not walking"
    assert "served_ledger" not in reached, f"the scoring path reaches served_ledger via {reached.get('served_ledger')}"


_RAIL_CHILD = """
import json, sys
import mind_mem.recall  # noqa: F401  — the scoring path, imported as a consumer would
import mind_mem._recall_core  # noqa: F401
import mind_mem.recall_attestation  # noqa: F401
print(json.dumps(sorted(m for m in sys.modules if m.startswith("mind_mem."))))
"""


def test_t12_importing_the_scoring_path_does_not_load_a_ledger_module() -> None:
    """The rail at RUNTIME, in a fresh interpreter — not only in the AST.

    A static walk sees ``import`` statements. It does not see ``importlib``,
    a plugin hook, or a re-export that pulls a module in sideways. Loading the
    scoring path in a clean process and reading ``sys.modules`` catches all
    three, and the two checks disagree only when something interesting is
    happening.
    """
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(mind_mem.__file__).parent.parent)}
    out = subprocess.run(
        [sys.executable, "-c", _RAIL_CHILD],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=True,
    ).stdout
    loaded = set(json.loads(out))
    assert "mind_mem.recall_attestation" in loaded, "child did not import the scoring path — the check is vacuous"
    for ledger in LEDGER_MODULES:
        if ledger == "retrieval_graph":
            continue  # the pinned, pre-existing breach; see the ratchet below
        assert f"mind_mem.{ledger}" not in loaded, f"{ledger} was loaded by importing the scoring path"


def test_t12_the_served_set_ledger_owns_nothing_the_attestation_owns() -> None:
    """One-way ownership: the ledger imports the encodings, never the record.

    ``recall_attestation`` re-exports nothing to the ledger — it must reach
    ``served_set_digest`` at its owner, ``recall_digests``. Importing it
    through the attestation module would put the attestation on the ledger's
    import path and make the direction ambiguous, which is the whole reason
    the encodings were split into a leaf.
    """
    source = (pathlib.Path(mind_mem.__file__).parent / "served_ledger.py").read_text(encoding="utf-8")
    modules = {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) and node.module}
    assert "recall_digests" in modules, "the ledger must reuse the canonical served-set encoding"
    assert "recall_attestation" not in modules
    assert "recall_attestation" not in _eager_closure("served_ledger")


def test_t12_the_scoring_path_ledger_surface_is_pinned() -> None:
    """A ratchet over a breach this bump does not own.

    ``_recall_core`` genuinely imports the retrieval ledger, and two of the
    names it imports are *reads* of prior-run state that move the current
    ranking. That is the rail's real violation; deleting it is a behaviour
    change well outside a preimage bump. What is enforceable here — and
    what this asserts — is that the surface cannot GROW: a new ledger
    symbol on the scoring path fails the build and has to be argued for.
    """
    source = (pathlib.Path(mind_mem.__file__).parent / "_recall_core.py").read_text(encoding="utf-8")
    named: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module in LEDGER_MODULES:
            named.update(alias.name for alias in node.names)

    assert named == LEDGER_WRITES | LEDGER_READS | LEDGER_PURE
    assert named & LEDGER_READS == LEDGER_READS, "a pinned prior-run read vanished — re-derive the rail"


# ---------------------------------------------------------------------------
# The live surface — a binding nothing threads through binds nothing
# ---------------------------------------------------------------------------


def _fixed_backend(monkeypatch: pytest.MonkeyPatch, ws: str) -> Any:
    """An MCP recall whose leg returns ONE list whatever it is asked.

    Holding the answer constant is the point: it isolates the query as the
    only thing that differs between the two runs below, which is precisely
    the collision v1 could not see.
    """
    import mind_mem.hybrid_recall as hr
    import mind_mem.mcp.tools.recall as mcp_recall

    class _FixedHB:
        vector_enabled = False
        vector_available = False

        @staticmethod
        def from_config(config: Any) -> Any:
            return _FixedHB()

        def search(self, query: str, workspace: str, limit: int = 10, **kwargs: Any) -> Any:
            return _as_results([{"_id": "D-1", "score": 1.0}], None)

    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    monkeypatch.setattr(hr, "HybridBackend", _FixedHB)
    monkeypatch.setattr(mcp_recall, "_load_config", lambda _ws: {"cache": {"enabled": False}})
    return mcp_recall


def test_the_served_envelope_binds_the_question_it_was_asked(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: the query reaches the record on the real recall path.

    The binding exists in the builder either way; what this pins is the
    *wiring*. A parameter no caller threads through is a field that is
    always the same constant, which is the v1 record wearing a new field
    name.
    """
    mcp_recall = _fixed_backend(monkeypatch, _workspace(tmp_path, "wired", poison=None))

    asked = "what did we decide about latency"
    envelope = json.loads(mcp_recall._recall_impl(asked, limit=5, backend="hybrid"))
    att = envelope["attestation"]

    assert att["schema"] == RECALL_ATTEST_TAG
    assert att["query_hash"] == query_hash(asked)
    assert verify_recall_attestation(att) is True

    other = json.loads(mcp_recall._recall_impl("who owns the ingest gate", limit=5, backend="hybrid"))["attestation"]
    assert other["results_digest"] == att["results_digest"], "fixture must hold the answer fixed"
    assert other["attestation_hash"] != att["attestation_hash"]


# ---------------------------------------------------------------------------
# T15 — withheld content cannot move the fingerprint
# ---------------------------------------------------------------------------


def _block(bid: str, statement: str, status: str) -> str:
    return f"[{bid}]\nStatement: {statement}\nDate: 2026-08-29\nStatus: {status}\n\n---\n\n"


def _workspace(tmp_path: Any, name: str, *, poison: str | None) -> str:
    ws = os.path.join(str(tmp_path), name)
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(_block("D-20260829-001", f"The {QUERY} decision", "active"))
    if poison is not None:
        with open(os.path.join(ws, "intelligence", "SIGNALS.md"), "w", encoding="utf-8") as fh:
            # Exactly the query text, so it would out-rank the servable seed.
            fh.write(_block("SIG-20260829-999", QUERY, poison))
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump({"recall": {"vector_enabled": False, "provider": "local"}}, fh)
    return ws


def _attest(ws: str) -> RecallAttestation:
    results = recall(ws, QUERY, limit=10, scoring_instant=INSTANT)
    return derive_recall_attestation(
        _as_results(list(results), None),
        vector_requested=False,
        vector_available=False,
        config_hash="CFG",
        index_anchor=GENESIS_ANCHOR,
        query=QUERY,
        scoring_instant=INSTANT,
    )


def test_t15_a_withheld_top_one_block_changes_nothing_in_the_record(tmp_path: Any) -> None:
    """The fingerprint of an answer must not depend on what was withheld.

    If a quarantined block that would have ranked first moved the hash,
    the record would be a side channel: its value would carry a fact about
    content the caller was never allowed to see.
    """
    clean = _attest(_workspace(tmp_path, "clean", poison=None))
    poisoned = _attest(_workspace(tmp_path, "poisoned", poison="quarantined"))

    assert "SIG-20260829-999" not in json.dumps(poisoned.to_dict())
    assert poisoned.result_count == clean.result_count
    assert poisoned.results_digest == clean.results_digest
    assert poisoned.attestation_hash == clean.attestation_hash


def test_t15_the_control_run_proves_the_poison_would_have_ranked_first(tmp_path: Any) -> None:
    """Without the control, T15 passes whenever the fixture never fired."""
    served = [r.get("_id") for r in recall(_workspace(tmp_path, "control", poison="active"), QUERY, limit=10, scoring_instant=INSTANT)]
    assert served and served[0] == "SIG-20260829-999", f"poison did not rank first: {served}"
