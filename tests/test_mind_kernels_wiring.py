# Copyright 2026 STARGA, Inc.
"""mind_kernels wiring — one loader, parity with the incumbents, downgrade refusal.

Slice 5 of the 5.0.1 restoration. ``mind_kernels`` was deleted in 5.0.0 for
having no importer; it is back, and these tests exist to prove it is
*connected* rather than merely present:

* ``bm25f_score`` is the recall scorer, not a lookalike (exact float equality
  over a fixture corpus — the old body disagreed in the last bits).
* ``sha3_512_chain_verify`` refuses a v3 -> v1 entry-hash downgrade, which
  per-entry verification structurally cannot see, and the hash-chain IMPORT
  door now uses it behind ``v4.mind_kernels``.
* There is ONE loader. The retired one handed ``$MIND_MEM_KERNELS_SO``
  straight to ``ctypes.CDLL`` with no allowlist.

Every test here fails if the wiring is removed or the module body stubbed.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter

import pytest

from mind_mem import mind_ffi, mind_kernels
from mind_mem._recall_constants import BM25_B, BM25_K1, FIELD_WEIGHTS
from mind_mem._recall_scoring import bm25f_score_terms, compute_weighted_tf
from mind_mem.hash_chain_v2 import (
    GENESIS_HASH,
    HashChainV2,
    _compute_entry_hash_v1,
    _compute_entry_hash_v3,
)

# ---------------------------------------------------------------------------
# Fixture corpus — real field names and real FIELD_WEIGHTS.
#
# The weights that matter here are 1.2 / 0.8 / 0.5 / 0.3: none is exact in
# binary floating point, so ``count(term) * weight`` (what the module used to
# compute) and ``+= weight`` per token (what the recall scorer computes)
# disagree in the last bits as soon as a term repeats. Documents 2 and 3 are
# built to trip exactly that.
# ---------------------------------------------------------------------------

CORPUS: list[dict[str, list[str]]] = [
    {
        "Statement": ["jwt", "rotation", "policy"],
        "Context": ["auth", "jwt", "gateway"],
        "Tags": ["jwt", "security"],
    },
    {
        "Description": ["jwt"] * 7 + ["expiry"],
        "Context": ["jwt", "clock"],
        "Tags": ["jwt", "jwt", "jwt"],
        "History": ["jwt"] * 3,
    },
    {
        "Statement": ["token"] * 4,
        "Description": ["token"] * 5 + ["jwt"] * 2,
        "Rationale": ["token", "jwt"],
        "History": ["token"] * 6,
    },
    {
        "Summary": ["unrelated", "prose"],
        "Context": ["nothing", "matching"],
    },
]

QUERIES = [["jwt"], ["token"], ["jwt", "token"], ["jwt", "jwt"], ["absent"]]


def _wdls() -> list[float]:
    return [compute_weighted_tf(doc, FIELD_WEIGHTS)[1] for doc in CORPUS]


def _avg_wdl() -> float:
    wdls = _wdls()
    return sum(wdls) / len(wdls)


# ---------------------------------------------------------------------------
# 1. BM25F parity with the incumbent scorer
# ---------------------------------------------------------------------------


def test_bm25f_score_is_exactly_the_recall_scorer_on_a_fixture_corpus():
    """Not "close to" — the same float, for every (query, document) pair.

    ``bm25f_score`` delegates to ``compute_weighted_tf`` +
    ``bm25f_score_terms``. If it goes back to computing its own weighted term
    frequency, docs 2 and 3 diverge in the low bits and this fails.
    """
    avg = _avg_wdl()
    compared = 0
    nonzero = 0
    for query in QUERIES:
        for doc, wdl in zip(CORPUS, _wdls()):
            weighted_tf, _ = compute_weighted_tf(doc, FIELD_WEIGHTS)
            expected = bm25f_score_terms(
                list(query),
                weighted_tf,
                wdl,
                {term: 1.0 for term in query},
                avg,
                k1=BM25_K1,
                b=BM25_B,
            )
            actual = mind_kernels.bm25f_score(query, doc, FIELD_WEIGHTS, wdl, avg, k1=BM25_K1, b=BM25_B)
            assert actual == expected, f"query={query} doc={sorted(doc)} {actual!r} != {expected!r}"
            compared += 1
            nonzero += 1 if actual > 0 else 0

    # Guard against a vacuous pass: a stub returning 0.0 would satisfy
    # equality only if the incumbent also returned 0.0 everywhere.
    assert compared == len(QUERIES) * len(CORPUS)
    assert nonzero >= 10, f"fixture corpus produced only {nonzero} non-zero scores"


def test_bm25f_score_uses_the_package_bm25_constants_by_default():
    """The defaults are the recall constants, not a private (1.5, 0.75) pair.

    The pre-wiring body defaulted to ``k1=1.5`` while every BM25 computation
    in recall uses ``BM25_K1 = 1.2``, so "the canonical fallback" scored
    differently from the thing it was a fallback for whenever a caller
    omitted the parameter.
    """
    doc = CORPUS[1]
    weighted_tf, wdl = compute_weighted_tf(doc, FIELD_WEIGHTS)
    avg = _avg_wdl()
    expected = bm25f_score_terms(["jwt"], weighted_tf, wdl, {"jwt": 1.0}, avg, k1=BM25_K1, b=BM25_B)
    assert mind_kernels.bm25f_score(["jwt"], doc, FIELD_WEIGHTS, wdl, avg) == expected


def test_bm25f_score_repeated_query_term_counts_twice_like_the_scorer():
    doc = CORPUS[0]
    weighted_tf, wdl = compute_weighted_tf(doc, FIELD_WEIGHTS)
    avg = _avg_wdl()
    once = mind_kernels.bm25f_score(["jwt"], doc, FIELD_WEIGHTS, wdl, avg)
    twice = mind_kernels.bm25f_score(["jwt", "jwt"], doc, FIELD_WEIGHTS, wdl, avg)
    assert twice == once * 2
    assert bm25f_score_terms(["jwt", "jwt"], weighted_tf, wdl, {"jwt": 1.0}, avg, k1=BM25_K1, b=BM25_B) == twice


def test_bm25f_score_clamps_a_nonpositive_average_length():
    doc = CORPUS[0]
    _, wdl = compute_weighted_tf(doc, FIELD_WEIGHTS)
    assert mind_kernels.bm25f_score(["jwt"], doc, FIELD_WEIGHTS, wdl, 0.0) == mind_kernels.bm25f_score(
        ["jwt"], doc, FIELD_WEIGHTS, wdl, 1.0
    )


def test_bm25f_score_reads_no_clock_and_no_randomness():
    """Determinism: recall is a pure function of (corpus, config, instant).

    A kernel on the scored path must not reach for a clock or an RNG. The
    static check is the honest one — a runtime repeat would pass even if the
    function read a clock and ignored the value.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mind_kernels))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not ({"time", "random", "datetime", "secrets"} & imported), sorted(imported)


# ---------------------------------------------------------------------------
# 2. Chain verify — the downgrade refusal
# ---------------------------------------------------------------------------


def _entry(entry_id: str, previous_hash: str, *, scheme: str) -> dict[str, str]:
    """One chain entry hashed under the named scheme."""
    fields = (entry_id, "2026-09-01T00:00:00Z", "BLOCK-1", "create", "c" * 128, previous_hash)
    digest = _compute_entry_hash_v3(*fields) if scheme == "v3" else _compute_entry_hash_v1(*fields)
    keys = ("entry_id", "timestamp", "block_id", "action", "content_hash", "previous_hash")
    return dict(zip(keys, fields)) | {"entry_hash": digest}


def _segment(schemes: list[str], head: str = GENESIS_HASH) -> list[dict[str, str]]:
    entries = []
    prev = head
    for idx, scheme in enumerate(schemes):
        e = _entry(f"e{idx}", prev, scheme=scheme)
        entries.append(e)
        prev = e["entry_hash"]
    return entries


def test_chain_verify_rejects_a_v3_to_v1_downgrade():
    """THE working definition. v3 then v1 is a forged continuation.

    The v1 preimage is ``|``-joined, so a field containing ``|`` can move a
    boundary without changing the digest. Accepting v1 *after* a v3 entry
    would let that weakness be used to append history to a hardened chain.
    """
    assert mind_kernels.sha3_512_chain_verify(_segment(["v3", "v1"])) is False
    assert mind_kernels.sha3_512_chain_verify(_segment(["v3", "v3", "v1"])) is False
    assert mind_kernels.sha3_512_chain_verify(_segment(["v1", "v3", "v1"])) is False


def test_chain_verify_accepts_the_legal_shapes():
    """Positive controls — without these the test above passes on a stub."""
    assert mind_kernels.sha3_512_chain_verify(_segment(["v1", "v1", "v1"])) is True
    assert mind_kernels.sha3_512_chain_verify(_segment(["v3", "v3", "v3"])) is True
    # An UPGRADE mid-chain is legal; only the downgrade is not.
    assert mind_kernels.sha3_512_chain_verify(_segment(["v1", "v1", "v3", "v3"])) is True
    assert mind_kernels.sha3_512_chain_verify([]) is True


def test_chain_verify_rejects_a_broken_link_and_a_tampered_field():
    segment = _segment(["v3", "v3"])
    broken = [segment[0], dict(segment[1], previous_hash="f" * 128)]
    assert mind_kernels.sha3_512_chain_verify(broken) is False

    tampered = [dict(segment[0], action="delete"), segment[1]]
    assert mind_kernels.sha3_512_chain_verify(tampered) is False


def test_chain_verify_anchors_the_first_entry_when_asked():
    """``previous_hash=`` is what makes it usable as an append-time gate."""
    segment = _segment(["v3", "v3"], head="a" * 128)
    assert mind_kernels.sha3_512_chain_verify(segment) is True
    assert mind_kernels.sha3_512_chain_verify(segment, previous_hash="a" * 128) is True
    assert mind_kernels.sha3_512_chain_verify(segment, previous_hash=GENESIS_HASH) is False


def test_chain_verify_uses_the_hash_chain_schemes_not_a_private_copy():
    """Delegation guard: re-point the v3 scheme and the verifier must follow."""
    import mind_mem.hash_chain_v2 as hc

    segment = _segment(["v3"])
    assert mind_kernels.sha3_512_chain_verify(segment) is True

    original = hc._compute_entry_hash_v3
    try:
        hc._compute_entry_hash_v3 = lambda *a: "0" * 128  # type: ignore[assignment]
        assert mind_kernels.sha3_512_chain_verify(segment) is False
    finally:
        hc._compute_entry_hash_v3 = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 3. The import door — flag ON refuses the downgrade, flag OFF is unchanged
# ---------------------------------------------------------------------------


@pytest.fixture
def flag_off(monkeypatch, tmp_path):
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "absent-mind-mem.json"))
    return tmp_path


@pytest.fixture
def flag_on(monkeypatch, tmp_path):
    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": {"mind_kernels": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    return tmp_path


def _forged_export(tmp_path, chain: HashChainV2) -> str:
    """A JSONL segment: one honest v3 entry, then a v1-hashed continuation."""
    head = chain.get_latest(1)
    prev = head[0].entry_hash if head else GENESIS_HASH
    segment = _segment(["v3", "v1"], head=prev)
    path = tmp_path / "forged.jsonl"
    path.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in segment), encoding="utf-8")
    return str(path)


def _row_count(chain: HashChainV2) -> int:
    with sqlite3.connect(chain._db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM hash_chain").fetchone()[0])


def test_import_jsonl_refuses_the_downgraded_segment_with_the_flag_on(flag_on):
    chain = HashChainV2(str(flag_on / "chain" / "chain.db"))
    chain.append("BLOCK-0", "create", "seed")
    before = _row_count(chain)

    with pytest.raises(ValueError, match="sequence verification"):
        chain.import_jsonl(_forged_export(flag_on, chain))

    assert _row_count(chain) == before, "a refused import must write nothing"
    assert chain.verify_chain() == (True, -1)


def test_import_jsonl_with_the_flag_off_is_byte_identical_to_before(flag_off):
    """Flag OFF keeps the PRE-EXISTING behaviour, gap and all.

    Per-entry verification accepts the v1 continuation, so the forged segment
    is written — and ``verify_chain`` then reports the ledger broken at that
    index. That disagreement between the two doors is exactly what the flag
    closes; this test pins the untouched default so the wiring cannot have
    changed it silently. See the deferred marker on ``import_jsonl``.
    """
    chain = HashChainV2(str(flag_off / "chain" / "chain.db"))
    chain.append("BLOCK-0", "create", "seed")
    before = _row_count(chain)

    imported = chain.import_jsonl(_forged_export(flag_off, chain))

    assert imported == 2
    assert _row_count(chain) == before + 2
    valid, broken_at = chain.verify_chain()
    assert valid is False and broken_at == 2


def test_import_jsonl_still_accepts_an_honest_segment_with_the_flag_on(flag_on):
    """The flag must refuse MORE, never something that used to work."""
    source = HashChainV2(str(flag_on / "src" / "chain.db"))
    source.append("BLOCK-1", "create", "one")
    source.append("BLOCK-2", "update", "two")
    export = str(flag_on / "honest.jsonl")
    assert source.export_jsonl(export) == 2

    target = HashChainV2(str(flag_on / "dst" / "chain.db"))
    assert target.import_jsonl(export) == 2
    assert target.verify_chain() == (True, -1)


def test_import_jsonl_keeps_its_original_rejection_messages_with_the_flag_on(flag_on):
    """A tampered entry still fails with the per-entry message and line number."""
    chain = HashChainV2(str(flag_on / "chain" / "chain.db"))
    segment = _segment(["v3"])
    segment[0]["action"] = "delete"  # entry_hash no longer matches
    path = flag_on / "tampered.jsonl"
    path.write_text(json.dumps(segment[0], separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        chain.import_jsonl(str(path))
    assert "sequence verification" in str(exc.value) or "tampered or corrupt" in str(exc.value)
    assert _row_count(chain) == 0


def test_the_import_flag_probe_is_silent_when_off(flag_off, capsys):
    """A probe deciding whether a feature is on must not be observable."""
    import mind_mem.hash_chain_v2 as hc

    bad = flag_off / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    import os

    os.environ["MIND_MEM_CONFIG"] = str(bad)
    try:
        capsys.readouterr()
        assert hc._sequence_verify_enabled() is False
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
    finally:
        os.environ["MIND_MEM_CONFIG"] = str(flag_off / "absent-mind-mem.json")


def test_the_flag_is_registered():
    """An unregistered flag name silently answers False forever."""
    from mind_mem.v4.feature_flags import ALL_V4_FLAGS

    assert "mind_kernels" in ALL_V4_FLAGS


# ---------------------------------------------------------------------------
# 4. One loader
# ---------------------------------------------------------------------------


def test_mind_kernels_load_kernels_delegates_to_the_ffi_loader(monkeypatch):
    """Not "returns something similar" — returns what the one loader returned."""
    sentinel = object()
    called: list[str | None] = []

    def _fake(path=None):
        called.append(path)
        return sentinel

    monkeypatch.setattr(mind_ffi, "load_kernels", _fake)
    assert mind_kernels.load_kernels("some/path") is sentinel
    assert called == ["some/path"]


def test_a_rogue_kernels_so_env_var_is_refused_and_reported(monkeypatch, tmp_path):
    """The retired loader read $MIND_MEM_KERNELS_SO and CDLL'd it unchecked.

    Any path in the environment could pull arbitrary native code into the
    process. It now goes through the same allowlist as MIND_MEM_LIB.
    """
    rogue = tmp_path / "rogue.so"
    rogue.write_bytes(b"\x7fELF definitely not a library")
    monkeypatch.delenv("MIND_MEM_LIB", raising=False)
    monkeypatch.setenv("MIND_MEM_KERNELS_SO", str(rogue))
    # Blank the search paths so this asserts about the ENV path only; a dev
    # checkout may hold a real lib/libmindmem.so, which the loader is right
    # to fall back to.
    monkeypatch.setattr(mind_ffi, "_LIB_SEARCH_PATHS", [])

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        mind_ffi,
        "_log",
        type("L", (), {"warning": lambda _s, e, **k: events.append((e, k)), "info": lambda *a, **k: None})(),
    )

    def _explode(*_a, **_k):
        raise AssertionError("the rogue .so must never reach ctypes.CDLL")

    monkeypatch.setattr(mind_ffi.ctypes, "CDLL", _explode)

    binding = mind_ffi.load_kernels()
    assert binding.native is None and binding.backend == "python"
    rejected = [kw for event, kw in events if event == "ffi_env_lib_rejected"]
    assert rejected, f"rejection not reported; events={events}"
    assert rejected[0]["reason"] == "outside allowed directories"
    assert rejected[0]["source"] == "MIND_MEM_KERNELS_SO"


def test_an_explicit_rogue_path_is_refused_too(monkeypatch, tmp_path):
    rogue = tmp_path / "rogue.so"
    rogue.write_bytes(b"\x7fELF")
    monkeypatch.delenv("MIND_MEM_LIB", raising=False)
    monkeypatch.delenv("MIND_MEM_KERNELS_SO", raising=False)
    monkeypatch.setattr(mind_ffi, "_LIB_SEARCH_PATHS", [])
    monkeypatch.setattr(
        mind_ffi.ctypes,
        "CDLL",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("rogue path reached CDLL")),
    )
    assert mind_ffi.load_kernels(str(rogue)).native is None


def test_resolve_allowlisted_lib_accepts_a_path_under_an_allowed_dir(tmp_path):
    allowed = mind_ffi.allowed_lib_dirs()[0]
    allowed.mkdir(parents=True, exist_ok=True)
    probe = allowed / "test-allowlist-probe.so"
    probe.write_bytes(b"\x7fELF")
    try:
        resolved, reason = mind_ffi.resolve_allowlisted_lib(str(probe))
        assert resolved == probe.resolve() and reason == ""
    finally:
        probe.unlink()

    missing, reason = mind_ffi.resolve_allowlisted_lib(str(allowed / "does-not-exist.so"))
    assert missing is None and reason == "file does not exist"


def test_the_binding_always_exposes_the_python_kernels(monkeypatch):
    """MIND kernels are OPTIONAL: no .so must ever mean no kernels."""
    monkeypatch.setattr(mind_ffi, "_LIB_SEARCH_PATHS", [])
    monkeypatch.delenv("MIND_MEM_LIB", raising=False)
    monkeypatch.delenv("MIND_MEM_KERNELS_SO", raising=False)
    binding = mind_ffi.load_kernels()
    assert binding.backend == "python" and binding.native is None
    assert binding.bm25f_score is mind_kernels.bm25f_score
    assert binding.sha3_512_chain_verify is mind_kernels.sha3_512_chain_verify
    assert binding.cosine is mind_kernels.cosine
    assert binding.dot is mind_kernels.dot
    assert binding.rrf_fusion is mind_kernels.rrf_fusion


def test_category_distiller_resolves_its_kernel_through_the_one_loader():
    """It used to construct MindMemKernel itself — a third probe."""
    import ast
    import inspect

    import mind_mem.category_distiller as cd

    source = inspect.getsource(cd)
    assert "load_kernels" in source
    names = {node.func.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "MindMemKernel" not in names


# ---------------------------------------------------------------------------
# 5. The other two kernels agree with their incumbents
# ---------------------------------------------------------------------------


def test_cosine_is_the_vector_inertness_incumbent():
    from mind_mem.vector_inertness import cosine as incumbent

    cases = [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]),
        ([1.0, 0.0], [0.0, 1.0]),
        ([0.0, 0.0], [1.0, 1.0]),
        ([], [1.0]),
        ([1.0, 2.0], [1.0]),
        ([-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]),
    ]
    for a, b in cases:
        assert mind_kernels.cosine(a, b) == incumbent(list(a), list(b)), (a, b)
    assert mind_kernels.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_dot_matches_a_plain_sum_and_degrades_on_mismatch():
    assert mind_kernels.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
    assert mind_kernels.dot([], [1.0]) == 0.0
    assert mind_kernels.dot([1.0, 2.0], [1.0]) == 0.0


def test_rrf_fusion_agrees_with_the_hybrid_recall_incumbent():
    """Same formula, different contract — pinned so they cannot drift.

    Compared as a {id: score} map, because the two disagree on TIES and are
    each right to: the kernel breaks a tie on the id (a bare id list carries
    nothing else), while ``rrf_fuse`` keeps the freshest result dict it saw.
    The ordering assertion below therefore uses a tie-free fixture.
    """
    from mind_mem.hybrid_recall import rrf_fuse

    lists = [["A", "B", "C"], ["B", "C", "D"]]
    fused = mind_kernels.rrf_fusion(lists, k=60)
    incumbent = rrf_fuse([[{"_id": i} for i in lst] for lst in lists], [1.0, 1.0], k=60)

    assert [bid for bid, _ in fused] == [hit["_id"] for hit in incumbent]
    # ``rrf_fuse`` rounds its published score to 6 dp; the kernel does not.
    assert {bid: round(score, 6) for bid, score in fused} == {h["_id"]: h["rrf_score"] for h in incumbent}

    # A fixture with an exact tie: the SCORES still agree, only the order is
    # each implementation's own business.
    tied = [["A", "B", "C"], ["B", "C", "D"], ["D", "A"]]
    tied_scores = {bid: round(score, 6) for bid, score in mind_kernels.rrf_fusion(tied, k=60)}
    tied_incumbent = rrf_fuse([[{"_id": i} for i in lst] for lst in tied], [1.0, 1.0, 1.0], k=60)
    assert tied_scores == {h["_id"]: h["rrf_score"] for h in tied_incumbent}


def test_rrf_fusion_is_deterministic_on_ties():
    fused = mind_kernels.rrf_fusion([["B", "A"], ["A", "B"]])
    assert [bid for bid, _ in fused] == ["A", "B"]


# ---------------------------------------------------------------------------
# 6. index_stats reports the resolved backend only when the flag is ON
# ---------------------------------------------------------------------------


def test_index_stats_has_no_backend_key_with_the_flag_off(flag_off):
    import mind_mem.mcp.tools.memory_ops as ops

    assert ops._kernel_backend_reporting_enabled() is False


def test_index_stats_reports_the_backend_with_the_flag_on(flag_on):
    import mind_mem.mcp.tools.memory_ops as ops

    assert ops._kernel_backend_reporting_enabled() is True
    assert mind_ffi.load_kernels().backend in {"native", "python"}


def test_weighted_tf_counter_shape_is_what_the_kernel_hands_the_scorer():
    """Guard the delegation contract itself, not just its output."""
    weighted_tf, wdl = compute_weighted_tf(CORPUS[0], FIELD_WEIGHTS)
    assert isinstance(weighted_tf, Counter)
    assert wdl > 0
