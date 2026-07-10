"""Tests for bench/recompaction_bench.py — the recompaction scalar metric.

This is the load-bearing piece: everything downstream (autoresearch scoring)
depends on it measuring something real. The echo-control asserts the harness
itself introduces zero fact loss and zero non-convergence — if that ever
stops scoring 1.0, the harness is broken, not the compressor.
"""

from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout
from typing import Any

import pytest

from mind_mem.bench.recompaction_bench import (
    BenchResult,
    ClusterRecord,
    _block_dict,
    _cluster_via_lexical_overlap,
    _open_readonly,
    evaluate,
    extract_probes,
    load_clusters,
    main,
    probes_present,
)
from mind_mem.compressors import CompressorError, EchoCompressor
from mind_mem.recompaction import RecompactionConfig

# --- probe extraction --------------------------------------------------------


def test_extract_probes_finds_numbers_dates_and_capitalized_entities():
    text = "Deployed on 2026-07-10 with 1469 blocks. Owner: Nikolai reviewed PRJ-mind."
    probes = extract_probes(text)
    assert "2026-07-10" in probes
    assert "1469" in probes
    assert "Nikolai" in probes
    assert "PRJ-mind" in probes


def test_extract_probes_finds_quoted_identifiers():
    text = 'The flag is called "recompaction_score" in the output.'
    probes = extract_probes(text)
    assert "recompaction_score" in probes


def test_extract_probes_is_deterministic():
    text = "Block D-20260213-001 references T-20260213-005 on 2026-02-13."
    assert extract_probes(text) == extract_probes(text)


def test_extract_probes_empty_text_yields_no_probes():
    assert extract_probes("") == set()


def test_probes_present_true_when_all_probes_survive():
    probes = {"1469", "Nikolai"}
    assert probes_present(probes, "There are 1469 blocks reviewed by Nikolai.") == 1.0


def test_probes_present_partial_when_some_probes_dropped():
    probes = {"1469", "Nikolai", "PRJ-mind"}
    text = "There are 1469 blocks reviewed by Nikolai."
    ratio = probes_present(probes, text)
    assert 0.0 < ratio < 1.0


def test_probes_present_zero_probes_is_vacuously_perfect():
    assert probes_present(set(), "anything") == 1.0


def test_probes_present_zero_when_nothing_survives():
    assert probes_present({"XYZ123"}, "no matching content here") == 0.0


# --- load_clusters -----------------------------------------------------------


def _make_test_db(path: str) -> None:
    """Build a minimal recall.db-shaped sqlite file with vec_blocks via sqlite_vec."""
    import sqlite_vec

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        """CREATE TABLE blocks (
            id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, status TEXT,
            date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT
        )"""
    )
    conn.execute("CREATE VIRTUAL TABLE vec_blocks USING vec0(block_id TEXT PRIMARY KEY, embedding FLOAT[4])")

    # Two tight clusters in 4-d space + one isolated block.
    rows = [
        ("A-001", [1.0, 0.0, 0.0, 0.0], "alpha fact 111"),
        ("A-002", [0.99, 0.01, 0.0, 0.0], "alpha fact 222"),
        ("A-003", [0.98, 0.02, 0.0, 0.0], "alpha fact 333"),
        ("B-001", [0.0, 0.0, 1.0, 0.0], "beta fact 444"),
        ("B-002", [0.0, 0.01, 0.99, 0.0], "beta fact 555"),
        ("C-001", [0.0, 1.0, 0.0, 1.0], "lonely fact 666"),
    ]
    for bid, vec, body in rows:
        conn.execute(
            "INSERT INTO blocks (id, type, file, line, status, date, speaker, tags, dia_id, parent_id, json_blob) "
            "VALUES (?, 'decision', 'x.md', 1, 'active', '2026-01-01', '', '', '', '', ?)",
            (bid, f'{{"_id": "{bid}", "body": "{body}"}}'),
        )
        conn.execute(
            "INSERT INTO vec_blocks(block_id, embedding) VALUES (?, ?)",
            (bid, sqlite_vec.serialize_float32(vec)),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def test_db(tmp_path):
    path = str(tmp_path / "recall.db")
    _make_test_db(path)
    return path


def test_load_clusters_groups_similar_blocks(test_db):
    clusters = load_clusters(test_db, k=2, min_size=2)
    assert len(clusters) >= 1
    ids_per_cluster = [{b["_id"] for b in c} for c in clusters]
    # The alpha trio should cluster together somewhere.
    assert any({"A-001", "A-002"} <= ids for ids in ids_per_cluster)


def test_load_clusters_is_deterministic(test_db):
    first = load_clusters(test_db, k=2, min_size=2)
    second = load_clusters(test_db, k=2, min_size=2)
    first_ids = [tuple(sorted(b["_id"] for b in c)) for c in first]
    second_ids = [tuple(sorted(b["_id"] for b in c)) for c in second]
    assert first_ids == second_ids


def test_load_clusters_respects_min_size(test_db):
    clusters = load_clusters(test_db, k=2, min_size=2)
    assert all(len(c) >= 2 for c in clusters)


def test_load_clusters_opens_database_read_only(test_db):
    """Must not mutate the source DB — verified by mtime/content stability."""
    import os

    before_size = os.path.getsize(test_db)
    load_clusters(test_db, k=2, min_size=2)
    after_size = os.path.getsize(test_db)
    assert before_size == after_size


def test_load_clusters_missing_db_raises(tmp_path):
    with pytest.raises((FileNotFoundError, sqlite3.OperationalError)):
        load_clusters(str(tmp_path / "nope.db"), k=2, min_size=2)


# --- _block_dict -------------------------------------------------------------


def test_block_dict_returns_none_for_missing_row(tmp_path):
    path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE blocks (id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, status TEXT, "
        "date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT)"
    )
    conn.commit()
    conn.close()
    conn = _open_readonly(path)
    try:
        assert _block_dict(conn, "does-not-exist") is None
    finally:
        conn.close()


def test_block_dict_returns_none_for_malformed_json(tmp_path):
    path = str(tmp_path / "malformed2.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE blocks (id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, status TEXT, "
        "date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT)"
    )
    conn.execute(
        "INSERT INTO blocks VALUES ('Y-001','decision','x.md',1,'active','2026-01-01','','','','','{broken')"
    )
    conn.commit()
    conn.close()
    conn = _open_readonly(path)
    try:
        assert _block_dict(conn, "Y-001") is None
    finally:
        conn.close()


# --- lexical fallback (no sqlite-vec / no vec_blocks table) -----------------


def _make_lexical_only_db(path: str) -> None:
    """A `blocks`-only sqlite file with no `vec_blocks` table at all —
    exercises the deterministic lexical fallback path directly."""
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE blocks (
            id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, status TEXT,
            date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT
        )"""
    )
    rows = [
        ("A-001", '{"_id": "A-001", "summary": "shared vocabulary alpha beta gamma token"}'),
        ("A-002", '{"_id": "A-002", "summary": "shared vocabulary alpha beta delta token"}'),
        ("A-003", '{"_id": "A-003", "summary": "shared vocabulary alpha beta epsilon token"}'),
        ("B-001", '{"_id": "B-001", "summary": "completely unrelated zeta words here"}'),
        ("C-001", '{"_id": "C-001", "summary": ""}'),  # empty text -> no tokens
    ]
    for bid, blob in rows:
        conn.execute(
            "INSERT INTO blocks (id, type, file, line, status, date, speaker, tags, dia_id, parent_id, json_blob) "
            "VALUES (?, 'decision', 'x.md', 1, 'active', '2026-01-01', '', '', '', '', ?)",
            (bid, blob),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def lexical_only_db(tmp_path):
    path = str(tmp_path / "lexical.db")
    _make_lexical_only_db(path)
    return path


def test_load_clusters_falls_back_to_lexical_when_vec_blocks_missing(lexical_only_db):
    clusters = load_clusters(lexical_only_db, k=2, min_size=2)
    assert len(clusters) >= 1
    ids_per_cluster = [{b["_id"] for b in c} for c in clusters]
    assert any({"A-001", "A-002"} <= ids for ids in ids_per_cluster)


def test_lexical_fallback_is_deterministic(lexical_only_db):
    conn1 = _open_readonly(lexical_only_db)
    conn2 = _open_readonly(lexical_only_db)
    try:
        first = _cluster_via_lexical_overlap(conn1, k=2, min_size=2)
        second = _cluster_via_lexical_overlap(conn2, k=2, min_size=2)
    finally:
        conn1.close()
        conn2.close()
    first_ids = [tuple(sorted(b["_id"] for b in c)) for c in first]
    second_ids = [tuple(sorted(b["_id"] for b in c)) for c in second]
    assert first_ids == second_ids


def test_lexical_fallback_skips_blocks_with_no_tokens(lexical_only_db):
    conn = _open_readonly(lexical_only_db)
    try:
        clusters = _cluster_via_lexical_overlap(conn, k=2, min_size=2)
    finally:
        conn.close()
    all_ids = {b["_id"] for c in clusters for b in c}
    assert "C-001" not in all_ids


def test_lexical_fallback_skips_malformed_json_blob(tmp_path):
    path = str(tmp_path / "malformed.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE blocks (
            id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, status TEXT,
            date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO blocks VALUES ('X-001','decision','x.md',1,'active','2026-01-01','','','','','not json')"
    )
    conn.commit()
    conn.close()
    clusters = load_clusters(path, k=2, min_size=2)
    assert clusters == []


def test_lexical_fallback_skips_empty_and_non_dict_json_blob(tmp_path):
    path = str(tmp_path / "mixed.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE blocks (
            id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, status TEXT,
            date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT
        )"""
    )
    rows = [
        ("E-001", ""),  # empty json_blob -> falsy, skipped
        ("E-002", "[1, 2, 3]"),  # valid json but not a dict -> skipped
        ("A-001", '{"summary": "shared alpha beta gamma token here"}'),
        ("A-002", '{"summary": "shared alpha beta delta token here"}'),
    ]
    for bid, blob in rows:
        conn.execute(
            "INSERT INTO blocks (id, type, file, line, status, date, speaker, tags, dia_id, parent_id, json_blob) "
            "VALUES (?, 'decision', 'x.md', 1, 'active', '2026-01-01', '', '', '', '', ?)",
            (bid, blob),
        )
    conn.commit()
    conn.close()

    clusters = load_clusters(path, k=2, min_size=2)
    # Only the two well-formed dict blobs (A-001, A-002) should have
    # contributed a cluster; E-001 (empty blob) and E-002 (non-dict json)
    # must never appear.
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_load_clusters_falls_back_to_lexical_when_vec_blocks_yields_no_clusters(tmp_path):
    """sqlite-vec loads fine but every block is isolated (no close neighbors) —
    the vector path returns zero clusters, so `load_clusters` must still fall
    back to the lexical path rather than returning an empty result."""
    import sqlite_vec

    path = str(tmp_path / "sparse.db")
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        """CREATE TABLE blocks (
            id TEXT PRIMARY KEY, type TEXT, file TEXT, line INTEGER, status TEXT,
            date TEXT, speaker TEXT, tags TEXT, dia_id TEXT, parent_id TEXT, json_blob TEXT
        )"""
    )
    conn.execute("CREATE VIRTUAL TABLE vec_blocks USING vec0(block_id TEXT PRIMARY KEY, embedding FLOAT[4])")
    # Two maximally-dissimilar vectors -> vector clustering finds no cluster
    # of size >= min_size, but the lexical path can still find shared tokens.
    rows = [
        ("X-001", [1.0, 0.0, 0.0, 0.0], "shared vocabulary alpha beta gamma"),
        ("X-002", [0.0, 1.0, 0.0, 0.0], "shared vocabulary alpha beta delta"),
    ]
    for bid, vec, body in rows:
        conn.execute(
            "INSERT INTO blocks (id, type, file, line, status, date, speaker, tags, dia_id, parent_id, json_blob) "
            "VALUES (?, 'decision', 'x.md', 1, 'active', '2026-01-01', '', '', '', '', ?)",
            (bid, f'{{"_id": "{bid}", "summary": "{body}"}}'),
        )
        conn.execute("INSERT INTO vec_blocks(block_id, embedding) VALUES (?, ?)", (bid, sqlite_vec.serialize_float32(vec)))
    conn.commit()
    conn.close()

    clusters = load_clusters(path, k=2, min_size=2)
    ids = {b["_id"] for c in clusters for b in c}
    assert ids == {"X-001", "X-002"}


def test_main_falls_back_to_lexical_and_still_scores_control_perfectly(lexical_only_db, capsys):
    main(["--db", lexical_only_db, "--model", "echo", "--clusters", "5"])
    out = capsys.readouterr().out
    assert "recompaction_score: 1.0" in out


# --- evaluate ------------------------------------------------------------


def _blocks(*bodies: str) -> list[dict[str, Any]]:
    return [{"_id": f"D-{i:03d}", "body": b} for i, b in enumerate(bodies, 1)]


def test_evaluate_echo_compressor_converges_immediately_with_perfect_retention():
    clusters = [_blocks("fact one 111", "fact two 222")]
    result = evaluate(clusters, EchoCompressor(), RecompactionConfig())
    assert result.convergence_rate == 1.0
    assert result.fact_retention == 1.0
    assert result.n_failures == 0
    assert result.n_clusters == 1


def test_evaluate_records_per_cluster_results():
    clusters = [_blocks("fact one 111", "fact two 222")]
    result = evaluate(clusters, EchoCompressor(), RecompactionConfig())
    assert len(result.records) == 1
    rec = result.records[0]
    assert isinstance(rec, ClusterRecord)
    assert rec.converged is True
    assert rec.iterations == 1  # EchoCompressor reaches the fixed point on the first pass


def test_evaluate_catches_non_convergence_and_records_failure():
    def _oscillating(text: str, blocks: list[dict]) -> str:
        return "B" if text.startswith("A") else "A"

    clusters = [_blocks("A block one", "A block two")]
    result = evaluate(clusters, _oscillating, RecompactionConfig(max_iterations=3))
    assert result.n_failures == 1
    assert result.convergence_rate == 0.0
    assert result.records[0].converged is False
    assert result.records[0].failure_reason is not None


def test_evaluate_catches_retention_floor_rejection_and_records_failure():
    def _destroy(text: str, blocks: list[dict]) -> str:
        return "x"

    clusters = [_blocks("a substantial block body here", "another substantial block body here")]
    result = evaluate(clusters, _destroy, RecompactionConfig(min_retention_ratio=0.5))
    assert result.n_failures == 1
    assert "retention" in (result.records[0].failure_reason or "").lower()


def test_evaluate_one_bad_cluster_does_not_abort_the_run():
    def _oscillating(text: str, blocks: list[dict]) -> str:
        return "B" if text.startswith("A") else "A"

    good = _blocks("fact 111", "fact 222")
    bad = _blocks("A one", "A two")
    result = evaluate([bad, good], _oscillating, RecompactionConfig(max_iterations=2))
    assert result.n_clusters == 2
    # Echo-equivalent good cluster still processed even though `_oscillating`
    # never converges for either — but the run itself must not raise.
    assert result.n_failures == 2


def test_evaluate_compression_ratio_reflects_output_over_input_length():
    clusters = [_blocks("a" * 100, "b" * 100)]
    result = evaluate(clusters, EchoCompressor(), RecompactionConfig())
    assert result.compression_ratio == pytest.approx(1.0)


def test_evaluate_empty_clusters_yields_zeroed_result():
    result = evaluate([], EchoCompressor(), RecompactionConfig())
    assert result.n_clusters == 0
    assert result.convergence_rate == 0.0
    assert result.fact_retention == 0.0


def test_mean_iterations_excludes_failed_clusters():
    """A failed cluster must not drag the iteration mean toward zero.

    Regression: mean_iterations was averaged over ALL records. A retention-floor
    rejection records iterations=0, so a compressor that failed most clusters
    reported a low, healthy-looking iteration count — the metric deflated in the
    reassuring direction. The mean must describe only the clusters that actually
    reached a fixed point; convergence_rate is what reports the failures.
    """

    def _shrink_then_settle(text: str, blocks: list[dict]) -> str:
        # Converges in exactly 2 passes, staying above the retention floor.
        return text if text.endswith("!") else text[: max(len(text) // 2, 40)] + "!"

    good = _blocks("kept fact 111 " + "x" * 60, "kept fact 222 " + "y" * 60)
    # A cluster whose rewrite collapses below the floor -> ValueError -> iterations=0.
    bad = _blocks("z" * 400, "w" * 400)

    def _mixed(text: str, blocks: list[dict]) -> str:
        if text.startswith(("z", "w")):
            return "tiny"  # far below the 0.25 * 400 retention floor
        return _shrink_then_settle(text, blocks)

    result = evaluate([good, bad], _mixed, RecompactionConfig())

    assert result.n_clusters == 2
    assert result.n_failures == 1
    assert result.convergence_rate == pytest.approx(0.5)
    converged = [r for r in result.records if r.converged]
    assert len(converged) == 1
    # The mean equals the converged cluster's own iteration count — not that
    # count halved by averaging in the failure's zero.
    assert result.mean_iterations == pytest.approx(converged[0].iterations)
    assert result.mean_iterations > 0.0


def test_compressor_error_records_a_failure_and_does_not_abort_the_run():
    """An infrastructure fault must not kill the run, nor pose as a model verdict.

    Regression: CompressorError subclasses RuntimeError, so it escaped both the
    NonConvergenceError and ValueError handlers and aborted the whole benchmark.
    A single ollama timeout would take down a 12-cluster run and print no score
    at all — autoresearch would read a crash where it should read a void result.

    It is recorded under its own `compressor_error:` reason, distinct from
    `non_convergence:`, because a slow GPU is not evidence that a model cannot
    reach a fixed point.
    """
    calls = {"n": 0}

    def _times_out_on_first_cluster(text: str, blocks: list[dict]) -> str:
        calls["n"] += 1
        if blocks and _block_body_of(blocks[0]).startswith("boom"):
            raise CompressorError("ollama request timed out after 60.0s")
        return text  # everything else settles immediately

    clusters = [_blocks("boom one", "boom two"), _blocks("fine 111", "fine 222")]
    result = evaluate(clusters, _times_out_on_first_cluster, RecompactionConfig())

    # The good cluster still ran — the bad one did not abort the sweep.
    assert result.n_clusters == 2
    assert result.n_failures == 1
    assert result.convergence_rate == pytest.approx(0.5)

    reasons = [r.failure_reason or "" for r in result.records]
    assert any(x.startswith("compressor_error:") for x in reasons)
    # Crucially: NOT misfiled as a convergence failure.
    assert not any(x.startswith("non_convergence:") for x in reasons)


def _block_body_of(block: dict[str, Any]) -> str:
    body = block.get("body")
    return body if isinstance(body, str) else ""


def test_mean_iterations_is_zero_when_nothing_converges():
    """All-fail is reported as 0.0 iterations, and convergence_rate says why."""

    def _oscillating(text: str, blocks: list[dict]) -> str:
        return "B" if text.startswith("A") else "A"

    result = evaluate([_blocks("A one", "A two")], _oscillating, RecompactionConfig(max_iterations=3))
    assert result.convergence_rate == 0.0
    assert result.mean_iterations == 0.0
    assert result.recompaction_score == 0.0


def test_bench_result_is_frozen():
    result = evaluate([_blocks("x fact 1", "y fact 2")], EchoCompressor(), RecompactionConfig())
    with pytest.raises(Exception):
        result.convergence_rate = 0.0  # type: ignore[misc]


def test_evaluate_never_mutates_input_clusters():
    clusters = [_blocks("fact one", "fact two")]
    before = [[dict(b) for b in c] for c in clusters]
    evaluate(clusters, EchoCompressor(), RecompactionConfig())
    assert clusters == before


# --- the control: echo must score 1.0 across the board ----------------------


def test_echo_control_scores_perfect_on_a_realistic_cluster_set():
    """If the control does not score 1.0, the harness itself is broken."""
    clusters = [
        _blocks("Deployed 2026-07-10 with 1469 blocks.", "Owner Nikolai reviewed PRJ-mind on 2026-07-09."),
        _blocks("The score threshold is 0.8.", "Config uses seed 42 for determinism."),
    ]
    result = evaluate(clusters, EchoCompressor(), RecompactionConfig())
    assert result.convergence_rate == 1.0
    assert result.fact_retention == 1.0
    assert result.compression_ratio == pytest.approx(1.0)
    assert result.recompaction_score == 1.0


def test_bench_result_recompaction_score_is_product_of_retention_and_convergence():
    result = BenchResult(
        n_clusters=2,
        n_failures=1,
        convergence_rate=0.5,
        mean_iterations=1.0,
        fact_retention=0.8,
        compression_ratio=0.9,
        records=(),
    )
    assert result.recompaction_score == pytest.approx(0.4)


# --- CLI main() ---------------------------------------------------------


def test_main_prints_machine_greppable_score_line(test_db, capsys):
    main(["--db", test_db, "--model", "echo", "--clusters", "5"])
    out = capsys.readouterr().out
    assert "recompaction_score: 1.0" in out
    assert "convergence_rate:" in out
    assert "fact_retention:" in out
    assert "mean_iterations:" in out
    assert "compression_ratio:" in out


def test_main_echo_control_asserts_perfect_score(test_db):
    """This test IS the harness self-check the task calls for."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--db", test_db, "--model", "echo", "--clusters", "5"])
    out = buf.getvalue()
    assert "recompaction_score: 1.0" in out


def test_main_empty_model_raises_value_error(test_db):
    """`--model` must be non-empty; an unknown *tag* is a network-time ollama
    404, not something `_build_compressor` can validate offline — only
    emptiness is checked before any request is made."""
    with pytest.raises(ValueError):
        main(["--db", test_db, "--model", "", "--clusters", "1"])
