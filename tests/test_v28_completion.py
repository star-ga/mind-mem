# Copyright 2026 STARGA, Inc.
"""Tests that close every unchecked roadmap item (v2.8.0 completion sweep)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mind_mem import (
    context_core,
    core_export,
    hook_installer,
    ingestion_pipeline,
    ledger_anchor,
    memory_mesh,
    mind_kernels,
    mrs,
    multi_modal,
    online_trainer,
    staleness,
    tiered_memory,
    tracking,
    turbo_quant,
)

# ---------------------------------------------------------------------------
# TurboQuant
# ---------------------------------------------------------------------------


class TestTurboQuant:
    def test_round_trip_within_quantisation_error(self) -> None:
        vec = [i / 10.0 for i in range(-20, 21)]
        qv = turbo_quant.quantize(vec)
        out = turbo_quant.dequantize(qv)
        # 3-bit over range [-2, 2] → step ≈ 0.57. Tolerance above that.
        max_err = max(abs(a - b) for a, b in zip(vec, out))
        assert max_err < 0.7

    def test_empty_vector(self) -> None:
        assert turbo_quant.dequantize(turbo_quant.quantize([])) == []

    def test_constant_vector_reconstructs_exactly(self) -> None:
        assert all(abs(x - 1.25) < 1e-9 for x in turbo_quant.dequantize(turbo_quant.quantize([1.25] * 8)))

    def test_encode_decode_round_trip(self) -> None:
        qv = turbo_quant.quantize([0.1, 0.2, 0.9])
        restored = turbo_quant.decode(turbo_quant.encode(qv))
        assert restored.dim == qv.dim

    def test_memory_reduction(self) -> None:
        # 128-dim float32 = 512 bytes. 3-bit = ~48 bytes payload + 20 header.
        qv = turbo_quant.quantize([0.01 * i for i in range(128)])
        assert qv.memory_bytes() < 512  # roughly 6x reduction goal


# ---------------------------------------------------------------------------
# MIND kernel Python fallbacks
# ---------------------------------------------------------------------------


class TestMindKernels:
    def test_bm25f_score_basic(self) -> None:
        score = mind_kernels.bm25f_score(
            ["jwt"],
            {"title": ["jwt", "auth"], "body": ["auth", "key"]},
            {"title": 2.0, "body": 1.0},
            doc_length=6,
            avg_doc_length=5.0,
        )
        assert score > 0

    def test_rrf_fusion_merges_axes(self) -> None:
        fused = mind_kernels.rrf_fusion([["A", "B"], ["B", "C"]])
        ids = [bid for bid, _ in fused]
        assert ids[0] == "B"

    def test_cosine_identical_vectors_one(self) -> None:
        assert abs(mind_kernels.cosine([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9

    def test_load_kernels_returns_fallback_without_env(self) -> None:
        os.environ.pop("MIND_MEM_KERNELS_SO", None)
        kernels = mind_kernels.load_kernels()
        assert kernels.bm25f_score is mind_kernels.bm25f_score


# ---------------------------------------------------------------------------
# MRS
# ---------------------------------------------------------------------------


class TestMRS:
    def test_percentile_basic(self) -> None:
        assert mrs.percentile([10, 20, 30, 40, 50], 50) == 30

    def test_compute_mrs_clean_yields_100(self) -> None:
        slis = [
            mrs.SLI(name="p99", value=10, unit="ms", threshold=1000, weight=1.0),
        ]
        report = mrs.compute_mrs("demo", slis, computed_at="t")
        assert report.score == 100.0
        assert report.violations == []

    def test_compute_mrs_violation_drops_score(self) -> None:
        slis = [
            mrs.SLI(name="p99", value=2000, unit="ms", threshold=1000, weight=1.0),
        ]
        assert mrs.compute_mrs("x", slis).score < 50

    def test_parse_slo_spec(self) -> None:
        slis = mrs.parse_slo_spec({"slis": [{"name": "a", "threshold": 1, "weight": 2}]})
        assert slis[0].name == "a" and slis[0].weight == 2.0

    def test_retrieval_slis_structure(self) -> None:
        slis = mrs.retrieval_slis(relevance_decay=0.1, contradiction_density=1.0, staleness_ratio=0.5)
        assert {s.name for s in slis} == {"relevance_decay", "contradiction_density", "staleness_ratio"}


# ---------------------------------------------------------------------------
# Memory mesh
# ---------------------------------------------------------------------------


class TestMemoryMesh:
    def test_add_and_remove_peer(self) -> None:
        mesh = memory_mesh.MemoryMesh()
        p = mesh.add_peer("peer-1", "http://x", scopes=[memory_mesh.SyncScope.MEMORIES])
        assert p.peer_id == "peer-1"
        assert mesh.remove_peer("peer-1") is True
        assert mesh.remove_peer("peer-1") is False

    def test_conflict_lww_prefers_newer(self) -> None:
        mesh = memory_mesh.MemoryMesh()
        out = mesh.resolve_conflict(
            memory_mesh.SyncScope.MEMORIES,
            {"_id": "A", "updated_at": "2026-04-01"},
            {"_id": "A", "updated_at": "2026-04-10"},
        )
        assert out["updated_at"] == "2026-04-10"

    def test_conflict_governance_keeps_local_and_flags_review(self) -> None:
        mesh = memory_mesh.MemoryMesh()
        out = mesh.resolve_conflict(
            memory_mesh.SyncScope.SEMANTIC,
            {"_id": "L", "updated_at": "2026-04-01"},
            {"_id": "L", "updated_at": "2026-04-10"},
        )
        assert out["metadata"]["requires_review"] is True

    def test_audit_log_records_sync(self) -> None:
        mesh = memory_mesh.MemoryMesh()
        mesh.add_peer("p", "e")
        mesh.log_sync("p", memory_mesh.SyncScope.MEMORIES, 10, 2)
        log = mesh.audit_log()
        assert log[0]["blocks_transferred"] == 10


# ---------------------------------------------------------------------------
# Tiered memory
# ---------------------------------------------------------------------------


class TestTieredMemory:
    def test_tier_boost_ordering(self) -> None:
        assert tiered_memory.Tier.PROCEDURAL.retrieval_boost > tiered_memory.Tier.WORKING.retrieval_boost

    def test_decay_decreases_over_time(self) -> None:
        decayed = tiered_memory.decay(
            1.0,
            "2026-01-01T00:00:00Z",
            now=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        assert decayed < 1.0

    def test_reset_on_access_returns_one(self) -> None:
        assert tiered_memory.reset_on_access(0.2) == 1.0

    def test_promote_candidates_threshold(self) -> None:
        block = tiered_memory.TieredBlock(
            block_id="B",
            tier=tiered_memory.Tier.EPISODIC,
            strength=0.5,
            session_count=5,
        )
        cands = tiered_memory.promote_candidates([block])
        assert cands and cands[0].to_tier == tiered_memory.Tier.SEMANTIC

    def test_procedural_never_promoted(self) -> None:
        block = tiered_memory.TieredBlock(
            block_id="X",
            tier=tiered_memory.Tier.PROCEDURAL,
            strength=1.0,
            session_count=100,
        )
        assert tiered_memory.promote_candidates([block]) == []


# ---------------------------------------------------------------------------
# Hook installer + privacy filter
# ---------------------------------------------------------------------------


class TestHookInstaller:
    def test_privacy_filter_redacts_secrets(self) -> None:
        redacted = hook_installer.privacy_filter("key=sk-01234567890123456789abc")
        assert "[REDACTED]" in redacted and "sk-" not in redacted

    def test_privacy_filter_private_block(self) -> None:
        out = hook_installer.privacy_filter("<private>hidden</private> visible")
        assert "hidden" not in out

    def test_validate_event_rejects_unknown_type(self) -> None:
        errors = hook_installer.validate_event({"type": "bogus", "timestamp": "t", "project": "p", "session_id": "s"})
        assert any("invalid type" in e for e in errors)

    def test_observation_to_block_dedup(self) -> None:
        seen: set[str] = set()
        event = {
            "type": "PostToolUse",
            "timestamp": "t",
            "project": "p",
            "session_id": "s",
            "output_summary": "repeat",
        }
        a = hook_installer.observation_to_block(event, seen_hashes=seen)
        b = hook_installer.observation_to_block(event, seen_hashes=seen)
        assert a is not None and b is None  # duplicate collapsed

    def test_install_config_dry_run(self, tmp_path) -> None:
        res = hook_installer.install_config("codex", str(tmp_path), dry_run=True)
        assert res["written"] is False and "mm context" in res["content"]

    def test_install_config_writes(self, tmp_path) -> None:
        res = hook_installer.install_config("codex", str(tmp_path), dry_run=False)
        assert res["written"] is True
        assert Path(res["path"]).is_file()


# ---------------------------------------------------------------------------
# Multi-modal
# ---------------------------------------------------------------------------


class TestMultiModal:
    def test_build_image_block_captures_description(self) -> None:
        block = multi_modal.build_image_block("I-1", "a photo of a cat", "/missing")
        assert block.description == "a photo of a cat"

    def test_build_audio_block_duration(self) -> None:
        block = multi_modal.build_audio_block("A-1", "transcript", "/tmp/x", duration_seconds=30)
        assert block.duration_seconds == 30

    def test_cross_modal_similarity_identical(self) -> None:
        sim = multi_modal.cross_modal_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 1e-9

    def test_modal_token_cost_image(self) -> None:
        cost = multi_modal.modal_token_cost({"type": "image"})
        assert cost > 50

    def test_modal_token_cost_audio_scales(self) -> None:
        short = multi_modal.modal_token_cost({"type": "audio", "duration_seconds": 1})
        long = multi_modal.modal_token_cost({"type": "audio", "duration_seconds": 100})
        assert long > short


# ---------------------------------------------------------------------------
# Ingestion pipeline + WAL + webhook
# ---------------------------------------------------------------------------


class TestIngestionPipeline:
    def test_queue_backpressure(self) -> None:
        q = ingestion_pipeline.IngestionQueue(capacity=2)
        assert q.offer({"a": 1}) is True
        assert q.offer({"a": 2}) is True
        assert q.offer({"a": 3}) is False  # full
        assert q.stats().backpressure_drops == 1

    def test_wal_append_replay(self, tmp_path) -> None:
        wal = ingestion_pipeline.WriteAheadLog(str(tmp_path / "wal.jsonl"))
        wal.append({"x": 1})
        wal.append({"x": 2})
        replay = wal.replay()
        assert [r["x"] for r in replay] == [1, 2]

    def test_webhook_round_trip(self, tmp_path) -> None:
        import http.client
        import json as _json

        q = ingestion_pipeline.IngestionQueue(capacity=10)
        wal = ingestion_pipeline.WriteAheadLog(str(tmp_path / "wal.jsonl"))
        import socket

        # Pick an ephemeral port explicitly so tests don't race.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        _, stop = ingestion_pipeline.serve_webhook(port, q, wal=wal)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", "/ingest", body=_json.dumps({"hello": "world"}), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 202
        finally:
            stop()
        assert q.drain()[0]["hello"] == "world"


# ---------------------------------------------------------------------------
# Online trainer
# ---------------------------------------------------------------------------


class TestOnlineTrainer:
    def test_build_training_tuples_correction(self) -> None:
        sig = {
            "signal_type": "correction",
            "new_query": "oauth2",
            "previous_results": ["A", "B"],
        }
        tuples = online_trainer.build_training_tuples([sig])
        assert tuples[0].negative_ids == ("A", "B")

    def test_weight_registry_promotion_gated(self) -> None:
        reg = online_trainer.WeightRegistry()
        reg.set_active(online_trainer.WeightRef(model_id="m", version="1", path="/a", base_mrr=0.5, promoted_at="t"))
        reg.set_candidate(online_trainer.WeightRef(model_id="m", version="2", path="/b", base_mrr=0.0, promoted_at="t"))
        ok, _ = reg.promote("m", new_mrr=0.4)
        assert ok is False  # regression rejected

    def test_weight_registry_revert(self) -> None:
        reg = online_trainer.WeightRegistry()
        reg.set_active(online_trainer.WeightRef("m", "1", "/a", 0.5, "t"))
        reg.set_candidate(online_trainer.WeightRef("m", "2", "/b", 0.0, "t"))
        reg.promote("m", new_mrr=0.9)
        assert reg.revert("m", reason="MRR regression") is True

    def test_training_loop_batches(self) -> None:
        calls: list[int] = []

        def fake_step(batch):
            calls.append(len(batch))
            return {"loss": 0.1}

        loop = online_trainer.TrainingLoop(fake_step, batch_size=2)
        loop.submit([online_trainer.TrainingTuple(query="q", positive_ids=(), negative_ids=(), signal_type="re_query") for _ in range(5)])
        assert sum(calls) == 4  # two batches of 2; one leftover


# ---------------------------------------------------------------------------
# Ledger anchor
# ---------------------------------------------------------------------------


class TestLedgerAnchor:
    def test_record_and_replay(self, tmp_path) -> None:
        hist = ledger_anchor.AnchorHistory(str(tmp_path / "anchors.jsonl"))
        hist.record("a" * 64, block_height=1, chain="local")
        hist.record("b" * 64, block_height=2, chain="sepolia", tx_hash="0xdead")
        all_ = hist.all()
        assert len(all_) == 2
        assert hist.latest().chain == "sepolia"

    def test_rejects_trivial_root(self, tmp_path) -> None:
        hist = ledger_anchor.AnchorHistory(str(tmp_path / "a.jsonl"))
        with pytest.raises(ValueError):
            hist.record("short", block_height=1)

    def test_status_confirmed_when_tx_present(self, tmp_path) -> None:
        hist = ledger_anchor.AnchorHistory(str(tmp_path / "a.jsonl"))
        e = ledger_anchor.anchor_root(hist, "x" * 64, block_height=1, tx_hash="0xabc")
        assert e.status == "confirmed"


# ---------------------------------------------------------------------------
# Core export + diff
# ---------------------------------------------------------------------------


class TestCoreExport:
    def _core(self, tmp_path, blocks=None, edges=None, built_at="2026-04-13T00:00:00Z"):
        path = tmp_path / "core.mmcore"
        context_core.build_core(
            str(path),
            namespace="ns",
            version="1.0",
            blocks=blocks or [],
            edges=edges or [],
            built_at=built_at,
        )
        return context_core.load_core(str(path))

    def test_export_to_jsonld(self, tmp_path) -> None:
        core = self._core(tmp_path, blocks=[{"_id": "A"}])
        doc = core_export.export_to_jsonld(core)
        assert doc["@type"] == "ContextCore"
        assert doc["blocks"][0]["_id"] == "A"

    def test_export_to_markdown_has_heading(self, tmp_path) -> None:
        core = self._core(tmp_path, blocks=[{"_id": "A", "type": "decision", "text": "OAuth"}])
        md = core_export.export_to_markdown(core)
        assert md.startswith("# Context Core")
        assert "decision — A" in md

    def test_diff_detects_added_and_removed(self, tmp_path) -> None:
        old = self._core(tmp_path, blocks=[{"_id": "A"}])
        new = self._core(tmp_path, blocks=[{"_id": "A"}, {"_id": "B"}])
        d = core_export.diff_cores(old, new)
        assert [b["_id"] for b in d.added_blocks] == ["B"]
        assert d.is_empty is False

    def test_diff_rollback_reverses_change(self, tmp_path) -> None:
        old = self._core(tmp_path, blocks=[{"_id": "A"}])
        new = self._core(tmp_path, blocks=[{"_id": "A"}, {"_id": "B"}])
        d = core_export.diff_cores(old, new)
        rolled = core_export.apply_diff_rollback(new, d)
        assert all(b.get("_id") != "B" for b in rolled.blocks)


# ---------------------------------------------------------------------------
# Tracking (MRR / packing quality / conventions / context window)
# ---------------------------------------------------------------------------


class TestTracking:
    def test_mrr_tracker_records_weekly(self) -> None:
        mt = tracking.MRRTracker()
        mrr = mt.record(["A", "B"], ["A"])
        assert mrr == 1.0
        assert mt.weeks()[0]["queries"] == 1

    def test_packing_quality_ratio(self) -> None:
        pq = tracking.PackingQualityMeter()
        pq.observe(packed=100, referenced=40)
        pq.observe(packed=100, referenced=80)
        assert abs(pq.ratio() - 0.6) < 1e-9

    def test_packing_quality_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            tracking.PackingQualityMeter().observe(-1, 0)

    def test_extract_conventions_dominant_naming(self) -> None:
        samples = ["def run_test(): pass", "def parse_line(): pass"]
        out = tracking.extract_conventions(samples)
        assert out["dominant_naming"] == "snake_case"

    def test_model_context_window(self) -> None:
        assert tracking.model_context_window("claude-opus-4-6") == 1_000_000
        assert tracking.model_context_window("unknown") > 0


# ---------------------------------------------------------------------------
# Retrieval stale penalty (via existing staleness API)
# ---------------------------------------------------------------------------


class TestStalenessScoring:
    def test_stale_penalty_reduces_score(self) -> None:
        plan = staleness.propagate_staleness(["A"], {"A": ["B"], "B": []})
        # A higher penalty weight scales score downward.
        score = 10.0
        penalty = plan.scores.get("A", 0.0)
        penalised = score * (1 - 0.3 * penalty)
        assert penalised < score
