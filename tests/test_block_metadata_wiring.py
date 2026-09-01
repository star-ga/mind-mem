# Copyright 2026 STARGA, Inc.
"""Acceptance gate for the ``v4.block_metadata`` wiring (restoration slice).

``v4/block_metadata.py`` shipped two surfaces — key/value tags with a TTL, and
schema-validation hooks — kept a test file of its own, and was imported by
nothing. The hooks half is the one with a missing caller rather than a missing
consumer: it is a *pre-write gate*, and the product has exactly one governed
write door for agent-authored blocks, ``propose_update``. That door validated
the statement TEXT (the quality gate) and nothing at all about the FIELDS,
even though ``block_type``, ``tags`` and every provenance value arrive from
outside the store.

This file pins the join: with the flag on, ``propose_update`` runs the
registered validator for the block kind and the workspace's controlled
vocabularies before anything reaches SIGNALS.md.

Four contracts, one class each:

1. with the flag OFF the leg does not exist — a validator that refuses
   everything and a vocabulary that forbids the value are both ignored, and
   the proposal is written exactly as before. The teeth of that comparison are
   checked by an explicit positive control, so a vacuous "it was written"
   cannot pass;
2. a registered schema validator decides, sees the real external fields, and a
   refusal writes NOTHING;
3. the workspace's controlled vocabularies decide the same way — this is the
   ``v4.vocabulary`` half reached through this call site — with ``reject``
   blocking and ``flag`` reporting-but-writing;
4. an undeclared field, or no declarations at all, restricts nothing.

Every assertion in classes 2-4 fails if the ``_v4_validate_block(...)`` call is
removed from ``mcp.tools.governance.propose_update``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.v4 import block_metadata as bm
from mind_mem.v4.block_metadata import FLAG as BM_FLAG
from mind_mem.v4.block_metadata import SchemaValidationResult
from mind_mem.v4.vocabulary import FLAG as VOCAB_FLAG

_SIGNALS = "intelligence/SIGNALS.md"

#: Long enough and specific enough to clear the deterministic quality gate, so
#: a rejection in these tests can only have come from the v4 leg.
_STATEMENT = (
    "STARGA wires the v4 block-metadata schema hooks into propose_update so field values are validated before the SIGNALS.md append."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _admin_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """propose_update is admin-scoped; the ACL gate fires before everything."""
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


@pytest.fixture(autouse=True)
def _clean_validator_registry() -> Any:
    """The validator registry is process-global — restore it between tests."""
    saved = dict(bm._validators)
    try:
        yield
    finally:
        bm._validators.clear()
        bm._validators.update(saved)


def _make_ws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    flags: dict[str, bool],
    vocabularies: dict | None = None,
) -> str:
    """An initialised workspace whose own ``mind-mem.json`` carries the flags.

    ``MIND_MEM_CONFIG`` points at that same file so the ambient probe and the
    workspace-scoped vocabulary loader read one document, not two.
    """
    ws = tmp_path / name
    ws.mkdir(parents=True)
    init(str(ws))
    cfg: dict[str, Any] = {"v4": {k: {"enabled": v} for k, v in flags.items()}}
    if vocabularies is not None:
        cfg["vocabularies"] = vocabularies
    (ws / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))
    return str(ws)


def _propose(ws: str, **kwargs: Any) -> dict[str, Any]:
    import mind_mem.mcp.tools.governance as gov

    params: dict[str, Any] = {
        "block_type": "decision",
        "statement": _STATEMENT,
        "rationale": "a written reason, required for decision proposals",
    }
    params.update(kwargs)
    with use_workspace(ws):
        return json.loads(gov.propose_update(**params))


def _signals(ws: str) -> str:
    path = Path(ws) / _SIGNALS
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _always_reject(reason: str = "refused by the test validator") -> Any:
    return lambda payload: SchemaValidationResult(ok=False, reason=reason)


def _always_accept(payload: dict[str, Any]) -> SchemaValidationResult:
    return SchemaValidationResult(ok=True)


# ===========================================================================
# 1. Flag OFF is inert
# ===========================================================================


class TestFlagOffIsInert:
    def test_off_ignores_a_rejecting_validator(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Register while the surface is on (the registrar is flag-gated too),
        # then propose against a workspace whose flag is off.
        _make_ws(tmp_path, monkeypatch, "on", flags={BM_FLAG: True})
        bm.register_schema_validator("decision", _always_reject())

        ws = _make_ws(tmp_path, monkeypatch, "off", flags={BM_FLAG: False})
        envelope = _propose(ws)
        assert envelope["status"] == "proposed"
        assert _STATEMENT[:40] in _signals(ws)

    def test_off_ignores_a_reject_vocabulary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws(
            tmp_path,
            monkeypatch,
            "off",
            flags={BM_FLAG: False, VOCAB_FLAG: False},
            vocabularies={"confidence": ["low"]},
        )
        envelope = _propose(ws, confidence="high")
        assert envelope["status"] == "proposed"
        assert _STATEMENT[:40] in _signals(ws)

    def test_the_off_comparison_has_teeth(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive control for both assertions above.

        They are of the form "the write still happened", which is what a
        completely unwired build also does. Turn the flags on against the same
        inputs and both must now be FALSE — that is what proves they were
        measuring the gate rather than measuring nothing.
        """
        ws = _make_ws(
            tmp_path,
            monkeypatch,
            "on",
            flags={BM_FLAG: True, VOCAB_FLAG: True},
            vocabularies={"confidence": ["low"]},
        )
        envelope = _propose(ws, confidence="high")
        assert envelope["error"] == "schema_validation_rejection"
        assert _signals(ws).count(_STATEMENT[:40]) == 0

        bm.register_schema_validator("decision", _always_reject())
        clean = _make_ws(tmp_path, monkeypatch, "on2", flags={BM_FLAG: True})
        assert _propose(clean)["error"] == "schema_validation_rejection"


# ===========================================================================
# 2. The schema validator decides
# ===========================================================================


class TestSchemaValidatorGate:
    def test_a_refusing_validator_blocks_the_proposal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws(tmp_path, monkeypatch, "ws", flags={BM_FLAG: True})
        bm.register_schema_validator("decision", _always_reject("decision blocks need an owner"))

        envelope = _propose(ws)
        assert envelope["error"] == "schema_validation_rejection"
        assert "decision blocks need an owner" in envelope["reason"]

    def test_a_refusal_writes_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The gate must fire BEFORE the append, not after it.

        Paired with its own positive control: the same workspace, the same
        statement, an accepting validator — and SIGNALS.md grows. Without that
        half, "the file did not change" would also pass against a workspace
        where the write silently failed for an unrelated reason.
        """
        ws = _make_ws(tmp_path, monkeypatch, "ws", flags={BM_FLAG: True})
        before = _signals(ws)

        bm.register_schema_validator("decision", _always_reject())
        assert _propose(ws)["error"] == "schema_validation_rejection"
        assert _signals(ws) == before

        bm.register_schema_validator("decision", _always_accept)
        assert _propose(ws)["status"] == "proposed"
        assert _signals(ws) != before

    def test_the_validator_sees_the_externally_supplied_fields(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A gate handed an empty payload would pass every test above."""
        ws = _make_ws(tmp_path, monkeypatch, "ws", flags={BM_FLAG: True})
        seen: list[dict[str, Any]] = []

        def _capture(payload: dict[str, Any]) -> SchemaValidationResult:
            seen.append(dict(payload))
            return SchemaValidationResult(ok=True)

        bm.register_schema_validator("task", _capture)
        envelope = _propose(
            ws,
            block_type="task",
            tags="alpha, beta",
            confidence="high",
            actor_role="reviewer",
            purpose="regression guard",
        )
        assert envelope["status"] == "proposed"
        assert len(seen) == 1
        payload = seen[0]
        assert payload["tags"] == ["alpha", "beta"]
        assert payload["confidence"] == "high"
        assert payload["actor_role"] == "reviewer"
        assert payload["purpose"] == "regression guard"
        assert payload["statement"].startswith("STARGA wires")

    def test_a_kind_with_no_validator_is_unrestricted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws(tmp_path, monkeypatch, "ws", flags={BM_FLAG: True})
        bm.register_schema_validator("task", _always_reject())
        assert _propose(ws, block_type="decision")["status"] == "proposed"

    def test_a_raising_validator_refuses_rather_than_crashing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws(tmp_path, monkeypatch, "ws", flags={BM_FLAG: True})

        def _boom(payload: dict[str, Any]) -> SchemaValidationResult:
            raise RuntimeError("validator bug")

        bm.register_schema_validator("decision", _boom)
        envelope = _propose(ws)
        assert envelope["error"] == "schema_validation_rejection"
        assert "validator_raised" in envelope["reason"]


# ===========================================================================
# 3. The workspace vocabulary decides (v4.vocabulary through this call site)
# ===========================================================================


class TestVocabularyGate:
    def test_reject_mode_blocks_an_out_of_vocabulary_tag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws(
            tmp_path,
            monkeypatch,
            "ws",
            flags={BM_FLAG: True, VOCAB_FLAG: True},
            vocabularies={"tags": {"values": ["alpha", "beta"], "mode": "reject"}},
        )
        envelope = _propose(ws, tags="alpha, gamma")
        assert envelope["error"] == "schema_validation_rejection"
        assert "'gamma'" in envelope["reason"]
        assert _signals(ws).count(_STATEMENT[:40]) == 0

        # Positive control: the in-vocabulary tag set goes through.
        assert _propose(ws, tags="alpha, beta")["status"] == "proposed"

    def test_block_kind_is_checked_implicitly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``block_kind`` is defaulted from the proposal's block_type."""
        ws = _make_ws(
            tmp_path,
            monkeypatch,
            "ws",
            flags={BM_FLAG: True, VOCAB_FLAG: True},
            vocabularies={"block_kind": ["decision"]},
        )
        assert _propose(ws, block_type="task")["error"] == "schema_validation_rejection"
        assert _propose(ws, block_type="decision")["status"] == "proposed"

    def test_flag_mode_reports_but_writes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws(
            tmp_path,
            monkeypatch,
            "ws",
            flags={BM_FLAG: True, VOCAB_FLAG: True},
            vocabularies={"tags": {"values": ["alpha"], "mode": "flag"}},
        )
        envelope = _propose(ws, tags="gamma")
        assert envelope["status"] == "proposed"
        assert _STATEMENT[:40] in _signals(ws)

    def test_vocabulary_flag_off_leaves_declarations_unenforced(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``v4.vocabulary`` gates itself even when block_metadata is on."""
        ws = _make_ws(
            tmp_path,
            monkeypatch,
            "ws",
            flags={BM_FLAG: True, VOCAB_FLAG: False},
            vocabularies={"tags": {"values": ["alpha"], "mode": "reject"}},
        )
        assert _propose(ws, tags="gamma")["status"] == "proposed"


# ===========================================================================
# 4. Backward compatibility
# ===========================================================================


class TestBackwardCompatible:
    def test_no_declarations_and_no_validators_restrict_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws(tmp_path, monkeypatch, "ws", flags={BM_FLAG: True, VOCAB_FLAG: True})
        envelope = _propose(ws, tags="anything, at, all", confidence="high", actor_role="whoever")
        assert envelope["status"] == "proposed"

    def test_undeclared_fields_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws(
            tmp_path,
            monkeypatch,
            "ws",
            flags={BM_FLAG: True, VOCAB_FLAG: True},
            vocabularies={"tags": ["alpha"]},
        )
        # ``confidence`` has no declaration, so no value of it is a violation.
        assert _propose(ws, tags="alpha", confidence="high")["status"] == "proposed"

    def test_the_content_source_refusal_still_fires_first(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The v4 leg must not displace the existing provenance validation."""
        ws = _make_ws(tmp_path, monkeypatch, "ws", flags={BM_FLAG: True})
        bm.register_schema_validator("decision", _always_accept)
        envelope = _propose(ws, content_source="not-a-real-class")
        assert "allowed" in envelope
        assert envelope["field"] == "content_source"
