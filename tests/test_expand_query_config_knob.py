# Copyright 2026 STARGA, Inc.
"""`recall.expand_query` must be a real, honoured key -- and may only NARROW.

The full-set gate (docs/benchmarks/2026-09-07-no-expansion-gate.md) measured
disabling query expansion as MRR 0.8954 -> 0.9048, 27 questions better against
11 worse, p=0.0139, with `recall_all@5` IDENTICAL at 0.8447 -- free, and it
centres the last gap against the zero-dependency floor.

That win is not shippable, because the lever it used was
`benchmarks/ablation_mask.py`, which REBINDS a product symbol in the
benchmark's own child process:

    for params in _recall_detection._QUERY_TYPE_PARAMS.values():
        params["expand_query"] = False

An operator cannot do that. `recall.expand_query` was not in
`_VALID_RECALL_KEYS` and neither read site consulted user config, so the key
was rejected as unknown. These controls pin the knob that makes the measured
behaviour reachable from configuration.

NARROWING-ONLY, on purpose. `adversarial` pins `expand_query: "morph_only"` to
suppress semantic drift on distractor-prone queries. A plain override would let
`expand_query: true` silently widen that back to full expansion and remove a
deliberate safeguard, so an override may only lower expansion, never raise it:

    False (none)  <  "morph_only"  <  True / "full"
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from mind_mem._recall_detection import _QUERY_TYPE_PARAMS, resolve_expand_mode


def test_absent_config_preserves_every_per_type_default():
    for qtype, params in _QUERY_TYPE_PARAMS.items():
        assert resolve_expand_mode(params, {}) == params.get("expand_query", True), qtype


def test_false_disables_expansion_for_every_query_type():
    # This is exactly what the measured `no_expansion` mask did.
    for qtype, params in _QUERY_TYPE_PARAMS.items():
        assert resolve_expand_mode(params, {"expand_query": False}) is False, qtype


def test_morph_only_narrows_full_types_but_leaves_adversarial_alone():
    assert resolve_expand_mode(_QUERY_TYPE_PARAMS["single-hop"], {"expand_query": "morph_only"}) == "morph_only"
    assert resolve_expand_mode(_QUERY_TYPE_PARAMS["adversarial"], {"expand_query": "morph_only"}) == "morph_only"


def test_true_may_not_widen_the_adversarial_safeguard():
    assert resolve_expand_mode(_QUERY_TYPE_PARAMS["adversarial"], {"expand_query": True}) == "morph_only"
    assert resolve_expand_mode(_QUERY_TYPE_PARAMS["single-hop"], {"expand_query": True}) is True


def test_unknown_value_is_ignored_rather_than_treated_as_truthy():
    # A typo must not silently become "full expansion".
    params = _QUERY_TYPE_PARAMS["single-hop"]
    assert resolve_expand_mode(params, {"expand_query": "nonsense"}) == params["expand_query"]
    assert resolve_expand_mode(params, {"expand_query": 3}) == params["expand_query"]


def test_key_is_accepted_by_the_recall_config_validator():
    from mind_mem._recall_constants import _VALID_RECALL_KEYS

    assert "expand_query" in _VALID_RECALL_KEYS


def test_both_product_read_sites_use_the_resolver():
    """`imported` is not `wired` -- neither site may keep the bare lookup."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "mind_mem"
    for rel in ("_recall_core.py", "sqlite_index.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "resolve_expand_mode(" in src, f"{rel} does not call the resolver"
        assert 'qparams.get("expand_query", True)' not in src, f"{rel} still reads the per-type value directly, bypassing user config"


# ---------------------------------------------------------------------------
# Dispatch proof. The controls above pin the resolver; these pin that the real
# product path CONSULTS it. A resolver nothing calls is not a shipped knob.
# ---------------------------------------------------------------------------
def _workspace(recall_cfg: dict) -> str:
    """An INDEXED workspace. Without a built index `query_index` takes the
    `index_missing_fallback` branch and returns before the expansion stage, so
    a spy would record nothing and the negative assertion below would be
    vacuous. The positive control catches exactly that."""
    ws = tempfile.mkdtemp(prefix="mm-expand-")
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump({"recall": recall_cfg}, fh)
    dec = os.path.join(ws, "decisions")
    os.makedirs(dec, exist_ok=True)
    with open(os.path.join(dec, "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "[D-1]\n"
            "Statement: The vendor quoted a March price for the storage contract.\n"
            "Status: active\n\n"
            "[D-2]\n"
            "Statement: Renewal terms were agreed with the supplier in spring.\n"
            "Status: active\n"
        )
    from mind_mem.sqlite_index import build_index

    build_index(ws, incremental=False)
    return ws


def _expansion_calls(monkeypatch, module, ws: str, query: str = "what did the vendor quote in March"):
    """Count real expand_query() calls made by `module` for one query."""
    calls: list[dict] = []
    real = module.expand_query

    def spy(tokens, *a, **kw):
        calls.append({"mode": kw.get("mode")})
        return real(tokens, *a, **kw)

    monkeypatch.setattr(module, "expand_query", spy)
    return calls


def test_config_off_actually_suppresses_expansion_in_sqlite_index(monkeypatch):
    """POSITIVE CONTROL FIRST: prove expansion fires when the key is absent."""
    from mind_mem import sqlite_index

    on_ws = _workspace({})
    calls_on = _expansion_calls(monkeypatch, sqlite_index, on_ws)
    # No try/except: the workspace is indexed by _workspace(), so this must
    # succeed. Swallowing here would hide a crash and let the negative
    # assertion below pass for the wrong reason.
    sqlite_index.query_index(on_ws, "what did the vendor quote in March", limit=5)
    assert calls_on, (
        "positive control FAILED: expansion never ran even with the key absent, so the negative assertion below would prove nothing"
    )

    # Now the real assertion, on the same code path.
    off_ws = _workspace({"expand_query": False})
    calls_off = _expansion_calls(monkeypatch, sqlite_index, off_ws)
    sqlite_index.query_index(off_ws, "what did the vendor quote in March", limit=5)
    assert calls_off == [], f"expand_query: false did not suppress expansion; ran {calls_off}"


# ---------------------------------------------------------------------------
# Hostile config values. Found in review: `configured in _EXPAND_RANK` raised
# TypeError on an unhashable value, so `expand_query: []` CRASHED a real
# query_index call rather than being ignored; and because bool subclasses int
# and hash(0.0) == hash(False), numeric values ALIASED booleans through that
# lookup -- 0.0 came back as the mode itself (a float, into
# `expand_query(tokens, mode=...)`) and 1.0 came back as True.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [[], {}, 0.0, 1.0, 0, 1, None, 3, ("a",), set(), object()])
def test_non_bool_non_str_values_are_ignored_and_never_crash(value):
    params = _QUERY_TYPE_PARAMS["single-hop"]
    assert resolve_expand_mode(params, {"expand_query": value}) == params["expand_query"]


@pytest.mark.parametrize("value", ["nonsense", "", "  ", "TRUE", "none", "off"])
def test_unknown_strings_are_ignored(value):
    params = _QUERY_TYPE_PARAMS["single-hop"]
    assert resolve_expand_mode(params, {"expand_query": value}) == params["expand_query"]


def test_a_hostile_value_does_not_crash_a_real_query(tmp_path):
    """The reported crash, through the REAL product path rather than the resolver."""
    from mind_mem import sqlite_index

    ws = _workspace({"expand_query": []})
    sqlite_index.query_index(ws, "vendor quote March", limit=5)


def test_case_and_whitespace_are_normalised_but_cannot_widen():
    params = _QUERY_TYPE_PARAMS["single-hop"]
    assert resolve_expand_mode(params, {"expand_query": "  MORPH_ONLY  "}) == "morph_only"
    assert resolve_expand_mode(params, {"expand_query": "FULL"}) is True
