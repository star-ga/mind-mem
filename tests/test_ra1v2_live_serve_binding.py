"""The live serve binding: a real serve records a v2 row, and a failed bind is visible.

WHAT THIS COVERS THAT THE LANE DID NOT. `tests/test_ra1_v2_versioned_rows.py` already proves
`context_digest` separates configurations, is injective under a delimiter-containing field, and
puts its version tag inside the preimage; `tests/test_ra1v2_outcome_acceptance_binding.py` proves
a v2 reference binds an outcome and refuses a foreign workspace. All of that is about the v2
SURFACE.

None of it exercised the BINDING, because there wasn't one: `context_digest` had zero production
callers, and `attach_served_run` accepted neither `serve_kind` nor a generation, so every live
serve recorded a v1 row no matter what the surface could express. These tests are about the three
states the binding can be in, and about the one property that makes the whole thing worth having:
a failed bind must be VISIBLE rather than silently downgraded.

THE THREE STATES, and why `generation` is a required keyword-only parameter rather than defaulted:

* a real identity -> bind it, write a v2 row carrying serve_kind + context_digest;
* ``NOT_BOUND``   -> this serving path captures no config snapshot yet, so it writes a v1 row,
                     exactly as it did before. Explicit and greppable at the call site;
* ``None`` / ""   -> the path TRIED to bind and could not (unrepresentable config, uncacheable
                     fingerprint). A failure, not a fallback: UNPROVEN reference with a reason,
                     and no row.

`append_served_run` defaults `serve_kind`/`context_digest` to `""` and writes v1, so a DEFAULT on
`attach_served_run` would let a call site that forgets keep emitting v1 while looking bound.
Required-and-loud turns that into a TypeError. The three-state split exists because collapsing
the middle case into the last would turn working v1 recording paths into unproven references — a
regression in ledger coverage — and collapsing it into the first would fabricate a context.

MUTATION EVIDENCE (each assertion below fails if the named change is reverted):
  * drop `serve_kind=`/`context_digest=` from the `append_served_run` call in
    `attach_served_run`  -> `a_real_generation_records_a_v2_row` goes red (row stays v1).
  * replace the `not generation` guard with `generation = ""`
    -> `an_unrepresentable_generation_is_unproven_not_v1` goes red (a v1 row appears instead of
       an unproven reference), which is the exact silent downgrade the guard exists to prevent.
  * (An `attach_served_run`-level `serve_kind` guard was REMOVED after mutation testing showed
    deleting it changed nothing: `ServedRunV2.__post_init__` already rejects an undefined kind and
    the caller gets the same unproven reference. `an_unknown_serve_kind_is_refused` therefore
    tests the ROW's validation, reached through the binding — which is the layer that should own
    the rule.)
  * give `generation` a default
    -> `the_binding_parameters_are_required_not_defaulted` goes red.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from mind_mem.served_ledger import (
    GENERATION_UNAVAILABLE,
    LEDGER_ERROR_KEY,
    NOT_BOUND,
    PROOF_RECORDED,
    PROOF_UNPROVEN,
    SERVE_KINDS,
    SERVED_PROOF_KEY,
    SERVED_SEQ_KEY,
    attach_served_run,
    context_digest,
    read_served_runs,
    row_serve_kind,
    served_set_digest,
)

#: A syntactically valid attestation record, with the four fields the binding reads.
_HEX = "a" * 64


#: The ids every test serves. The ledger REFUSES a row whose served_digest is not the digest of
#: its ids ("served_digest does not match ids"), so the record must carry the real one — a guard
#: to respect, not to route around.
_IDS = ["D-1"]


def _record() -> dict:
    return {
        "query_hash": "a" * 64,  # must be lowercase HEX: "q"/"i" are not hex digits
        "results_digest": served_set_digest(_IDS),
        "config_hash": "c" * 64,
        "index_anchor": "d" * 64,
        "scoring_instant": "2026-09-12",
    }


def _ws(tmp_path: Path) -> str:
    ws = tmp_path / "wsp"
    (ws / "decisions").mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text("[D-1]\nStatement: seeded\nStatus: active\n", encoding="utf-8")
    with open(ws / "mind-mem.json", "w", encoding="utf-8") as fh:
        json.dump({"served_ledger": {"enabled": True}}, fh)
    return str(ws)


def _only_row(ws: str):
    rows = read_served_runs(ws)
    assert len(rows) == 1, f"expected exactly one row, got {len(rows)}"
    return rows[0]


def test_a_real_generation_records_a_v2_row(tmp_path):
    """The point of the whole change: a live serve's row carries its context."""
    ws = _ws(tmp_path)
    out = attach_served_run(_record(), ws, ids=_IDS, serve_kind="attested", generation="PV:1:fingerprint")

    assert out[SERVED_PROOF_KEY] == PROOF_RECORDED, out
    assert out[SERVED_SEQ_KEY] is not None, out
    assert out[LEDGER_ERROR_KEY] is None, out

    row = _only_row(ws)
    assert row_serve_kind(row) == "attested", "the row must record WHICH kind of serve it was"
    digest = getattr(row, "context_digest", "")
    assert digest, "a v2 row must carry a context digest; an empty one binds nothing"

    # And it must be THE digest of these exact inputs — not merely non-empty.
    assert digest == context_digest(
        workspace=ws,
        config_hash=_record()["config_hash"],
        generation="PV:1:fingerprint",
        index_anchor=_record()["index_anchor"],
    ), "the stored digest must be derived from the serve's own workspace/config/generation/anchor"


def test_a_different_generation_gives_a_different_row_context(tmp_path):
    """Same query, same ids, same head — a changed policy identity must be distinguishable."""
    ws_a = _ws(tmp_path / "a")
    ws_b = _ws(tmp_path / "b")
    a = attach_served_run(_record(), ws_a, ids=_IDS, serve_kind="attested", generation="PV:1:one")
    b = attach_served_run(_record(), ws_b, ids=_IDS, serve_kind="attested", generation="PV:1:two")
    assert a[SERVED_PROOF_KEY] == b[SERVED_PROOF_KEY] == PROOF_RECORDED

    da = getattr(_only_row(ws_a), "context_digest", "")
    db = getattr(_only_row(ws_b), "context_digest", "")
    assert da and db and da != db, (
        "two serves differing ONLY in generation identity produced the same context digest, so the ledger cannot tell the policies apart"
    )


def test_an_unrepresentable_generation_is_unproven_not_v1(tmp_path):
    """THE load-bearing test. A failed bind must not quietly become a v1 row.

    ``anticipation_generation_identity`` returns None when the retrieval config cannot be
    represented by the shared projection. Writing a v1 row there would be a serve that looks
    recorded, verifies clean, and binds no context — which is the defect v2 exists to close.
    """
    ws = _ws(tmp_path)
    out = attach_served_run(_record(), ws, ids=_IDS, serve_kind="attested", generation=None)

    assert out[SERVED_PROOF_KEY] == PROOF_UNPROVEN, out
    assert out[SERVED_SEQ_KEY] is None, "an unproven reference must not carry a seq"
    assert out[LEDGER_ERROR_KEY] == GENERATION_UNAVAILABLE, out
    assert not read_served_runs(ws), "a failed bind must write NO row, not a v1 row"


def test_not_bound_still_records_a_v1_row(tmp_path):
    """A serving path that captures no config yet keeps working, and says so explicitly."""
    ws = _ws(tmp_path)
    out = attach_served_run(_record(), ws, ids=_IDS, serve_kind="attested", generation=NOT_BOUND)

    assert out[SERVED_PROOF_KEY] == PROOF_RECORDED, out
    row = _only_row(ws)
    assert not getattr(row, "context_digest", ""), "an unthreaded path must write a v1 row; a context digest here would be fabricated"


def test_an_unknown_serve_kind_is_refused_with_its_own_reason(tmp_path):
    """A kind the ledger does not define is refused, and the reason names it.

    The refusal comes from ``ServedRunV2``'s own constructor, reached through the binding — not
    from a duplicate check in ``attach_served_run``. That duplicate existed and was removed when
    mutation testing showed deleting it changed no observable behaviour.
    """
    ws = _ws(tmp_path)
    out = attach_served_run(_record(), ws, ids=_IDS, serve_kind="made_up", generation="PV:1:fingerprint")
    assert out[SERVED_PROOF_KEY] == PROOF_UNPROVEN, out
    assert "made_up" in (out[LEDGER_ERROR_KEY] or ""), "the reason must name the offending kind, or an operator cannot fix it"
    assert out[LEDGER_ERROR_KEY] != GENERATION_UNAVAILABLE, "a bad serve_kind must not be reported as a generation failure"
    assert not read_served_runs(ws)


@pytest.mark.parametrize("kind", sorted(SERVE_KINDS))
def test_every_defined_serve_kind_is_accepted(kind, tmp_path):
    """Positive control for the guard above: the guard must not reject valid kinds.

    Without this, the serve_kind test would pass if the guard rejected EVERYTHING.
    """
    ws = _ws(tmp_path / kind)
    out = attach_served_run(_record(), ws, ids=_IDS, serve_kind=kind, generation="PV:1:fingerprint")
    assert out[SERVED_PROOF_KEY] == PROOF_RECORDED, (kind, out)
    assert row_serve_kind(_only_row(ws)) == kind


def test_the_binding_parameters_are_required_not_defaulted():
    """Structural: a call site that forgets must be a TypeError, not a silent v1 row.

    This is the test that keeps the API honest. ``append_served_run`` already defaults
    ``serve_kind``/``context_digest`` to "" and writes v1, so if these two ever acquire defaults
    a forgotten call site downgrades in silence and nothing here would notice.
    """
    sig = inspect.signature(attach_served_run)
    for name in ("serve_kind", "generation"):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{name} must be keyword-only"
        assert param.default is inspect.Parameter.empty, (
            f"{name} must have NO default: a default lets a call site that forgets keep emitting v1 rows while appearing bound"
        )


def test_no_scoring_path_door_declares_its_generation_unbound_while_holding_a_snapshot():
    """A door that captured a snapshot must DERIVE its generation — never declare it unbound.

    WHAT THIS REPLACED, AND WHY. The earlier version asserted three scoring-path doors each passed
    a literal ``generation="__not_bound__"`` equal to ``served_ledger.NOT_BOUND``, so a duplicated
    literal could not become a duplicated truth. That was the right guard for a transitional state
    and the wrong thing to keep: its ``seen >= 3`` count pinned HOW MANY doors were still
    unthreaded, a number that must fall to zero as each door gets bound. Left as it was, the test
    would have failed the build for making the code better.

    So the count becomes a ratchet in the direction the work goes. Every door that passes
    ``config_hash=`` holds a snapshot, and a door holding a snapshot owes a derived generation:
    passing NOT_BOUND alongside a captured hash is exactly the mixed-coordinate row an independent
    control caught on the axis door — a bound hash and an unbound generation in one RECORDED row,
    asserting a context that existed at neither moment.

    The literal-vs-sentinel check is kept for any door that legitimately still declares itself
    unbound, because the scoring-path import rail has not moved: ``_recall_core`` may not import
    ``served_ledger``, so such a door would have to spell the value out.
    """
    import io
    import re
    import tokenize
    from pathlib import Path

    def _code_only(text: str) -> str:
        """The module with COMMENT tokens removed.

        A plain regex over the file also matches prose: the axis door's comment *explains* that it
        used to pass ``generation="__not_bound__"``, and a scan that counts that as code reports a
        defect that was already fixed. Tokenising is exact where stripping from the first ``#``
        would mangle a ``#`` inside a string literal.
        """
        out: list[str] = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type != tokenize.COMMENT:
                    out.append(tok.string)
        except (tokenize.TokenError, IndentationError):  # pragma: no cover — fall back to raw text
            return text
        return "\n".join(out)

    src_root = Path(__file__).resolve().parent.parent / "src" / "mind_mem"
    doors = ("_recall_core.py", "axis_recall.py", "recall.py")
    literals: dict[str, list[str]] = {}
    for mod in doors:
        raw = (src_root / mod).read_text(encoding="utf-8")
        # Two views of the same file, deliberately. The literal scan wants code only, but the
        # import-rail check below is a substring test on SOURCE LINES — run against the tokenised
        # view it can never match, because tokenising joins tokens with newlines and
        # "from .served_ledger import" arrives as four separate tokens. That would leave the rail
        # assertion permanently, invisibly true: a guard that cannot fail.
        text = _code_only(raw)
        found = re.findall(r'generation="([^"]*)"', text)
        literals[mod] = found
        for literal in found:
            assert literal == NOT_BOUND, (
                f"{mod} passes generation={literal!r}, but served_ledger.NOT_BOUND is "
                f"{NOT_BOUND!r} — the scoring path and the ledger disagree about what "
                f"'not bound' means, so those rows would be misclassified"
            )
        assert "from .served_ledger import" not in raw or mod == "recall.py", (
            f"{mod} must not import from served_ledger at module scope — the scoring-path rail"
        )

    # THE RATCHET. Each of these doors captures a policy snapshot (they all pass config_hash), so
    # none of them may also declare the generation unbound. A door added later that captures
    # nothing is free to pass NOT_BOUND — it just may not also pass a captured hash.
    for mod in doors:
        text = _code_only((src_root / mod).read_text(encoding="utf-8"))
        if "config_hash=" not in text:
            continue
        assert not literals[mod], (
            f"{mod} passes a captured config_hash AND declares generation={literals[mod]!r}: a "
            f"recorded row would bind a hash from the snapshot to a generation from nowhere. "
            f"Derive it from the captured config (mind_mem.recall._derive_generation) or capture "
            f"nothing at all."
        )

    # Positive control for the regex itself: it must be able to see a literal when one is there.
    # Without this the two loops above pass just as happily on a pattern that matches nothing.
    # The import-rail substring must also be able to match when it is really there. Without this
    # the check above silently survives any future change to how the file is read.
    assert "from .served_ledger import" in "from .served_ledger import NOT_BOUND\n", (
        "the import-rail substring check can no longer match a known-positive sample"
    )
    probe = 'attest_and_record(ws, q, r, generation="__not_bound__")'
    assert re.findall(r'generation="([^"]*)"', probe) == [NOT_BOUND], (
        "the literal-detecting pattern no longer matches a known-positive sample, so every assertion above is vacuous"
    )


def test_a_v2_row_hands_its_coordinates_back_to_the_caller(tmp_path):
    """A v2 row a client cannot present back is unusable. This is the hole I made.

    Binding the MCP attested path to v2 made the outcome path correctly demand the digest the run
    was bound to — "this served run records a workspace/config context; the outcome must present
    the context_digest it was bound to" — while ``_ledger_fields`` still returned only
    seq/row_hash/error/proof. So the row verified, no caller could ever obtain the digest, and the
    outcome could not be reported at all. It surfaced far away, as a KeyError in a v1-era test,
    which is why this assertion lives next to the binding instead.
    """
    ws = _ws(tmp_path)
    out = attach_served_run(_record(), ws, ids=_IDS, serve_kind="attested", generation="PV:1:fingerprint")
    assert out[SERVED_PROOF_KEY] == PROOF_RECORDED, out

    assert out.get("served_serve_kind") == "attested", "a v2 row must tell the caller WHICH kind it recorded"
    digest = out.get("served_context_digest")
    assert digest, "a v2 row must hand back the context digest, or no outcome can be bound to it"
    assert digest == getattr(_only_row(ws), "context_digest", ""), (
        "the digest handed to the caller must be the one actually stored on the row"
    )


def test_a_v1_row_omits_the_v2_coordinates_rather_than_emptying_them(tmp_path):
    """Absent, not empty — so a consumer can tell "no context" from "empty context".

    An empty string would be indistinguishable from a real-but-blank digest, and the hashing layer
    rejects a blank one anyway, so emitting one would invite a caller to present something that
    can never validate.
    """
    ws = _ws(tmp_path)
    out = attach_served_run(_record(), ws, ids=_IDS, serve_kind="attested", generation=NOT_BOUND)
    assert out[SERVED_PROOF_KEY] == PROOF_RECORDED, out
    assert "served_context_digest" not in out, out
    assert "served_serve_kind" not in out, out
