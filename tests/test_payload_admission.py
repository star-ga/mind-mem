# Copyright 2026 STARGA, Inc.
"""A NEW payload that cannot be hashed exactly is refused, and nothing is written.

Both ledgers hash a payload's RENDERING: `json.dumps(..., default=str)` runs
for any value JSON cannot encode, so a datetime and its string form produce
one digest. The repair is prevention at admission, not a format change --
verification reads the STORED payload_hash and never re-hashes the original
object, so refusing an unsupported value changes no accepted preimage and no
historical row.

Every refusal control asserts the THREE things that make "refused" mean
something: the file bytes, the record count, and the in-memory tail are all
exactly what they were. A writer that refuses after appending has not refused.
"""

from __future__ import annotations

import datetime
import decimal
import hashlib
import json
import os

import pytest

from mind_mem.audit_chain import AuditChain
from mind_mem.evidence_objects import EvidenceAction, EvidenceChain
from mind_mem.payload_admission import (
    MAX_DEPTH,
    MAX_INT_BITS,
    MAX_NODES,
    MAX_PAYLOAD_BYTES,
    MAX_STRING_CHARS,
    PayloadRejected,
    admit_payload,
    validate_payload,
)

#: Every value that would be stringified, one per class of the defect.
UNSUPPORTED = {
    "datetime": {"when": datetime.datetime(2026, 9, 8, 11, 0, 0)},
    "decimal": {"amt": decimal.Decimal("1.10")},
    "set": {"who": {"a", "b"}},
    "bytes_nested": {"blob": b"x"},
    "non_str_key": {1: "a"},
    "oversized_key": {"k" * (MAX_STRING_CHARS + 1): "value"},
    "nan": {"x": float("nan")},
    "infinity": {"x": float("inf")},
}

GOOD = {"op": "release", "n": 3, "ok": True, "none": None, "list": [1, "two", 2.5, False], "nested": {"deep": {"er": "yes"}}}


def _state(path: str) -> tuple[bytes, int]:
    if not os.path.isfile(path):
        return (b"", 0)
    raw = open(path, "rb").read()
    return (raw, len([ln for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()]))


# ---------------------------------------------------------------------------
# The validator itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_the_validator_refuses_each_unsupported_class(name) -> None:
    with pytest.raises(PayloadRejected) as ei:
        validate_payload(UNSUPPORTED[name])
    assert ei.value.reason, "a refusal must say why"


def test_a_cycle_is_refused_by_name_not_by_recursion_error() -> None:
    """A RecursionError names the symptom; this names the payload."""
    cyc: dict = {}
    cyc["self"] = cyc
    with pytest.raises(PayloadRejected, match="cycle"):
        validate_payload(cyc)


def test_depth_and_size_are_refused_explicitly() -> None:
    deep: dict = {}
    cur = deep
    for _ in range(MAX_DEPTH + 5):
        cur["n"] = {}
        cur = cur["n"]
    with pytest.raises(PayloadRejected, match="nesting deeper"):
        validate_payload(deep)

    with pytest.raises(PayloadRejected, match="exceeds"):
        validate_payload({"big": list(range(MAX_NODES + 10))})


def test_bool_is_accepted_and_stays_distinct_from_int() -> None:
    """bool and int must both survive as themselves.

    A PROPERTY control, not a guard control, and the difference is worth
    stating: the validator's explicit `bool` branch is redundant today --
    `bool` is in the accepted scalar tuple, so removing that branch changes no
    outcome and this control does NOT go red. Mutation-checked, so the claim
    is measured rather than assumed.

    The branch is kept as an ordering guard against a future int-specific
    check being added above the generic one, where `isinstance(True, int)`
    would start mattering. What this control pins is the property itself:
    True and 1 remain distinguishable in the preimage.
    """
    validate_payload({"flag": True, "count": 1})
    rendered = json.dumps({"flag": True, "count": 1}, sort_keys=True)
    assert rendered == '{"count": 1, "flag": true}', rendered


def test_documented_top_level_semantics_are_untouched() -> None:
    """bytes / str / None at the TOP level are hashed directly, never coerced.

    Asserted on the PREIMAGE, not merely on "it did not raise": the content
    semantics these roots carry are exactly the bytes each one hashes.
    """
    assert admit_payload(b"raw bytes", accepts_bytes=True).preimage == b"raw bytes"
    assert admit_payload("a string", accepts_bytes=True).preimage == b"a string"
    assert admit_payload(None, accepts_bytes=True).preimage == b""


def test_validate_payload_refuses_a_non_dict_instead_of_returning_quietly() -> None:
    """The shape of the original defect, pinned so it cannot come back.

    The first version of this checker returned silently for EVERY non-dict, so
    an arbitrary object, a top-level list and a bare int all passed straight
    through to a real write. Returning quietly for anything it does not
    recognise is the failure mode; refusing is the fix.
    """
    for payload in (object(), [1, 2], 5, b"bytes"):
        with pytest.raises(PayloadRejected):
            validate_payload(payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The payload ROOT, by each ledger's own documented contract
# ---------------------------------------------------------------------------


class _Opaque:
    def __repr__(self) -> str:
        return "<opaque>"


def _cyclic_list() -> list:
    c: list = []
    c.append(c)
    return c


#: Roots outside BOTH contracts. Each one reached a real write before this
#: change: `<opaque>` and `[{1: "x"}]` were appended to an evidence chain with
#: a digest of their repr, and the cyclic list reached a bare
#: `ValueError: Circular reference detected` from inside `json.dumps` --
#: past the admission point, with no path in the message.
UNDOCUMENTED_ROOTS = {
    "arbitrary_object": _Opaque,
    "top_level_list": lambda: [{1: "x"}],
    "cyclic_list": _cyclic_list,
    "bare_int": lambda: 5,
    "tuple": lambda: (1, 2),
}


@pytest.mark.parametrize("name", sorted(UNDOCUMENTED_ROOTS))
def test_evidence_refuses_an_undocumented_root_and_writes_nothing(tmp_path, name) -> None:
    path = str(tmp_path / "memory" / "evidence_chain.jsonl")
    chain = EvidenceChain(store_path=path)
    chain.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-1", target_file="b.md", payload=GOOD)

    before_bytes, before_lines = _state(path)
    before_tail = chain._entries[-1].evidence_hash
    assert before_lines == 1, "positive control: the good payload did not land"

    with pytest.raises(PayloadRejected):
        chain.create(
            action=EvidenceAction.APPLY,
            actor="t",
            target_block_id="B-2",
            target_file="b.md",
            payload=UNDOCUMENTED_ROOTS[name](),
        )

    assert _state(path) == (before_bytes, before_lines), "the file changed on a refused root"
    assert chain._entries[-1].evidence_hash == before_tail, "the in-memory tail moved on a refused root"


@pytest.mark.parametrize("name", sorted(UNDOCUMENTED_ROOTS))
def test_audit_refuses_an_undocumented_root_and_writes_nothing(tmp_path, name) -> None:
    chain = AuditChain(str(tmp_path))
    chain.append(operation="create_block", target="t.md", agent="a", reason="r", payload=GOOD)
    path = chain._chain_path

    before = _state(path)
    assert before[1] == 1, "positive control: the good payload did not land"

    with pytest.raises(PayloadRejected):
        chain.append(operation="create_block", target="t.md", agent="a", reason="r", payload=UNDOCUMENTED_ROOTS[name]())

    assert _state(path) == before, "the audit file changed on a refused root"


def test_each_ledger_is_bound_to_its_own_documented_contract(tmp_path) -> None:
    """`bytes` is a documented evidence root and is NOT a documented audit root.

    The audit chain's signature says `dict | str | None`. Passing bytes there
    reached `json.dumps(..., default=str)` and recorded a digest of
    ``"b'raw'"`` -- the repr, not the content. The two doors are therefore not
    given one merged contract; each passes its own.
    """
    evidence = EvidenceChain(store_path=str(tmp_path / "memory" / "evidence_chain.jsonl"))
    ev = evidence.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-1", target_file="b.md", payload=b"raw")
    assert ev.payload_hash == hashlib.sha256(b"raw").hexdigest(), "the documented bytes preimage moved"

    audit = AuditChain(str(tmp_path))
    audit.append(operation="create_block", target="t.md", agent="a", reason="r", payload=GOOD)
    before = _state(audit._chain_path)
    with pytest.raises(PayloadRejected, match="dict \\| str \\| None"):
        audit.append(operation="create_block", target="t.md", agent="a", reason="r", payload=b"raw")
    assert _state(audit._chain_path) == before


# ---------------------------------------------------------------------------
# Bounds that a node COUNT does not give
# ---------------------------------------------------------------------------


def test_an_integer_too_wide_to_render_is_refused_by_name() -> None:
    """CPython raises from INSIDE json.dumps, after admission said yes.

    Measured before the bound existed: ``{"n": 10 ** 5000}`` passed admission
    and then died on ``ValueError: Exceeds the limit (4300 digits) for integer
    string conversion`` -- an uncontrolled error, from the wrong layer, naming
    no path in the payload.
    """
    with pytest.raises(PayloadRejected, match="bits wide"):
        validate_payload({"n": 10**5000})
    assert (10**5000).bit_length() > MAX_INT_BITS, "positive control: the fixture is not actually over the bound"


def test_a_huge_string_and_a_huge_serialization_are_refused_by_name() -> None:
    """MAX_NODES is a COUNT. Ten thousand large strings satisfy it and still

    serialize to gigabytes, so size is bounded separately and while encoding,
    not after the whole string has been built.
    """
    with pytest.raises(PayloadRejected, match="characters, past"):
        validate_payload({"s": "x" * (MAX_STRING_CHARS + 1)})

    wide = {f"k{i}": "y" * (MAX_STRING_CHARS - 1) for i in range(64)}
    assert len(wide) < MAX_NODES, "positive control: this payload passes the node bound"
    assert all(len(v) <= MAX_STRING_CHARS for v in wide.values()), "and passes the per-string bound"
    assert len(json.dumps(wide, sort_keys=True)) > MAX_PAYLOAD_BYTES, "positive control: it really is over the size bound"
    with pytest.raises(PayloadRejected, match="more than"):
        validate_payload(wide)


def test_key_length_is_bounded_before_serialization(monkeypatch) -> None:
    key = "k" * (MAX_STRING_CHARS + 1)
    real_dumps = json.dumps
    serialized_keys = []

    def observed_dumps(value, *args, **kwargs):
        if value is key:
            serialized_keys.append(value)
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", observed_dumps)
    with pytest.raises(PayloadRejected, match="key.*characters"):
        admit_payload({key: None}, accepts_bytes=False)
    assert serialized_keys == [], "oversized keys must be refused before allocating their encoded form"
    accepted = {key[:-1]: None}
    assert admit_payload(accepted, accepts_bytes=False).preimage == real_dumps(accepted, sort_keys=True).encode()


def test_key_inventory_stops_at_the_remaining_node_budget() -> None:
    class CountedKeys(dict):
        visits = 0

        def __iter__(self):
            for key in super().__iter__():
                self.visits += 1
                yield key

    payload = CountedKeys((str(i), None) for i in range(MAX_NODES + 2))
    with pytest.raises(PayloadRejected, match="exceeds"):
        admit_payload(payload, accepts_bytes=False)
    assert payload.visits == MAX_NODES, "do not materialize and sort an unbounded key inventory before admission"
    accepted = {str(i): None for i in range(MAX_NODES - 1)}
    assert admit_payload(accepted, accepts_bytes=False).preimage == json.dumps(accepted, sort_keys=True).encode()


# ---------------------------------------------------------------------------
# One traversal: what is checked IS what is hashed
# ---------------------------------------------------------------------------


class _MutatesAfterFirstLook(dict):
    """Presents clean content, then gains an unencodable value.

    Deterministic by construction -- it counts its own traversals rather than
    relying on scheduler timing -- and it models the real window: any caller
    holding a reference can mutate the dict between the check and the hash.
    """

    def __init__(self) -> None:
        super().__init__({"ok": 1})
        self.traversals = 0

    def __iter__(self):
        self.traversals += 1
        if self.traversals > 1:
            dict.__setitem__(self, "when", datetime.datetime(2026, 9, 8))
        return dict.__iter__(self)


def test_the_digest_is_of_the_checked_bytes_not_a_later_look_at_the_object(tmp_path) -> None:
    """Admission checks and ENCODES in one pass, and the writer hashes THAT.

    Measured on the previous shape: a payload validated as ``{"ok": 1}`` and
    then given a ``datetime`` was hashed as
    ``{"ok": 1, "when": "2026-09-08 00:00:00"}`` -- the stringified value the
    validator exists to refuse, admitted because hashing was a SECOND
    traversal of the same mutable object.
    """
    payload = _MutatesAfterFirstLook()
    chain = EvidenceChain(store_path=str(tmp_path / "memory" / "evidence_chain.jsonl"))
    ev = chain.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-1", target_file="b.md", payload=payload)

    checked = hashlib.sha256(json.dumps({"ok": 1}, sort_keys=True).encode("utf-8")).hexdigest()
    mutated = hashlib.sha256(
        json.dumps({"ok": 1, "when": datetime.datetime(2026, 9, 8)}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert ev.payload_hash != mutated, "the post-check mutation reached the digest"
    assert ev.payload_hash == checked, "the digest is not of the bytes admission checked"
    assert payload.traversals == 1, f"the writer traversed the caller's object {payload.traversals} times, not once"


def test_the_audit_writer_also_traverses_once(tmp_path) -> None:
    """The same property on the second ledger. One fixed and one not is the hole."""
    payload = _MutatesAfterFirstLook()
    chain = AuditChain(str(tmp_path))
    entry = chain.append(operation="create_block", target="t.md", agent="a", reason="r", payload=payload)

    assert entry.payload_hash == hashlib.sha256(json.dumps({"ok": 1}, sort_keys=True).encode("utf-8")).hexdigest()
    assert payload.traversals == 1, f"the audit writer traversed the object {payload.traversals} times"


def test_the_transactional_writer_also_traverses_once(tmp_path) -> None:
    """And the third door."""
    payload = _MutatesAfterFirstLook()
    chain = EvidenceChain(store_path=str(tmp_path / "memory" / "evidence_chain.jsonl"))
    with chain.transaction() as txn:
        ev = txn.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-1", target_file="b.md", payload=payload)

    assert ev.payload_hash == hashlib.sha256(json.dumps({"ok": 1}, sort_keys=True).encode("utf-8")).hexdigest()
    assert payload.traversals == 1, f"the transactional writer traversed the object {payload.traversals} times"


# ---------------------------------------------------------------------------
# The canonical encoding is the legacy rendering, byte for byte
# ---------------------------------------------------------------------------


CANONICAL_CASES = [
    {},
    {"a": 1},
    {"b": 2, "a": 1},
    {"z": [1, 2, {"y": None}]},
    {"s": "\u00e9\u4e2d\U0001f600"},
    {"esc": 'quote" back\\ nl\n tab\t'},
    {"f": 1e300},
    {"f": -0.0},
    {"f": 3.141592653589793},
    {"big": 12345678901234567890123456789},
    {"t": True, "f": False, "n": None},
    {"nested": {"a": {"b": {"c": [1, [2, [3]]]}}}},
    {"tuple": (1, 2, 3)},
    {"empty_list": [], "empty_dict": {}},
]


@pytest.mark.parametrize("case", CANONICAL_CASES, ids=range(len(CANONICAL_CASES)))
def test_the_canonical_encoding_equals_the_legacy_rendering(case) -> None:
    """Byte-identity is what makes this prevention rather than a format change.

    A one-pass encoder that produced even slightly different bytes -- a
    separator, an escape, a float repr -- would silently move the digest of
    every future accepted payload while claiming to preserve it.
    """
    from mind_mem.payload_admission import _canonical_json

    assert _canonical_json(case) == json.dumps(case, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# The public writers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_the_evidence_create_refuses_and_writes_nothing(tmp_path, name) -> None:
    path = str(tmp_path / "memory" / "evidence_chain.jsonl")
    chain = EvidenceChain(store_path=path)
    chain.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-1", target_file="b.md", payload=GOOD)

    before_bytes, before_lines = _state(path)
    before_tail = chain._entries[-1].evidence_hash
    assert before_lines == 1, "positive control: the good payload did not land"

    with pytest.raises(PayloadRejected):
        chain.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-2", target_file="b.md", payload=UNSUPPORTED[name])

    after_bytes, after_lines = _state(path)
    assert after_bytes == before_bytes, "the file changed on a refused write"
    assert after_lines == before_lines, "a record was appended on a refused write"
    assert chain._entries[-1].evidence_hash == before_tail, "the in-memory tail moved on a refused write"

    # And the writer still works afterwards -- a refusal must not wedge it.
    chain.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-3", target_file="b.md", payload=GOOD)
    assert _state(path)[1] == before_lines + 1


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_the_transactional_create_refuses_and_writes_nothing(tmp_path, name) -> None:
    """The second public entry point. Refusing in one and not the other IS the hole."""
    path = str(tmp_path / "memory" / "evidence_chain.jsonl")
    chain = EvidenceChain(store_path=path)
    with chain.transaction() as txn:
        txn.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-1", target_file="b.md", payload=GOOD)

    before_bytes, before_lines = _state(path)
    assert before_lines == 1, "positive control: the transactional writer did not land the good payload"

    with pytest.raises(PayloadRejected):
        with chain.transaction() as txn:
            txn.create(action=EvidenceAction.APPLY, actor="t", target_block_id="B-2", target_file="b.md", payload=UNSUPPORTED[name])

    after_bytes, after_lines = _state(path)
    assert after_bytes == before_bytes, "the file changed on a refused transactional write"
    assert after_lines == before_lines


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_the_audit_append_refuses_and_writes_nothing(tmp_path, name) -> None:
    chain = AuditChain(str(tmp_path))
    chain.append(operation="create_block", target="t.md", agent="a", reason="r", payload=GOOD)
    path = chain._chain_path

    before_bytes, before_lines = _state(path)
    assert before_lines == 1, "positive control: the good payload did not land"

    with pytest.raises(PayloadRejected):
        chain.append(operation="create_block", target="t.md", agent="a", reason="r", payload=UNSUPPORTED[name])

    after_bytes, after_lines = _state(path)
    assert after_bytes == before_bytes, "the audit file changed on a refused append"
    assert after_lines == before_lines


# ---------------------------------------------------------------------------
# Nothing historical moved
# ---------------------------------------------------------------------------


def test_an_accepted_payload_hashes_exactly_as_before(tmp_path) -> None:
    """The preimage of an ACCEPTED payload is unchanged, byte for byte.

    This is the whole claim of the repair: prevention costs the accepted path
    nothing. Compared against the digest computed the old way, directly.
    """
    from mind_mem.evidence_objects import _compute_payload_hash

    expected = hashlib.sha256(json.dumps(GOOD, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    assert _compute_payload_hash(GOOD) == expected, "an accepted payload's digest moved"


def test_an_old_entry_still_verifies_after_the_change(tmp_path) -> None:
    """A chain written before this change must still load and verify intact.

    The fixture is written through the ordinary writer and then re-read by a
    fresh reader, which is the path a historical file takes.
    """
    path = str(tmp_path / "memory" / "evidence_chain.jsonl")
    chain = EvidenceChain(store_path=path)
    for i in range(3):
        chain.create(action=EvidenceAction.APPLY, actor="t", target_block_id=f"B-{i}", target_file="b.md", payload={"i": i})

    raw = open(path, "rb").read()
    # The payload repair does not include the separate read-only verifier
    # feature. A fresh ordinary reader still exercises historical loading and
    # verification while keeping this focused test independent of that branch.
    fresh = EvidenceChain(store_path=path)
    assert len(fresh._entries) == 3
    assert fresh.verify_chain()[0] is True
    assert fresh.integrity_compromised is False
    assert open(path, "rb").read() == raw, "reading the chain rewrote it"
