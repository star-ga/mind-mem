# Copyright 2026 STARGA, Inc.
"""Strict admission for NEW ledger payloads. Historical bytes untouched.

Two ledgers hash a payload's RENDERING rather than the payload:
``audit_chain._payload_hash`` and ``evidence_objects._compute_payload_hash``
both called ``json.dumps(payload, sort_keys=True, default=str)``. ``default=str``
runs for any value JSON cannot encode, so a value and its string rendering
produce the SAME digest -- a datetime and ``"2026-09-08 11:00:00"`` are one
record as far as either chain can tell.

WHY THIS IS PREVENTION, NOT A FORMAT CHANGE. Verification reads the STORED
``payload_hash`` out of the entry; it never recovers the original Python
object and re-hashes it. Verified by reading the call graph rather than taking
it on faith: every call to either hash function is on a WRITE path --
``AuditChain.append``, ``EvidenceChain.create`` and ``ChainTransaction.create``
-- and none is on a verify path. So refusing an unsupported object at
admission changes no accepted preimage, no stored row and no historical
result. Old entries keep verifying exactly as they did.

ADMISSION RETURNS THE PREIMAGE, AND THAT IS THE POINT. An earlier shape of
this module validated the caller's object and then let the writer hash the
same object in a SECOND traversal. Two traversals of one mutable object is a
gap, and it was demonstrated rather than argued: a payload that passed
validation and then gained a ``datetime`` was hashed with the stringified
value anyway. :func:`admit_payload` therefore performs ONE traversal that both
checks and ENCODES, and hands back immutable bytes. Every writer hashes those
bytes and never looks at the caller's object again, so there is no window in
which the checked value and the hashed value can differ.

WHAT IS REFUSED, and each is a way two payloads become one digest, one payload
becomes two, or the serializer fails past the admission point:

* a value JSON cannot encode -- datetime, Decimal, set, bytes nested in a
  dict -- refused rather than stringified;
* a non-string dict key -- ``{1: "a"}`` and ``{"1": "a"}`` are different
  payloads that ``sort_keys`` renders identically;
* a non-finite float -- NaN, Infinity: ``NaN != NaN``, so a payload
  containing one is not even equal to itself;
* a cycle, or depth, node count, string length, integer width or total
  serialized size past an explicit bound -- refused BY NAME rather than met
  with a ``RecursionError``, a bare ``ValueError: Circular reference
  detected``, or CPython's ``Exceeds the limit (4300 digits) for integer
  string conversion``, all of which fire from inside ``json.dumps`` after
  admission has already said yes;
* a payload root the calling API does not document. Each door passes its own
  contract: the audit chain takes ``dict | str | None``, the evidence chain
  additionally takes ``bytes``. Anything else -- an arbitrary object, a
  top-level list, a bare int -- is refused there rather than stringified into
  a digest.

``bool`` is deliberately NOT refused and deliberately NOT treated as ``int``:
it is JSON-native, ``True`` and ``1`` must stay distinguishable, and Python's
``isinstance(True, int)`` is exactly the trap that would collapse them.

WHAT THIS DOES NOT CLAIM. It does not make a historical content hash into a
typed payload identity. Cross-top-level equivalence -- the same bytes admitted
as ``bytes`` and as ``str`` hashing alike -- and unbound record fields are
separate contract issues and are untouched here. The size bound applies to the
STRUCTURED (dict) root only: top-level ``bytes`` and ``str`` are block content,
hashed directly, never reaching ``default=``, and bounding them would refuse
large-but-legitimate content writes. That exclusion is deliberate and is
flagged rather than silently assumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

#: Deepest nesting a payload may carry. Past this, refuse by name rather than
#: let the serializer meet a RecursionError with no useful message.
MAX_DEPTH = 32

#: Most values a payload may contain in total, counted across the whole tree.
#: A ledger entry is a record of one operation, not a data dump.
MAX_NODES = 10_000

#: Largest canonical serialization a STRUCTURED payload may produce, in bytes.
#: `MAX_NODES` is a count, not a size: ten thousand one-megabyte strings pass a
#: node bound and still produce ten gigabytes. Enforced while encoding, so a
#: refusal happens before the whole string is built.
MAX_PAYLOAD_BYTES = 1_048_576

#: Longest single string a structured payload may carry. Subsumed by
#: `MAX_PAYLOAD_BYTES`, but refused separately so the message names the path.
MAX_STRING_CHARS = 65_536

#: Widest integer, in bits. CPython refuses to render an int wider than
#: ``sys.get_int_max_str_digits()`` (4300 by default) and raises a bare
#: ValueError from inside ``json.dumps`` -- past admission, with no path in the
#: message. log10(2) = 0.30103, so 14_000 bits is at most 4_215 digits and
#: stays under that limit on a default interpreter. Measured with
#: ``bit_length()``, which costs nothing; ``len(str(n))`` would raise the very
#: error this bound exists to prevent.
MAX_INT_BITS = 14_000

#: Types JSON encodes natively. `bool` is listed FIRST and checked before
#: `int`, because `isinstance(True, int)` is true and the order is what keeps
#: a bool a bool.
_SCALARS = (bool, int, float, str, type(None))


class PayloadRejected(TypeError):
    """A new payload was refused at admission. Nothing was written.

    A TypeError subclass on purpose: the caller passed a value the ledger
    cannot record, which is the same class of programming error as passing one
    to ``json.dumps`` without ``default=``. Callers that need a typed value in
    a payload should encode it explicitly at the call site -- that is a
    decision with an owner, unlike ``default=str``, which is the same decision
    made silently everywhere at once.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


@dataclass(frozen=True)
class AdmittedPayload:
    """The exact bytes a writer hashes, produced by the check that admitted it.

    Frozen, and carrying ``bytes`` rather than a reference to the caller's
    object, so that what was checked and what is hashed are the same thing by
    construction rather than by two traversals agreeing.
    """

    kind: str
    preimage: bytes

    def digest(self) -> str:
        """SHA-256 of the preimage -- the value both ledgers store."""
        return hashlib.sha256(self.preimage).hexdigest()


def _canonical_json(payload: dict) -> str:
    """Check and ENCODE in one traversal; return the canonical rendering.

    Byte-identical to ``json.dumps(payload, sort_keys=True)`` for everything it
    admits, which is what keeps every accepted preimage exactly where it was:
    scalars are encoded by ``json.dumps`` itself, so string escaping and float
    repr are the stdlib's, and containers use the stdlib's own default
    separators ``", "`` and ``": "`` with ``sorted()`` keys.

    Written as one pass on purpose. Validating in one traversal and serializing
    in another lets a mutable object present one shape to each; here the value
    that is checked IS the value that is emitted.
    """
    out: list[str] = []
    seen: set[int] = set()
    state = {"nodes": 0, "size": 0}

    def emit(text: str, path: str) -> None:
        state["size"] += len(text)
        if state["size"] > MAX_PAYLOAD_BYTES:
            raise PayloadRejected(
                path or "<payload>",
                f"serializes to more than {MAX_PAYLOAD_BYTES} bytes; a ledger entry records an operation, not a dump",
            )
        out.append(text)

    def walk(node: Any, path: str, depth: int) -> None:
        state["nodes"] += 1
        if state["nodes"] > MAX_NODES:
            raise PayloadRejected(path, f"payload exceeds {MAX_NODES} values; a ledger entry records an operation, not a dump")
        if depth > MAX_DEPTH:
            raise PayloadRejected(path, f"nesting deeper than {MAX_DEPTH}")

        if isinstance(node, (dict, list, tuple)):
            marker = id(node)
            if marker in seen:
                raise PayloadRejected(path, "payload contains a cycle")
            seen.add(marker)

        if isinstance(node, dict):
            keys = []
            key_budget = max(MAX_NODES - state["nodes"], 0)
            for key in node:
                if not isinstance(key, str):
                    raise PayloadRejected(
                        f"{path or '<payload>'}[key]",
                        f"dict key is {type(key).__name__}, not str; non-string keys would render identically to their text form",
                    )
                if len(key) > MAX_STRING_CHARS:
                    raise PayloadRejected(
                        f"{path or '<payload>'}[key]",
                        f"dict key is {len(key)} characters, past the {MAX_STRING_CHARS}-character bound",
                    )
                if len(keys) >= key_budget:
                    raise PayloadRejected(
                        path or "<payload>",
                        f"payload exceeds {MAX_NODES} values; a ledger entry records an operation, not a dump",
                    )
                keys.append(key)
            emit("{", path)
            for i, key in enumerate(sorted(keys)):
                if i:
                    emit(", ", path)
                emit(json.dumps(key), path)
                emit(": ", path)
                walk(node[key], f"{path}.{key}" if path else key, depth + 1)
            emit("}", path)
            seen.discard(marker)
            return

        if isinstance(node, (list, tuple)):
            emit("[", path)
            for i, value in enumerate(node):
                if i:
                    emit(", ", path)
                walk(value, f"{path}[{i}]", depth + 1)
            emit("]", path)
            seen.discard(marker)
            return

        # bool BEFORE int: `isinstance(True, int)` is true, and this ordering
        # is what keeps True from being measured as an integer by the width
        # bound below. Unlike the previous shape of this check, the branch is
        # now LOAD-BEARING -- remove it and `True` reaches the int arm.
        if isinstance(node, bool):
            emit("true" if node else "false", path)
            return
        if isinstance(node, int):
            if node.bit_length() > MAX_INT_BITS:
                raise PayloadRejected(
                    path,
                    f"integer is {node.bit_length()} bits wide, past the {MAX_INT_BITS}-bit bound; "
                    f"CPython refuses to render it at all and would raise from inside the serializer",
                )
            emit(json.dumps(node), path)
            return
        if isinstance(node, float):
            if not math.isfinite(node):
                raise PayloadRejected(path, f"non-finite float {node!r}; NaN is not even equal to itself")
            emit(json.dumps(node), path)
            return
        if isinstance(node, str):
            if len(node) > MAX_STRING_CHARS:
                raise PayloadRejected(path, f"string is {len(node)} characters, past the {MAX_STRING_CHARS}-character bound")
            emit(json.dumps(node), path)
            return
        if node is None:
            emit("null", path)
            return

        raise PayloadRejected(
            path,
            f"{type(node).__name__} is not JSON-native. It would be stringified, and the value and its "
            f"rendering would then share one digest. Encode it explicitly at the call site instead.",
        )

    walk(payload, "", 0)
    return "".join(out)


def admit_payload(payload: Any, *, accepts_bytes: bool) -> AdmittedPayload:
    """Admit *payload* and return the exact preimage bytes to hash.

    THE DOOR. Called before any lock, tail read or byte is written, so a
    refusal leaves the file bytes, the record count and the in-memory tail
    exactly as they were.

    *accepts_bytes* is the calling API's own contract, not a preference: the
    evidence chain documents ``bytes | str | dict | None``, the audit chain
    documents ``dict | str | None``. A root outside the caller's contract is
    refused here rather than reaching ``default=str`` and being recorded as
    its own ``repr``.
    """
    if payload is None:
        return AdmittedPayload("none", b"")
    if isinstance(payload, bytes):
        if not accepts_bytes:
            raise PayloadRejected(
                "<payload>",
                "this ledger's payload contract is dict | str | None; bytes would be recorded as its repr, "
                "not its content. Decode it at the call site, or use the evidence chain, which documents bytes.",
            )
        # `bytes` is already immutable, so this IS the snapshot.
        return AdmittedPayload("bytes", payload)
    if isinstance(payload, str):
        return AdmittedPayload("str", payload.encode("utf-8"))
    if isinstance(payload, dict):
        return AdmittedPayload("json", _canonical_json(payload).encode("utf-8"))

    accepted = "bytes | str | dict | None" if accepts_bytes else "dict | str | None"
    raise PayloadRejected(
        "<payload>",
        f"{type(payload).__name__} is not a documented payload root; this ledger accepts {accepted}. "
        f"A root outside that contract would be stringified into a digest that records its repr rather than its content.",
    )


def validate_payload(payload: dict) -> None:
    """Structural check of a JSON-object payload. NOT the admission door.

    Kept as the standalone checker for a ``dict`` payload; :func:`admit_payload`
    is what a writer calls, because only it returns the preimage that closes the
    validate-then-hash gap. Passing a non-dict here raises rather than returning
    quietly: returning quietly for every unrecognised root is precisely how the
    earlier version let an arbitrary object through.
    """
    if not isinstance(payload, dict):
        raise PayloadRejected(
            "<payload>",
            f"validate_payload checks a dict payload; {type(payload).__name__} has no structure to check. "
            f"Call admit_payload(), which is the door and handles every documented root.",
        )
    _canonical_json(payload)
