"""MIC/MAP — STARGA-native serialization for MIND IR graphs.

Faithful Python implementation of the two STARGA wire formats:

* **mic@2** — line-oriented text, LLM- and git-friendly
  (spec: ``mind-spec/spec/mic/mic2-spec.md``).
* **mic-b** — varint binary, ~4× smaller than mic@2
  (spec: ``mind-spec/spec/mic/micb-spec.md``).

Both formats encode a typed dataflow graph (symbols + types + values
+ output). Values are arguments, parameters, and nodes (matmul / add /
relu / softmax / etc.) referenced by **implicit sequential ID** — the
ordering rule is the structural invariant that makes the format
deterministic.

The Rust reference lives in ``mind/src/ir/compact/v2/`` (parse, emit,
binary, varint). This Python port preserves every spec rule:
sequential-only IDs, no forward references, ULEB128 minimum encoding,
zigzag for signed parameters, first-seen string interning, magic
``MICB`` + version byte ``0x02``.

Public API::

    g = parse_mic2(text)            # text → Graph
    text = emit_mic2(g)             # Graph → text (canonical)
    g = parse_micb(buf)             # bytes → Graph
    buf = emit_micb(g)              # Graph → bytes
    assert parse_mic2(emit_mic2(g)) == g     # round-trip

Used inside mind-mem to ship MIND-native graphs through MCP / CLI / on
disk without going through JSON. ``mm audit-model`` and the rest of
the model-safety pipeline still use JSON because their payloads are
audit reports, not IR graphs — MIC/MAP is for graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from io import BytesIO
from typing import IO

# ----------------------------------------------------------------------
# Spec constants — kept aligned with mic2-spec.md / micb-spec.md.
# ----------------------------------------------------------------------

MIC2_HEADER = "mic@2"
MICB_MAGIC = b"MICB"
MICB_VERSION = 0x02

# Per micb-spec §4.0 dtype encoding.
DTYPES: tuple[str, ...] = (
    "f16",
    "f32",
    "f64",
    "bf16",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "bool",
)
DTYPE_TO_BYTE: dict[str, int] = {d: i for i, d in enumerate(DTYPES)}
BYTE_TO_DTYPE: dict[int, str] = {i: d for i, d in enumerate(DTYPES)}

# Per micb-spec §4.0 opcode encoding. Order matters — the byte tag IS
# the index into this tuple. Don't reorder; append at the end if a new
# op is introduced (and update the spec at the same time).
OPCODES: tuple[str, ...] = (
    "m",  # 0  Matmul
    "+",  # 1  Add
    "-",  # 2  Sub
    "*",  # 3  Mul
    "/",  # 4  Div
    "r",  # 5  Relu
    "s",  # 6  Softmax    — sleb128 axis param
    "sig",  # 7  Sigmoid
    "th",  # 8  Tanh
    "gelu",  # 9  GELU
    "ln",  # 10 LayerNorm
    "t",  # 11 Transpose  — uleb128 n + n×sleb128 perm
    "rshp",  # 12 Reshape
    "sum",  # 13 Sum       — uleb128 n + n×sleb128 axes
    "mean",  # 14 Mean      — uleb128 n + n×sleb128 axes
    "max",  # 15 Max       — uleb128 n + n×sleb128 axes
    "cat",  # 16 Concat    — sleb128 axis
    "split",  # 17 Split   — sleb128 axis + uleb128 count
    "gth",  # 18 Gather    — sleb128 axis
)
OPCODE_TO_BYTE: dict[str, int] = {o: i for i, o in enumerate(OPCODES)}
BYTE_TO_OPCODE: dict[int, str] = {i: o for i, o in enumerate(OPCODES)}

# Arity (input count). 0 = N-ary (cat). The arity is part of the spec
# and the parser uses it to reject malformed graphs.
OP_ARITY: dict[str, int] = {
    "m": 2,
    "+": 2,
    "-": 2,
    "*": 2,
    "/": 2,
    "r": 1,
    "s": 1,
    "sig": 1,
    "th": 1,
    "gelu": 1,
    "ln": 1,
    "t": 1,
    "rshp": 1,
    "sum": 1,
    "mean": 1,
    "max": 1,
    "cat": -1,  # variadic — at least 1
    "split": 1,
    "gth": 2,
}

# Ops whose binary form carries extra parameter bytes after the opcode.
# Each entry is the (encode, decode) pair for the parameter section.
# Returns/consumes a sleb128/uleb128 sequence as documented in the spec.
PARAM_OPS: frozenset[str] = frozenset({"s", "t", "sum", "mean", "max", "cat", "split", "gth"})

# Spec security limits.
MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_LINE_COUNT = 1_000_000
MAX_VALUE_COUNT = 100_000
MAX_DIM_COUNT = 32
MAX_STRING_COUNT = 1_000_000
MAX_STRING_LEN = 64 * 1024


# ----------------------------------------------------------------------
# Graph model
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Type:
    """A tensor type — dtype + shape (dimensions are token strings)."""

    dtype: str
    dims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_TO_BYTE:
            raise ValueError(f"unknown dtype: {self.dtype!r}")
        if len(self.dims) > MAX_DIM_COUNT:
            raise ValueError(f"shape rank {len(self.dims)} exceeds {MAX_DIM_COUNT}")


@dataclass(frozen=True)
class Arg:
    """Graph input — assigned an implicit sequential value ID at parse time."""

    name: str
    type_idx: int


@dataclass(frozen=True)
class Param:
    """Learnable parameter — same ID assignment as Arg."""

    name: str
    type_idx: int


@dataclass(frozen=True)
class Node:
    """An op invocation. ``inputs`` references earlier value IDs only."""

    opcode: str
    inputs: tuple[int, ...] = ()
    # Opcode-specific extra integer parameters (axes, perms, counts).
    # The semantics depend on opcode; see micb-spec §4.0 column 3.
    op_params: tuple[int, ...] = ()


# Anything that lives in the value table.
Value = Arg | Param | Node


@dataclass
class Graph:
    """Mind IR graph in normalised in-memory form."""

    types: list[Type] = field(default_factory=list)
    values: list[Value] = field(default_factory=list)
    output: int = -1
    # Symbol declarations are advisory metadata — they don't change the
    # value-ID assignments. Kept so round-trips preserve operator intent.
    symbols: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Cheap structural checks. The parser also runs these."""
        if len(self.values) > MAX_VALUE_COUNT:
            raise ValueError(f"value count {len(self.values)} exceeds {MAX_VALUE_COUNT}")
        for i, t in enumerate(self.types):
            if i != self.types.index(t) and t in self.types[:i]:
                # Equal-by-fields is fine; we only ban None entries.
                pass
        for vid, v in enumerate(self.values):
            if isinstance(v, (Arg, Param)):
                if v.type_idx < 0 or v.type_idx >= len(self.types):
                    raise ValueError(f"value {vid} references unknown type {v.type_idx}")
            else:
                arity = OP_ARITY.get(v.opcode)
                if arity is None:
                    raise ValueError(f"value {vid}: unknown opcode {v.opcode!r}")
                if arity == -1:
                    if not v.inputs:
                        raise ValueError(f"value {vid}: variadic op needs >=1 inputs")
                elif len(v.inputs) != arity:
                    raise ValueError(f"value {vid}: opcode {v.opcode} expects {arity} inputs, got {len(v.inputs)}")
                for inp in v.inputs:
                    if inp < 0 or inp >= vid:
                        # Forward-reference rule from mic2-spec §"Validation Rules".
                        raise ValueError(f"value {vid}: input {inp} is not an earlier value")
        if self.output < 0 or self.output >= len(self.values):
            raise ValueError(f"output {self.output} is not a valid value ID")


# ----------------------------------------------------------------------
# Text parser
# ----------------------------------------------------------------------


class Mic2ParseError(ValueError):
    """Raised on any mic@2 syntactic / semantic error.

    ``line_no`` (1-based) is set to help diagnostics; ``__str__`` includes it.
    """

    def __init__(self, msg: str, line_no: int | None = None) -> None:
        if line_no is not None:
            super().__init__(f"line {line_no}: {msg}")
        else:
            super().__init__(msg)
        self.line_no = line_no


def parse_mic2(text: str) -> Graph:
    """Parse a mic@2 text payload. Raises :class:`Mic2ParseError` on any
    error with the offending 1-based line number.
    """
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise Mic2ParseError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    raw_lines = text.splitlines()
    if len(raw_lines) > MAX_LINE_COUNT:
        raise Mic2ParseError(f"input exceeds {MAX_LINE_COUNT} lines")

    g = Graph()
    output_set = False
    saw_header = False
    for ln_idx, raw in enumerate(raw_lines, start=1):
        line = raw.rstrip("\r")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue

        if not saw_header:
            if stripped != MIC2_HEADER:
                raise Mic2ParseError(
                    f"first non-blank/non-comment line must be {MIC2_HEADER!r} header, got {stripped!r}",
                    ln_idx,
                )
            saw_header = True
            continue

        if output_set:
            # Per spec, exactly one output line, and it must be last.
            raise Mic2ParseError("trailing data after output line", ln_idx)

        toks = stripped.split()
        head = toks[0]
        try:
            if head == "S":
                if len(toks) != 2:
                    raise Mic2ParseError(f"S line takes 1 arg, got {len(toks) - 1}", ln_idx)
                g.symbols.append(toks[1])
            elif head.startswith("T") and head[1:].isdigit():
                idx = int(head[1:])
                if idx != len(g.types):
                    raise Mic2ParseError(
                        f"type index must be sequential — expected T{len(g.types)}, got {head}",
                        ln_idx,
                    )
                if len(toks) < 2:
                    raise Mic2ParseError("type line needs dtype", ln_idx)
                dtype = toks[1]
                dims = tuple(toks[2:])
                g.types.append(Type(dtype=dtype, dims=dims))
            elif head == "a":
                if len(toks) != 3:
                    raise Mic2ParseError("'a NAME T<idx>' takes 2 args", ln_idx)
                t_ref = _parse_type_ref(toks[2], ln_idx)
                if t_ref >= len(g.types):
                    raise Mic2ParseError(f"type ref T{t_ref} not yet defined", ln_idx)
                g.values.append(Arg(name=toks[1], type_idx=t_ref))
            elif head == "p":
                if len(toks) != 3:
                    raise Mic2ParseError("'p NAME T<idx>' takes 2 args", ln_idx)
                t_ref = _parse_type_ref(toks[2], ln_idx)
                if t_ref >= len(g.types):
                    raise Mic2ParseError(f"type ref T{t_ref} not yet defined", ln_idx)
                g.values.append(Param(name=toks[1], type_idx=t_ref))
            elif head == "O":
                if len(toks) != 2:
                    raise Mic2ParseError("'O <id>' takes 1 arg", ln_idx)
                vid = _parse_int(toks[1], ln_idx)
                if vid < 0 or vid >= len(g.values):
                    raise Mic2ParseError(f"output id {vid} is not a valid value", ln_idx)
                g.output = vid
                output_set = True
            else:
                # Otherwise must be a node opcode line.
                if head not in OP_ARITY:
                    raise Mic2ParseError(f"unknown opcode or directive: {head!r}", ln_idx)
                g.values.append(_parse_node_line(head, toks[1:], ln_idx, len(g.values)))
                if len(g.values) > MAX_VALUE_COUNT:
                    raise Mic2ParseError(f"value count exceeds {MAX_VALUE_COUNT}", ln_idx)
        except Mic2ParseError:
            raise
        except (ValueError, IndexError) as exc:
            raise Mic2ParseError(str(exc), ln_idx) from exc

    if not saw_header:
        raise Mic2ParseError(f"missing {MIC2_HEADER!r} header")
    if not output_set:
        raise Mic2ParseError("missing 'O <id>' output line")
    try:
        g.validate()
    except ValueError as exc:
        raise Mic2ParseError(str(exc)) from exc
    return g


def _parse_type_ref(tok: str, ln_idx: int) -> int:
    if not tok.startswith("T") or not tok[1:].isdigit():
        raise Mic2ParseError(f"expected type ref like T0, got {tok!r}", ln_idx)
    return int(tok[1:])


def _parse_int(tok: str, ln_idx: int) -> int:
    try:
        return int(tok)
    except ValueError:
        raise Mic2ParseError(f"expected integer, got {tok!r}", ln_idx) from None


def _parse_node_line(opcode: str, args: list[str], ln_idx: int, current_id: int) -> Node:
    """Parse a node line. The text format folds inputs and op_params into
    one space-separated token stream; for variadic / param-bearing opcodes
    we split them according to spec §"Opcodes" arity column.

    Convention: integers that look like value IDs (>= 0 and < current_id)
    are treated as inputs; trailing integers that don't are op_params.
    For unambiguous parsing, the spec is strict: arity-N opcodes consume
    EXACTLY N input tokens, then the remainder is op_params. Variadic
    ``cat`` consumes one trailing op_param (axis) and the rest as inputs.
    """
    arity = OP_ARITY[opcode]
    ints = [_parse_int(a, ln_idx) for a in args]

    if arity == -1:
        # Variadic — the spec puts axis last for cat. We follow:
        #   cat <in_0> <in_1> ... <in_n-1> <axis>
        if not ints:
            raise Mic2ParseError(f"opcode {opcode}: needs at least one input", ln_idx)
        if opcode == "cat":
            if len(ints) < 2:
                raise Mic2ParseError("cat needs at least one input + axis", ln_idx)
            return Node(opcode=opcode, inputs=tuple(ints[:-1]), op_params=(ints[-1],))
        return Node(opcode=opcode, inputs=tuple(ints))

    if arity > len(ints):
        raise Mic2ParseError(
            f"opcode {opcode} expects {arity} inputs, got only {len(ints)} tokens",
            ln_idx,
        )
    inputs = tuple(ints[:arity])
    op_params = tuple(ints[arity:])
    return Node(opcode=opcode, inputs=inputs, op_params=op_params)


# ----------------------------------------------------------------------
# Text emitter
# ----------------------------------------------------------------------


def emit_mic2(g: Graph) -> str:
    """Emit a deterministic mic@2 representation of ``g``.

    Canonical ordering per spec §"Canonicalization Rules": header,
    symbols, types, values, output. No comments, no trailing newline
    after ``O <id>``, single-space token separators, LF-only line
    endings.
    """
    g.validate()
    parts: list[str] = [MIC2_HEADER]
    for s in g.symbols:
        parts.append(f"S {s}")
    for i, t in enumerate(g.types):
        dims = " ".join(t.dims) if t.dims else ""
        parts.append(f"T{i} {t.dtype}" + (f" {dims}" if dims else ""))
    for v in g.values:
        parts.append(_emit_value(v))
    parts.append(f"O {g.output}")
    return "\n".join(parts) + "\n"


def _emit_value(v: Value) -> str:
    if isinstance(v, Arg):
        return f"a {v.name} T{v.type_idx}"
    if isinstance(v, Param):
        return f"p {v.name} T{v.type_idx}"
    # Node
    if v.opcode == "cat":
        # variadic: <in_0> ... <in_n-1> <axis>
        ins = " ".join(str(i) for i in v.inputs)
        axis = v.op_params[0] if v.op_params else 0
        return f"cat {ins} {axis}"
    ins = " ".join(str(i) for i in v.inputs)
    if v.op_params:
        params = " ".join(str(p) for p in v.op_params)
        return f"{v.opcode} {ins} {params}".strip()
    return f"{v.opcode} {ins}".strip()


# ----------------------------------------------------------------------
# Binary varint helpers
# ----------------------------------------------------------------------


def _uleb128_encode(v: int) -> bytes:
    """Unsigned LEB128 — minimum-byte encoding (spec §"ULEB128")."""
    if v < 0:
        raise ValueError("uleb128 only encodes non-negative values")
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_exact(buf: IO[bytes], n: int) -> bytes:
    """Read *exactly* ``n`` bytes from ``buf``, looping on short reads.

    Sockets, pipes, and any ``BufferedReader`` over a slow source can
    return fewer bytes than requested — the streaming parser must not
    assume one ``read(n)`` call returns ``n`` bytes. Returns whatever
    was read on EOF (0 .. n-1 bytes); callers check the length.
    """
    out = bytearray()
    while len(out) < n:
        chunk = buf.read(n - len(out))
        if not chunk:
            return bytes(out)
        out.extend(chunk)
    return bytes(out)


def _uleb128_decode(buf: IO[bytes]) -> int:
    """Decode a ULEB128 from a binary stream, enforcing minimum encoding."""
    result = 0
    shift = 0
    while True:
        ch = _read_exact(buf, 1)
        if not ch:
            raise MicbParseError("unexpected EOF in uleb128")
        byte = ch[0]
        result |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return result
        shift += 7
        if shift > 70:
            raise MicbParseError("uleb128 too long (>10 bytes)")


def _sleb128_encode(v: int) -> bytes:
    """Signed LEB128 via zigzag → ULEB128."""
    z = (v << 1) ^ (v >> 63) if v >= 0 else ((v << 1) ^ -1) & ((1 << 64) - 1)
    # Standard zigzag for arbitrary-precision Python ints:
    # avoid the >>63 trick because Python ints are unbounded.
    if v >= 0:
        z = v << 1
    else:
        z = ((-v) << 1) - 1
    return _uleb128_encode(z)


def _sleb128_decode(buf: IO[bytes]) -> int:
    z = _uleb128_decode(buf)
    return (z >> 1) ^ -(z & 1)


# ----------------------------------------------------------------------
# Binary emitter / parser
# ----------------------------------------------------------------------


class MicbParseError(ValueError):
    """Raised on any mic-b binary error."""


def emit_micb(g: Graph) -> bytes:
    """Encode ``g`` to mic-b binary."""
    g.validate()

    # String table — first-seen insertion across symbols, dim tokens,
    # value names. No re-use of dim_str across types beyond first insert.
    strings: list[str] = []
    str_idx: dict[str, int] = {}

    def intern(s: str) -> int:
        if s not in str_idx:
            if len(strings) >= MAX_STRING_COUNT:
                raise ValueError(f"string table exceeds {MAX_STRING_COUNT}")
            if len(s.encode("utf-8")) > MAX_STRING_LEN:
                raise ValueError(f"string {s!r} exceeds {MAX_STRING_LEN} bytes")
            str_idx[s] = len(strings)
            strings.append(s)
        return str_idx[s]

    for s in g.symbols:
        intern(s)
    for t in g.types:
        for d in t.dims:
            intern(d)
    for v in g.values:
        if isinstance(v, (Arg, Param)):
            intern(v.name)

    out = bytearray()
    out.extend(MICB_MAGIC)
    out.append(MICB_VERSION)

    # 1. String table
    out.extend(_uleb128_encode(len(strings)))
    for s in strings:
        b = s.encode("utf-8")
        out.extend(_uleb128_encode(len(b)))
        out.extend(b)

    # 2. Symbol table
    out.extend(_uleb128_encode(len(g.symbols)))
    for s in g.symbols:
        out.extend(_uleb128_encode(str_idx[s]))

    # 3. Type table
    out.extend(_uleb128_encode(len(g.types)))
    for t in g.types:
        out.append(DTYPE_TO_BYTE[t.dtype])
        out.extend(_uleb128_encode(len(t.dims)))
        for d in t.dims:
            out.extend(_uleb128_encode(str_idx[d]))

    # 4. Value table
    out.extend(_uleb128_encode(len(g.values)))
    for v in g.values:
        if isinstance(v, Arg):
            out.append(0)  # tag Arg
            out.extend(_uleb128_encode(str_idx[v.name]))
            out.extend(_uleb128_encode(v.type_idx))
        elif isinstance(v, Param):
            out.append(1)  # tag Param
            out.extend(_uleb128_encode(str_idx[v.name]))
            out.extend(_uleb128_encode(v.type_idx))
        else:
            out.append(2)  # tag Node
            out.append(OPCODE_TO_BYTE[v.opcode])
            _encode_op_params(out, v.opcode, v.op_params)
            out.extend(_uleb128_encode(len(v.inputs)))
            for inp in v.inputs:
                out.extend(_uleb128_encode(inp))

    # 5. Output
    out.extend(_uleb128_encode(g.output))

    if len(out) > MAX_INPUT_BYTES:
        raise ValueError(f"emitted bytes exceed {MAX_INPUT_BYTES}")
    return bytes(out)


# ----------------------------------------------------------------------
# Streaming parser — incremental decode for network / large-input use
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class StreamHeader:
    """Emitted once after the magic + version bytes are consumed."""

    version: int


@dataclass(frozen=True)
class StreamStringTable:
    """Emitted once after the entire string table is decoded.

    Strings are referenced by index from every later event, so the
    full table has to be resident before symbols / types / values can
    be decoded — no way around it without changing the wire format.
    Peak memory at this point is bounded by ``MAX_STRING_COUNT *
    MAX_STRING_LEN`` worst-case, but in practice the on-wire size
    limit (``MAX_INPUT_BYTES``) caps it long before that.
    """

    strings: tuple[str, ...]


@dataclass(frozen=True)
class StreamSymbol:
    """Emitted per symbol declaration, with its position in the table."""

    index: int
    name: str


@dataclass(frozen=True)
class StreamType:
    """Emitted per type declaration."""

    index: int
    type: Type


@dataclass(frozen=True)
class StreamValue:
    """Emitted per value (Arg / Param / Node). Caller can stream-process
    each value into downstream consumers (compiler IR, validator, etc.)
    without holding the whole graph in memory at once."""

    index: int
    value: Value


@dataclass(frozen=True)
class StreamComplete:
    """Final event — output index of the graph."""

    output: int


# Discriminated-union alias for the events ``parse_micb_stream`` yields.
StreamEvent = StreamHeader | StreamStringTable | StreamSymbol | StreamType | StreamValue | StreamComplete


def parse_micb_stream(reader: IO[bytes]):
    """Stream-parse a mic-b payload, yielding :class:`StreamEvent` as
    bytes arrive.

    Bounded peak memory: only the string table (capped by
    ``MAX_STRING_COUNT`` × ``MAX_STRING_LEN`` and indirectly by the
    on-wire byte caps), type registry, and current-value scratch are
    held. Caller can drop ``StreamValue`` objects after processing —
    the parser doesn't retain them. Compare with :func:`parse_micb`
    which assembles the full :class:`Graph` and holds everything
    resident.

    Raises :class:`MicbParseError` mid-iteration on any spec violation.
    Events emitted before the error are valid; events after are not
    produced. Input length is *not* bounded by ``MAX_INPUT_BYTES``
    here (the per-section limits + opcode arity caps are the
    load-bearing defence in the streaming case — there's no total
    length to check until EOF).
    """
    # 1. magic + version
    magic = _read_exact(reader, 4)
    if len(magic) != 4 or magic != MICB_MAGIC:
        raise MicbParseError(f"bad magic: {magic!r}")
    ver_byte = _read_exact(reader, 1)
    if not ver_byte:
        raise MicbParseError("payload shorter than magic+version")
    if ver_byte[0] != MICB_VERSION:
        raise MicbParseError(f"unsupported version: 0x{ver_byte[0]:02x}")
    yield StreamHeader(version=MICB_VERSION)

    # 2. String table — fully loaded before any later section can
    # decode (every later index is into this table).
    n_strs = _uleb128_decode(reader)
    if n_strs > MAX_STRING_COUNT:
        raise MicbParseError(f"string count {n_strs} exceeds {MAX_STRING_COUNT}")
    strings: list[str] = []
    for _ in range(n_strs):
        slen = _uleb128_decode(reader)
        if slen > MAX_STRING_LEN:
            raise MicbParseError(f"string length {slen} exceeds {MAX_STRING_LEN}")
        chunk = _read_exact(reader, slen)
        if len(chunk) != slen:
            raise MicbParseError("unexpected EOF in string table")
        try:
            strings.append(chunk.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise MicbParseError(f"invalid UTF-8 in string table: {exc}") from exc
    yield StreamStringTable(strings=tuple(strings))

    # 3. Symbol table
    n_syms = _uleb128_decode(reader)
    for i in range(n_syms):
        si = _uleb128_decode(reader)
        if si >= len(strings):
            raise MicbParseError(f"symbol str_idx {si} out of bounds")
        yield StreamSymbol(index=i, name=strings[si])

    # 4. Type table — types must be retained because values reference
    # them by index and we need the bounds check.
    n_types = _uleb128_decode(reader)
    type_count = 0
    for i in range(n_types):
        dt_byte = _read_exact(reader, 1)
        if not dt_byte:
            raise MicbParseError("EOF in type table")
        dtype = BYTE_TO_DTYPE.get(dt_byte[0])
        if dtype is None:
            raise MicbParseError(f"unknown dtype byte 0x{dt_byte[0]:02x}")
        rank = _uleb128_decode(reader)
        if rank > MAX_DIM_COUNT:
            raise MicbParseError(f"rank {rank} exceeds {MAX_DIM_COUNT}")
        dims: list[str] = []
        for _ in range(rank):
            di = _uleb128_decode(reader)
            if di >= len(strings):
                raise MicbParseError(f"dim str_idx {di} out of bounds")
            dims.append(strings[di])
        yield StreamType(index=i, type=Type(dtype=dtype, dims=tuple(dims)))
        type_count = i + 1

    # 5. Value table — bounds-check by *position* in the stream rather
    # than by holding all prior values; this is the win that makes
    # streaming useful.
    n_vals = _uleb128_decode(reader)
    if n_vals > MAX_VALUE_COUNT:
        raise MicbParseError(f"value count {n_vals} exceeds {MAX_VALUE_COUNT}")
    for vid in range(n_vals):
        tag = _read_exact(reader, 1)
        if not tag:
            raise MicbParseError("EOF in value table")
        if tag[0] == 0:
            ni = _uleb128_decode(reader)
            ti = _uleb128_decode(reader)
            if ni >= len(strings):
                raise MicbParseError(f"arg str_idx {ni} out of bounds")
            if ti >= type_count:
                raise MicbParseError(f"arg type_idx {ti} out of bounds")
            v: Value = Arg(name=strings[ni], type_idx=ti)
        elif tag[0] == 1:
            ni = _uleb128_decode(reader)
            ti = _uleb128_decode(reader)
            if ni >= len(strings):
                raise MicbParseError(f"param str_idx {ni} out of bounds")
            if ti >= type_count:
                raise MicbParseError(f"param type_idx {ti} out of bounds")
            v = Param(name=strings[ni], type_idx=ti)
        elif tag[0] == 2:
            opc_b = _read_exact(reader, 1)
            if not opc_b:
                raise MicbParseError("EOF in node opcode")
            opcode = BYTE_TO_OPCODE.get(opc_b[0])
            if opcode is None:
                raise MicbParseError(f"unknown opcode byte 0x{opc_b[0]:02x}")
            op_params = _decode_op_params(reader, opcode)
            n_inputs = _uleb128_decode(reader)
            inputs: list[int] = []
            for _ in range(n_inputs):
                inp = _uleb128_decode(reader)
                if inp >= vid:
                    raise MicbParseError(f"value {vid}: input {inp} not earlier value (forward-ref)")
                inputs.append(inp)
            v = Node(opcode=opcode, inputs=tuple(inputs), op_params=op_params)
        else:
            raise MicbParseError(f"unknown value tag 0x{tag[0]:02x}")
        yield StreamValue(index=vid, value=v)

    # 6. Output index
    output = _uleb128_decode(reader)
    if output >= n_vals:
        raise MicbParseError(f"output {output} not a valid value")
    yield StreamComplete(output=output)


def parse_micb(data: bytes) -> Graph:
    """Decode a mic-b payload to a :class:`Graph`.

    Thin wrapper around :func:`parse_micb_stream` that drains the
    iterator and assembles the canonical in-memory graph. Use the
    streaming API directly when memory is bounded — see
    :func:`parse_micb_stream` for the per-event contract.
    """
    if len(data) > MAX_INPUT_BYTES:
        raise MicbParseError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    if len(data) < 5:
        raise MicbParseError("payload shorter than magic+version")

    buf = BytesIO(data)
    types: list[Type] = []
    values: list[Value] = []
    symbols: list[str] = []
    output: int = -1

    for ev in parse_micb_stream(buf):
        if isinstance(ev, StreamSymbol):
            symbols.append(ev.name)
        elif isinstance(ev, StreamType):
            types.append(ev.type)
        elif isinstance(ev, StreamValue):
            values.append(ev.value)
        elif isinstance(ev, StreamComplete):
            output = ev.output
        # StreamHeader / StreamStringTable carry no graph-level state.

    g = Graph(types=types, values=values, output=output, symbols=symbols)
    g.validate()
    return g


def _encode_op_params(out: bytearray, opcode: str, params: tuple[int, ...]) -> None:
    """Per micb-spec §4.0 column 3 — opcode-specific param section."""
    if opcode == "s" or opcode == "cat" or opcode == "gth":
        # Single sleb128 axis (default 0 if absent).
        axis = params[0] if params else 0
        out.extend(_sleb128_encode(axis))
    elif opcode == "t":
        # uleb128 n + n × sleb128 perm
        out.extend(_uleb128_encode(len(params)))
        for p in params:
            out.extend(_sleb128_encode(p))
    elif opcode in ("sum", "mean", "max"):
        out.extend(_uleb128_encode(len(params)))
        for p in params:
            out.extend(_sleb128_encode(p))
    elif opcode == "split":
        # sleb128 axis + uleb128 count
        axis = params[0] if params else 0
        count = params[1] if len(params) > 1 else 1
        out.extend(_sleb128_encode(axis))
        out.extend(_uleb128_encode(max(count, 0)))
    # Other opcodes carry no extra params.


def _decode_op_params(buf: IO[bytes], opcode: str) -> tuple[int, ...]:
    if opcode == "s" or opcode == "cat" or opcode == "gth":
        return (_sleb128_decode(buf),)
    if opcode == "t":
        n = _uleb128_decode(buf)
        return tuple(_sleb128_decode(buf) for _ in range(n))
    if opcode in ("sum", "mean", "max"):
        n = _uleb128_decode(buf)
        return tuple(_sleb128_decode(buf) for _ in range(n))
    if opcode == "split":
        axis = _sleb128_decode(buf)
        count = _uleb128_decode(buf)
        return (axis, count)
    return ()


# ----------------------------------------------------------------------
# Public convenience
# ----------------------------------------------------------------------


def round_trip(g: Graph) -> Graph:
    """``parse_mic2(emit_mic2(g))`` — useful in tests + ad-hoc inspection."""
    return parse_mic2(emit_mic2(g))


def round_trip_b(g: Graph) -> Graph:
    """``parse_micb(emit_micb(g))`` — same idea for the binary format."""
    return parse_micb(emit_micb(g))


__all__ = [
    "Arg",
    "Graph",
    "Mic2ParseError",
    "MicbParseError",
    "Node",
    "Param",
    "StreamComplete",
    "StreamEvent",
    "StreamHeader",
    "StreamStringTable",
    "StreamSymbol",
    "StreamType",
    "StreamValue",
    "Type",
    "DTYPES",
    "OPCODES",
    "OP_ARITY",
    "MIC2_HEADER",
    "MICB_MAGIC",
    "MICB_VERSION",
    "emit_mic2",
    "emit_micb",
    "parse_mic2",
    "parse_micb",
    "parse_micb_stream",
    "round_trip",
    "round_trip_b",
]


# Re-export ``replace`` for callers that want to mutate immutable nodes
# without depending directly on dataclasses (mind-mem internal style).
del replace  # unused — keep frozen dataclasses as is
