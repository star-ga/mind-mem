# MIC/MAP — MIND IR Graph Serialization

MIND-Mem ships **MIC/MAP**, the STARGA-native serialization formats
for MIND IR graphs (typed dataflow graphs: symbols + types + values
+ output). Two wire formats:

| Format    | Purpose                                                              |
|-----------|----------------------------------------------------------------------|
| **mic@2** | Line-oriented text, LLM-readable, git-friendly                       |
| **mic-b** | Varint binary, ~4x smaller than mic@2, byte-stream parseable         |

Both encode the same graph; round-trip is byte-identical.

The **canonical wire-format spec** lives at
[`star-ga/mind-spec/spec/mic/`](https://github.com/star-ga/mind-spec/tree/main/spec/mic).
The Rust reference impl is at
[`star-ga/mind/src/ir/compact/v2/`](https://github.com/star-ga/mind/tree/main/src/ir/compact/v2).
This package implements the same wire formats faithfully —
implementations are interchangeable on the wire.

## Status

- v3.8.5 — pure-Python codec landed (parse + emit, both formats)
- v3.8.8 — fuzz harness + adversarial DoS corpus + benchmarks
- v3.8.9 — streaming parser (`parse_micb_stream`) for bounded peak memory
- v3.8.10 — optional Cython accelerator (`mind-mem[accelerated]`)
- v3.8.11 — MCP tools + `mm mic` CLI + this doc

## Python API

```python
from mind_mem.mic_map import (
    Graph, Type, Arg, Param, Node,
    parse_mic2, emit_mic2,
    parse_micb, emit_micb,
    parse_micb_stream,
)

g = Graph(
    types=[Type(dtype="f16", dims=("128", "128"))],
    values=[
        Arg(name="X", type_idx=0),
        Param(name="W", type_idx=0),
        Node(opcode="m", inputs=(0, 1)),    # matmul X @ W
    ],
    output=2,
)

text   = emit_mic2(g)
binary = emit_micb(g)
assert parse_mic2(text)  == g
assert parse_micb(binary) == g
```

A runnable end-to-end example with the streaming parser lives at
[`examples/mic_map_quickstart.py`](../examples/mic_map_quickstart.py).

## CLI: `mm mic`

Two subcommands. Both auto-detect the input format from the file's
header (mic@2 plain text vs. `MICB` magic bytes).

### `mm mic convert <file> --to {mic2|micb} [-o <file>]`

Convert a graph between mic@2 and mic-b. Round-trips byte-identically.
Output goes to stdout by default; `-o <file>` writes to disk (binary
written raw, text as UTF-8).

```bash
mm mic convert graph.mic2 --to micb -o graph.micb
mm mic convert graph.micb --to mic2          # stdout
```

### `mm mic inspect <file> [--json]`

Print a structural summary — type count, value count, output index,
per-value tag (Arg / Param / Node + opcode):

```bash
$ mm mic inspect graph.mic2
format:        mic2
types:         1
values:        3
output:        #2

Types:
  T0: f16(128, 128)

Values:
  #  0 arg     X : T0
  #  1 param   W : T0
  #  2 node    m(#0, #1)
```

`--json` emits machine-readable JSON instead.

## MCP tools

Two MCP tools are registered on the MIND-Mem MCP server (visible to
any agent connected via the standard MCP wiring):

### `mic_convert`

Convert between mic@2 (text) and mic-b (binary). Round-trips
byte-for-byte. Inputs are size-bounded at 8 MiB.

```json
{
  "input": "mic@2\nT 0 f16 128 128\na X 0\np W 0\nm 0 1\no 2\n",
  "input_format": "auto",
  "output_format": "micb"
}
```

Returns:

```json
{
  "ok": true,
  "output": "<base64-encoded mic-b>",
  "input_format": "mic2",
  "output_format": "micb",
  "byte_size": 33,
  "value_count": 3,
  "type_count": 1
}
```

For mic-b inputs, the `input` field is base64-encoded bytes (the
on-the-wire format is binary).

### `mic_inspect`

Structural summary of any conforming MIC payload. Same shape as the
`mm mic inspect --json` output:

```json
{
  "ok": true,
  "input_format": "mic2",
  "type_count": 1,
  "value_count": 3,
  "output_idx": 2,
  "types": [{"index": 0, "dtype": "f16", "dims": ["128", "128"]}],
  "values": [
    {"index": 0, "kind": "arg", "name": "X", "type_idx": 0},
    {"index": 1, "kind": "param", "name": "W", "type_idx": 0},
    {"index": 2, "kind": "node", "opcode": "m", "inputs": [0, 1]}
  ]
}
```

## Performance

The codec is pure-Python by default — zero new runtime dependencies,
works on every platform MIND-Mem ships to. For higher throughput,
opt into the Cython accelerator:

```bash
pip install --upgrade 'mind-mem[accelerated]'
```

This compiles `_mic_map_accel.pyx` into a per-platform extension at
install time. Bench delta on the residual block (single-core,
accelerator vs pure Python):

| Graph size                     | parse_micb |
|--------------------------------|------------|
| Small (residual block, 7 vals) | +16 %      |
| Medium (transformer layer)     | +20 %      |
| Large (200-layer deep stack)   | +36 %      |

The pure-Python fallback is always present; behaviour is bit-identical
either way. `_ACCEL_AVAILABLE: bool` reports the truth at runtime.

## Streaming parser

For large or live-arriving payloads, parse incrementally:

```python
from io import BytesIO
from mind_mem.mic_map import parse_micb_stream, StreamValue

reader = BytesIO(binary)   # or socket, file, BufferedReader
for event in parse_micb_stream(reader):
    if isinstance(event, StreamValue):
        process(event.value)   # drop after processing
```

Six event types in spec order: `StreamHeader`, `StreamStringTable`,
`StreamSymbol`, `StreamType`, `StreamValue`, `StreamComplete`.
Bounded peak memory regardless of input size.

## Wire-format invariants

The codec enforces every spec rule (tested in `tests/test_mic_map*.py`):

- **Round-trip identity** — `parse(emit(g)) == g` for both formats
- **Sequential IDs only** — values may only reference earlier values;
  forward references are rejected at parse time
- **Minimum varint encoding** — ULEB128 / SLEB128 inputs that use more
  bytes than necessary (or extend past the 70-bit shift cap) are rejected
- **Magic + version match** — `mic-b` requires `MICB` + version `0x02`
- **No exception leakage** — fuzz harness asserts the parsers raise only
  `Mic2ParseError` / `MicbParseError` (or succeed); arbitrary bytes never
  produce `UnicodeDecodeError`, `IndexError`, etc.
- **Adversarial DoS bound** — every adversarial input returns within
  500 ms wall clock; varint bombs, length-prefix overflow, truncation
  at every layer covered

## Use cases

- **MCP / agents**: serialize a model graph for storage / transport,
  parse one received from another agent
- **CI**: assert "this graph has at most N nodes" or "no `auto_map`
  opcode" via `mm mic inspect --json | jq`
- **Debugging**: convert mic-b ↔ mic@2 to read in a text editor
- **Cross-substrate identity** (future): canonical hash of the mic-b
  bytes is the substrate-independent model identity (see the MIC/MAP
  v15 patent provisional)
