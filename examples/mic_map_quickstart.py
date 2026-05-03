"""MIC/MAP quickstart — emit, parse, round-trip, stream.

Run with::

    python3 examples/mic_map_quickstart.py

mic-map is the STARGA serialization for MIND IR graphs (a typed
dataflow graph: symbols + types + values + output). It ships in two
wire formats:

* **mic@2** — line-oriented text, LLM-readable, git-friendly
* **mic-b** — varint binary, ~4x smaller than mic@2

Both formats encode the same Graph; round-trip is byte-identical.

Spec lives at https://github.com/star-ga/mind-spec/tree/main/spec/mic
"""

from io import BytesIO

from mind_mem.mic_map import (
    Arg,
    Graph,
    Node,
    Param,
    StreamValue,
    Type,
    emit_mic2,
    emit_micb,
    parse_mic2,
    parse_micb,
    parse_micb_stream,
)


def build_residual_block() -> Graph:
    """Tiny residual-block graph: matmul + bias + relu + add."""
    return Graph(
        types=[
            Type(dtype="f16", dims=("128", "128")),
            Type(dtype="f16", dims=("128",)),
        ],
        values=[
            Arg(name="X", type_idx=0),
            Param(name="W", type_idx=0),
            Param(name="b", type_idx=1),
            Node(opcode="m", inputs=(0, 1)),     # matmul X @ W
            Node(opcode="+", inputs=(3, 2)),     # + bias
            Node(opcode="r", inputs=(4,)),       # relu
            Node(opcode="+", inputs=(5, 0)),     # residual: + X
        ],
        output=6,
    )


def main() -> None:
    g = build_residual_block()

    # ---- emit ---------------------------------------------------------
    text = emit_mic2(g)
    binary = emit_micb(g)
    print(f"mic@2 text:   {len(text.encode('utf-8'))} bytes")
    print(f"mic-b binary: {len(binary)} bytes  ({len(binary)/len(text.encode('utf-8')):.0%})")

    # ---- round-trip ---------------------------------------------------
    assert parse_mic2(text) == g
    assert parse_micb(binary) == g
    print("round-trip: OK (both formats parse back to the original Graph)")

    # ---- streaming parser --------------------------------------------
    # parse_micb_stream yields events as bytes arrive — bounded peak
    # memory, regardless of input size. Use it for sockets, slow pipes,
    # or any case where you want to drop processed values rather than
    # holding the whole graph resident.
    print("\nstreaming events:")
    reader = BytesIO(binary)
    for event in parse_micb_stream(reader):
        if isinstance(event, StreamValue):
            kind = type(event.value).__name__
            print(f"  StreamValue #{event.index}: {kind}")
        else:
            print(f"  {type(event).__name__}")

    # ---- inspect via mm CLI ------------------------------------------
    # The same payload can be inspected from the command line:
    #   mm mic inspect graph.mic2
    #   mm mic inspect graph.micb --json
    # Or converted between formats:
    #   mm mic convert graph.mic2 --to micb -o graph.micb


if __name__ == "__main__":
    main()
