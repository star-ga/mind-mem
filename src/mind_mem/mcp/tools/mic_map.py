"""MIC/MAP serialization MCP tools — wraps ``mind_mem.mic_map``.

Two tools, both stateless, both pure-Python (zero new dependencies):

* ``mic_convert_tool``  — convert between mic@2 (text) and mic-b
  (binary) representations of a MIND IR graph; round-trips byte-for-byte.
* ``mic_inspect_tool``  — return a structural summary of any
  conforming MIC payload (version, type count, value count, output
  index, per-value tag) without forcing the caller to reason about
  the wire format.

The MIC/MAP spec is at ``mind-spec/spec/mic/`` (canonical) — this
package implements the same wire formats faithfully (see
``mind_mem.mic_map``).

Inputs are size-bounded by ``MAX_INPUT_BYTES`` so an MCP caller
cannot blow up the server with a 10 GiB payload; the limit is
generous (8 MiB) but bounded.
"""

from __future__ import annotations

import base64
import binascii
import json

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")

MAX_INPUT_BYTES = 8 * 1024 * 1024  # 8 MiB — generous; protects MCP server from DoS.

_VALID_INPUT_FMTS = {"auto", "mic2", "micb"}
_VALID_OUTPUT_FMTS = {"mic2", "micb"}


def _err(msg: str) -> str:
    """Wrap a one-line error in the canonical MCP envelope."""
    return json.dumps(
        {"_schema_version": MCP_SCHEMA_VERSION, "ok": False, "error": msg},
        indent=2,
    )


def _decode_input(payload: str, fmt: str) -> tuple[bytes | str | None, str | None]:
    """Decode an MCP-tool ``input`` string into the appropriate Python
    value for the chosen format.

    For ``mic2`` the payload is taken verbatim as text.
    For ``micb`` the payload is base64-decoded into bytes.
    For ``auto`` we sniff: payload starting with ``mic@2`` is text,
    otherwise we try base64-decode and fall back to UTF-8 bytes.

    Returns ``(decoded, error)``. On success ``error`` is ``None`` and
    ``decoded`` is either ``str`` (for mic@2) or ``bytes`` (for mic-b).
    """
    if not isinstance(payload, str):
        return None, "input must be a string"
    if len(payload.encode("utf-8", errors="ignore")) > MAX_INPUT_BYTES:
        return None, f"input exceeds {MAX_INPUT_BYTES} bytes"
    if fmt == "mic2":
        return payload, None
    if fmt == "micb":
        try:
            return base64.b64decode(payload, validate=True), None
        except (binascii.Error, ValueError) as exc:
            return None, f"micb input must be valid base64: {exc}"
    # auto-detect
    if payload.lstrip().startswith("mic@2"):
        return payload, None
    # Try base64 first; if it parses to something starting with MICB magic,
    # it's mic-b. Otherwise assume text.
    try:
        decoded = base64.b64decode(payload, validate=True)
        if decoded.startswith(b"MICB"):
            return decoded, None
    except (binascii.Error, ValueError):
        pass
    return payload, None  # treat as text mic@2


def _detect_format(decoded: bytes | str) -> str:
    """Return ``mic2`` or ``micb`` based on the decoded payload's shape."""
    if isinstance(decoded, bytes):
        return "micb"
    return "mic2"


def _summarize(graph) -> dict:
    """Return a structural summary of a parsed Graph as plain dict."""
    from mind_mem.mic_map import Arg, Node, Param

    values: list[dict] = []
    for i, v in enumerate(graph.values):
        if isinstance(v, Arg):
            values.append({"index": i, "kind": "arg", "name": v.name, "type_idx": v.type_idx})
        elif isinstance(v, Param):
            values.append({"index": i, "kind": "param", "name": v.name, "type_idx": v.type_idx})
        elif isinstance(v, Node):
            values.append(
                {
                    "index": i,
                    "kind": "node",
                    "opcode": v.opcode,
                    "inputs": list(v.inputs),
                }
            )
        else:  # pragma: no cover - defensive
            values.append({"index": i, "kind": "unknown"})
    return {
        "type_count": len(graph.types),
        "value_count": len(graph.values),
        "output_idx": graph.output,
        "types": [{"index": i, "dtype": t.dtype, "dims": list(t.dims)} for i, t in enumerate(graph.types)],
        "values": values,
    }


@mcp_tool_observe
def mic_convert_tool(
    input: str,
    input_format: str = "auto",
    output_format: str = "mic2",
) -> str:
    """Convert a MIND IR graph between mic@2 (text) and mic-b (binary).

    Round-trips byte-for-byte: ``mic_convert(emit(g, X), input_format=X,
    output_format=Y)`` produces the same bytes as ``emit(g, Y)``.

    Args:
        input: The graph payload. For ``mic2`` this is the plain-text
            content. For ``micb`` this is base64-encoded bytes (the
            on-the-wire format). For ``auto`` the format is sniffed
            from the first bytes (``mic@2`` prefix → mic2; ``MICB``
            magic after base64-decode → micb).
        input_format: One of ``auto`` | ``mic2`` | ``micb``.
            Default ``auto``.
        output_format: One of ``mic2`` | ``micb``.
            Default ``mic2``.

    Returns:
        JSON object with: ``ok`` (bool), ``output`` (the converted
        payload — text for mic2, base64 for micb), ``input_format``
        (the resolved input format), ``output_format``, ``byte_size``
        (output size in bytes), ``value_count``, ``type_count``.
    """
    if input_format not in _VALID_INPUT_FMTS:
        return _err(f"input_format must be one of {sorted(_VALID_INPUT_FMTS)}")
    if output_format not in _VALID_OUTPUT_FMTS:
        return _err(f"output_format must be one of {sorted(_VALID_OUTPUT_FMTS)}")

    decoded, err = _decode_input(input, input_format)
    if err is not None:
        return _err(err)
    assert decoded is not None  # narrow for mypy (err was None → decoded set)
    resolved_in = input_format if input_format != "auto" else _detect_format(decoded)

    try:
        from mind_mem.mic_map import (
            Mic2ParseError,
            MicbParseError,
            emit_mic2,
            emit_micb,
            parse_mic2,
            parse_micb,
        )

        if resolved_in == "mic2":
            assert isinstance(decoded, str)
            graph = parse_mic2(decoded)
        else:
            assert isinstance(decoded, bytes)
            graph = parse_micb(decoded)
    except (Mic2ParseError, MicbParseError) as exc:
        return _err(f"parse failed: {exc}")

    if output_format == "mic2":
        out_text = emit_mic2(graph)
        out_bytes = out_text.encode("utf-8")
        out_payload = out_text
    else:
        out_bin = emit_micb(graph)
        out_bytes = out_bin
        out_payload = base64.b64encode(out_bin).decode("ascii")

    metrics.inc("mcp_mic_convert")
    _log.info(
        "mcp_mic_convert",
        input_format=resolved_in,
        output_format=output_format,
        value_count=len(graph.values),
        byte_size=len(out_bytes),
    )

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "ok": True,
            "output": out_payload,
            "input_format": resolved_in,
            "output_format": output_format,
            "byte_size": len(out_bytes),
            "value_count": len(graph.values),
            "type_count": len(graph.types),
        },
        indent=2,
    )


@mcp_tool_observe
def mic_inspect_tool(input: str, input_format: str = "auto") -> str:
    """Return a structural summary of a MIC/MAP payload without
    re-emitting it.

    Useful for: agents that want to know the shape of a graph (how
    many values, which opcodes, which dtypes) before deciding whether
    to forward it; CI checks that need to assert "this graph has at
    most N nodes"; debugging when a payload looks malformed.

    Args:
        input: The graph payload. Same conventions as
            :func:`mic_convert_tool` — text for ``mic2``, base64 for
            ``micb``, ``auto`` sniffs.
        input_format: One of ``auto`` | ``mic2`` | ``micb``.
            Default ``auto``.

    Returns:
        JSON object with: ``ok``, ``input_format``, ``type_count``,
        ``value_count``, ``output_idx``, ``types`` (list of
        ``{index, dtype, dims}``), ``values`` (list of
        ``{index, kind, ...}`` per Arg / Param / Node).
    """
    if input_format not in _VALID_INPUT_FMTS:
        return _err(f"input_format must be one of {sorted(_VALID_INPUT_FMTS)}")

    decoded, err = _decode_input(input, input_format)
    if err is not None:
        return _err(err)
    assert decoded is not None  # narrow for mypy (err was None → decoded set)
    resolved_in = input_format if input_format != "auto" else _detect_format(decoded)

    try:
        from mind_mem.mic_map import (
            Mic2ParseError,
            MicbParseError,
            parse_mic2,
            parse_micb,
        )

        if resolved_in == "mic2":
            assert isinstance(decoded, str)
            graph = parse_mic2(decoded)
        else:
            assert isinstance(decoded, bytes)
            graph = parse_micb(decoded)
    except (Mic2ParseError, MicbParseError) as exc:
        return _err(f"parse failed: {exc}")

    summary = _summarize(graph)
    summary["_schema_version"] = MCP_SCHEMA_VERSION
    summary["ok"] = True
    summary["input_format"] = resolved_in

    metrics.inc("mcp_mic_inspect")
    _log.info(
        "mcp_mic_inspect",
        input_format=resolved_in,
        value_count=summary["value_count"],
    )

    return json.dumps(summary, indent=2)


def register(mcp) -> None:
    """Wire the two MIC/MAP tools onto *mcp*."""
    mcp.tool(mic_convert_tool)
    mcp.tool(mic_inspect_tool)
