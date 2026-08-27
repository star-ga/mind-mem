"""``mind-mem-chat`` — ask a workspace a question, get cited answers.

Thin argparse wrapper over :func:`mind_mem.chat_memory.chat_with_memory`.
The default generator is the deterministic offline extractive one, so
the command works on a bare install with no accelerator and no network;
``--generator service`` opts into the local generation service.

Exit codes:
    0 — a grounded answer was produced
    1 — bad input (missing workspace, empty question, unknown generator)
    2 — recall found nothing (``no record found``)
    3 — the answer failed the citation-grounding contract
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .chat_citations import CitationError
from .chat_generators import GeneratorError, resolve_generator
from .chat_memory import ChatAnswer, chat_with_memory

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_NO_RECORD = 2
EXIT_UNGROUNDED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mind-mem-chat",
        description="Ask a mind-mem workspace a question; every claim is cited with [[block_id]].",
    )
    parser.add_argument("question", help="the question to answer")
    parser.add_argument(
        "-w",
        "--workspace",
        default=".",
        help="workspace root (default: current directory)",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=8,
        help="max blocks to recall (default: 8)",
    )
    parser.add_argument(
        "-g",
        "--generator",
        default="extractive",
        help="generator backend: 'extractive' (default, offline+deterministic) or 'service'",
    )
    parser.add_argument(
        "--on-invalid",
        choices=("raise", "reject"),
        default="reject",
        help="what to do with an ungrounded answer (default: reject)",
    )
    parser.add_argument(
        "--require-in-evidence",
        action="store_true",
        help="also reject citations that resolve but were not in the recalled evidence",
    )
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    return parser


def _render(result: ChatAnswer) -> str:
    lines = [result.answer]
    if result.citations:
        lines.append("")
        lines.append("Sources:")
        by_id = {item.block_id: item for item in result.evidence}
        for block_id in result.citations:
            item = by_id.get(block_id)
            location = f" ({item.source})" if item is not None and item.source else ""
            lines.append(f"  [[{block_id}]]{location}")
    for warning in result.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        generator = resolve_generator(args.generator)
        result = chat_with_memory(
            args.workspace,
            args.question,
            generator=generator,
            limit=args.limit,
            on_invalid=args.on_invalid,
            require_in_evidence=args.require_in_evidence,
        )
    except (ValueError, TypeError) as exc:
        print(f"mind-mem-chat: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except GeneratorError as exc:
        print(f"mind-mem-chat: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except CitationError as exc:
        print(f"mind-mem-chat: ungrounded answer rejected — {exc}", file=sys.stderr)
        return EXIT_UNGROUNDED

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_render(result))

    if result.rejected:
        return EXIT_UNGROUNDED
    if result.no_record:
        return EXIT_NO_RECORD
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
