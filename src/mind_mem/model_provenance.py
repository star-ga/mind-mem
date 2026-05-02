"""Provenance allowlist check for ``mm audit-model`` checkpoints.

Companion to :mod:`mind_mem.model_audit`. ``audit_model`` now exposes a
seventh check (``check_provenance``) that reads the ``base_model``
claim from ``config.json`` and refuses any checkpoint whose declared
upstream publisher isn't in a known-good allowlist.

Threat model: an attacker re-uploads a tampered Qwen / Llama / Mistral
weight-bundle under a non-canonical HF organisation, hoping a
downstream pipeline pins the namespace string ("alibaba-qwen/...")
without realising it doesn't match the real publisher. ``mm
audit-model`` is the load-gate; this check is the namespace whitelist.

The default allowlist covers the seven publishers mind-mem
operationally supports plus three more that show up in the local-fleet
ecosystem (DeepSeek, Microsoft Phi, TII Falcon). Operators with
internal fine-tunes can extend the list via ``allow_extra=`` (in
Python) or ``--allow-extra <slug>`` (in CLI).

When ``config.json`` does not declare ``base_model`` the check passes
with a "no base_model declared" status — HF does not require the
field, and pretrain checkpoints legitimately omit it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Publisher:
    """Canonical allowlist entry — name + the HF org slugs we accept."""

    name: str
    slugs: frozenset[str]
    description: str


# Default allowlist. Slug match is case-insensitive on the leading namespace
# of ``base_model`` (everything before the first ``/``). The same publisher
# can ship under multiple slugs (e.g. IBM Granite uses both ``ibm`` and
# ``ibm-granite``).
DEFAULT_PUBLISHERS: tuple[Publisher, ...] = (
    Publisher(
        name="Alibaba Qwen",
        slugs=frozenset({"qwen"}),
        description="Qwen / Qwen2 / Qwen2.5 / Qwen3 family from Alibaba.",
    ),
    Publisher(
        name="Meta Llama",
        slugs=frozenset({"meta-llama", "facebook"}),
        description="Llama 2 / 3 / 3.1 / 3.2 family from Meta AI.",
    ),
    Publisher(
        name="Mistral AI",
        slugs=frozenset({"mistralai"}),
        description="Mistral / Mixtral / Codestral family.",
    ),
    Publisher(
        name="Google Gemma",
        slugs=frozenset({"google"}),
        description="Gemma / Gemma-2 / PaLM family from Google.",
    ),
    Publisher(
        name="IBM Granite",
        slugs=frozenset({"ibm-granite", "ibm"}),
        description="Granite-3 / Granite-Code family from IBM.",
    ),
    Publisher(
        name="OpenAI",
        slugs=frozenset({"openai"}),
        description="Whisper / CLIP and other open-weight OpenAI releases.",
    ),
    Publisher(
        name="Anthropic",
        slugs=frozenset({"anthropic"}),
        description="Anthropic-published artifacts (no open Claude weights yet).",
    ),
    Publisher(
        name="DeepSeek",
        slugs=frozenset({"deepseek-ai"}),
        description="DeepSeek-V2 / V3 / R1 family.",
    ),
    Publisher(
        name="Microsoft Phi",
        slugs=frozenset({"microsoft"}),
        description="Phi-2 / Phi-3 / Phi-4 family from Microsoft Research.",
    ),
    Publisher(
        name="TII Falcon",
        slugs=frozenset({"tiiuae"}),
        description="Falcon-7B / 40B / 180B family from TII Abu Dhabi.",
    ),
)


@dataclass
class ProvenanceFinding:
    """Result of ``check_provenance`` — wired into ``model_audit.CheckResult``.

    ``passed`` is ``True`` when either:
      * ``base_model`` is missing (pretrain or undeclared), OR
      * ``base_model`` namespace matches an allowlisted publisher.
    """

    passed: bool
    detail: str
    base_model: str | None = None
    matched_publisher: str | None = None
    evidence: list[str] = field(default_factory=list)


def _slug_set(publishers: tuple[Publisher, ...]) -> dict[str, str]:
    """Flatten a tuple of publishers into a {slug -> publisher_name} map.

    Slugs are stored lowercased so the namespace match is
    case-insensitive (HF org slugs are case-sensitive in URLs but
    operators frequently mis-case them in configs).
    """
    out: dict[str, str] = {}
    for pub in publishers:
        for slug in pub.slugs:
            out[slug.lower()] = pub.name
    return out


def _read_base_model(root: Path) -> str | None:
    """Extract ``base_model`` from ``config.json`` (or return ``None``)."""
    cfg = root / "config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    base = data.get("base_model")
    if isinstance(base, str) and base.strip():
        return base.strip()
    return None


def check_provenance(
    root: Path,
    *,
    publishers: tuple[Publisher, ...] | None = None,
    allow_extra: tuple[str, ...] | None = None,
) -> ProvenanceFinding:
    """Verify the ``base_model`` claim against an allowlist of publishers.

    ``publishers`` overrides the default allowlist entirely; ``allow_extra``
    augments it with operator-specific HF org slugs (e.g. an internal
    fine-tune org). Both are optional.

    The slug match is case-insensitive on the leading namespace of
    ``base_model`` (the substring before the first ``/``).
    """
    used_publishers = publishers if publishers is not None else DEFAULT_PUBLISHERS
    slug_map = _slug_set(used_publishers)
    if allow_extra:
        for slug in allow_extra:
            slug_map[slug.lower()] = f"operator-allowlist:{slug}"

    base = _read_base_model(root)
    if base is None:
        return ProvenanceFinding(
            passed=True,
            detail="no base_model declared (pretrain or undeclared)",
        )

    namespace = base.split("/", 1)[0].strip()
    if not namespace:
        return ProvenanceFinding(
            passed=False,
            detail="base_model has no namespace (malformed)",
            base_model=base,
            evidence=[f"base_model={base!r}"],
        )

    matched = slug_map.get(namespace.lower())
    if matched is None:
        return ProvenanceFinding(
            passed=False,
            detail=f"base_model namespace {namespace!r} not in allowlist",
            base_model=base,
            evidence=[
                f"base_model={base!r}",
                f"namespace={namespace!r}",
                f"allowlist_size={len(slug_map)}",
            ],
        )

    return ProvenanceFinding(
        passed=True,
        detail=f"base_model from {matched}",
        base_model=base,
        matched_publisher=matched,
    )


def list_publishers(publishers: tuple[Publisher, ...] | None = None) -> list[dict[str, object]]:
    """Return a JSON-serialisable view of the active allowlist."""
    used = publishers if publishers is not None else DEFAULT_PUBLISHERS
    return [
        {
            "name": p.name,
            "slugs": sorted(p.slugs),
            "description": p.description,
        }
        for p in used
    ]
