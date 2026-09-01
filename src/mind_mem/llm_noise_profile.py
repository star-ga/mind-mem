# Copyright 2026 STARGA, Inc.
"""LLM Noise Profiler — per-provider, per-domain reliability tracking.

Models each LLM as a noisy sensor with domain-specific reliability scores.
Scores are updated via an Exponential Moving Average (EMA) so recent
observations carry more weight while old data decays gracefully.

EMA formula (alpha = 0.95):
    reliability = reliability * alpha + outcome * (1 - alpha)
    where outcome is 1.0 for correct, 0.0 for incorrect.

Usage::

    from mind_mem.llm_noise_profile import LLMNoiseProfiler

    profiler = LLMNoiseProfiler()
    profiler.register_provider("gpt-4", initial_reliability=0.9)
    profiler.record_outcome("gpt-4", domain="code", was_correct=True)
    print(profiler.get_reliability("gpt-4", domain="code"))
    profiler.save("/path/to/profiles.json")
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final

from .observability import get_logger

# EMA smoothing factor — higher means slower adaptation (more history weight)
_EMA_ALPHA: float = 0.95

_DEFAULT_RELIABILITY: float = 0.7


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NoiseProfile:
    """Reliability profile for a single LLM provider.

    Attributes:
        provider_id: Unique string identifying the LLM provider/model.
        global_reliability: Overall correctness rate in [0, 1].
        domain_reliability: Per-domain correctness rates in [0, 1].
        total_observations: Count of all record_outcome calls.
        error_count: Count of record_outcome calls where was_correct=False.
        last_calibrated: Unix timestamp of the most recent update.
    """

    provider_id: str
    global_reliability: float = _DEFAULT_RELIABILITY
    domain_reliability: dict[str, float] = field(default_factory=dict)
    total_observations: int = 0
    error_count: int = 0
    last_calibrated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "global_reliability": self.global_reliability,
            "domain_reliability": self.domain_reliability,
            "total_observations": self.total_observations,
            "error_count": self.error_count,
            "last_calibrated": self.last_calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoiseProfile:
        return cls(
            provider_id=data["provider_id"],
            global_reliability=data.get("global_reliability", _DEFAULT_RELIABILITY),
            domain_reliability=data.get("domain_reliability", {}),
            total_observations=data.get("total_observations", 0),
            error_count=data.get("error_count", 0),
            last_calibrated=data.get("last_calibrated", time.time()),
        )


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class LLMNoiseProfiler:
    """Track and update per-LLM, per-domain reliability scores.

    Thread-safety: not thread-safe. Wrap with a lock for concurrent use.
    """

    def __init__(self, alpha: float = _EMA_ALPHA) -> None:
        self._alpha = alpha
        self._profiles: dict[str, NoiseProfile] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_provider(self, provider_id: str, initial_reliability: float = _DEFAULT_RELIABILITY) -> None:
        """Register an LLM provider. Ignored if already registered."""
        if provider_id in self._profiles:
            return
        self._profiles[provider_id] = NoiseProfile(
            provider_id=provider_id,
            global_reliability=initial_reliability,
        )

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        provider_id: str,
        domain: str,
        *,
        was_correct: bool,
        at: float | None = None,
    ) -> None:
        """Update reliability scores for a provider after an observed outcome.

        Uses EMA to blend the new binary signal into both global and
        domain-level scores.

        Args:
            provider_id: Must already be registered.
            domain: Domain label (e.g. "code", "math", "summarization").
            was_correct: True if the LLM's output was correct/useful.
            at: Injectable observation instant (UTC epoch seconds) recorded
                as ``last_calibrated``. Defaults to the wall clock, which is
                what every pre-5.1.0 caller gets. Injecting it makes the
                persisted profile a pure function of the outcome stream —
                the same reports replayed on another machine produce the
                same file, byte for byte. The value is provenance only: no
                score, here or downstream, reads it.

        Raises:
            KeyError: If provider_id has not been registered.
        """
        profile = self._profiles[provider_id]  # raises KeyError if unknown
        signal = 1.0 if was_correct else 0.0
        alpha = self._alpha

        # Update global reliability via EMA
        profile.global_reliability = profile.global_reliability * alpha + signal * (1.0 - alpha)

        # Update domain reliability via EMA; seed directly from global if new (no EMA on first entry)
        if domain in profile.domain_reliability:
            prior = profile.domain_reliability[domain]
            profile.domain_reliability[domain] = prior * alpha + signal * (1.0 - alpha)
        else:
            profile.domain_reliability[domain] = profile.global_reliability

        profile.total_observations += 1
        if not was_correct:
            profile.error_count += 1
        profile.last_calibrated = time.time() if at is None else at

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_reliability(self, provider_id: str, domain: str | None = None) -> float:
        """Return reliability for a provider, optionally scoped to a domain.

        Falls back to global reliability when no domain-specific data exists.

        Raises:
            KeyError: If provider_id is not registered.
        """
        profile = self._profiles[provider_id]  # raises KeyError if unknown
        if domain is None:
            return profile.global_reliability
        return profile.domain_reliability.get(domain, profile.global_reliability)

    def get_observation_noise(self, provider_id: str, domain: str | None = None) -> float:
        """Return the noise level (1 - reliability) for a provider.

        A higher value means the provider is less trustworthy for this domain.

        Raises:
            KeyError: If provider_id is not registered.
        """
        return max(0.0, 1.0 - self.get_reliability(provider_id, domain))

    def get_best_provider(self, domain: str | None = None) -> str:
        """Return the provider_id with the highest reliability for a domain.

        Args:
            domain: Optional domain to scope the comparison. Falls back to
                    global reliability when domain data is absent.

        Raises:
            ValueError: If no providers are registered.
        """
        if not self._profiles:
            raise ValueError("no providers registered")
        return max(
            self._profiles,
            key=lambda pid: self.get_reliability(pid, domain),
        )

    def ranking(self, domain: str | None = None) -> list[tuple[str, float]]:
        """Return all providers sorted by reliability (descending).

        Args:
            domain: Optional domain to scope reliability scores.

        Returns:
            List of (provider_id, reliability) tuples, highest first.
        """
        return sorted(
            ((pid, self.get_reliability(pid, domain)) for pid in self._profiles),
            key=lambda pair: pair[1],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, *, at: float | None = None) -> None:
        """Persist all profiles to a JSON file.

        Creates parent directories as needed. Atomic write is approximated
        via a temp file + rename on POSIX systems.

        Args:
            path: Destination file path.
            at: Injectable value for the file's ``saved_at`` provenance
                field; defaults to the wall clock. See
                :meth:`record_outcome` — with both injected, two runs over
                the same reports write byte-identical files.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data: dict[str, Any] = {
            "version": 1,
            "saved_at": time.time() if at is None else at,
            "profiles": {pid: profile.to_dict() for pid, profile in self._profiles.items()},
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def load(self, path: str) -> None:
        """Load profiles from a JSON file, merging into current state.

        Silently ignores missing or malformed files.

        Args:
            path: Source file path.
        """
        if not os.path.isfile(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        profiles_raw = data.get("profiles", {})
        if not isinstance(profiles_raw, dict):
            return
        for pid, raw in profiles_raw.items():
            if isinstance(raw, dict):
                self._profiles[pid] = NoiseProfile.from_dict(raw)


# ---------------------------------------------------------------------------
# Wiring: the ``report_outcome`` → profile leg (5.1.0 restoration, slice 5)
# ---------------------------------------------------------------------------
#
# Until 5.1.0 this module had no production caller: everything above was
# imported only by ``tests/test_llm_noise_profile.py``. The event source it
# was always missing is outcome attribution — ``report_outcome`` already
# carries a verdict ("did acting on these blocks actually work?") plus the
# provenance of who reported it, which is exactly a ``was_correct``
# observation about a noisy sensor.
#
# Everything below is the adapter between that event and the EMA above. It
# is deliberately in *this* module rather than in ``outcome_store``: the
# derivation rules (who is the provider, what is the domain, when does an
# observation count at all) are the profiler's contract, and they are
# testable here without a database.

#: v4 flag guarding the whole leg. OFF by default; see :mod:`mind_mem.v4.feature_flags`.
NOISE_PROFILE_FLAG: Final[str] = "llm_noise_profile"

#: Where the profile is persisted, relative to the workspace root.
#: ``intelligence/`` is a corpus directory, but only ``.md`` files are corpus
#: (``corpus_registry.BLOCK_EXTENSIONS``), so a JSON sidecar here is state,
#: not content — it is never parsed as a block and never reaches recall.
PROFILE_REL_PATH: Final[str] = "intelligence/llm_profiles.json"

#: Provider used when a report carries neither ``tool_id`` nor ``actor_id``.
#: One shared bucket, mirroring ``outcome_store._ANONYMOUS_ACTOR``: an
#: unattributed report must not mint a fresh provider each time.
UNATTRIBUTED_PROVIDER: Final[str] = "unattributed"

#: Domain used for a block id with no leading alphabetic family.
DEFAULT_DOMAIN: Final[str] = "general"

#: Verdicts that carry information about correctness. ``neutral`` means "not
#: attributable to these blocks", which is not evidence about the reporter's
#: accuracy either — it moves nothing.
_SCORED_VERDICTS: Final[dict[str, bool]] = {"success": True, "failure": False}

_FAMILY_RE = re.compile(r"^[A-Za-z]+")

_log = get_logger("llm_noise_profile")


def block_domain(block_id: str) -> str:
    """Return the reliability *domain* a block id belongs to.

    The domain is the block id's family — the leading alphabetic run,
    upper-cased: ``D-20260401-001`` → ``"D"``, ``INBOX-2026-…`` → ``"INBOX"``.
    That is the coarsest partition mind-mem already has that means something
    ("decisions", "tasks", "inbox drops"), and it is derived from the id
    alone, so it needs no corpus read — which matters, because this leg must
    never touch block content.

    An id with no alphabetic prefix falls back to :data:`DEFAULT_DOMAIN`
    rather than minting a domain per id; a per-id domain would make every
    observation the first one in its bucket and the EMA would never run.
    """
    match = _FAMILY_RE.match(block_id.strip())
    return match.group(0).upper() if match else DEFAULT_DOMAIN


def report_domains(block_ids: Iterable[str]) -> list[str]:
    """Return the sorted distinct domains one outcome report speaks about.

    Sorted, because the EMA is order-dependent and two identical reports
    must move the profile identically.

    DISTINCT is the load-bearing word. A report naming fifty decision blocks
    is *one* verdict about the ``D`` domain, not fifty: counting per block
    would let a reporter amplify its own vote simply by listing more ids —
    the same unbounded-influence bug ``outcome_store._projection_query_id``
    exists to prevent on the calibration side.
    """
    return sorted({block_domain(bid) for bid in block_ids if str(bid).strip()})


def provider_for(*, tool_id: str = "", actor_id: str = "") -> str:
    """Return the provider id an outcome report is evidence about.

    ``tool_id`` first (it names the reporting tool — in practice the model or
    agent whose work was judged), then ``actor_id``, then the shared
    :data:`UNATTRIBUTED_PROVIDER` bucket. Never empty: an empty provider id
    would collide with nothing and grow the file without meaning.
    """
    return tool_id.strip() or actor_id.strip() or UNATTRIBUTED_PROVIDER


def profile_path(workspace: str) -> str:
    """Absolute path of the persisted profile for *workspace*."""
    return os.path.join(os.path.abspath(workspace), *PROFILE_REL_PATH.split("/"))


def profiling_enabled() -> bool:
    """Whether the outcome → profile leg is switched on.

    Uses ``is_enabled_quiet``, never ``is_enabled``: this is the PROBE that
    decides whether an OFF-by-default surface runs at all, and a probe that
    logs (``is_enabled`` warns ``v4_config_unreadable`` on a malformed
    config) would make the flag-off build observably different from the
    build that never had the feature. That regression was caught in slice 1
    and is pinned here by ``tests/test_llm_noise_profile_wiring.py``.
    """
    try:
        from .v4.feature_flags import is_enabled_quiet

        return is_enabled_quiet(NOISE_PROFILE_FLAG)
    except Exception:  # pragma: no cover — a broken import means OFF, silently
        return False


def stamp_to_epoch(stamp: str | None) -> float | None:
    """Parse an outcome's ``recorded_at`` into a UTC epoch, or ``None``.

    ``report_outcome`` takes an injectable ISO-8601 ``recorded_at``; feeding
    it through to the profile is what makes the persisted file a pure
    function of the outcome stream rather than of when the stream was
    replayed. ``None`` (unparseable or absent) lets the caller fall back to
    the wall clock, which is the pre-existing behaviour of this module.

    Timezone-independent by construction: the parsed value is stamped UTC
    before ``.timestamp()``, so two machines in different zones agree.
    """
    if not stamp:
        return None
    try:
        parsed = datetime.strptime(stamp.strip(), "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def record_report(
    workspace: str,
    block_ids: Iterable[str],
    outcome: str,
    *,
    tool_id: str = "",
    actor_id: str = "",
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Fold one **newly recorded** outcome report into the persisted profile.

    Load → update → save on every call, rather than holding a profiler in
    memory: an MCP tool call is frequently its own process, so "survives a
    restart" is only true if the file is the state. The file is tiny and the
    save is atomic (temp + ``os.replace``).

    Called only when ``outcome_store.record_outcome`` actually inserted a
    row. A replayed report conflicts on the outcome-id primary key and
    changes nothing in the store; it must change nothing here either, or a
    reporter could move its own reliability without bound by re-sending one
    report.

    Returns a summary dict (never raises): ``provider_id``, the ``domains``
    observed, the number of ``observations`` applied, and the resulting
    per-domain ``reliability``. ``observations`` is 0 for a ``neutral``
    verdict and for any failure to persist.
    """
    was_correct = _SCORED_VERDICTS.get(outcome)
    provider = provider_for(tool_id=tool_id, actor_id=actor_id)
    if was_correct is None:
        return {"provider_id": provider, "domains": [], "observations": 0, "reliability": {}}

    domains = report_domains(block_ids)
    if not domains:
        return {"provider_id": provider, "domains": [], "observations": 0, "reliability": {}}

    at = stamp_to_epoch(recorded_at)
    path = profile_path(workspace)
    profiler = LLMNoiseProfiler()
    try:
        profiler.load(path)
        profiler.register_provider(provider)
        for domain in domains:
            profiler.record_outcome(provider, domain, was_correct=was_correct, at=at)
        profiler.save(path, at=at)
    except OSError as exc:
        # The flag is ON, so a persistence failure is worth saying out loud —
        # but it must never take the outcome write down with it. The row is
        # already committed; the profile is a sidecar.
        _log.warning("llm_noise_profile_persist_failed", path=path, error=str(exc))
        return {"provider_id": provider, "domains": domains, "observations": 0, "reliability": {}}

    return {
        "provider_id": provider,
        "domains": domains,
        "observations": len(domains),
        "reliability": {domain: profiler.get_reliability(provider, domain) for domain in domains},
    }


def reliability_report(workspace: str, *, top_n: int = 20) -> dict[str, Any] | None:
    """Per-provider reliability for ``calibration_stats``, or ``None`` when OFF.

    ``None`` — not an empty dict — is the OFF answer, so the caller can leave
    the key out of its response entirely and keep flag-off output
    byte-identical. An ON workspace that has recorded nothing yet reports an
    empty ``providers`` list, which is a different statement.

    Reads only the persisted file: no clock, no corpus, no block content.
    """
    if not profiling_enabled():
        return None
    profiler = LLMNoiseProfiler()
    profiler.load(profile_path(workspace))
    # Order the PROFILES, then render. Sorting the rendered dicts instead
    # means sorting values typed as ``object``, which needs a cast per key to
    # type-check and hides the fact that the ordering is a property of the
    # profile, not of its JSON shape. Deterministic: reliability descending,
    # provider_id as the tie-break, so equal-reliability providers keep a
    # stable order rather than dict-insertion order.
    ordered = sorted(
        profiler._profiles.values(),
        key=lambda profile: (-profile.global_reliability, profile.provider_id),
    )
    providers = [
        {
            "provider_id": profile.provider_id,
            "reliability": profile.global_reliability,
            "observation_noise": max(0.0, 1.0 - profile.global_reliability),
            "observations": profile.total_observations,
            "errors": profile.error_count,
            "domains": dict(sorted(profile.domain_reliability.items())),
        }
        for profile in ordered
    ]
    return {
        "flag": f"v4.{NOISE_PROFILE_FLAG}",
        "path": PROFILE_REL_PATH,
        "providers": providers[: max(1, int(top_n))],
        "provider_count": len(providers),
    }
