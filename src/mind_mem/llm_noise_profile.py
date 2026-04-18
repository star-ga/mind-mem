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
import time
from dataclasses import dataclass, field
from typing import Any

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

    def record_outcome(self, provider_id: str, domain: str, *, was_correct: bool) -> None:
        """Update reliability scores for a provider after an observed outcome.

        Uses EMA to blend the new binary signal into both global and
        domain-level scores.

        Args:
            provider_id: Must already be registered.
            domain: Domain label (e.g. "code", "math", "summarization").
            was_correct: True if the LLM's output was correct/useful.

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
        profile.last_calibrated = time.time()

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

    def save(self, path: str) -> None:
        """Persist all profiles to a JSON file.

        Creates parent directories as needed. Atomic write is approximated
        via a temp file + rename on POSIX systems.

        Args:
            path: Destination file path.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data: dict[str, Any] = {
            "version": 1,
            "saved_at": time.time(),
            "profiles": {pid: profile.to_dict() for pid, profile in self._profiles.items()},
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
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
