# Copyright 2026 STARGA, Inc.
"""Thin async bridge over the multi-LLM orchestrator providers."""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
from dataclasses import dataclass
from typing import Any

from ..observability import get_logger
from .config import ORCHESTRATOR_PATH

_log = get_logger("skill_opt.fleet_bridge")


@dataclass(frozen=True)
class FleetResponse:
    """Result from a single fleet model query."""

    model: str
    content: str
    latency_ms: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.content)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "content": self.content,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


# model_key -> (provider_class_name, api_key_name, model_string)
#
# Model IDs verified 2026-05-27 (see ~/CLAUDE.md model table). Stale IDs
# return 404 from the upstream provider, so this dict is REPLACE-don't-keep
# for renamed models. The Perplexity seat is left in for when its billing
# resumes; the orchestrator's health check will route around dead seats.
FLEET_MODELS: dict[str, tuple[str, str, str]] = {
    "grok-4.3": ("XAIProvider", "xai", "grok-4.3"),
    "mistral-large-latest": ("MistralProvider", "mistral", "mistral-large-latest"),
    "deepseek-v4-pro": ("DeepSeekProvider", "deepseek", "deepseek-v4-pro"),
    "sonar-pro": ("PerplexityProvider", "perplexity", "sonar-pro"),
    "glm-5.1": ("ZhipuProvider", "zhipu", "glm-5.1"),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": (
        "NvidiaProvider",
        "nvidia",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    ),
    "kimi-k2.6": ("MoonshotProvider", "moonshot", "kimi-k2.6"),
}


def _load_orchestrator() -> tuple[Any, dict[str, str]]:
    """Dynamically import orchestrator providers and load API keys.

    Returns (providers_module, api_keys_dict).
    Raises ImportError if orchestrator is not installed.
    """
    if ORCHESTRATOR_PATH not in sys.path:
        sys.path.insert(0, ORCHESTRATOR_PATH)
    providers_mod = importlib.import_module("providers")
    config_mod = importlib.import_module("config")
    keys: dict[str, str] = config_mod.get_api_keys()
    return providers_mod, keys


class FleetBridge:
    """Async bridge to the multi-LLM fleet."""

    def __init__(
        self,
        models: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._timeout_s = timeout_s
        self._requested_models = models or list(FLEET_MODELS.keys())
        self._providers: dict[str, Any] = {}
        self._unavailable: dict[str, str] = {}
        self._initialized = False

    def _ensure_init(self) -> None:
        """Build a provider per requested model, recording every drop.

        A requested seat can fail to materialise three ways — the key
        names no entry in :data:`FLEET_MODELS`, the orchestrator holds no
        API key for its vendor, or the provider class is missing from the
        installed orchestrator. All three used to be a bare ``continue``:
        no counter, no log, and no way for a caller to compare
        ``available_models`` against what it asked for. A run that
        reached zero seats then looked exactly like a run that measured a
        genuinely worthless skill.

        Every drop is now recorded in :attr:`unavailable_models` with its
        reason and logged; reaching zero seats is logged at error level.
        """
        if self._initialized:
            return
        try:
            providers_mod, keys = _load_orchestrator()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"Multi-LLM orchestrator not found at {ORCHESTRATOR_PATH}. Install it or update ORCHESTRATOR_PATH in skill_opt/config.py."
            ) from exc
        rate_cls = getattr(providers_mod, "RateLimitConfig")
        for model_key in self._requested_models:
            spec = FLEET_MODELS.get(model_key)
            if spec is None:
                self._unavailable[model_key] = "unknown model key (not in FLEET_MODELS)"
                continue
            cls_name, key_name, model_str = spec
            api_key = keys.get(key_name, "")
            if not api_key:
                self._unavailable[model_key] = f"no API key configured for vendor {key_name!r}"
                continue
            cls = getattr(providers_mod, cls_name, None)
            if cls is None:
                self._unavailable[model_key] = f"provider class {cls_name!r} missing from the installed orchestrator"
                continue
            self._providers[model_key] = cls(
                api_key=api_key,
                model=model_str,
                rate_config=rate_cls(max_concurrent=2, min_request_spacing_s=1.0),
                timeout_s=self._timeout_s,
            )
        self._initialized = True

        for model_key, reason in self._unavailable.items():
            _log.warning("skill_opt_fleet_seat_unavailable", model=model_key, reason=reason)
        if not self._providers:
            # Distinct from "every seat answered badly": nothing was even
            # contactable, so any downstream score is a verdict on a fleet
            # that never ran.
            _log.error(
                "skill_opt_fleet_no_seats_available",
                requested=len(self._requested_models),
                reasons="; ".join(f"{k}: {v}" for k, v in self._unavailable.items()),
            )

    async def query(
        self,
        prompt: str,
        models: list[str] | None = None,
    ) -> list[FleetResponse]:
        """Send prompt to multiple fleet models in parallel.

        ``models=None`` means "every available model". An explicitly empty
        list means "no model" and yields no responses -- it must NEVER widen
        back to the full fleet. A truthiness fallback here would silently turn
        a caller that filtered every model out (see :meth:`query_excluding`)
        into a caller that queries all of them.
        """
        self._ensure_init()
        targets = list(self._providers.keys()) if models is None else models
        tasks = [self._query_one(m, prompt) for m in targets if m in self._providers]
        return list(await asyncio.gather(*tasks))

    async def query_excluding(
        self,
        prompt: str,
        exclude: set[str],
    ) -> list[FleetResponse]:
        """Query all fleet models except those in the exclude set.

        The exclusion is absolute: when ``exclude`` covers the whole fleet the
        result is empty. An empty result is the honest answer and callers must
        not read it as a fleet verdict.

        ``_ensure_init`` runs here and not only inside :meth:`query` because
        the target list is built from ``self._providers``, which is empty until
        initialization -- reading it too early would exclude everything.
        """
        self._ensure_init()
        targets = [m for m in self._providers if m not in exclude]
        return await self.query(prompt, models=targets)

    async def _query_one(self, model_key: str, prompt: str) -> FleetResponse:
        provider = self._providers.get(model_key)
        if provider is None:
            return FleetResponse(model=model_key, content="", latency_ms=0, error="no provider")
        t0 = time.monotonic()
        try:
            result = await provider.request(prompt)
            elapsed = (time.monotonic() - t0) * 1000
            if result.status.value == "ok" and result.content:
                return FleetResponse(
                    model=model_key,
                    content=result.content,
                    latency_ms=elapsed,
                )
            return FleetResponse(
                model=model_key,
                content=result.content or "",
                latency_ms=elapsed,
                error=result.error or f"status={result.status.value}",
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return FleetResponse(model=model_key, content="", latency_ms=elapsed, error=str(exc))

    @property
    def available_models(self) -> list[str]:
        """Models that actually have a provider bound.

        Compare against :attr:`unavailable_models` before reading an empty
        or short result as a fleet verdict.
        """
        self._ensure_init()
        return list(self._providers.keys())

    @property
    def unavailable_models(self) -> dict[str, str]:
        """Requested models that could not be initialised, and why.

        The counterpart to :attr:`available_models`: together they account
        for every entry the caller asked for, so a caller can report
        "0 of 7 seats — no API key configured" instead of a bare 0.0.
        """
        self._ensure_init()
        return dict(self._unavailable)
