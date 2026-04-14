# Copyright 2026 STARGA, Inc.
"""Thin async bridge over the multi-LLM orchestrator providers."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from .config import ENV_PATH, ORCHESTRATOR_PATH


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
FLEET_MODELS: dict[str, tuple[str, str, str]] = {
    "grok-4-1-fast-reasoning": ("XAIProvider", "xai", "grok-4-1-fast-reasoning"),
    "mistral-large-latest": ("MistralProvider", "mistral", "mistral-large-latest"),
    "deepseek-reasoner": ("DeepSeekProvider", "deepseek", "deepseek-reasoner"),
    "sonar-pro": ("PerplexityProvider", "perplexity", "sonar-pro"),
    "glm-5": ("ZhipuProvider", "zhipu", "glm-5"),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": (
        "NvidiaProvider",
        "nvidia",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    ),
    "kimi-k2.5-preview": ("MoonshotProvider", "moonshot", "kimi-k2.5-preview"),
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
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        try:
            providers_mod, keys = _load_orchestrator()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"Multi-LLM orchestrator not found at {ORCHESTRATOR_PATH}. "
                "Install it or update ORCHESTRATOR_PATH in skill_opt/config.py."
            ) from exc
        rate_cls = getattr(providers_mod, "RateLimitConfig")
        for model_key in self._requested_models:
            spec = FLEET_MODELS.get(model_key)
            if spec is None:
                continue
            cls_name, key_name, model_str = spec
            api_key = keys.get(key_name, "")
            if not api_key:
                continue
            cls = getattr(providers_mod, cls_name, None)
            if cls is None:
                continue
            self._providers[model_key] = cls(
                api_key=api_key,
                model=model_str,
                rate_config=rate_cls(max_concurrent=2, min_request_spacing_s=1.0),
                timeout_s=self._timeout_s,
            )
        self._initialized = True

    async def query(
        self,
        prompt: str,
        models: list[str] | None = None,
    ) -> list[FleetResponse]:
        """Send prompt to multiple fleet models in parallel."""
        self._ensure_init()
        targets = models or list(self._providers.keys())
        tasks = [self._query_one(m, prompt) for m in targets if m in self._providers]
        return list(await asyncio.gather(*tasks))

    async def query_excluding(
        self,
        prompt: str,
        exclude: set[str],
    ) -> list[FleetResponse]:
        """Query all fleet models except those in the exclude set."""
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
            return FleetResponse(
                model=model_key, content="", latency_ms=elapsed, error=str(exc)
            )

    @property
    def available_models(self) -> list[str]:
        self._ensure_init()
        return list(self._providers.keys())
