"""Load causal language models with the narrow Qwen3.5 config workaround.

Transformers exposes Qwen3.5 as a composite vision/text configuration.  Some
released AutoModelForCausalLM factories do not select its nested text config,
so the causal model constructor receives the composite object and fails before
weights are read.  This module normalizes only that experimentally verified
``qwen3_5`` shape.  All other configurations are passed through unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

_QWEN35_MODEL_TYPE = "qwen3_5"


def normalize_causal_lm_config(config: Any) -> Any:
    """Return the text config required by the builtin Qwen3.5 causal model.

    The check intentionally keys on Transformers' model type rather than a
    model name or architecture suffix.  A config without the verified
    ``qwen3_5`` type, including an ordinary GPT2 config, is returned by
    identity.  A malformed Qwen3.5 config without a callable text-config
    accessor also fails closed by returning it unchanged; the model loader
    then reports its native error.
    """

    if getattr(config, "model_type", None) != _QWEN35_MODEL_TYPE:
        return config
    get_text_config = getattr(config, "get_text_config", None)
    if not callable(get_text_config):
        return config
    text_config = get_text_config()
    return config if text_config is None else text_config


def load_causal_lm(
    source: str,
    *,
    auto_config: Any,
    auto_model: Any,
    config_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> Any:
    """Materialize and load a causal model, preserving caller load options.

    ``config_kwargs`` is forwarded verbatim to ``AutoConfig`` and
    ``model_kwargs`` is forwarded verbatim to ``AutoModelForCausalLM``.  This
    keeps revision, token, trust, and local-files-only policy explicit at both
    stages rather than silently retrying or consulting the Hub.
    """

    config_options = dict(config_kwargs or {})
    config = auto_config.from_pretrained(source, **config_options)
    normalized = normalize_causal_lm_config(config)
    return auto_model.from_pretrained(source, config=normalized, **model_kwargs)
