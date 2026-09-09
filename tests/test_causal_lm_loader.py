from __future__ import annotations

from mind_mem.causal_lm_loader import load_causal_lm, normalize_causal_lm_config


class _Config:
    def __init__(self, model_type: str, text_config: object | None = None) -> None:
        self.model_type = model_type
        self._text_config = text_config
        self.calls = 0

    def get_text_config(self) -> object | None:
        self.calls += 1
        return self._text_config


def test_only_qwen35_composite_config_is_normalized() -> None:
    text = object()
    composite = _Config("qwen3_5", text)
    assert normalize_causal_lm_config(composite) is text
    assert composite.calls == 1

    gpt2 = _Config("gpt2", text)
    assert normalize_causal_lm_config(gpt2) is gpt2
    assert gpt2.calls == 0

    malformed = _Config("qwen3_5", None)
    assert normalize_causal_lm_config(malformed) is malformed


def test_loader_forwards_config_and_model_options_without_mutation() -> None:
    text = object()
    composite = _Config("qwen3_5", text)

    class AutoConfig:
        seen: list[tuple[str, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, source: str, **kwargs: object) -> _Config:
            cls.seen.append((source, kwargs))
            return composite

    class AutoModel:
        seen: list[tuple[str, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, source: str, **kwargs: object) -> object:
            cls.seen.append((source, kwargs))
            return "model"

    config_options = {
        "revision": "sha256:config",
        "token": "secret-token",
        "trust_remote_code": False,
        "local_files_only": True,
    }
    model_options = {
        "revision": "sha256:model",
        "token": "secret-token",
        "trust_remote_code": False,
        "local_files_only": True,
        "torch_dtype": "bf16",
    }
    original_config_options = dict(config_options)
    original_model_options = dict(model_options)

    assert load_causal_lm(
        "local/checkpoint",
        auto_config=AutoConfig,
        auto_model=AutoModel,
        config_kwargs=config_options,
        **model_options,
    ) == "model"
    assert AutoConfig.seen == [("local/checkpoint", original_config_options)]
    assert AutoModel.seen == [
        ("local/checkpoint", {"config": text, **original_model_options})
    ]
    assert config_options == original_config_options
    assert model_options == original_model_options

