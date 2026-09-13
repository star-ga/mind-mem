"""Offline controls for GGUF source selection and adapter/base binding."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "train" / "export_gguf.py"


def _load_export(monkeypatch: pytest.MonkeyPatch, root: Path, base: str):
    monkeypatch.setenv("MM_TRAIN_ROOT", str(root / "train-output"))
    monkeypatch.setenv("MM_BASE_MODEL", base)
    monkeypatch.delenv("MM_GGUF_SOURCE", raising=False)
    spec = importlib.util.spec_from_file_location("export_gguf_test", EXPORT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adapter(root: Path, declared: object) -> Path:
    adapter = root / "train-output" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": declared}), encoding="utf-8"
    )
    return adapter


def test_mismatched_adapter_refuses_before_heavy_import_or_merge_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "selected-base"
    base.mkdir()
    _adapter(tmp_path, "different/base")
    module = _load_export(monkeypatch, tmp_path, str(base))
    deleted = False

    def fail_delete(_path):
        nonlocal deleted
        deleted = True
        raise AssertionError("merge directory was deleted before binding check")

    real_import = builtins.__import__

    def forbid_heavy(name, *args, **kwargs):
        if name in {"torch", "peft", "transformers"}:
            raise AssertionError(f"heavy import reached before binding check: {name}")
        return real_import(name, *args, **kwargs)

    with patch.object(module.shutil, "rmtree", side_effect=fail_delete):
        with patch("builtins.__import__", side_effect=forbid_heavy):
            with pytest.raises(SystemExit, match="binds base"):
                module._merge_adapter_to_disk()
    assert deleted is False


def test_malformed_adapter_config_refuses_before_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "selected-base"
    base.mkdir()
    adapter = tmp_path / "train-output" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("[]", encoding="utf-8")
    module = _load_export(monkeypatch, tmp_path, str(base))
    with pytest.raises(SystemExit, match="must contain an object"):
        module._merge_adapter_to_disk()
    assert not (tmp_path / "train-output" / "mm_merged").exists()
    assert adapter.is_dir()


def test_unknown_gguf_source_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_export(monkeypatch, tmp_path, "selected/base")
    monkeypatch.setenv("MM_GGUF_SOURCE", "mystery")
    with pytest.raises(SystemExit, match="unknown MM_GGUF_SOURCE"):
        module._resolve_source()


def test_matching_adapter_uses_selected_base_through_canonical_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "selected-base"
    base.mkdir()
    adapter = _adapter(tmp_path, str(base))
    module = _load_export(monkeypatch, tmp_path, str(base))
    merged = tmp_path / "merged"
    module.MERGED = merged
    calls: list[tuple] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(("tokenizer", source, kwargs))
            return cls()

        def save_pretrained(self, destination):
            calls.append(("tokenizer-save", destination))

    class FakeModel:
        def merge_and_unload(self):
            calls.append(("merge",))
            return self

        def save_pretrained(self, destination, **kwargs):
            calls.append(("model-save", destination, kwargs))

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(("config", source, kwargs))
            return types.SimpleNamespace(model_type="fixture")

    class FakeModelFactory:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(("model", source, kwargs))
            return FakeModel()

    class FakePeft:
        @classmethod
        def from_pretrained(cls, model, source, **kwargs):
            calls.append(("adapter", model, source, kwargs))
            return model

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeConfig
    fake_transformers.AutoModelForCausalLM = FakeModelFactory
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeft

    with patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers, "peft": fake_peft},
    ):
        result = module._merge_adapter_to_disk()

    assert result == merged
    assert ("tokenizer", str(base), {"trust_remote_code": True}) in calls
    assert any(call[0] == "config" and call[1] == str(base) for call in calls)
    assert any(call[0] == "model" and call[1] == str(base) for call in calls)
    assert any(call[0] == "adapter" and call[2] == str(adapter) for call in calls)
    assert ("merge",) in calls
