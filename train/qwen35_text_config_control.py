"""Offline smoke control for the Qwen3.5 composite-config adapter.

This creates a tiny random full composite checkpoint in a temporary directory,
reloads it through the shared causal loader, and runs one forward pass.  It
never contacts the Hub or downloads weights; missing optional ML dependencies
are reported as a skip.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def main() -> int:
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    except ImportError as exc:
        print(json.dumps({"status": "skipped", "reason": f"optional dependency missing: {exc}"}))
        return 0

    from _causal_lm_import import load_causal_lm

    torch.set_num_threads(1)
    config_data = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 64,
        },
        "vision_config": {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4, "out_hidden_size": 32},
    }
    with tempfile.TemporaryDirectory(prefix="mind-mem-qwen35-") as temp_dir:
        path = Path(temp_dir)
        (path / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
        composite = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        seed = AutoModelForImageTextToText.from_config(composite, trust_remote_code=False)
        seed.save_pretrained(path)
        # Preserve the composite source config after save_pretrained emits the
        # model's generated config representation.
        (path / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
        del seed
        model = load_causal_lm(
            str(path),
            auto_config=AutoConfig,
            auto_model=AutoModelForCausalLM,
            config_kwargs={"local_files_only": True, "trust_remote_code": False},
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        # output_loading_info is useful for checking the real dispatch, but
        # the adapter deliberately leaves the public Transformers return shape
        # untouched.  Unpack only this control's known tuple response.
        model, loading = model
        assert not loading.get("missing_keys"), loading
        assert not loading.get("unexpected_keys"), loading
        assert not loading.get("mismatched_keys"), loading
        model.eval()
        with torch.no_grad():
            output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
        print(json.dumps({
            "status": "pass",
            "transformers_version": transformers.__version__,
            "input_model_type": composite.model_type,
            "input_has_vision_config": hasattr(composite, "vision_config"),
            "loaded_class": type(model).__name__,
            "loaded_config_class": type(model.config).__name__,
            "parameters": sum(value.numel() for value in model.parameters()),
            "logits_shape": list(output.logits.shape),
            "scope": "tiny random CPU model; local-only dispatch and forward; no downloaded weights",
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

