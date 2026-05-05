"""Export the trained model to GGUF for Ollama / LM Studio / llama.cpp.

Two layouts supported:

  * QLoRA adapter (legacy v3.0):  ``${MM_TRAIN_ROOT}/adapter/``
      → load BASE in bf16, apply adapter via PeftModel.merge_and_unload,
        save merged model, then convert.
  * Full fine-tune (v3.9 onward):  ``${MM_FULLFT_DIR}`` (default
      ``/data/checkpoints/mm-workspace/mind-mem-4b-fullft/full-ft``)
      → already a self-contained HF model; convert directly, no merge.

The full-FT path is auto-detected when a `model.safetensors` (or
`model.safetensors.index.json` shard set) lives at MM_FULLFT_DIR.
Override layout choice with ``MM_GGUF_SOURCE`` (`fullft` or `adapter`).

Prerequisites:
    ``llama.cpp`` cloned at /home/n/llama.cpp  (or set MM_LLAMA_CPP_DIR).
    ``llama.cpp`` built:  cmake -B build && cmake --build build.

Output:
    ${MM_TRAIN_ROOT}/mind-mem-4b-Q4_K_M.gguf
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = "Qwen/Qwen3.5-4B"
_BASE_DIR = Path(os.environ.get("MM_TRAIN_ROOT", "/data/checkpoints/mm-workspace/train-output"))
ADAPTER = _BASE_DIR / "adapter"
FULLFT = Path(
    os.environ.get(
        "MM_FULLFT_DIR",
        "/data/checkpoints/mm-workspace/mind-mem-4b-fullft/full-ft",
    )
)
# Everything stays on /data — 4B base merged weights (~18 GB) + F16
# GGUF (~17 GB) + Q4_K_M (~5 GB) would blow out / in a heartbeat.
MERGED = Path("/data/checkpoints/mm-workspace/mm_merged")
OUT_GGUF_F16 = Path("/data/checkpoints/mm-workspace/mind-mem-4b-F16.gguf")
OUT_GGUF_Q4 = _BASE_DIR / "mind-mem-4b-Q4_K_M.gguf"
LLAMA_CPP = Path(os.environ.get("MM_LLAMA_CPP_DIR", "/home/n/llama.cpp"))


def _resolve_source() -> Path:
    """Pick the model dir to convert. Full-FT preferred over QLoRA merge."""
    pref = os.environ.get("MM_GGUF_SOURCE", "").strip().lower()
    fullft_ready = (FULLFT / "model.safetensors").is_file() or (FULLFT / "model.safetensors.index.json").is_file()
    if pref == "adapter":
        if not ADAPTER.is_dir():
            sys.exit(f"MM_GGUF_SOURCE=adapter but {ADAPTER} missing")
        return _merge_adapter_to_disk()
    if pref == "fullft":
        if not fullft_ready:
            sys.exit(f"MM_GGUF_SOURCE=fullft but no model.safetensors at {FULLFT}")
        return FULLFT
    if fullft_ready:
        print(f"using full-FT weights at {FULLFT}")
        return FULLFT
    if ADAPTER.is_dir():
        print(f"no full-FT found; falling back to QLoRA adapter at {ADAPTER}")
        return _merge_adapter_to_disk()
    sys.exit(f"no weights — checked {FULLFT} and {ADAPTER}")


def _merge_adapter_to_disk() -> Path:
    # Heavy imports only when QLoRA path is chosen.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if MERGED.is_dir():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    print("loading base on CPU (bf16) …")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True)
    print("applying adapter …")
    model = PeftModel.from_pretrained(model, str(ADAPTER), device_map=None)
    print("merging adapter into base weights …")
    model = model.merge_and_unload()
    print(f"saving merged model → {MERGED}")
    model.save_pretrained(str(MERGED), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED))
    return MERGED


def _convert_to_gguf(source: Path) -> None:
    convert = LLAMA_CPP / "convert_hf_to_gguf.py"
    if not convert.is_file():
        sys.exit(
            f"llama.cpp converter missing at {convert}. "
            "Clone https://github.com/ggml-org/llama.cpp to /home/n/llama.cpp "
            "or set MM_LLAMA_CPP_DIR."
        )
    cmd = [
        sys.executable,
        str(convert),
        str(source),
        "--outfile",
        str(OUT_GGUF_F16),
        "--outtype",
        "f16",
    ]
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _quantize() -> None:
    # llama.cpp builds land either at build/bin/llama-quantize or
    # just build/llama-quantize depending on version. Probe both.
    candidates = [
        LLAMA_CPP / "build" / "bin" / "llama-quantize",
        LLAMA_CPP / "build" / "llama-quantize",
    ]
    binary = next((p for p in candidates if p.is_file()), None)
    if binary is None:
        sys.exit(f"llama-quantize not found under {LLAMA_CPP}/build. Run: cd llama.cpp && cmake -B build && cmake --build build -j")
    cmd = [str(binary), str(OUT_GGUF_F16), str(OUT_GGUF_Q4), "Q4_K_M"]
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    source = _resolve_source()
    _convert_to_gguf(source)
    _quantize()
    if OUT_GGUF_F16.is_file():
        OUT_GGUF_F16.unlink()  # keep only the Q4 build
    print(f"\nGGUF Q4_K_M → {OUT_GGUF_Q4}")
    print(f"size        → {OUT_GGUF_Q4.stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
