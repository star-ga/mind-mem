"""Merge the LoRA adapter into the base weights, then export to GGUF.

This is a one-shot script — it is *only* necessary when publishing a
release. Ollama / LM Studio / llama.cpp users prefer a single
self-contained GGUF over the adapter + base combo.

Pipeline:
    1. Load BASE in bf16, apply the adapter via PeftModel.merge_and_unload.
    2. Save the merged HF model to /tmp/mm_merged.
    3. Invoke llama.cpp's ``convert_hf_to_gguf.py`` to produce an FP16 GGUF.
    4. Quantize to Q4_K_M (default) via llama.cpp's ``llama-quantize``.

Prerequisites:
    ``llama.cpp`` cloned at /home/n/llama.cpp  (or set MM_LLAMA_CPP_DIR).
    ``llama.cpp`` built in release mode:  cmake -B build && cmake --build build.

Output:
    /home/n/mm-train-output/mind-mem-7b-Q4_K_M.gguf
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = Path("/home/n/mm-train-output/adapter")
MERGED = Path("/tmp/mm_merged")
OUT_GGUF_F16 = Path("/home/n/mm-train-output/mind-mem-7b-F16.gguf")
OUT_GGUF_Q4 = Path("/home/n/mm-train-output/mind-mem-7b-Q4_K_M.gguf")
LLAMA_CPP = Path(os.environ.get("MM_LLAMA_CPP_DIR", "/home/n/llama.cpp"))


def _merge_adapter() -> None:
    if MERGED.is_dir():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    print("merging adapter into base weights …")
    model = model.merge_and_unload()
    print(f"saving merged model → {MERGED}")
    model.save_pretrained(str(MERGED), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED))


def _convert_to_gguf() -> None:
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
        str(MERGED),
        "--outfile", str(OUT_GGUF_F16),
        "--outtype", "f16",
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
        sys.exit(
            f"llama-quantize not found under {LLAMA_CPP}/build. "
            "Run: cd llama.cpp && cmake -B build && cmake --build build -j"
        )
    cmd = [str(binary), str(OUT_GGUF_F16), str(OUT_GGUF_Q4), "Q4_K_M"]
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    if not ADAPTER.is_dir():
        sys.exit(f"adapter missing at {ADAPTER}. run train_qlora.py first.")
    _merge_adapter()
    _convert_to_gguf()
    _quantize()
    if OUT_GGUF_F16.is_file():
        OUT_GGUF_F16.unlink()  # keep only the Q4 build
    print(f"\nGGUF Q4_K_M → {OUT_GGUF_Q4}")
    print(f"size        → {OUT_GGUF_Q4.stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
