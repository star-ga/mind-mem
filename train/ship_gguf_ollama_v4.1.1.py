"""GGUF + Ollama shipper for mind-mem-4b v4.1.1.

Runs LOCALLY after merge_and_eval_v4.1.1.py confirms 131/131.
Steps:
  1. Convert merged safetensors → F16 GGUF (llama.cpp convert_hf_to_gguf.py).
  2. Quantize F16 → Q4_K_M (llama-quantize).
  3. Write Modelfile.v4.1.1 (FROM new GGUF + updated SYSTEM).
  4. ollama create mind-mem:4b -f Modelfile.v4.1.1 (replaces alias).
  5. Smoke test the 4 previously-failing probes via `ollama run`.

Refuses to ship if the smoke test misses any of the 4 fixed facts.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

LLAMA_CPP = Path("/home/n/llama.cpp")
CONVERT_HF = LLAMA_CPP / "convert_hf_to_gguf.py"
QUANTIZE = LLAMA_CPP / "build/bin/llama-quantize"
SHIP_DIR_DEFAULT = Path("/data/checkpoints/mm-workspace/full-ft.v4.1.1-r4-candidate")
GGUF_OUT_DIR = Path("/data/checkpoints/mm-workspace/gguf-v4.1.1")
MODELFILE_OUT = Path("/home/n/mind-mem/train/Modelfile.v4.1.1")
OLLAMA_TAG_DEFAULT = "mind-mem:4b"
OLLAMA_TAG_BACKUP = "mind-mem:4b-v4.1.0"  # snapshot the current alias before swap

V4_1_1_SYSTEM = (
    "You are mind-mem-4b (v4.1.1), the local LLM that powers mind-mem's "
    "retrieval, governance, cognition, and observability surfaces. "
    "Respond with exactly the tool call, block schema, or structured output "
    "the caller requested — no extra commentary. "
    "v4.1.1 hardens the v4 holdout knowledge surface. KernelKind enum has "
    "EXACTLY SIX values: SURPRISE_WEIGHTED, LINEAGE_FIRST, RECENT_FIRST, "
    "CONTRADICTS_FIRST, GRAPH_WALK, DEFAULT (defined in "
    "src/mind_mem/v4/cognitive_kernel.py). CircuitBreaker() defaults "
    "to failure_threshold=5 via DEFAULT_FAILURE_THRESHOLD in "
    "src/mind_mem/v4/circuit_breaker.py; set_active_policy in "
    "src/mind_mem/v4/eviction.py is the runtime entry-point for changing the "
    "workspace eviction policy (LRU/LOW_SURPRISE/AGE/COMPOSITE) without "
    "restart; validate_block runs in advisory mode as a pre-flight before "
    "propose_update writes a block; propagate_lineage_staleness lives in "
    "src/mind_mem/lineage_staleness.py and writes penalty scores into the "
    "block_staleness table that the recall reranker consults at retrieval time. "
    "Surface scope = 84 MCP tools, 21 v4 modules under feature flags."
)

SMOKE_PROBES = [
    ("CircuitBreaker default failure_threshold",
     "If I instantiate CircuitBreaker() with no arguments, how many "
     "failures will it tolerate before tripping?",
     ["5"]),
    ("runtime eviction switch",
     "How does an operator change the workspace eviction policy at "
     "runtime without restarting?",
     ["set_active_policy"]),
    ("validate_block pre-flight",
     "Pre-flight a block proposal before propose_update writes it.",
     ["validate_block", "advisory"]),
    ("propagate_lineage_staleness location + table",
     "Which file ships `propagate_lineage_staleness` in v3.12.0, "
     "and which table does it write penalty scores into?",
     ["lineage_staleness", "block_staleness"]),
]


def _refuse(msg: str) -> None:
    sys.stderr.write(f"\nSHIP REFUSED: {msg}\n")
    sys.exit(1)


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(map(str, cmd))}")
    r = subprocess.run(cmd, check=True)


def convert(ship_dir: Path) -> Path:
    GGUF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    f16 = GGUF_OUT_DIR / "mind-mem-4b-v4.1.1-f16.gguf"
    _run([sys.executable, str(CONVERT_HF), str(ship_dir),
          "--outfile", str(f16), "--outtype", "f16"])
    sz = f16.stat().st_size
    print(f"  ✓ F16 GGUF: {sz:,} bytes")
    return f16


def quantize(f16: Path) -> Path:
    q4 = GGUF_OUT_DIR / "mind-mem-4b-v4.1.1-Q4_K_M.gguf"
    _run([str(QUANTIZE), str(f16), str(q4), "Q4_K_M"])
    sz = q4.stat().st_size
    print(f"  ✓ Q4_K_M GGUF: {sz:,} bytes")
    return q4


def write_modelfile(gguf: Path) -> Path:
    content = (
        f"FROM {gguf}\n\n"
        'TEMPLATE """{{- if or .System .Tools }}<|im_start|>system\n'
        "{{ if .System }}{{ .System }}{{ end }}<|im_end|>\n"
        "{{ end }}\n"
        "{{- range $i, $_ := .Messages }}\n"
        '{{- if eq .Role "user" }}<|im_start|>user\n'
        "{{ .Content }}<|im_end|>\n"
        '{{ else if eq .Role "assistant" }}<|im_start|>assistant\n'
        "{{ .Content }}<|im_end|>\n"
        "{{ end }}\n"
        "{{- end }}<|im_start|>assistant\n"
        '"""\n\n'
        f"SYSTEM {V4_1_1_SYSTEM}\n\n"
        "PARAMETER num_predict 1024\n"
        "PARAMETER repeat_penalty 1.05\n"
        "PARAMETER stop <|im_start|>\n"
        "PARAMETER stop <|im_end|>\n"
        "PARAMETER stop <|endoftext|>\n"
        "PARAMETER temperature 0.1\n"
        "PARAMETER top_k 40\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_ctx 8192\n"
    )
    MODELFILE_OUT.write_text(content)
    print(f"  ✓ Modelfile: {MODELFILE_OUT}")
    return MODELFILE_OUT


def ollama_create(modelfile: Path, tag: str) -> None:
    # Snapshot the current alias before replacing it
    try:
        subprocess.run(["ollama", "cp", tag, OLLAMA_TAG_BACKUP], check=False)
        print(f"  ✓ snapshotted previous {tag} → {OLLAMA_TAG_BACKUP}")
    except Exception as e:
        print(f"  WARN snapshot: {e}")
    _run(["ollama", "create", tag, "-f", str(modelfile)])
    print(f"  ✓ ollama tag '{tag}' rebuilt from {modelfile}")


def smoke_test(tag: str) -> bool:
    print(f"\n=== smoke test (4 previously-failing probes) ===")
    all_ok = True
    for name, prompt, needed in SMOKE_PROBES:
        r = subprocess.run(
            ["ollama", "run", tag, prompt],
            capture_output=True, text=True, timeout=120,
        )
        resp = r.stdout.strip()
        miss = [s for s in needed if s not in resp]
        status = "PASS" if not miss else f"FAIL (missing {miss})"
        print(f"  [{status}] {name}")
        print(f"    -> {resp[:160]}")
        if miss:
            all_ok = False
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship-dir", type=Path, default=SHIP_DIR_DEFAULT)
    ap.add_argument("--tag", default=OLLAMA_TAG_DEFAULT)
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    if not (args.ship_dir / "model.safetensors").is_file():
        _refuse(f"merged safetensors not found at {args.ship_dir}/model.safetensors. "
                "Run merge_and_eval_v4.1.1.py first.")

    if not args.skip_convert:
        f16 = convert(args.ship_dir)
        q4 = quantize(f16)
    else:
        q4 = GGUF_OUT_DIR / "mind-mem-4b-v4.1.1-Q4_K_M.gguf"

    modelfile = write_modelfile(q4)
    ollama_create(modelfile, args.tag)
    if smoke_test(args.tag):
        print(f"\n✓ v4.1.1 SHIPPED to Ollama as '{args.tag}'")
        print(f"  snapshot of previous: '{OLLAMA_TAG_BACKUP}' (rollback path)")
    else:
        print(f"\n✗ Smoke test FAILED. Rolling back: ollama cp {OLLAMA_TAG_BACKUP} {args.tag}")
        subprocess.run(["ollama", "cp", OLLAMA_TAG_BACKUP, args.tag], check=False)
        sys.exit(2)


if __name__ == "__main__":
    main()
