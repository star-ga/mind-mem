# Setting up the mind-mem-4b model

`star-ga/mind-mem-4b` is the **full fine-tune** of `Qwen/Qwen3.5-4B`
on the mind-mem domain — all 4.2B parameters trained (not a LoRA
adapter). It knows the 57 MCP tools, 14 block schemas, governance
workflows, and CHANGELOG history through v3.0.0.

You don't *need* this model to use mind-mem — every client integrates
through the MCP server and works with any LLM. The model is there
when you want **fast, local, mind-mem-aware inference** for queries,
recall composition, or captured-fact compression.

## TL;DR

```bash
# 1. Install mind-mem itself (Python package + CLI)
pip install --upgrade "mind-mem[all]"

# 2. Initialise a workspace
mind-mem-init ~/.openclaw/workspace
export MIND_MEM_WORKSPACE=~/.openclaw/workspace

# 3. Auto-configure every AI client on this machine
mm install-all

# 4. Pull the model (~8.4 GB, full-FT bf16)
hf download star-ga/mind-mem-4b --local-dir ~/mm-models/mind-mem-4b

# 5. Sanity check — it should talk
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('~/mm-models/mind-mem-4b')
m = AutoModelForCausalLM.from_pretrained(
    '~/mm-models/mind-mem-4b',
    dtype='bfloat16',
    device_map='auto',
)
msg = [
    {'role': 'system', 'content': 'You are mind-mem-4b.'},
    {'role': 'user',   'content': 'Which MCP tool verifies the audit chain?'},
]
ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors='pt').to(m.device)
print(tok.decode(m.generate(ids, max_new_tokens=128)[0][ids.shape[1]:], skip_special_tokens=True))
"
```

If that last command prints something about `verify_chain`, you're done.

## Hardware requirements

| Precision | Disk | VRAM (load) | Speed (RTX 3080) |
|---|---|---|---|
| bf16 (default) | 8.4 GB | ~8.5 GB | ~40 tok/s |
| fp16 | 8.4 GB | ~8.5 GB | ~40 tok/s |
| int8 (bnb) | 8.4 GB (disk) | ~5 GB | ~60 tok/s |
| int4 (bnb NF4) | 8.4 GB (disk) | ~3 GB | ~80 tok/s |
| EXL2 4.0 bpw | ~2.8 GB | ~3.5 GB | **~180 tok/s** |
| EXL2 6.0 bpw | ~3.6 GB | ~4.5 GB | ~150 tok/s |

**Sweet spot for a consumer NVIDIA card: EXL2 4.0 bpw via exllamav2.**
CPU-only inference works (llama.cpp after GGUF conversion) but is
~5-10 tok/s — usable for batch jobs, painful for interactive.

## Option 1 — Transformers (simplest, slowest)

Fine for one-off recall composition. Handles any backend (CPU, MPS,
CUDA, ROCm). No quantization needed on a 10 GB+ NVIDIA card.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "star-ga/mind-mem-4b"  # or a local path

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",  # falls back to eager if sdpa unavailable
)
model.eval()

def chat(user: str, system: str = "You are mind-mem-4b.") -> str:
    msgs = [{"role": "system", "content": system},
            {"role": "user",   "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=256, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

print(chat("Show me a DREF block template."))
```

For 4-bit load on a tight card, add `bitsandbytes` and switch the
loader:

```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto"
)
```

## Option 2 — exllamav2 (fastest on NVIDIA)

STARGA's default for local inference. 5× faster than transformers
bf16 on the same card.

### Convert the model to EXL2

One-time step (takes ~10 min on an RTX 3080):

```bash
# Clone exllamav2 if you haven't already
git clone https://github.com/turboderp/exllamav2 ~/src/exllamav2

# Download calibration data (wikitext by default)
# Quantize to 4.0 bpw — the sweet spot for 4B models.
python ~/src/exllamav2/convert.py \
    -i ~/mm-models/mind-mem-4b \
    -o ~/mm-models/mind-mem-4b-exl2-4bpw \
    -cf ~/mm-models/mind-mem-4b-exl2-4bpw \
    -b 4.0
```

For quality-sensitive workloads bump to `-b 6.0` (adds ~1 GB, gains
~0.5 bpw effective precision).

### Inference via exllamav2

```python
from exllamav2 import ExLlamaV2, ExLlamaV2Cache_Q4, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler

config = ExLlamaV2Config("~/mm-models/mind-mem-4b-exl2-4bpw")
model = ExLlamaV2(config)
tok = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache_Q4(model, lazy=True)
model.load_autosplit(cache, progress=True)

gen = ExLlamaV2DynamicGenerator(model=model, cache=cache, tokenizer=tok)
settings = ExLlamaV2Sampler.Settings.greedy()

prompt = (
    "<|im_start|>system\nYou are mind-mem-4b.<|im_end|>\n"
    "<|im_start|>user\nWhich tool verifies the audit chain?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
print(gen.generate(prompt=prompt, max_new_tokens=256, gen_settings=settings))
```

Wrap this in a long-running process and you get ~180 tok/s end-to-end
on a 3080.

## Option 3 — vLLM (best for serving multiple concurrent users)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model star-ga/mind-mem-4b \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

Exposes an OpenAI-compatible endpoint on `localhost:8000`. Point any
HTTP client (including the mind-mem MCP server via an optional
adapter) at it.

## Option 4 — GGUF + llama.cpp (CPU-friendly)

```bash
# Install llama.cpp tools
brew install llama.cpp  # or: apt install llama.cpp, or build from source

# Convert + quantize
cd ~/src/llama.cpp
python convert_hf_to_gguf.py ~/mm-models/mind-mem-4b \
    --outfile ~/mm-models/mind-mem-4b-f16.gguf \
    --outtype f16

./build/bin/llama-quantize \
    ~/mm-models/mind-mem-4b-f16.gguf \
    ~/mm-models/mind-mem-4b-Q4_K_M.gguf \
    Q4_K_M

# Interactive shell
./build/bin/llama-cli -m ~/mm-models/mind-mem-4b-Q4_K_M.gguf \
    -sys "You are mind-mem-4b." \
    -p "Show me a CODE block template."
```

## Option 5 — Ollama (one-command local service)

```bash
ollama pull star-ga/mind-mem-4b  # if STARGA ships a GGUF on the Ollama registry
# or, from a local GGUF:
cat > ~/mm-models/mind-mem-4b.Modelfile << 'EOF'
FROM ~/mm-models/mind-mem-4b-Q4_K_M.gguf
SYSTEM "You are mind-mem-4b."
PARAMETER num_ctx 4096
PARAMETER temperature 0.2
EOF
ollama create mind-mem-4b -f ~/mm-models/mind-mem-4b.Modelfile
ollama run mind-mem-4b "What MCP tool lists contradictions?"
```

## Wiring the model into mind-mem's workflows

The model is genuinely useful when it sits next to the mind-mem MCP
server and handles the LLM-side operations (recall composition,
capture enrichment, contradiction classification).

Typical stack:

```
┌──────────────┐       ┌─────────────────┐       ┌──────────────┐
│  Claude Code │──────▶│ mind-mem MCP    │──────▶│  SQLite +    │
│ (or any CLI) │       │  (57 tools)     │       │  FTS + vec   │
└──────────────┘       └────────┬────────┘       └──────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  mind-mem-4b    │   <- this model
                       │ via transformers│
                       │   / exllamav2   │
                       └─────────────────┘
```

Hook it in via `configuration.md` → `llm.backend`. Default is
"ollama" pointing at `localhost:11434/api/generate`; switch to
"exllamav2" (or a generic `openai-compatible` backend targeting your
vLLM endpoint) in the workspace's `mind-mem.json`.

```json
{
  "llm": {
    "backend": "openai-compatible",
    "base_url": "http://localhost:8000/v1",
    "model": "star-ga/mind-mem-4b"
  }
}
```

## Checklist

- [ ] `pip install --upgrade "mind-mem[all]"` (Python 3.10+)
- [ ] `mind-mem-init <workspace>`
- [ ] `mm install-all` to wire every local AI client
- [ ] `hf download star-ga/mind-mem-4b --local-dir <path>`
- [ ] Pick a runtime (transformers / exllamav2 / vLLM / llama.cpp / Ollama)
- [ ] Point mind-mem's `llm.backend` at the runtime if you want the
  model to handle mind-mem's internal LLM tasks
- [ ] Add the Memory Protocol block to your CLI's CLAUDE.md / AGENTS.md
  / .cursorrules (the `mm install` commands do this automatically)

## Known limitations

- **Imperative phrasing** still occasionally triggers role-play
  instead of clean tool recall ("roll back the apply" vs "which tool
  rolls back an apply?"). Phrase questions as queries until v3.1
  lands a bigger corpus.
- **Out-of-domain open-chat** — the model is specialised. Use the
  base `Qwen/Qwen3.5-4B` for anything non-memory.
- **Context length** — tested at 4K; pushing to 8K+ may degrade
  structured output quality.
