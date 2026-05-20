# Companion Tools

External tools that solve adjacent memory/inference problems
mind-mem deliberately does not solve. Documented here so users see
them as **complements**, not competitors. mind-mem **does not depend
on any of these** — license, scope, and substrate-of-record concerns
make co-existence the right pattern.

## MindLLM — pure-MIND deterministic inference (STARGA)

[`star-ga/MindLLM`](https://github.com/star-ga/MindLLM) is STARGA's
local, governed, **deterministic** inference server written in pure
MIND. It exposes OpenAI-compatible endpoints
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`)
plus a first-party RFN (Resonance Field Network) classifier endpoint
(`/v1/rfn/classify`).

mind-mem treats MindLLM as a first-class LLM backend (alongside
Ollama and vLLM). Set `backend: "mindllm"` in `mind-mem.json` or
`MIND_MEM_LLM_BACKEND=mindllm` in the environment; mind-mem will
talk to `http://localhost:8080/v1` by default (override with
`MIND_MEM_MINDLLM_URL`).

**Why pick MindLLM over Ollama/vLLM for mind-mem workloads:**

| Property                                    | Ollama / vLLM       | MindLLM             |
| ------------------------------------------- | ------------------- | ------------------- |
| Bit-identical output, same GPU              | no                  | **yes** (Q16.16 evidence path + deterministic-reduction kernels) |
| Per-token cryptographic evidence chain      | no                  | **yes** |
| Compiled governance constraints (I11–I15)   | no                  | **yes** (via 512-mind) |
| Mid-request detection of silent reconfig    | no                  | **yes** (HOLD→RESUME) |
| Raw tokens/sec advantage vs vLLM            | —                   | **±10%** at best |

mind-mem's `mind-mem-4b` retrieval-helper model runs equally on
Ollama or MindLLM at the protocol layer; the deterministic +
evidence-chain guarantees that MindLLM adds are what differentiate
audit-grade deployments from best-effort ones.

**Quick start (mind-mem ↔ MindLLM):**

```bash
# 1. Build MindLLM (requires mind-inference + 512-mind siblings —
#    see https://github.com/star-ga/MindLLM Quick start).
git clone https://github.com/star-ga/MindLLM ~/MindLLM
cd ~/MindLLM && make build && ./build/MindLLM serve --port 8080 &

# 2. Point mind-mem at it.
export MIND_MEM_LLM_BACKEND=mindllm
# (optional override) export MIND_MEM_MINDLLM_URL=http://localhost:8080/v1

# 3. Verify reachability.
mm doctor
```

mind-mem's existing `auto`-discovery order also probes MindLLM at
`localhost:8080` before vLLM at `localhost:8000`, so deployments
that already run MindLLM are picked up with zero config.

## GitNexus — code knowledge-graph indexer (third-party, MCP)

[`h4ckf0r0day/GitNexus`](https://github.com/h4ckf0r0day/GitNexus)
is a code knowledge-graph indexer exposed as an MCP server. It
parses repo structure (call graphs, dependencies, clusters) and
serves architectural-awareness tools to coding agents.

**What it solves vs what mind-mem solves:**

| Question                                    | Tool                 |
| ------------------------------------------- | -------------------- |
| "what does the code do at this point in time" | **GitNexus** |
| "what did we decide and why, over time"    | **mind-mem** |

The two are orthogonal — code structure today vs governed decision
history. License: **PolyForm Noncommercial**, incompatible with
mind-mem's Apache-2.0 as a programmatic dependency.

**Recommendation:** install GitNexus as a separate MCP server
alongside mind-mem. Both end up in your Claude Code / Cursor /
Windsurf / Zed MCP lists — no integration code required.

```bash
# GitNexus install (per its own README — typical pattern):
git clone https://github.com/h4ckf0r0day/GitNexus
# follow upstream install + MCP-registration instructions

# mind-mem install (Apache-2.0, this repo):
pip install "mind-mem[all]"
mm install-all  # auto-wires MCP for Claude Code, Cursor, Windsurf, ...
```

Both tools then appear in your MCP client's tool list and answer
their respective question domains side-by-side.

---

## Out of scope here

This document only covers tools mind-mem can be configured to
**talk to** (MindLLM as a backend) or that share a **deployment
pattern** (GitNexus as a sibling MCP server). For mind-mem's
internal architecture — block schema, recall pipeline, governance
engine, federation — see `ARCHITECTURE.md` and `ROADMAP.md`.
