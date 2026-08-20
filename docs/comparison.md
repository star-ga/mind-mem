# Comparison with Alternatives

> LoCoMo numbers for MIND-Mem below are the canonical full-10-conversation
> (1986-question), BM25-only, external-LLM-judge result — see
> [`docs/benchmarks.md`](benchmarks.md#locomo-benchmark--canonical-number)
> for scope, reproduction command, and raw data. Do not cite a different
> MIND-Mem LoCoMo number.

## MIND-Mem vs Mem0

| Feature | MIND-Mem | Mem0 |
|---------|----------|------|
| Dependencies | Zero | Redis, PostgreSQL |
| Retrieval | BM25F + vector hybrid | Vector only |
| Audit trail | Full proposal system | Limited |
| LoCoMo benchmark (mean score, full 10-conv) | 70.5 | 66.88¹ |
| Contradiction detection | Built-in | No |

## MIND-Mem vs Zep

| Feature | MIND-Mem | Zep |
|---------|----------|-----|
| Dependencies | Zero | Cloud service |
| Scoring | BM25F with field weights | Proprietary |
| LoCoMo benchmark (mean score, full 10-conv) | 70.5 | 65.99 |
| Self-hosted | Yes (files only) | Requires Zep Cloud |
| Open source | Fully open | Partial |

## MIND-Mem vs LangMem

| Feature | MIND-Mem | LangMem |
|---------|----------|---------|
| Dependencies | Zero | LangChain |
| Retrieval | BM25F + hybrid | Vector-based |
| LoCoMo benchmark (mean score, full 10-conv) | 70.5 | 58.10 |
| MIND kernels | Yes | No |
| MCP tools | 89 | N/A |

## MIND-Mem vs Full Context

| Feature | MIND-Mem | Full Context |
|---------|----------|-------------|
| Scalability | O(log n) retrieval | O(n) context |
| Token cost | Low (top-K only) | High (all tokens) |
| LoCoMo benchmark (mean score, full 10-conv) | 70.5 | 72.90 |
| LoCoMo Acc>=50 (full 10-conv) | 73.8% | 72.9% |

MIND-Mem and a full-context baseline score within noise of each other on raw
LoCoMo accuracy — the benchmark does not capture MIND-Mem's actual value
proposition against full-context, which is **O(log n) token cost at scale**
plus governance (contradiction detection, audit trail, byte-identical replay)
that full-context has no equivalent of.

---

¹ `66.88` is Mem0's own published LoCoMo-paper number. Mem0's separate **2026
managed platform** self-reports **91.6** on LoCoMo — that is a different
setup/judge (their hosted product benchmark, not the open LoCoMo-paper
config) and is not apples-to-apples with the number in this table. See
[`docs/benchmarks.md`](benchmarks.md#locomo-benchmark--canonical-number) for
the full comparison table and honesty notes.
