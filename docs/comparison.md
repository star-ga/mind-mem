# Comparison with Alternatives

## MIND-Mem vs Mem0

| Feature | MIND-Mem | Mem0 |
|---------|----------|------|
| Dependencies | Zero | Redis, PostgreSQL |
| Retrieval | BM25F + vector hybrid | Vector only |
| Audit trail | Full proposal system | Limited |
| LoCoMo benchmark | 77.9 mean | 66.88 mean |
| Contradiction detection | Built-in | No |

## MIND-Mem vs Zep

| Feature | MIND-Mem | Zep |
|---------|----------|-----|
| Dependencies | Zero | Cloud service |
| Scoring | BM25F with field weights | Proprietary |
| LoCoMo benchmark | 77.9 mean | 65.99 mean |
| Self-hosted | Yes (files only) | Requires Zep Cloud |
| Open source | Fully open | Partial |

## MIND-Mem vs LangMem

| Feature | MIND-Mem | LangMem |
|---------|----------|---------|
| Dependencies | Zero | LangChain |
| Retrieval | BM25F + hybrid | Vector-based |
| LoCoMo benchmark | 77.9 mean | 58.10 mean |
| MIND kernels | Yes | No |
| MCP tools | 54 | N/A |

## MIND-Mem vs Full Context

| Feature | MIND-Mem | Full Context |
|---------|----------|-------------|
| Scalability | O(log n) retrieval | O(n) context |
| Token cost | Low (top-K only) | High (all tokens) |
| LoCoMo benchmark | 77.9 mean | 72.90 mean |
| Adversarial | 82.3 | Lower |
| Temporal | 88.5 | Lower |
