# Benchmarks

## LoCoMo Benchmark Results

mind-mem v1.8.0, evaluated with Mistral Large (LoCoMo, 10 conversations):

| Metric | Score |
|--------|-------|
| Mean | 77.9 |
| Adversarial | 82.3 |
| Temporal | 88.5 |

### Comparison

| System | Mean | Adversarial | Temporal |
|--------|------|-------------|----------|
| **mind-mem** | **77.9** | **82.3** | **88.5** |
| Full context | 72.90 | - | - |
| Mem0 | 66.88 | - | - |
| Zep | 65.99 | - | - |
| LangMem | 58.10 | - | - |

## Recall Latency

CI benchmark with 50 blocks:
- Average: <500ms
- P95: <500ms

## How to Run

```bash
# Run CI benchmark
pytest tests/ -k "benchmark" --benchmark-only

# Run recall timing
make benchmark
```
