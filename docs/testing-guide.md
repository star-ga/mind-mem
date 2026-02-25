# Testing Guide

## Running Tests

```bash
# Full suite
pytest tests/

# Specific file
pytest tests/test_recall_edge_cases.py

# By marker
pytest tests/ -k "benchmark"

# With coverage
pytest tests/ --cov=scripts --cov-report=term-missing

# Verbose
pytest tests/ -v
```

## Test Categories

### Unit Tests
Test individual functions in isolation:
- `test_tokenization.py` — Tokenizer functions
- `test_constants.py` — Constant validation
- `test_bigrams.py` — Bigram extraction
- `test_stopwords.py` — Stopword handling

### Integration Tests
Test module interactions:
- `test_recall_edge_cases.py` — Recall with edge inputs
- `test_hybrid_search.py` — End-to-end search
- `test_multi_file_recall.py` — Cross-directory recall

### Edge Case Tests
- `test_block_parser_edge.py` — Parser edge cases
- `test_unicode_edge_cases.py` — Unicode handling

## Writing Tests

### Workspace Fixture
Use `init()` to create temporary workspaces:

```python
import tempfile
from scripts.init_workspace import init

def test_example():
    ws = tempfile.mkdtemp()
    init(ws)
    # ... test code
```

### Module Import Pattern
For modules that may not exist:

```python
def test_optional():
    try:
        from scripts.some_module import some_func
        result = some_func()
        assert result is not None
    except ImportError:
        pass
```

## CI Matrix

Tests run on:
- Python: 3.10, 3.12, 3.13, 3.14
- OS: Ubuntu, macOS, Windows
- Total: 14 CI jobs
