# Workspace Structure

## Directory Layout

```
workspace/
├── .mind/              # MIND kernel configs (optional)
│   └── recall.mind     # BM25F parameter overrides
├── decisions/          # Decision blocks
├── tasks/              # Task blocks
├── entities/           # Entity blocks (people, tools, projects)
├── memory/             # Daily logs and observations
├── intelligence/       # Signals and analysis
├── summaries/          # Category summaries
└── MEMORY.md           # Stable long-term facts
```

## Directory Purposes

| Directory | Block Types | Purpose |
|-----------|------------|---------|
| `decisions/` | Decision | Constraints, choices, architectural decisions |
| `tasks/` | Task | Action items, todos, tracked work |
| `entities/` | Entity | People, tools, projects, organizations |
| `memory/` | Memory | Daily logs, observations, conversations |
| `intelligence/` | Signal | Analysis, contradictions, drift detection |
| `summaries/` | Summary | Category-based topic summaries |

## File Naming

- Use descriptive names: `api-design.md`, `sprint-42.md`
- One topic per file when possible
- Multiple blocks per file are supported

## Initialization

```python
from scripts.init_workspace import init
init("/path/to/workspace")
```

This creates all directories and `MEMORY.md`. Safe to call multiple times (idempotent).
