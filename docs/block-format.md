# Block Format

## Overview

mind-mem stores information as structured blocks in markdown files. Each block represents a discrete unit of knowledge.

## Block Structure

```markdown
[BLOCK-ID]
Type: Decision
Statement: The main content of this block
Tags: tag1, tag2
Date: 2026-02-24
Status: Active
Priority: High
Speaker: user
References: OTHER-001, OTHER-002
```

## Required Fields

| Field | Description |
|-------|-------------|
| ID | Unique identifier in brackets (e.g., `[PROJ-001]`) |
| Type | Block classification (Decision, Task, Entity, etc.) |
| Statement | Main content of the block |

## Optional Fields

| Field | Description |
|-------|-------------|
| Tags | Comma-separated labels |
| Date | Creation or decision date (YYYY-MM-DD) |
| Status | Active, WIP, Archived, Superseded |
| Priority | High, Medium, Low |
| Speaker | Who created this block |
| References | Comma-separated block IDs |
| Supersedes | Block ID this replaces |

## Block Types

- **Decision** — A decision or constraint
- **Task** — A tracked task or action item
- **Entity** — A person, project, or tool
- **Memory** — A daily log or observation
- **Signal** — An intelligence signal
- **Summary** — A category summary

## File Organization

Blocks are organized by type in the workspace:

```
workspace/
  decisions/    → Decision blocks
  tasks/        → Task blocks
  entities/     → Entity blocks
  memory/       → Memory/log blocks
  intelligence/ → Signal blocks
  summaries/    → Summary blocks
```

## ID Conventions

- Use uppercase prefix matching the category (e.g., `DEC-`, `TASK-`, `ENT-`)
- Follow with a zero-padded number (e.g., `DEC-001`)
- IDs must be unique across the entire workspace
