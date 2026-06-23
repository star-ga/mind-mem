# Agent-to-agent messaging (`mm send` / `mm inbox`)

mind-mem is the **sanctioned cross-agent / cross-node comm channel** for a
STARGA fleet. There is no separate message-bus daemon: the *shared block
store is the channel*. One agent "sends" by writing a message block; another
"receives" by reading its inbox. Because every node points at the same store
(the `.193` Postgres federation hub), a message written on one node is
readable on every node — with no extra wiring.

> This is the supported channel. `claude-peers` is banned for fleet comms.

## TL;DR

```bash
# Send a directed message
mm send "deploy the patch" --to S1 --from U1 --subject "patch"

# Send a broadcast (no --to: every recipient sees it)
mm send "fleet-wide notice: gate is green" --from U1

# Read your inbox (your addressed mail + broadcasts), newest first
mm inbox --to S1
mm inbox --to S1 --since 2026-06-23 -n 50
```

The `mm` console script is on PATH in every wired CLI (claude / codex /
gemini / grok …) and in every sub-agent shell, so the same two commands work
identically everywhere. Sub-agents that can't reach MCP use exactly this bash
form.

## Mental model

| Verb | Mechanism |
|------|-----------|
| **send** | write an `MSG-` block (`type: AgentMessage`) to `memory/MESSAGES.md` (markdown default) or the Postgres store |
| **receive** | enumerate the messages corpus, filtered to your `To` (plus broadcasts) |

- **Send = write a block.** The block id is `MSG-<utc-stamp>-<rand>`, which
  routes through the `MSG` entry in `_BLOCK_PREFIX_MAP` to
  `memory/MESSAGES.md`. The block carries `To`, `From`, `Subject`, and the
  message body in `Statement`.
- **Receive = read your mail.** `mm inbox --to S1` returns every message
  addressed to `S1` plus every broadcast (a message with no `To`). It
  enumerates the corpus directly (via the backend-aware
  `storage.iter_active_blocks`), so all message fields are preserved and it
  works identically on SQLite-markdown and Postgres.
- **Search still works too.** Sent messages are also indexed, so
  `mm recall "<keyword>"` / `hybrid_search` find them like any other block.

## Verified send → receive

Run against a fresh markdown workspace:

```text
$ mm send "deploy the patch" --to S1 --from U1 --subject "patch"
{"sent": "MSG-20260623T093228Z-ff9d2b71", "to": "S1", "from": "U1"}

$ mm inbox --to S1
[
  {
    "_id": "MSG-20260623T093228Z-ff9d2b71",
    "Statement": "deploy the patch",
    "Status": "active",
    "Subject": "patch",
    "From": "U1",
    "Timestamp": "20260623T093228Z",
    "To": "S1",
    "_source_file": "memory/MESSAGES.md",
    "_source_label": "messages"
  }
]
```

Recipient scoping is real: `mm inbox --to G1` against the same workspace
returns only broadcasts, never `S1`'s addressed mail.

## Cross-node (the federation hub)

Set every node's workspace to the shared store and the two commands become a
fleet message bus:

```bash
export MIND_MEM_WORKSPACE="$HOME/.openclaw/workspace"   # mind-mem.json -> postgres .193
# On U1:
mm send "starting the keystone gate run" --from U1 --to S1
# On S1 (same hub):
mm inbox --to S1
```

On Postgres, the store returns every active block regardless of which node
wrote it, so no node-local index step is needed — the write is itself the
delivery.

## Python API

```python
from mind_mem.agent_messaging import send_message, read_inbox

mid = send_message(ws, "deploy the patch", to="S1", sender="U1", subject="patch")
for msg in read_inbox(ws, to="S1", since="2026-06-23"):
    print(msg["From"], "->", msg["To"], ":", msg["Statement"])
```

## Notes / limits

- **Broadcasts** are messages with no `To`. Every recipient sees them.
- **`--since`** is an ISO-8601 lower bound; string comparison works for both
  the `YYYYMMDDTHHMMSSZ` stamp and `YYYY-MM-DD` dates.
- **No read receipts / deletion-on-read.** Messages are durable blocks; a
  recipient reading its inbox does not consume them. Prune with the normal
  block-lifecycle tools (`delete_memory_item`) if needed.
- **Identification.** `mm inbox` recognises messages by the `MSG-` id prefix
  (the markdown parser drops the lowercase `type:` field on re-parse, same as
  inbox-ingested documents), so the prefix is the stable contract.
