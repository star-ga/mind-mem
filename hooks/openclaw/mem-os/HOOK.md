---
name: mind-mem
description: "Memory OS health check on bootstrap + auto-capture on /new"
homepage: https://github.com/star-ga/mind-mem
metadata:
  { "openclaw": { "emoji": "🧠", "events": ["agent:bootstrap", "command:new"], "requires": { "bins": ["python3"] } } }
---
# Mind Mem

Memory + Immune System for OpenClaw agents. Injects health context on agent bootstrap and auto-captures session signals on `/new`.

## Events

- **agent:bootstrap**: Reads `intel-state.json` and pushes health summary into bootstrap context
- **command:new**: Runs `capture.py` to extract decision/task signals from the session

## Configuration

In `~/.openclaw/openclaw.json`:

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "mind-mem": {
          "enabled": true,
          "env": {
            "MIND_MEM_WORKSPACE": "/path/to/your/workspace"
          }
        }
      }
    }
  }
}
```

## Install

```bash
# Copy hook to managed hooks directory
cp -r /path/to/mind-mem/hooks/openclaw/mind-mem ~/.openclaw/hooks/mind-mem

# Enable
openclaw hooks enable mind-mem
```
