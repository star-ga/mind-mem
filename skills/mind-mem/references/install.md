# Installing mind-mem

## Requirements

* Python 3.10 or newer.
* The core package and the `mm` CLI have no required third-party
  dependencies. Optional features are pulled in through extras.

## Install the package

```bash
pip install mind-mem                 # CLI + engine
pipx install "mind-mem[mcp]"         # isolated install, adds the MCP server
pip install "mind-mem[postgres]"     # Postgres block-store backend
```

Available extras: `mcp`, `api`, `postgres`, `embeddings`, `cross-encoder`,
`otel`, `accelerated`, `benchmark`, `red-team`, `test`, and `all`.

Check the install:

```bash
mm --help
mind-mem-mcp --help        # only with the [mcp] extra
```

## Create a workspace

A workspace is a directory holding the Markdown corpus, `mind-mem.json` and a
local index.

```bash
mind-mem-init ~/mind-mem-workspace
export MIND_MEM_WORKSPACE=~/mind-mem-workspace
mm status
```

`mind-mem-init` options: `--backend {markdown,postgres,encrypted}` (default
`markdown`), `--dsn` and `--schema` for Postgres, `--ensure-schema` to create
the Postgres tables now. The backend can also come from `MIND_MEM_BACKEND`
and the DSN from `MIND_MEM_DSN`. The `encrypted` backend needs
`MIND_MEM_ENCRYPTION_PASSPHRASE` set.

`mm` itself never guesses a workspace: it uses `$MIND_MEM_WORKSPACE`, or the
current directory when that is unset.

## Wire mind-mem into AI coding clients

```bash
mm detect                       # which clients are installed (JSON)
mm install-all --dry-run        # preview every change
mm install-all                  # configure every detected client
mm install claude-code          # configure one client
```

* Install is a non-destructive merge; `--force` overwrites instead.
* `mm install-all --no-mcp` writes only the text/hook integration and skips
  native MCP server registration.
* `mm install-all --agent codex --agent gemini` restricts the run to named
  clients.
* Both commands configure the workspace `mm` resolves (see above), so set
  `MIND_MEM_WORKSPACE` first.
* Client keys accepted by `mm install` and `--agent`: `claude-code`, `codex`,
  `grok-build`, `vibe`, `gemini`, `cursor`, `windsurf`, `aider`, `openclaw`,
  `nanoclaw`, `nemoclaw`, `continue`, `cline`, `roo`, `zed`, `copilot`,
  `copilot-cli`, `cody`, `qodo`.

Restart a client after wiring it; most clients read their configuration only
at startup.

From a source checkout, `./install.sh` installs the package and wires clients
in one step (`./install.sh --all`, or per client: `--claude-code`, `--codex`,
`--gemini`, `--cursor`, `--windsurf`, `--zed`, `--openclaw`,
`--claude-desktop`; `--workspace PATH`; `--no-install` to only wire clients).

## Install this skill

The skill ships with the package. Copy it into an agent's skills directory:

```bash
mm skill install                             # -> ~/.claude/skills/mind-mem
mm skill install --target ~/.codex/skills    # any SKILL.md-folder agent
mm skill install --dry-run                   # show source and destination only
mm skill install --force                     # replace a different existing copy
```

The command prints JSON with a `status` of `installed`, `up_to_date`,
`would_install`, `exists` (a different copy is present; re-run with
`--force`) or `missing_bundle`. Without the package, copy `skills/mind-mem/`
from the source repository into the skills directory by hand.

## Optional local model

```bash
mm install-model --dry-run
mm install-model
```

Downloads the `mind-mem-4b` GGUF and registers it in Ollama as `mind-mem:4b`.
Requires `ollama` on `PATH`. Flags: `--model`, `--name`, `--dest`,
`--keep-alive`.

## Upgrading

```bash
mm self-update --check     # exit 10 when a newer release is available
mm self-update --yes       # upgrade without prompting
mm self-update --pre       # include pre-releases
```

Automatic update checks are off unless the workspace `mind-mem.json` has
`"auto_update": {"enabled": true}` (`mode` is `notify` or `auto`, `channel`
is `stable` or `pre`, `interval_hours` defaults to 24). Setting
`MIND_MEM_NO_AUTO_UPDATE=1` disables the check for a process.

After an upgrade that changes the workspace layout, run
`mind-mem-migrate "$MIND_MEM_WORKSPACE"`, then `mm doctor`.

## Removing

* `pip uninstall mind-mem` (or `pipx uninstall mind-mem`) removes the package.
* Client wiring lives in each client's own config file under a `mind-mem`
  entry; remove that entry to unwire a client.
* The workspace directory is plain files and is never deleted by uninstalling
  the package.
