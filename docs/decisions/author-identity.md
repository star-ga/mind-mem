# Commit author identity — policy, current state, and the open decision

Status: **decision pending (operator)** · Scope: all public STARGA repositories

## Policy

Every commit in a public repository must be authored *and* committed by
`STARGA Inc. <noreply@star.ga>`. Published work is single-author; no personal
identity belongs in public history. Two spellings of the name occur historically
(`STARGA Inc` and `STARGA Inc.`); the email is the load-bearing field.

## Current state

A scan of commits reachable from `origin/main`:

| Repo | Commits on `main` | Non-conforming | Dominant pattern |
|---|---:|---:|---|
| `mind-mem` | 1160 | 248 | 129 personal-identity (119 of them merge commits); 106 `info@star.ga` |
| `mind-spec` | 126 | 62 | 56 `info@star.ga`; 5 personal; 1 tool default |
| `mind-nerve` | 297 | 17 | 9 personal-authored merges; 8 bot |
| `mind` | 1736 | 7 | 5 bot; 2 personal |

This is **systemic, not a one-off.** Two distinct classes:

1. **Personal identity** (`Nikolai Nedovodin <info@cputer.com>`) — the actual
   leak. ~140 commits across the four repos, overwhelmingly merge commits created with the GitHub
   merge button, which stamps the clicking account as the commit author.
2. **Non-canonical org identity** (`info@star.ga` instead of `noreply@star.ga`)
   — a policy drift, not a personal leak, and materially less severe.

The commit that prompted this review, `28ffc5a`
(`docs(mhs): define device-memory freshness boundary`), is class 1 and is on
`origin/main`.

## Root cause

Not a single bad commit — a configuration surface that fails open, in three
layers:

- **The global fallback is itself wrong.** `~/.gitconfig` on the primary
  workstation reads `user.email = info@star.ga`. Any repository without a local
  override inherits a non-conforming identity by default. On that machine, 52
  STARGA repositories currently resolve to a non-canonical identity; `mind-mem`
  is correct only because it happens to carry a local override.
- **Other machines have no override at all.** `28ffc5a` does not appear in this
  clone's reflog; it arrived via `fetch origin: fast-forward`. It was created and
  pushed from a different checkout, where git fell through to that machine's
  personal identity. **No local hook can prevent this** — the hook was never
  installed there. Only a server-side or CI check catches it.
- **The merge button re-authors.** Landing a PR through the GitHub UI attributes
  the merge commit to the clicking human account regardless of local config.
  This produced the large majority of class-1 commits.

The recurrence is the defect. Fixing one commit while leaving these three paths
open simply leaks again.

## The decision

Two standing rules conflict here: *never rewrite public history* versus *no
personal identity in public history*.

**Option A — accept and fix forward (recommended).**
Leave history as-is; prevent recurrence with the guard described below.

- The email is already public: cloned, forked, cached by the GitHub events API,
  and mirrored by third-party services. A rewrite does **not** un-publish it.
- `info@cputer.com` is a business contact address, not a home address or a
  secret. The disclosure is a policy violation, not a security incident.
- Cost is zero.

**Option B — rewrite and force-push.**
Rewrite ~140 commits across four repos and force-push.

- Blast radius: every commit SHA from the earliest rewritten commit onward
  changes. Every clone and fork must re-clone or hard-reset. Every open PR must
  be rebased or reopened. Signed tags and release artifacts referencing old SHAs
  break, as does any external link to a commit. Since the earliest affected
  commit in `mind-mem` is `Initial commit`, this means the entire history of the
  repository.
- Benefit: the email disappears from the *current* tree only. Old SHAs stay
  reachable via the events API and existing forks.

**Assessment:** Option B pays the maximum possible cost for a partial remedy of
an already-published fact. Recommend Option A, with the guard made mandatory.
This is an operator decision and is **not** executed by this document.

## The guard (implemented)

- `scripts/check_author_identity.sh` — two modes. `staged` validates the
  identity the next commit would carry, including `GIT_AUTHOR_*` /
  `GIT_COMMITTER_*` overrides. `range <base>..<head>` validates a commit range.
- `scripts/pre-commit-hook.sh` — composite pre-commit hook. Install with
  `ln -sf ../../scripts/pre-commit-hook.sh .git/hooks/pre-commit`. It runs the
  identity check first, then the existing anatomy refresh; there is only one
  pre-commit slot and chaining here keeps a new check from silently displacing
  an old one.
- **`author-identity` job in `.github/workflows/ci.yml`** — the load-bearing
  leg. It checks the commits a push or PR *introduces*, which is the only layer
  that catches a commit made in a clone where the hook was never installed —
  precisely how `28ffc5a` got in. Scope is the introduced range, not all
  history, so pre-existing violations do not hold the gate permanently red.

Merge commits are checked on the author field but not the committer field, since
a GitHub-created merge is necessarily committed by `GitHub <noreply@github.com>`.
Recognised automation (dependabot) is exempt by allowlist: machine accounts leak
no personal identity.

## Remaining operator actions

1. Correct the global fallback so no repository can inherit a wrong identity:
   ```bash
   git config --global user.name  "STARGA Inc."
   git config --global user.email "noreply@star.ga"
   ```
   Apply on **every** machine that commits to a STARGA repo — the workstation
   and each fleet node. This is the single highest-value fix; it closes the
   default-inheritance path for all 52 exposed local repos at once.
2. Install the hook in each working clone (it is not transferred by `git clone`).
3. Stop landing PRs with the GitHub merge button on public repos; land via
   `git push`, which preserves the committed author.
4. Adopt the same `author-identity` CI job in `mind`, `mind-spec`, `mind-nerve`.
