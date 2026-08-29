# Commit author identity — policy, measured state, and the open decision

Status: **decision pending (operator)** · Scope: all public STARGA repositories
Last re-measured: 2026-08-28. Every number below is reproducible with the command
printed beside it; earlier revisions of this document carried figures that no
longer hold, and the corrections are called out explicitly.

## Policy

Every commit in a public repository must be authored *and* committed by
`STARGA Inc. <noreply@star.ga>`. Published work is single-author; no personal
identity belongs in public history.

**Three** spellings of the name occur in public history, and they are not
interchangeable to a string check:

| Spelling | Commits | Status |
|---|---:|---|
| `STARGA Inc`   | 3013 | in the workstation global config |
| `STARGA Inc.`  |  138 | in the documented commit recipe |
| `STARGA, Inc.` |    1 | `mind-spec` only; **not** in the guard's allowlist |

The period-less and period forms are both in active use — config supplies one,
the documented recipe supplies the other — so a guard hard-coded to either would
reject the other, and both are allowlisted. The comma form is deliberately *not*
allowlisted: it is a single historical commit, and blessing a third spelling to
make one commit pass would entrench the drift this section exists to remove.
**The email is the load-bearing field**; it is what identifies a person.

## Measured state

```bash
git -C <repo> log --format='%an <%ae>' origin/main | LC_ALL=C sort | uniq -c | sort -rn
```

`LC_ALL=C` matters: under a UTF-8 locale `sort` and `uniq` disagree about
adjacency and one identity is reported as up to seven separate groups.

| Repo (all public) | Commits on `main` | Personal identity | Non-canonical org email | Automation | Other |
|---|---:|---:|---:|---:|---:|
| `mind-mem`  | 1164 | 129 | 33 | 11 | 2 |
| `mind`      | 1738 |   4 |  0 |  5 | 0 |
| `mind-spec` |  126 |   4 | 40 |  0 | 1 |
| `mind-nerve`|  297 |   9 |  0 |  8 | 0 |

Totals: **146** personal-identity commits and **73** non-canonical-org-email
commits across the four public repositories.

Four distinct classes, in descending severity:

1. **Personal identity** — 147 commits. The actual leak.
2. **A third human-looking account** — 2 commits in `mind-mem`
   (`ee5b9f7`, `b840de0`), authored by a GitHub account that is neither the org
   account nor recognised automation, using a disposable-alias mail domain.
   See *Access control* below; this is the finding with the largest blast radius
   and it is not an identity-cosmetics issue.
3. **A coding tool's default identity** — `838e1c2` in `mind-spec`, authored and
   committed under a vendor `noreply@` address, reachable from `origin/main`.
   The vendor string is deliberately not reproduced here: writing it into a
   tracked file would itself trip the no-vendor-names CI gate this repository
   enforces. Resolve the SHA to see it.
4. **Non-canonical org email** (`info@star.ga` rather than `noreply@star.ga`) —
   33 commits. **Not a personal-identity leak:** it is the company contact
   address, it names the company rather than a person, and it is already
   published on the org's own profile. It is policy drift, and materially less
   severe than classes 1–3.

### Access control (supersedes the identity question in urgency)

```bash
gh api repos/star-ga/mind-mem/collaborators --jq '.[] | "\(.login) \(.role_name)"'
```

`star-ga` (admin) plus **two non-org accounts holding `write`**. One of them has
already authored two commits on `main`. No other public repo has extra
collaborators. Whether or not these are our own machine accounts, standing write
access on a public repo by accounts that do not commit under the org identity is
the path by which class 2 occurred, and it remains open. **Confirm or revoke
before spending any effort on history cosmetics.**

## Root cause

Three paths, and they need different fixes. The discriminator is GitHub's
web-flow signature: server-created commits are signed, client-created ones are not.

```bash
gh api repos/star-ga/mind-mem/commits/<sha> -q .commit.verification.reason
```

- **The merge button re-authors — the dominant path.** Of the 129 personal
  commits in `mind-mem`, **119 are merge commits, and all 119 are committed by
  `GitHub <noreply@github.com>`** — landed through the GitHub UI, which stamps
  the clicking account as author. Verified: `8431c03` carries a valid web-flow
  PGP signature. **The durable fix is on the account, not in the repo:** the org
  GitHub account's profile name is a person's name and its primary address is
  the personal one, so *every* server-side commit it creates leaks by
  construction. Renaming the profile and repointing the primary address closes
  this path for all repos at once; a repository-side hook can never reach it.
- **A clone on another machine.** The remaining commits are **unsigned**, so they
  came from a real git client. `28ffc5a` is not in this clone's reflog — it
  arrived by `fetch origin: fast-forward` — and the three most recent landed
  across two repos within four minutes, i.e. one session on one machine that is
  not this workstation. Only a CI or server-side check catches these; no local
  hook was ever installed there.
- **Correction to an earlier revision of this document.** It claimed the
  workstation global config was itself wrong (`user.email = info@star.ga`) and
  that 52 local repositories inherited a non-canonical identity. **Neither
  reproduces now.** `~/.gitconfig` reads `noreply@star.ga`, and of 179 local
  clones **176 resolve to `noreply@star.ga` and 3 to `info@star.ga`**
  (`skill-improver`, `MindLLM`, `drd.io`). Either the fix was applied or the
  measurement was wrong; either way the global-inheritance path is not the
  live cause and should not be cited as one.

## The decision

Two standing rules conflict: *never rewrite public history* versus *no personal
identity in public history*.

Scope first, because it is not uniform and it changes the answer per repo:

```bash
# earliest non-conforming commit, and how much history a rewrite would touch
git log --format='%H %ae' --reverse origin/main | grep -vE 'noreply@star\.ga|users\.noreply' | head -1
```

| Repo | Earliest non-conforming | Commits rewritten | Share of history |
|---|---|---:|---:|
| `mind`      | 2026-08-11 |  117 | **6.7%** |
| `mind-nerve`| 2026-08-06 |   58 | 19.5% |
| `mind-spec` | 2025-12-27 |  125 | 99.2% |
| `mind-mem`  | 2026-02-17 | 1164 | 100% |

**Option A — accept and fix forward (recommended).**
Leave history; prevent recurrence with the guard below and the account fix above.

- The addresses are already published: cloned, forked, and served by the commits
  API. A rewrite does not un-publish them.
- **A rewrite cannot even fully remove them here.** `mind-mem` has an external
  fork (`dhlqiang7/mind-mem`). Forks share an object network, so the old commits
  stay fetchable by SHA from the fork after a force-push. The remedy is partial
  by construction.
- The leaked address is a long-published business contact address, not a home
  address or a secret. This is a policy violation, not a security incident.
- Cost: zero.

**Option B — revert and re-land.**
Revert the offending commits' content and re-commit it under the org identity.

- **This does not work for this problem and should not be chosen.** A revert adds
  a new commit; it never removes the original. The leaking commit object stays
  reachable on `main` exactly as before, and history grows by two commits per
  leak. It buys nothing an identity leak cares about.
- Only worth considering if the *content* were the problem. It is not.

**Option C — rewrite history and force-push.**
Rewrite and force-push 1464 commits across four repos.

- Who breaks: every existing clone (hard-reset or re-clone), the external fork,
  the **2 open PRs** on `mind-mem`, and **227 tags / 215 published releases**
  whose artifacts and release notes reference commit SHAs — `mind-mem` alone has
  151 tags and 149 releases, and is published to PyPI, where release provenance
  points at SHAs that would cease to exist. Every external link to a commit 404s.
- What it buys: the addresses leave the *current* tree only. Old SHAs stay
  reachable through the fork network and through already-fetched clones.
- For `mind` alone the scope is small (6.7%, 117 commits, **0 open PRs**), so if
  the operator wants the flagship repo clean, `mind` is the one place where the
  cost is close to tractable. It still breaks 49 tags / 51 releases.

**Assessment:** Option C pays a large, partly irreversible cost for a partial
remedy of an already-published fact, and Option B pays a smaller cost for no
remedy at all. Recommend **Option A**, with the account fix and the guard made
mandatory, and with the access-control item treated as the actual priority.
This is an operator decision and is **not** executed by this document.

## The guard

- `scripts/check_author_identity.sh` — `staged` validates the identity the next
  commit would carry; `range <rev-args>` validates a set of commits.
- `scripts/pre-commit-hook.sh` — composite pre-commit hook.
- `scripts/pre-push-hook.sh` — **pre-push**, install with
  `ln -sf ../../scripts/pre-push-hook.sh .git/hooks/pre-push`. The pre-commit
  hook reads git *config* and the `GIT_AUTHOR_*` environment, so it cannot see
  `git commit --author="..."`, which sets the author directly and passes the
  hook. Push is where the real commit objects exist and where identity becomes
  public, so that is where the committed identity is checked. Fails closed if
  the checker is missing.
- **`author-identity` job in `.github/workflows/ci.yml`** — the load-bearing leg,
  and the only one that catches a commit made in a clone where no hook was
  installed. Scoped to the commits a push or PR introduces so pre-existing
  violations do not hold the gate permanently red.

Merge commits are checked on the author field only: a GitHub-created merge is
necessarily committed by `GitHub <noreply@github.com>`. Recognised automation
(dependabot) is exempt by allowlist — machine accounts leak no personal identity.

### Defects found in the first implementation, now fixed

Verified by running it, not by reading it:

- **A range that fails to resolve reported success.** `git log` was consumed
  through process substitution, which hides its exit status from `set -e`;
  `range deadbeef..cafebabe` printed `fatal: ambiguous argument` on stderr and
  still exited 0 with an all-clear. In CI, any `BASE` that no longer resolves —
  after a force-push, a rebased base, a deleted and recreated branch — turned the
  gate green while it checked nothing. The range is now resolved and its status
  checked before any commit is inspected.
- **An empty range reported "all commits conform".** Zero commits inspected, exit
  0, success message. The checker now counts what it inspected and prints that
  count, so a passing run proves it ran; `REQUIRE_NONEMPTY=1` makes an empty
  range an error where one is expected to be impossible.
- **`git commit --author=` bypassed the pre-commit hook entirely**, producing a
  commit under a personal identity while the hook reported success. Closed by
  the pre-push hook above.
- **Still open, in the CI leg:** when `BASE` is absent or all-zeros — a new
  branch's first push — the workflow falls back to `RANGE="$HEAD~1..$HEAD"` and
  therefore checks only the tip commit. A new branch pushed with twenty commits
  has nineteen of them unchecked. The pre-push hook uses
  `<sha> --not --remotes=origin` for this case, which is the correct range;
  the workflow should use the same. Left unchanged here because editing a public
  repo's CI is an operator decision, not a review artifact.

## Remaining operator actions

1. **Confirm or revoke the two non-org `write` collaborators on `mind-mem`.**
   Highest value, and unrelated to history cosmetics.
2. **Fix the GitHub account profile** — display name off a personal name, primary
   address to `noreply@star.ga`. This is what actually closes the merge-button
   path; it is an account setting, not a repo change.
3. Stop landing PRs with the merge button on public repos; land via `git push`.
4. Reconcile the name spelling (`STARGA Inc` vs `STARGA Inc.`) so config and the
   documented recipe agree; until then keep both in the allowlist.
5. Install the pre-commit **and** pre-push hooks in every clone on every machine
   that commits to a STARGA repo — hooks are not transferred by `git clone`.
6. Set the three drifted local clones to `noreply@star.ga`.
7. Adopt the `author-identity` CI job in `mind`, `mind-spec`, `mind-nerve`.
