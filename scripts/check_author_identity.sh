#!/usr/bin/env bash
# check_author_identity.sh — enforce the single-author identity rule.
#
# Policy: this repository is published under one identity. Every commit must be
# BOTH authored and committed by:
#
#     STARGA Inc. <noreply@star.ga>
#
# Any other identity — a personal name/email, a machine default, a tool default,
# a bot — is a leak into public history, and public history cannot be un-leaked.
#
# Two call modes:
#
#   staged                 Validate the identity that the NEXT commit would carry
#                          (pre-commit hook use). Reads the effective git config
#                          plus GIT_AUTHOR_*/GIT_COMMITTER_* env overrides.
#
#   range <base>..<head>   Validate every commit in a range (CI use). This is the
#                          leg that catches a commit made on another machine,
#                          where no local hook ever ran.
#
# Exit 0 = conforming. Exit 1 = violation (with the offending identity printed).

set -euo pipefail

readonly WANT_EMAIL="noreply@star.ga"

# Automation that GitHub itself authors and that we cannot re-author without
# giving up the automation. These are machine accounts, not people: they leak no
# personal identity, which is the actual thing this guard protects. Anything not
# on this list — including any human account — is a violation.
readonly BOT_AUTHORS=(
  "49699333+dependabot[bot]@users.noreply.github.com"
)
# Historically two spellings of the name have been used ("STARGA Inc" and
# "STARGA Inc."). The email is the load-bearing field — it is what identifies a
# person — so the name is checked against an allowlist and the email exactly.
readonly ALLOWED_NAMES=("STARGA Inc." "STARGA Inc")

die() { printf '%s\n' "$@" >&2; exit 1; }

is_bot() {
  local e="$1" b
  for b in "${BOT_AUTHORS[@]}"; do [ "$e" = "$b" ] && return 0; done
  return 1
}

name_ok() {
  local n="$1" a
  for a in "${ALLOWED_NAMES[@]}"; do [ "$n" = "$a" ] && return 0; done
  return 1
}

# Emits nothing on success; a description of the problem on failure.
check_identity() {
  local role="$1" name="$2" email="$3" ctx="$4" bad=0
  if [ "$email" != "$WANT_EMAIL" ]; then
    printf '  %s%s email is %s (must be %s)\n' "$ctx" "$role" "${email:-<empty>}" "$WANT_EMAIL" >&2
    bad=1
  fi
  if ! name_ok "$name"; then
    printf '  %s%s name is %s (must be %s)\n' "$ctx" "$role" "${name:-<empty>}" "${ALLOWED_NAMES[0]}" >&2
    bad=1
  fi
  return "$bad"
}

remediation() {
  cat >&2 <<'MSG'

Nothing was rewritten — this guard only refuses, it never edits history.

Commit with the identity stated explicitly:

    git -c user.name="STARGA Inc." -c user.email="noreply@star.ga" commit -m "..."

Or set it once for this clone (recommended on every machine that touches it):

    git config user.name  "STARGA Inc."
    git config user.email "noreply@star.ga"

Then verify:

    git log -1 --format='%an <%ae> | %cn <%ce>'
MSG
}

mode="${1:-staged}"

case "$mode" in
  staged)
    # Env overrides win over config, exactly as git resolves them. Note the
    # ${VAR-default} form (not ${VAR:-default}): git treats an env var that is
    # SET BUT EMPTY as an empty ident and refuses the commit, so an empty value
    # must reach the check rather than silently falling back to config.
    a_name="${GIT_AUTHOR_NAME-$(git config user.name || true)}"
    a_email="${GIT_AUTHOR_EMAIL-$(git config user.email || true)}"
    c_name="${GIT_COMMITTER_NAME-$(git config user.name || true)}"
    c_email="${GIT_COMMITTER_EMAIL-$(git config user.email || true)}"

    # $(...) strips trailing newlines, so join the two captures explicitly
    # rather than concatenating them into one run-on line.
    a_out="$(check_identity "author"    "$a_name" "$a_email" "" 2>&1 >/dev/null || true)"
    c_out="$(check_identity "committer" "$c_name" "$c_email" "" 2>&1 >/dev/null || true)"
    problems="$(printf '%s\n%s' "$a_out" "$c_out" | grep -v '^$' || true)"

    if [ -n "${problems//[[:space:]]/}" ]; then
      echo "identity guard: refusing this commit — it would enter history under a non-STARGA identity." >&2
      printf '%s\n' "$problems" >&2
      remediation
      exit 1
    fi
    exit 0
    ;;

  range)
    range="${2:-}"
    [ -n "$range" ] || die "usage: $0 range <base>..<head>"

    rc=0
    # Merges are INCLUDED, deliberately. In this repo's own history 119 of the
    # 129 personal-identity commits on main are merge commits created by the
    # GitHub merge button, which stamps the clicking account as the AUTHOR —
    # a `--no-merges` scan would have missed the dominant leak path entirely.
    #
    # The committer of such a merge is necessarily `GitHub <noreply@github.com>`
    # and cannot be otherwise, so for merge commits the COMMITTER field is not
    # checked; the AUTHOR field — the one that carries the personal identity —
    # still is. The real fix for that path is to land via `git push` rather than
    # the merge button; this check makes a regression visible either way.
    while IFS=$'\x01' read -r sha parents a_name a_email c_name c_email; do
      [ -n "$sha" ] || continue
      is_merge=0
      case "$parents" in *" "*) is_merge=1 ;; esac

      # Recognised automation accounts are exempt (see BOT_AUTHORS).
      if is_bot "$a_email"; then continue; fi
      # Both checks always run — a `||` here would short-circuit and hide a
      # committer violation behind an author violation on the same commit.
      check_identity "author"    "$a_name" "$a_email" "$sha " || rc=1
      if [ "$is_merge" -eq 0 ]; then
        check_identity "committer" "$c_name" "$c_email" "$sha " || rc=1
      fi
    done < <(git log --format="%H%x01%P%x01%an%x01%ae%x01%cn%x01%ce" "$range")

    if [ "$rc" -ne 0 ]; then
      printf '\nidentity guard: the commits above carry a non-STARGA identity.\n' >&2
      remediation
      exit 1
    fi
    echo "identity guard: all commits in $range are STARGA Inc. <$WANT_EMAIL>"
    exit 0
    ;;

  *)
    die "usage: $0 [staged | range <base>..<head>]"
    ;;
esac
