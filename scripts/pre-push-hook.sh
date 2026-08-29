#!/usr/bin/env bash
# pre-push-hook.sh — the last LOCAL gate before an identity becomes public.
#
# Why this exists in addition to the pre-commit hook: the pre-commit hook
# validates the identity that git *config* and the GIT_AUTHOR_*/GIT_COMMITTER_*
# environment would supply. It cannot see `git commit --author="..."`, which
# sets the author directly and bypasses both. Push is the moment the identity
# actually becomes public and the moment the real commit objects exist, so this
# is where the committed identity — not the configured one — can be checked.
#
# Install:
#   ln -sf ../../scripts/pre-push-hook.sh .git/hooks/pre-push
#
# stdin, one line per ref: <local ref> <local sha> <remote ref> <remote sha>
set -euo pipefail

ZERO='0000000000000000000000000000000000000000'
root="$(git rev-parse --show-toplevel)"
guard="$root/scripts/check_author_identity.sh"

# Fail CLOSED. A missing checker means the push is unverified, and an
# unverified push into public history is the exact thing being prevented.
if [ ! -x "$guard" ]; then
  echo "pre-push: $guard is missing or not executable — refusing to push unverified." >&2
  exit 1
fi

rc=0
while read -r _local_ref local_sha _remote_ref remote_sha; do
  [ -n "${local_sha:-}" ] || continue
  # Branch deletion pushes nothing new.
  [ "$local_sha" = "$ZERO" ] && continue

  if [ "$remote_sha" = "$ZERO" ]; then
    # New branch: everything it introduces that origin does not already have.
    "$guard" range "$local_sha" --not --remotes=origin || rc=1
  else
    "$guard" range "${remote_sha}..${local_sha}" || rc=1
  fi
done

if [ "$rc" -ne 0 ]; then
  echo "pre-push: refusing to push — the commits above carry a non-STARGA identity." >&2
  echo "Nothing was rewritten. Public history cannot be un-leaked, so this refuses before the push." >&2
  exit 1
fi
exit 0
