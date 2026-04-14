"""Push the retrained adapter + model card to star-ga/mind-mem-4b.

Requires a HuggingFace token with **write** scope — the default token
cached in ~/.cache/huggingface/token is typically read-only.  Pass one
via ``--token hf_...`` or ``HF_TOKEN=... python3 upload_to_hf.py``.

Files uploaded:
    adapter_config.json
    adapter_model.safetensors
    tokenizer_config.json
    tokenizer.json
    chat_template.jinja     (if present)
    README.md               (the model card)
    mind-mem-4b-Q4_K_M.gguf (if the GGUF export step succeeded)

Existing files in the repo are overwritten; the prior v2.8.x adapter
lives in git history on HF.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_commit, CommitOperationAdd

REPO_ID = "star-ga/mind-mem-4b"
OUT_DIR = Path(
    os.environ.get("MM_TRAIN_ROOT", "/data/checkpoints/mm-workspace/train-output")
)
ADAPTER_DIR = OUT_DIR / "adapter"


def _discover_upload_paths() -> list[tuple[Path, str]]:
    """Return (local_path, path_in_repo) for every file we want to push."""
    uploads: list[tuple[Path, str]] = []

    # Adapter artifacts
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        p = ADAPTER_DIR / name
        if p.is_file():
            uploads.append((p, name))

    # Model card
    card = OUT_DIR / "README.md"
    if card.is_file():
        uploads.append((card, "README.md"))

    # Optional GGUF build
    gguf = OUT_DIR / "mind-mem-4b-Q4_K_M.gguf"
    if gguf.is_file():
        uploads.append((gguf, gguf.name))

    return uploads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commit-message", default="Retrain mind-mem-4b on v2.9.0 corpus (393 examples)")
    args = parser.parse_args()

    uploads = _discover_upload_paths()
    if not uploads:
        sys.exit(f"no files to upload — {ADAPTER_DIR} is empty. Train first.")

    print("Upload plan:")
    for local, remote in uploads:
        print(f"  {local}  →  {REPO_ID}:{remote}  ({local.stat().st_size} bytes)")
    if args.dry_run:
        return

    if not args.token:
        sys.exit(
            "no HF token provided. Pass --token hf_... or set HF_TOKEN env. "
            "Token must have 'write' scope for star-ga/mind-mem-4b."
        )

    api = HfApi(token=args.token)
    # Verify write permission before we start uploading GB of weights
    try:
        who = api.whoami()
        role = who.get("auth", {}).get("accessToken", {}).get("role", "?")
        if role != "write":
            sys.exit(
                f"token role is {role!r} — need 'write' for {REPO_ID}. "
                "Generate a new token at https://huggingface.co/settings/tokens"
            )
    except Exception as exc:
        sys.exit(f"token check failed: {exc}")

    ops = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for local, remote in uploads
    ]
    result = create_commit(
        repo_id=REPO_ID,
        operations=ops,
        commit_message=args.commit_message,
        token=args.token,
    )
    print(f"\ncommit: {result.commit_url}")
    print(f"files:  https://huggingface.co/{REPO_ID}/tree/main")


if __name__ == "__main__":
    main()
