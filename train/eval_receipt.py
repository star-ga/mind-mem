"""Build a small, content-addressed receipt for the 4B evaluation gates.

The model evaluation itself remains in :mod:`eval_harness` and
:mod:`eval_holdout`.  This module only records what those runners evaluated so
that a score cannot be detached from its model, source, or data.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "mind-mem/eval-receipt@1"
_NON_MODEL_ARTIFACTS = {"eval_report.json", "eval_holdout_report.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[tuple[str, Path]]:
    if root.is_file():
        yield root.name, root
        return
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.name not in _NON_MODEL_ARTIFACTS and path.is_file() and not path.is_symlink():
            yield str(path.relative_to(root)), path


def file_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    """Return deterministic hashes and byte sizes for a file or directory."""
    return {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in _files(root)
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=repo_root, timeout=5
        ).returncode != 0
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _versions(names: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def build_receipt(
    *,
    repo_root: Path,
    model_root: Path,
    dataset_root: Path,
    source_paths: Iterable[Path],
    probe_sets: dict[str, Any],
    command: str,
) -> dict[str, Any]:
    """Build a receipt without loading a model or contacting a service."""
    model_root = model_root.resolve()
    dataset_root = dataset_root.resolve()
    source = {}
    for path in sorted({Path(path).resolve() for path in source_paths}):
        if path.is_file():
            source[str(path.relative_to(repo_root.resolve()))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    dataset = {
        "path": str(dataset_root),
        "present": dataset_root.is_file(),
        "sha256": sha256_file(dataset_root) if dataset_root.is_file() else None,
        "bytes": dataset_root.stat().st_size if dataset_root.is_file() else None,
    }
    probes = {name: _digest(value) for name, value in sorted(probe_sets.items())}
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "command": command,
        "source": {"repo_root": str(repo_root.resolve()), "git": _git(repo_root), "files": source},
        "model": {"path": str(model_root), "files": file_manifest(model_root)},
        "dataset": dataset,
        "probes": probes,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _versions(("torch", "transformers", "peft", "bitsandbytes")),
        },
    }
    receipt["complete"] = bool(
        receipt["source"]["git"]["commit"]
        and source
        and receipt["model"]["files"]
        and dataset["present"]
        and all(probes.values())
    )
    receipt["receipt_sha256"] = _digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    return receipt


def require_complete(receipt: dict[str, Any]) -> None:
    """Refuse to treat an unbound report as a shippable evaluation."""
    if not receipt.get("complete"):
        raise ValueError("evaluation receipt is incomplete: model/source/dataset binding is required")


def receipt_is_valid(receipt: dict[str, Any]) -> bool:
    """Check the receipt's self-digest before it is used as a release gate."""
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str):
        return False
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return claimed == _digest(body)


def bindings_match(receipt: dict[str, Any]) -> bool:
    """Re-hash the recorded paths so a post-eval replacement cannot publish."""
    try:
        source = receipt["source"]
        repo_root = Path(source["repo_root"]).resolve()
        if _git(repo_root) != source["git"]:
            return False
        for rel, meta in source["files"].items():
            path = repo_root / rel
            if not path.is_file() or sha256_file(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
                return False
        dataset = receipt["dataset"]
        dataset_path = Path(dataset["path"])
        if not dataset.get("present") or not dataset_path.is_file():
            return False
        if sha256_file(dataset_path) != dataset["sha256"] or dataset_path.stat().st_size != dataset["bytes"]:
            return False
        model_path = Path(receipt["model"]["path"])
        return file_manifest(model_path) == receipt["model"]["files"]
    except (KeyError, OSError, TypeError, ValueError):
        return False
