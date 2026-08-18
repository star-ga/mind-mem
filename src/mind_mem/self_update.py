# Copyright 2026 STARGA, Inc.
"""Self-update mechanism for mind-mem installs.

``mm self-update`` checks PyPI for a newer ``mind-mem`` release and upgrades
the current install in place via the pip that owns *this* interpreter.
An opt-in, interval-gated auto-check hook (see :func:`maybe_auto_check`) can
also run a lightweight, best-effort check on every ``mm`` invocation.

This module is stdlib-only and has **zero coupling** to any recall/evidence/
scoring module in this package — it only reads package metadata, prints
status, and shells out to ``pip``/``pipx``. All human-facing output goes to
**stderr**, prefixed ``[self-update]``, so it never corrupts JSON/text a
normal ``mm`` command prints to stdout.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import os
import re
import shutil
import site
import subprocess  # nosec B404 — every invocation below uses a fixed argument list (shell=False); no user input reaches argv
import sys
import sysconfig
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

PACKAGE = "mind-mem"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE}/json"

EXIT_UP_TO_DATE = 0
EXIT_UPDATE_AVAILABLE = 10  # --check only: a newer release exists
EXIT_CANNOT_CHECK = 2  # offline / PyPI unreachable / pipx missing (graceful, no traceback)

_LOG_PREFIX = "[self-update]"

STATE_DIR = Path(os.environ.get("MIND_MEM_STATE_DIR", "") or str(Path.home() / ".mind-mem"))
STATE_FILE = STATE_DIR / "update_state.json"

DEFAULT_INTERVAL_HOURS = 24.0
_MIN_INTERVAL_HOURS = 1.0
_VALID_MODES = {"notify", "auto"}
_VALID_CHANNELS = {"stable", "pre"}


def _log(message: str) -> None:
    """Write one human-facing line to stderr with the module prefix."""
    print(f"{_LOG_PREFIX} {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Version handling (PEP 440 subset — no ``packaging`` dependency)
# ---------------------------------------------------------------------------

# deferred: full PEP 440 (epochs, local version segments) is out of scope —
# stubbed to this regex subset because `packaging` is not a dependency of
# mind-mem and every version this project has ever published (X.Y.Z, optional
# a/b/rcN pre-release, optional .postN, optional .devN) fits it. Upgrade path:
# vendor packaging.version if a future release needs epoch/local support.
_VER_RE = re.compile(
    r"^(?P<release>\d+(?:\.\d+)*)"
    r"(?:(?P<pre_letter>a|b|rc)(?P<pre_num>\d+))?"
    r"(?:\.post(?P<post>\d+))?"
    r"(?:\.dev(?P<dev>\d+))?$"
)
_PRE_RANK = {"a": 0, "b": 1, "rc": 2}

# Sort key: (release_tuple, pre_key, post_key, dev_key). ``pre_key`` and
# ``dev_key`` use +/-infinity sentinels (mirroring the real PEP 440 algorithm)
# so a bare ``X.Y.Z.devN`` — which has no explicit pre-release segment — still
# sorts *before* ``X.Y.Za1``, and a final release sorts after every
# pre-release of the same release number:
#   1.5.0.dev1 < 1.5.0a1 < 1.5.0rc1 < 1.5.0 < 1.5.0.post1
VersionKey = tuple[tuple[int, ...], tuple[float, ...], float, float]


def parse_version(version: str) -> Optional[VersionKey]:
    """Parse *version* into a sortable key, or ``None`` if unparseable."""
    match = _VER_RE.match(version.strip())
    if not match:
        return None

    release = tuple(int(part) for part in match.group("release").split("."))
    pre_letter = match.group("pre_letter")
    post = match.group("post")
    dev = match.group("dev")

    if pre_letter is not None:
        pre_key: tuple[float, ...] = (_PRE_RANK[pre_letter], int(match.group("pre_num") or 0))
    elif post is None and dev is not None:
        pre_key = (-math.inf,)  # bare ".devN" — a pre-pre-release
    else:
        pre_key = (math.inf,)  # final or post release — after every pre-release

    post_key: float = int(post) if post is not None else -math.inf
    dev_key: float = int(dev) if dev is not None else math.inf

    return (release, pre_key, post_key, dev_key)


def is_prerelease(version: str) -> bool:
    """True if *version* has an a/b/rc segment or a ``.devN`` segment."""
    match = _VER_RE.match(version.strip())
    if not match:
        return False
    return match.group("pre_letter") is not None or match.group("dev") is not None


# ---------------------------------------------------------------------------
# Install introspection
# ---------------------------------------------------------------------------


def get_installed_version() -> str:
    """Return the installed ``mind-mem`` version, or ``"0.0.0"`` if unknown."""
    try:
        return importlib.metadata.version(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        _log(f"warning: {PACKAGE!r} distribution metadata not found; assuming 0.0.0")
        return "0.0.0"


def _package_dir() -> Path:
    """Directory this module lives in (== the installed ``mind_mem`` package dir)."""
    return Path(__file__).resolve().parent


def _site_dirs() -> list[str]:
    """Every known site-packages-style directory for this interpreter."""
    dirs: list[str] = []
    get_site = getattr(site, "getsitepackages", None)
    if callable(get_site):
        try:
            dirs.extend(get_site())
        except Exception:
            pass
    get_user_site = getattr(site, "getusersitepackages", None)
    if callable(get_user_site):
        try:
            dirs.append(get_user_site())
        except Exception:
            pass
    try:
        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            dirs.append(purelib)
    except Exception:
        pass
    return [d for d in dict.fromkeys(dirs) if d]


def _direct_url_editable() -> Optional[bool]:
    """PEP 660 check: does ``direct_url.json`` mark this install editable?"""
    try:
        dist = importlib.metadata.distribution(PACKAGE)
        raw = dist.read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    dir_info = data.get("dir_info")
    if not isinstance(dir_info, dict):
        return False
    return bool(dir_info.get("editable", False))


def _editable_pth_present() -> bool:
    """Does any site dir have a ``pip install -e`` ``__editable__*.pth`` for us?"""
    patterns = ("__editable__*mind_mem*.pth", "__editable__*mind-mem*.pth")
    for site_dir in _site_dirs():
        path = Path(site_dir)
        if not path.is_dir():
            continue
        if any(path.glob(pattern) for pattern in patterns):
            return True
    return False


def _running_from_source_tree() -> bool:
    """Is the package running from a checkout, not a site-packages install?"""
    pkg_dir = _package_dir()
    site_dirs = {str(Path(d).resolve()) for d in _site_dirs()}
    if any(str(pkg_dir).startswith(d) for d in site_dirs):
        return False
    for parent in pkg_dir.parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue
        if re.search(r'name\s*=\s*"mind-mem"', text):
            return True
    return False


def is_editable_install() -> bool:
    """True for a dev/editable install (``pip install -e .`` or a bare checkout).

    Upgrading over one of these would clobber a source tree, so callers must
    refuse rather than run pip.
    """
    if _direct_url_editable():
        return True
    if _editable_pth_present():
        return True
    return _running_from_source_tree()


def is_pipx_install() -> bool:
    """True if the current interpreter lives inside a pipx-managed venv."""
    return "pipx" in Path(sys.executable).resolve().parts


# ---------------------------------------------------------------------------
# PyPI query (offline-graceful)
# ---------------------------------------------------------------------------


def fetch_latest_version(include_pre: bool = False, timeout: float = 5.0) -> Optional[str]:
    """Return the newest published ``mind-mem`` version, or ``None`` if unknown.

    Never raises — any network, JSON, or shape error is swallowed and
    reported as "could not determine" via the ``None`` return.
    """
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": f"mind-mem-self-update/{get_installed_version()}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # nosec B310 — PYPI_JSON_URL is a fixed https constant
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

    try:
        data = json.loads(raw)
        releases: dict[str, Any] = data["releases"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    candidates = [
        version
        for version, files in releases.items()
        if not _all_yanked(files) and (include_pre or not is_prerelease(version)) and parse_version(version) is not None
    ]
    return max(candidates, key=parse_version) if candidates else None  # type: ignore[arg-type]


def _all_yanked(files: Any) -> bool:
    """True if *files* is empty or every file entry is marked ``yanked``."""
    if not isinstance(files, list) or not files:
        return True
    return all(isinstance(f, dict) and f.get("yanked") for f in files)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UpdateStatus:
    """Result of a single :func:`check` call."""

    installed: str
    latest: Optional[str]
    update_available: bool
    channel: str
    editable: bool
    pipx: bool
    error: Optional[str]


def check(include_pre: bool = False, timeout: float = 5.0) -> UpdateStatus:
    """Compose install-introspection + the PyPI query into an :class:`UpdateStatus`."""
    installed = get_installed_version()
    latest = fetch_latest_version(include_pre=include_pre, timeout=timeout)

    error = None if latest is not None else "could not reach PyPI (offline or request failed)"
    update_available = False
    if latest is not None:
        installed_key, latest_key = parse_version(installed), parse_version(latest)
        update_available = latest_key is not None and (installed_key is None or latest_key > installed_key)

    return UpdateStatus(
        installed=installed,
        latest=latest,
        update_available=update_available,
        channel="pre" if include_pre else "stable",
        editable=is_editable_install(),
        pipx=is_pipx_install(),
        error=error,
    )


# ---------------------------------------------------------------------------
# Upgrade execution
# ---------------------------------------------------------------------------


def _tail(text: str, n: int = 15) -> str:
    return "\n".join(text.splitlines()[-n:])


def _run_pip_cmd(cmd: list[str], log: Callable[[str], None]) -> subprocess.CompletedProcess[str]:
    log(f"running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # nosec B603 — fixed argv built from sys.executable/PACKAGE constants; shell=False


def _perform_pipx_upgrade(log: Callable[[str], None]) -> int:
    pipx_bin = shutil.which("pipx")
    if not pipx_bin:
        log(f"pipx install detected but 'pipx' is not on PATH — run: pipx upgrade {PACKAGE}")
        return EXIT_CANNOT_CHECK
    proc = _run_pip_cmd([pipx_bin, "upgrade", PACKAGE], log)
    log(_tail((proc.stdout or "") + (proc.stderr or "")))
    return proc.returncode


def _perform_pip_upgrade(include_pre: bool, log: Callable[[str], None]) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if include_pre:
        cmd.append("--pre")
    cmd.append(PACKAGE)

    proc = _run_pip_cmd(cmd, log)
    if proc.returncode != 0 and "externally-managed-environment" in (proc.stderr or ""):
        retry_cmd = [*cmd, "--break-system-packages"]
        log("externally-managed-environment detected; retrying with --break-system-packages")
        proc = _run_pip_cmd(retry_cmd, log)

    log(_tail((proc.stdout or "") + (proc.stderr or "")))
    return proc.returncode


def perform_upgrade(include_pre: bool = False, log: Callable[[str], None] = _log) -> int:
    """Run the upgrade (pipx or pip, with a PEP 668 retry) and return its exit code."""
    if is_pipx_install():
        return _perform_pipx_upgrade(log)
    return _perform_pip_upgrade(include_pre, log)


# ---------------------------------------------------------------------------
# CLI entry (``mm self-update``)
# ---------------------------------------------------------------------------


def _package_source_path() -> str:
    return str(_package_dir())


def _report_check(status: UpdateStatus) -> int:
    if status.error is not None:
        _log(f"installed {status.installed} — could not check latest ({status.error})")
        return EXIT_CANNOT_CHECK
    _log(f"installed {status.installed} / latest {status.latest} ({status.channel})")
    return EXIT_UPDATE_AVAILABLE if status.update_available else EXIT_UP_TO_DATE


def _confirm_upgrade(status: UpdateStatus) -> bool:
    if not sys.stdin.isatty():
        _log(f"update available ({status.installed} -> {status.latest}); rerun with --yes to upgrade non-interactively")
        return False
    try:
        answer = input(f"{_LOG_PREFIX} Upgrade mind-mem {status.installed} -> {status.latest}? [y/N] ")
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def cmd_self_update(args: argparse.Namespace) -> int:
    """Entry point for ``mm self-update``."""
    include_pre = bool(getattr(args, "pre", False))
    status = check(include_pre=include_pre)

    if getattr(args, "check", False):
        return _report_check(status)

    if status.editable:
        _log(
            f"editable/dev install detected at {_package_source_path()} — refusing to upgrade "
            "over a source tree; use git pull + pip install -e ."
        )
        return 0

    if status.error is not None:
        _log(f"installed {status.installed} — could not check latest ({status.error})")
        return EXIT_CANNOT_CHECK

    if not status.update_available:
        _log(f"mind-mem {status.installed} is already the latest {status.channel} release")
        return 0

    if not getattr(args, "yes", False) and not _confirm_upgrade(status):
        _log("upgrade cancelled")
        return 0

    rc = perform_upgrade(include_pre=include_pre)
    if rc == 0:
        _log(f"upgraded mind-mem {status.installed} -> {get_installed_version()}")
    else:
        _log(f"upgrade failed (exit {rc}); see pip output above")
    return rc


# ---------------------------------------------------------------------------
# Auto-update hook + state file
# ---------------------------------------------------------------------------


def _read_state() -> dict[str, Any]:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return dict(data) if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp_path, STATE_FILE)  # atomic on POSIX and Windows


def _refresh_state(channel: str) -> None:
    """Background-worker body: fetch latest, stamp state. Invoked via ``--refresh-state``."""
    latest = fetch_latest_version(include_pre=(channel == "pre"), timeout=3.0)
    state = _read_state()
    state["last_check"] = time.time()
    state["channel"] = channel
    if latest is not None:
        state["latest_seen"] = latest
    _write_state(state)


def _normalize_auto_update_config(raw: dict[str, Any]) -> tuple[str, str, float]:
    mode = raw.get("mode", "notify")
    if mode not in _VALID_MODES:
        _log(f"unknown auto_update.mode {mode!r}; falling back to 'notify'")
        mode = "notify"

    channel = raw.get("channel", "stable")
    if channel not in _VALID_CHANNELS:
        _log(f"unknown auto_update.channel {channel!r}; falling back to 'stable'")
        channel = "stable"

    try:
        interval_hours = max(_MIN_INTERVAL_HOURS, float(raw.get("interval_hours", DEFAULT_INTERVAL_HOURS)))
    except (TypeError, ValueError):
        interval_hours = DEFAULT_INTERVAL_HOURS

    return mode, channel, interval_hours


def _maybe_notify_or_upgrade(state: dict[str, Any], mode: str, channel: str) -> None:
    """Locally compare ``latest_seen`` (from the last background refresh) vs installed. Zero network."""
    latest_seen = state.get("latest_seen")
    if not latest_seen:
        return

    installed = get_installed_version()
    installed_key, latest_key = parse_version(installed), parse_version(str(latest_seen))
    if latest_key is None or (installed_key is not None and latest_key <= installed_key):
        if "latest_seen" in state:
            _write_state({k: v for k, v in state.items() if k != "latest_seen"})
        return

    blocked = is_editable_install() or (is_pipx_install() and not shutil.which("pipx"))
    if mode == "auto" and not blocked:
        _log(f"auto-update: {PACKAGE} {installed} -> {latest_seen} (mode=auto)")
        perform_upgrade(include_pre=(channel == "pre"))
        return

    _log(f"mind-mem {latest_seen} is available (installed {installed}) — run 'mm self-update'")


def _maybe_spawn_refresh(state: dict[str, Any], channel: str, interval_hours: float) -> None:
    """If the interval has elapsed, restamp ``last_check`` and spawn a detached refresh."""
    last_check = float(state.get("last_check") or 0.0)
    if time.time() - last_check < interval_hours * 3600:
        return
    _write_state({**state, "last_check": time.time()})  # stamp first: concurrent/failed runs don't stampede
    _spawn_refresh_worker(channel)


def _spawn_refresh_worker(channel: str) -> None:
    kwargs: dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    cmd = [sys.executable, "-m", "mind_mem.self_update", "--refresh-state", channel]
    subprocess.Popen(cmd, **kwargs)  # nosec B603 — fixed argv (sys.executable + literal module name + validated channel); shell=False; detached best-effort background refresh


def maybe_auto_check(config: dict[str, Any], argv0_command: Optional[str]) -> None:
    """Best-effort, interval-gated auto-update hook. Called once per ``mm`` invocation.

    Never raises: the whole body is guarded so a network hiccup, a malformed
    config, or a missing state directory can never break a normal ``mm``
    command. Default is a no-op unless ``auto_update.enabled`` is explicitly
    ``true`` in ``mind-mem.json``.
    """
    try:
        auto_cfg = config.get("auto_update") if isinstance(config, dict) else None
        if not isinstance(auto_cfg, dict) or auto_cfg.get("enabled") is not True:
            return
        if argv0_command == "self-update" or os.environ.get("MIND_MEM_NO_AUTO_UPDATE") == "1":
            return

        mode, channel, interval_hours = _normalize_auto_update_config(auto_cfg)
        state = _read_state()
        _maybe_notify_or_upgrade(state, mode, channel)
        _maybe_spawn_refresh(state, channel, interval_hours)
    except Exception:
        return


def _run_refresh_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m mind_mem.self_update")
    parser.add_argument("--refresh-state", dest="channel", metavar="CHANNEL", default=None)
    ns = parser.parse_args(argv)
    if not ns.channel:
        return 0
    try:
        _refresh_state(ns.channel)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_run_refresh_cli(sys.argv[1:]))
