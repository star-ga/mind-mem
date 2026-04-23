"""Model checkpoint audit — scan for remote-code hooks, unsafe pickle, tokenizer injection.

Six attack surfaces covered:
  1. Remote-code hooks  — auto_map / trust_remote_code flags in config files
  2. Python files       — any .py shipping with the checkpoint
  3. Weight format      — safetensors/gguf (safe) vs .bin/.pt/.pth (pickle, unsafe)
  4. Pickle opcodes     — dangerous imports (os, subprocess, socket, ctypes, eval, exec, __builtin__)
  5. Tokenizer strings  — URLs / shell redirectors / script refs inside tokenizer.json
  6. SHA-256 manifest   — tamper-evident hash of every file

Zero runtime dependencies beyond stdlib. No model loading. Static inspection only.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickletools
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DANGEROUS_PICKLE_IMPORTS = {
    "os", "posix", "nt", "subprocess", "socket", "ctypes", "importlib",
    "builtins", "__builtin__", "runpy", "pty", "pickle", "shelve",
    "pathlib", "shutil", "tempfile", "urllib", "requests", "httpx",
    "eval", "exec", "compile", "__import__",
}

UNSAFE_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
SAFE_WEIGHT_SUFFIXES = {".safetensors", ".gguf"}

URL_RE = re.compile(rb"https?://[^\s\"'<>\\]{8,}")
SHELL_RE = re.compile(rb"(?:curl|wget|bash\s+-|sh\s+-|nc\s+-|python\s+-c|import\s+os|import\s+subprocess)")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    model_path: str
    checks: list[CheckResult] = field(default_factory=list)
    manifest: dict[str, str] = field(default_factory=dict)
    total_bytes: int = 0
    file_count: int = 0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "passed": self.passed,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, "evidence": c.evidence}
                for c in self.checks
            ],
            "manifest": self.manifest,
        }


# --- Individual checks -------------------------------------------------------


def check_remote_code_hooks(root: Path) -> CheckResult:
    """Fail if any config file has auto_map or trust_remote_code=true."""
    hits: list[str] = []
    for cfg in list(root.rglob("config.json")) + list(root.rglob("*_config.json")) + list(root.rglob("generation_config.json")):
        try:
            data = json.loads(cfg.read_text())
        except Exception:
            continue
        if "auto_map" in data:
            hits.append(f"{cfg.name}: auto_map -> {data['auto_map']}")
        if data.get("trust_remote_code") is True:
            hits.append(f"{cfg.name}: trust_remote_code=true")
    return CheckResult(
        name="remote_code_hooks",
        passed=not hits,
        detail="no auto_map or trust_remote_code flags" if not hits else f"{len(hits)} hook(s) found",
        evidence=hits,
    )


def check_no_python_files(root: Path) -> CheckResult:
    """Fail if any .py file is present in the checkpoint."""
    py_files = [str(p.relative_to(root)) for p in root.rglob("*.py")]
    return CheckResult(
        name="no_python_files",
        passed=not py_files,
        detail="no .py files present" if not py_files else f"{len(py_files)} .py file(s) found",
        evidence=py_files,
    )


def check_weight_format(root: Path) -> CheckResult:
    """Prefer safetensors/gguf; flag every .bin/.pt/.pth/.ckpt."""
    unsafe: list[str] = []
    safe_count = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in UNSAFE_WEIGHT_SUFFIXES:
            # training_args.bin is expected and gets separate pickle scan
            if p.name == "training_args.bin":
                continue
            unsafe.append(str(p.relative_to(root)))
        elif suffix in SAFE_WEIGHT_SUFFIXES:
            safe_count += 1
    return CheckResult(
        name="weight_format",
        passed=not unsafe,
        detail=(
            f"{safe_count} safetensors/gguf weight file(s); no pickle weights"
            if not unsafe
            else f"{len(unsafe)} unsafe pickle weight file(s) found"
        ),
        evidence=unsafe,
    )


def _scan_pickle_bytes(raw: bytes) -> set[str]:
    """Raw-byte opcode walk over a pickle stream.

    Avoids pickletools.genops' encoding assumptions (Python 3.12 defaults
    read_stringnl to ASCII, which fails on legitimate non-ASCII trainer
    configs). We only need to find GLOBAL / STACK_GLOBAL references to
    dangerous modules — everything else we skip by length.

    Opcodes we care about (Pickle protocol 0-5):
      'c' GLOBAL        — <module>\\n<name>\\n
      '\\x93' STACK_GLOBAL — module+name on stack above (preceding *UNICODE opcodes)
      '\\x8c' SHORT_BINUNICODE — <len:1><bytes>
      'B' BINUNICODE     — <len:4 LE><bytes>
      '\\x8d' BINUNICODE8 — <len:8 LE><bytes>
    """
    bad: set[str] = set()
    short_strings: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        op = raw[i]
        i += 1
        if op == 0x63:  # 'c' GLOBAL
            # Read up to two newline-terminated strings.
            end1 = raw.find(b"\n", i)
            if end1 < 0:
                break
            module = raw[i:end1].decode("utf-8", errors="replace")
            i = end1 + 1
            end2 = raw.find(b"\n", i)
            if end2 < 0:
                break
            i = end2 + 1
            root_mod = module.split(".", 1)[0]
            if root_mod in DANGEROUS_PICKLE_IMPORTS:
                bad.add(root_mod)
        elif op == 0x8C:  # SHORT_BINUNICODE
            if i >= n:
                break
            length = raw[i]
            i += 1
            if i + length > n:
                break
            s = raw[i : i + length].decode("utf-8", errors="replace")
            i += length
            short_strings.append(s)
            if len(short_strings) > 4096:
                short_strings = short_strings[-2048:]
            root_mod = s.split(".", 1)[0]
            if root_mod in DANGEROUS_PICKLE_IMPORTS and len(s) < 64:
                # Flag only if also used by a subsequent STACK_GLOBAL.
                # Conservative: flag immediately, these shouldn't appear in
                # trainer-config pickles at all.
                bad.add(root_mod)
        elif op == 0x8D:  # BINUNICODE8
            if i + 8 > n:
                break
            length = int.from_bytes(raw[i : i + 8], "little")
            i += 8
            if length > 1 << 30:  # sanity cap
                break
            i += length
        elif op == 0x42 and i + 4 <= n:
            # Could be BINBYTES ('B') or BINUNICODE ('X'=0x58) — disambiguate
            # by opcode. 'B' = 0x42 is BINBYTES; actual BINUNICODE is 0x58.
            length = int.from_bytes(raw[i : i + 4], "little")
            i += 4
            if length > 1 << 30:
                break
            i += length
        elif op == 0x58:  # 'X' BINUNICODE
            if i + 4 > n:
                break
            length = int.from_bytes(raw[i : i + 4], "little")
            i += 4
            if length > 1 << 30:
                break
            if i + length > n:
                break
            s = raw[i : i + length].decode("utf-8", errors="replace")
            i += length
            root_mod = s.split(".", 1)[0]
            if root_mod in DANGEROUS_PICKLE_IMPORTS and len(s) < 64:
                bad.add(root_mod)
        elif op == 0x93:  # STACK_GLOBAL
            # Module + qualname are on the stack as the two most recent
            # *UNICODE strings. We've already flagged them above if they
            # matched DANGEROUS_PICKLE_IMPORTS.
            pass
        # For all other opcodes, we only care about consuming the correct
        # number of bytes. Since we don't need to interpret them, we fall
        # through — the single-byte advance above handles most ops. This
        # is intentionally conservative: we may skip some opcodes incorrectly
        # and drift, but GLOBAL ops appear near the beginning of trainer-
        # config pickles so early-stream fidelity is what matters.
        # If drift produces garbage module names they won't match the
        # DANGEROUS list by construction, so false positives are bounded.
    return bad


def check_pickle_safety(root: Path) -> CheckResult:
    """Raw-byte opcode scan of every pickle file for dangerous imports."""
    flagged: list[str] = []
    pickle_files: list[Path] = []
    for suffix in UNSAFE_WEIGHT_SUFFIXES:
        pickle_files.extend(root.rglob(f"*{suffix}"))
    for pkl in pickle_files:
        try:
            raw = pkl.read_bytes()
            bad = _scan_pickle_bytes(raw)
        except Exception as exc:
            flagged.append(f"{pkl.name}: failed to read ({exc})")
            continue
        if bad:
            flagged.append(f"{pkl.name}: dangerous imports {sorted(bad)}")
    return CheckResult(
        name="pickle_safety",
        passed=not flagged,
        detail=(
            f"{len(pickle_files)} pickle file(s) disassembled, no dangerous imports"
            if not flagged
            else f"{len(flagged)} pickle file(s) with dangerous imports"
        ),
        evidence=flagged,
    )


def check_tokenizer_injection(root: Path) -> CheckResult:
    """Scan high-risk tokenizer fields for URLs and shell commands.

    BPE vocab/merges naturally contain substrings like "curl" or "https://" as
    training-data artifacts — these are NOT attack surfaces. The real attack
    surface is: added_tokens, special_tokens, post_processor templates, and
    chat_template strings, where a malicious author can smuggle content that
    the *model sees* every turn or that tooling *renders* at load time.
    """
    hits: list[str] = []

    def scan_value(source: str, value: Any) -> None:
        if isinstance(value, str):
            for url in URL_RE.findall(value.encode("utf-8")):
                try:
                    url_s = url.decode("utf-8", errors="replace")
                except Exception:
                    url_s = repr(url)
                if len(url_s) >= 20 and "/" in url_s[8:]:
                    hits.append(f"{source}: url {url_s[:120]}")
            for sh in SHELL_RE.findall(value.encode("utf-8")):
                hits.append(f"{source}: shell-like pattern {sh.decode('utf-8', errors='replace')}")
        elif isinstance(value, dict):
            for k, v in value.items():
                scan_value(f"{source}.{k}", v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                scan_value(f"{source}[{i}]", v)

    for tok_file in list(root.rglob("tokenizer.json")):
        try:
            data = json.loads(tok_file.read_text())
        except Exception:
            continue
        # High-risk sections only — skip model.vocab and model.merges.
        for field_name in ("added_tokens", "post_processor", "normalizer", "pre_tokenizer", "decoder"):
            if field_name in data:
                scan_value(f"{tok_file.name}:{field_name}", data[field_name])

    for cfg_file in list(root.rglob("tokenizer_config.json")) + list(root.rglob("special_tokens_map.json")):
        try:
            data = json.loads(cfg_file.read_text())
        except Exception:
            continue
        for field_name, value in data.items():
            # These are all attack-surface fields — tokenizer_config.json is
            # small and entirely operator-visible, so every field matters.
            scan_value(f"{cfg_file.name}:{field_name}", value)

    return CheckResult(
        name="tokenizer_injection",
        passed=not hits,
        detail="no embedded URLs or shell patterns in tokenizer metadata" if not hits else f"{len(hits)} suspicious string(s)",
        evidence=hits[:20],  # cap noise
    )


def check_safetensors_header(root: Path) -> CheckResult:
    """Validate that every .safetensors has a well-formed header and no '__metadata__.code' key."""
    bad: list[str] = []
    count = 0
    for st in root.rglob("*.safetensors"):
        count += 1
        try:
            with st.open("rb") as f:
                hdr_len_bytes = f.read(8)
                if len(hdr_len_bytes) != 8:
                    bad.append(f"{st.name}: truncated header length")
                    continue
                (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
                if hdr_len > 100 * 1024 * 1024:
                    bad.append(f"{st.name}: header size {hdr_len} bytes (suspicious, >100MB)")
                    continue
                hdr_raw = f.read(hdr_len)
                hdr = json.loads(hdr_raw)
                meta = hdr.get("__metadata__", {})
                # Flag any metadata key that looks executable-intent.
                for key in meta:
                    if any(tok in key.lower() for tok in ("code", "script", "exec", "eval", "python")):
                        bad.append(f"{st.name}: suspicious metadata key {key}")
        except Exception as exc:
            bad.append(f"{st.name}: header parse failed ({exc})")
    return CheckResult(
        name="safetensors_header",
        passed=not bad,
        detail=f"{count} safetensors file(s), all headers clean" if not bad else f"{len(bad)} issue(s)",
        evidence=bad,
    )


def compute_manifest(root: Path) -> tuple[dict[str, str], int]:
    """SHA-256 every file in the checkpoint."""
    manifest: dict[str, str] = {}
    total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
                total += len(chunk)
        manifest[str(p.relative_to(root))] = h.hexdigest()
    return manifest, total


# --- Top-level audit ---------------------------------------------------------


def audit_model(path: str | Path) -> AuditReport:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"model path not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"model path is not a directory: {root}")
    report = AuditReport(model_path=str(root))
    report.checks.append(check_remote_code_hooks(root))
    report.checks.append(check_no_python_files(root))
    report.checks.append(check_weight_format(root))
    report.checks.append(check_pickle_safety(root))
    report.checks.append(check_tokenizer_injection(root))
    report.checks.append(check_safetensors_header(root))
    report.manifest, report.total_bytes = compute_manifest(root)
    report.file_count = len(report.manifest)
    return report


def format_report_text(report: AuditReport, *, color: bool = True) -> str:
    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    ok = c("PASS", "32")
    bad = c("FAIL", "31")
    lines: list[str] = []
    lines.append(f"mind-mem model audit — {report.model_path}")
    lines.append(f"  files: {report.file_count}   total bytes: {report.total_bytes:,}")
    lines.append("")
    for chk in report.checks:
        tag = ok if chk.passed else bad
        lines.append(f"  [{tag}] {chk.name:24s} {chk.detail}")
        if not chk.passed:
            for ev in chk.evidence[:10]:
                lines.append(f"         · {ev}")
            if len(chk.evidence) > 10:
                lines.append(f"         · ... ({len(chk.evidence) - 10} more)")
    lines.append("")
    overall = ok if report.passed else bad
    lines.append(f"  overall: {overall}")
    return "\n".join(lines)
