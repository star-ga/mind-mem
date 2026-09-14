# Copyright 2026 STARGA, Inc.
"""The pluggable detector chain — registration is structural, not polite.

A redaction layer whose detectors are wired up by hand has one failure
mode that no test of the detectors themselves can see: a detector that
was written, reviewed, and never added to the list. It scans nothing,
reports nothing, and the absence of findings reads exactly like a clean
document. So registration here is not a call an author has to remember —
it is a property of *being* a concrete detector:

    class MyDetector(RegexDetector):
        name = "my_thing"
        ...

is in :func:`registered_detectors` the moment the class statement
finishes, because :class:`_DetectorMeta` registers every concrete
subclass at class-creation time. There is no ``@register`` decorator to
forget and no list to edit. A concrete detector without a ``name``
refuses to be created at all, and two detectors claiming one name refuse
the second — the registry cannot silently hold one of a pair.

Abstract bases are skipped, and *abstract* means the language's answer
(``__abstractmethods__`` non-empty after :class:`abc.ABCMeta` has run),
not a class attribute an author sets. That distinction is why the hook
is a metaclass rather than ``__init_subclass__``: inside
``__init_subclass__`` the abstract-method set has not been computed yet
and still resolves to the *parent's*, so an abstract intermediate would
look concrete and a concrete leaf could look abstract.

**Findings carry no copy of what they matched, and no hash of it.** A
finding is ``(detector, category, start, end)`` — enough to audit *where*
and *what kind*, never enough to recover the value. The hash was the
tempting middle ground and it is not safe: a SHA-256 of an email address
or a 16-digit card number is a dictionary away from the plaintext, so a
ledger full of "commitments" would be a ledger full of recoverable
secrets. Provenance for the redaction as a whole is carried by the
digests of the whole document, before and after — see
:mod:`mind_mem.compliance.audit`.

No clock, no IO, no randomness: scanning is a pure function of the text
and the registry, which is what lets the export bundle be byte-identical
across runs.

An explicitly configured external detector is loaded through
:func:`load_external_detector`. Its class creation is kept out of the
process-wide in-tree registry; absent a plugin reference, no external module
is imported and the shipped chain is unchanged.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import abc
import contextvars
import importlib
import inspect
import re
from dataclasses import dataclass
from typing import ClassVar, Iterable, Sequence

__all__ = [
    "CATEGORY_PII",
    "CATEGORY_SECRET",
    "DetectorSpecError",
    "DuplicateDetectorError",
    "Finding",
    "Detector",
    "RegexDetector",
    "detector_names",
    "get_detector",
    "registered_detectors",
    "resolve_detectors",
    "load_external_detector",
    "scan_text",
]

#: The two classes of thing a detector may claim to have found. A
#: category is a routing hint for an operator ("which policy pack is
#: this?"), never an authorisation decision, so an unknown one is a
#: spec error rather than a silent third class.
CATEGORY_PII = "pii"
CATEGORY_SECRET = "secret"
_CATEGORIES = frozenset({CATEGORY_PII, CATEGORY_SECRET})


class DetectorSpecError(TypeError):
    """A concrete detector class is missing its name or category."""


class DuplicateDetectorError(ValueError):
    """Two distinct detector classes claim the same registry name."""


@dataclass(frozen=True, order=True)
class Finding:
    """One match: where it was, and what kind of thing it was.

    Ordered so a list of findings has one canonical sort — ``(start,
    end, detector)`` — which is what makes a redacted document a
    function of its input rather than of dict iteration order.
    """


    start: int
    end: int
    detector: str
    category: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        """The record shape written to bundles and audit payloads.

        Deliberately without the matched text and without a digest of
        it; see the module docstring.
        """
        return {
            "detector": self.detector,
            "category": self.category,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


# Importing a configured plugin must not make it an implicit member of the
# process-wide in-tree registry. A ContextVar keeps that rule true even when
# a caller resolves policies concurrently in different contexts.
_LOADING_EXTERNAL = contextvars.ContextVar("mind_mem_loading_external_detector", default=False)


#: name -> concrete detector class. Populated by :class:`_DetectorMeta`
#: at class-creation time; read only through the accessors below, which
#: sort by name so every caller sees one order.
_REGISTRY: dict[str, type["Detector"]] = {}


def _register(cls: type["Detector"]) -> None:
    _validate_detector_class(cls)
    name = cls.name
    existing = _REGISTRY.get(name)
    if existing is not None:
        same_class = existing.__module__ == cls.__module__ and existing.__qualname__ == cls.__qualname__
        if not same_class:
            raise DuplicateDetectorError(f"detector name {name!r} is claimed by both {existing.__qualname__} and {cls.__qualname__}")
    _REGISTRY[name] = cls


def _validate_detector_class(cls: type["Detector"]) -> None:
    """Validate the narrow class contract shared by built-ins and plugins."""
    name = getattr(cls, "name", "")
    if not isinstance(name, str) or not name:
        raise DetectorSpecError(
            f"{cls.__qualname__} is a concrete detector with no 'name'; a nameless detector cannot be addressed or audited"
        )
    category = getattr(cls, "category", "")
    if category not in _CATEGORIES:
        raise DetectorSpecError(f"{cls.__qualname__} declares category {category!r}; expected one of {sorted(_CATEGORIES)}")


class _DetectorMeta(abc.ABCMeta):
    """Registers every concrete detector as it is created.

    Runs *after* :class:`abc.ABCMeta` has computed ``__abstractmethods__``,
    so "is this abstract?" is answered by the language rather than by a
    flag the author sets — an author cannot opt a working detector out of
    the registry, and an abstract base cannot fall into it.
    """

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, object], **kwargs: object) -> "_DetectorMeta":
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if bases and not cls.__abstractmethods__ and not _LOADING_EXTERNAL.get():
            _register(cls)  # type: ignore[arg-type]
        return cls


class Detector(metaclass=_DetectorMeta):
    """One thing worth refusing to write down.

    Subclass, give it a ``name`` and a ``category``, implement
    :meth:`scan`. The class is registered by existing.
    """

    #: Registry key. Stable: it appears in audit reasons and in export
    #: envelopes, so renaming one is a corpus-visible change.
    name: ClassVar[str] = ""

    #: :data:`CATEGORY_PII` or :data:`CATEGORY_SECRET`.
    category: ClassVar[str] = ""

    @abc.abstractmethod
    def scan(self, text: str) -> list[Finding]:
        """Every match in *text*, in ``(start, end)`` order, no overlaps within one detector."""


class RegexDetector(Detector):
    """A detector defined by one compiled pattern plus an optional check.

    :attr:`pattern` is abstract, which keeps this class abstract and
    therefore out of the registry; a subclass satisfies it with a plain
    class attribute. :meth:`validate` is the hook for the matches a
    regex can shape but not decide — a 16-digit run is only a card
    number if it passes Luhn.
    """

    @property
    @abc.abstractmethod
    def pattern(self) -> re.Pattern[str]:
        """The compiled pattern this detector matches."""

    def validate(self, matched: str) -> bool:
        """Second-stage check on a regex hit. Default: accept."""
        return True

    def scan(self, text: str) -> list[Finding]:
        out: list[Finding] = []
        for m in self.pattern.finditer(text):
            if m.start() == m.end():
                continue
            if not self.validate(m.group(0)):
                continue
            out.append(Finding(start=m.start(), end=m.end(), detector=self.name, category=self.category))
        return out


# ---------------------------------------------------------------------------
# The shipped pack
#
# `hook_installer` keeps its own private copy of a few of these shapes for
# scrubbing agent transcripts on disk. That is a different job on a
# different path and it is left alone here.
# deferred: hook_installer's transcript scrubber could resolve its patterns
# from this registry instead of its own tuple — upgrade path: replace
# hook_installer._SECRET_PATTERNS with resolve_detectors(("secret",)) once a
# caller needs the two sets to agree.
# ---------------------------------------------------------------------------


class EmailDetector(RegexDetector):
    name = "email"
    category = CATEGORY_PII
    pattern = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


class AwsAccessKeyDetector(RegexDetector):
    name = "aws_access_key_id"
    category = CATEGORY_SECRET
    pattern = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")


class GithubTokenDetector(RegexDetector):
    name = "github_token"
    category = CATEGORY_SECRET
    pattern = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")


class SlackTokenDetector(RegexDetector):
    name = "slack_token"
    category = CATEGORY_SECRET
    pattern = re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")


class GoogleApiKeyDetector(RegexDetector):
    name = "google_api_key"
    category = CATEGORY_SECRET
    pattern = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")


class SecretKeyPrefixDetector(RegexDetector):
    """The ``sk-``/``pypi-`` family: vendor keys that announce themselves."""

    name = "secret_key_prefix"
    category = CATEGORY_SECRET
    pattern = re.compile(r"\b(?:sk|pypi)-[A-Za-z0-9_\-]{20,}")


class PrivateKeyBlockDetector(RegexDetector):
    name = "private_key_block"
    category = CATEGORY_SECRET
    pattern = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")


class CreditCardDetector(RegexDetector):
    """13-19 digit runs that pass Luhn.

    Without the check this fires on order numbers, build ids and hashes;
    a redactor that cries wolf gets turned off, which is a governance
    failure with extra steps.
    """

    name = "credit_card"
    category = CATEGORY_PII
    pattern = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")

    def validate(self, matched: str) -> bool:
        digits = [int(ch) for ch in matched if ch.isdigit()]
        if not 13 <= len(digits) <= 19:
            return False
        total = 0
        for index, digit in enumerate(reversed(digits)):
            if index % 2:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def detector_names() -> tuple[str, ...]:
    """Every registered name, sorted."""
    return tuple(sorted(_REGISTRY))


def get_detector(name: str) -> "Detector":
    """One detector instance by name.

    Raises :class:`KeyError` naming the known set, so a typo in a policy
    is a refusal rather than a silently smaller detector chain.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; registered: {', '.join(detector_names()) or '(none)'}") from None
    return cls()


def registered_detectors() -> tuple["Detector", ...]:
    """One instance of every registered detector, in name order."""
    return tuple(_REGISTRY[name]() for name in detector_names())


def resolve_detectors(names: Sequence[str] | None = None) -> tuple["Detector", ...]:
    """The chain to run: *names* in name order, or everything registered."""
    if names is None:
        return registered_detectors()
    return tuple(get_detector(name) for name in sorted(set(names)))


def load_external_detector(spec: str) -> Detector:
    """Load one explicitly configured ``module:DetectorClass`` reference.

    This is deliberately a module path contract rather than ambient package
    discovery.  A workspace that does not name a plugin performs no imports;
    a malformed, unavailable, abstract, non-detector, or non-instantiable
    plugin is a configuration error before any caller can persist text.
    """
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise DetectorSpecError("external detector must be a 'module:Class' string")
    module_name, object_name = (part.strip() for part in spec.split(":", 1))
    if not module_name or module_name.startswith(".") or not object_name:
        raise DetectorSpecError(f"invalid external detector reference {spec!r}")
    if any(not part.isidentifier() for part in module_name.split(".")):
        raise DetectorSpecError(f"invalid external detector module {module_name!r}")
    if any(not part.isidentifier() for part in object_name.split(".")):
        raise DetectorSpecError(f"invalid external detector object {object_name!r}")

    token = _LOADING_EXTERNAL.set(True)
    try:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise DetectorSpecError(f"could not load external detector {spec!r}: {exc}") from exc
        candidate: object = module
        try:
            for part in object_name.split("."):
                candidate = getattr(candidate, part)
        except AttributeError as exc:
            raise DetectorSpecError(f"external detector object not found: {spec!r}") from exc
    finally:
        _LOADING_EXTERNAL.reset(token)

    if not inspect.isclass(candidate) or not issubclass(candidate, Detector):
        raise DetectorSpecError(f"external detector {spec!r} must be a Detector subclass")
    if candidate.__abstractmethods__:
        raise DetectorSpecError(f"external detector {spec!r} is abstract")
    _validate_detector_class(candidate)
    try:
        instance = candidate()
    except Exception as exc:
        raise DetectorSpecError(f"external detector {spec!r} must have a zero-argument constructor") from exc
    return instance


def _dedupe(findings: Iterable[Finding]) -> list[Finding]:
    """Drop findings that overlap an earlier-kept one.

    Leftmost wins; on a tie the longer span wins, then the detector name
    breaks it. Every step is a total order over data already in hand, so
    two runs over one document keep the same set.
    """
    kept: list[Finding] = []
    end_so_far = -1
    for finding in sorted(findings, key=lambda f: (f.start, -f.end, f.detector)):
        if finding.start < end_so_far:
            continue
        kept.append(finding)
        end_so_far = finding.end
    return kept


def scan_text(text: str, detectors: Sequence[Detector] | None = None) -> list[Finding]:
    """Every non-overlapping finding in *text*, canonically ordered."""
    chain = registered_detectors() if detectors is None else tuple(detectors)
    found: list[Finding] = []
    for detector in chain:
        try:
            findings = detector.scan(text)
        except Exception as exc:
            raise DetectorSpecError(f"detector {detector.name!r} failed while scanning") from exc
        if not isinstance(findings, list):
            raise DetectorSpecError(f"detector {detector.name!r} must return a list of Finding objects")
        for finding in findings:
            if not isinstance(finding, Finding):
                raise DetectorSpecError(f"detector {detector.name!r} returned a non-Finding result")
            if finding.detector != detector.name or finding.category != detector.category:
                raise DetectorSpecError(f"detector {detector.name!r} returned a mismatched finding identity")
            if (
                isinstance(finding.start, bool)
                or isinstance(finding.end, bool)
                or not isinstance(finding.start, int)
                or not isinstance(finding.end, int)
            ):
                raise DetectorSpecError(f"detector {detector.name!r} returned non-integer finding bounds")
            if not 0 <= finding.start < finding.end <= len(text):
                raise DetectorSpecError(f"detector {detector.name!r} returned out-of-range finding bounds")
        found.extend(findings)
    return _dedupe(found)
