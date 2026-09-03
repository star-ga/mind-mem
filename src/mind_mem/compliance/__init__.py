# Copyright 2026 STARGA, Inc.
"""Compliance-sensitive opt-in extensions: redaction, provenance policy, export.

Three capabilities that a governed memory store is expected to have and
that this one, until now, only had names for. Each was declared as a
feature flag with no consumer, which is the failure this package closes:
an operator who set ``{"redaction": {"enabled": true}}`` got silence, and
silence is indistinguishable from success.

    :mod:`~mind_mem.compliance.detectors`
        The pluggable detector chain. Registration is structural — a
        concrete detector is in the registry because it exists, not
        because somebody remembered to add it.
    :mod:`~mind_mem.compliance.redaction`
        The pass itself: ``off`` / ``flag`` / ``redact`` / ``reject``,
        deterministic, with digests of the document before and after.
    :mod:`~mind_mem.compliance.audit`
        Every redaction lands in the tamper-evident ledger, carrying what
        kind of thing was found and never the thing itself.
    :mod:`~mind_mem.compliance.provenance_policy`
        ``off | recommended | required`` over the five provenance fields
        — the half of "provenance-rich blocks" that was missing.
    :mod:`~mind_mem.compliance.prewrite`
        One door that runs provenance policy, then the detector chain,
        then the ledger write, in that order.
    :mod:`~mind_mem.compliance.export`
        ``mm export`` — a byte-deterministic bundle over the *admitted*
        corpus, with a policy, a since-date and a withheld count.

Every surface here is opt-in and default-off. With the flags unset,
nothing in this package runs, reads a config twice, or writes a byte —
which is the property that lets it ship in a release whose keystone claim
is that the disabled build is the build that never had the feature.

Copyright STARGA, Inc.
"""

from __future__ import annotations

from .detectors import (
    CATEGORY_PII,
    CATEGORY_SECRET,
    Detector,
    DetectorSpecError,
    DuplicateDetectorError,
    Finding,
    RegexDetector,
    detector_names,
    get_detector,
    registered_detectors,
    resolve_detectors,
    scan_text,
)
from .export import (
    BUNDLE_SCHEMA,
    COMPLIANCE_EXPORT_FLAG,
    FORMATS,
    ExportBundle,
    ExportPolicy,
    UnknownExportPolicyError,
    build_bundle,
    policy_names,
    render_bundle,
    resolve_export_policy,
)
from .prewrite import PreWritePolicy, Screening, screen, screen_for_workspace
from .provenance_policy import (
    POLICY_OFF,
    POLICY_RECOMMENDED,
    POLICY_REQUIRED,
    PROVENANCE_FIELDS,
    PROVENANCE_FLAG,
    ProvenanceConfigError,
    ProvenanceDecision,
    ProvenanceRequired,
    evaluate_provenance,
    require_provenance,
    resolve_policy,
    resolve_required_fields,
)
from .redaction import (
    MODE_FLAG,
    MODE_OFF,
    MODE_REDACT,
    MODE_REJECT,
    REDACTION_FLAG,
    RedactionConfigError,
    RedactionRefused,
    RedactionResult,
    redact,
    redaction_chain_for_workspace,
    resolve_mode,
)

__all__ = [
    "BUNDLE_SCHEMA",
    "CATEGORY_PII",
    "CATEGORY_SECRET",
    "COMPLIANCE_EXPORT_FLAG",
    "Detector",
    "DetectorSpecError",
    "DuplicateDetectorError",
    "ExportBundle",
    "ExportPolicy",
    "FORMATS",
    "Finding",
    "MODE_FLAG",
    "MODE_OFF",
    "MODE_REDACT",
    "MODE_REJECT",
    "POLICY_OFF",
    "POLICY_RECOMMENDED",
    "POLICY_REQUIRED",
    "PROVENANCE_FIELDS",
    "PROVENANCE_FLAG",
    "PreWritePolicy",
    "ProvenanceConfigError",
    "ProvenanceDecision",
    "ProvenanceRequired",
    "REDACTION_FLAG",
    "RedactionConfigError",
    "RedactionRefused",
    "RedactionResult",
    "RegexDetector",
    "Screening",
    "UnknownExportPolicyError",
    "build_bundle",
    "detector_names",
    "evaluate_provenance",
    "get_detector",
    "policy_names",
    "redact",
    "redaction_chain_for_workspace",
    "registered_detectors",
    "render_bundle",
    "require_provenance",
    "resolve_detectors",
    "resolve_export_policy",
    "resolve_mode",
    "resolve_policy",
    "resolve_required_fields",
    "scan_text",
    "screen",
    "screen_for_workspace",
]
