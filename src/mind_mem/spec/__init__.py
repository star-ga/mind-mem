# Copyright 2026 STARGA, Inc.
"""Specification generators that read the live product surface.

The OpenAPI exporter lives HERE rather than under ``sdk/`` because it must
import :func:`mind_mem.api.rest.create_app` to derive the document from the
routes the server actually serves. A spec that is hand-maintained, or derived
from anything other than the live app, can silently disagree with it — and a
spec that can drift is worse than no spec, because it is trusted.

Putting the generator under ``sdk/`` made that import a cross-package
dependency edge, which the NO_CROSS_PKG rule forbids and which the repository's
own import-graph test caught. Moving the generator to the package it reads is
the fix by construction: ``sdk/spec/`` keeps the generated artifact, which is
what an SDK consumer needs, and nothing outside ``mind_mem`` imports into it.
"""
