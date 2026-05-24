"""Deprecated location. Moved to ``lazytools.skills`` (pip install lazytoolkit).

This shim keeps ``from lazybridge.external_tools.doc_skills import build_skill``
(and the other skill symbols) working with a :class:`DeprecationWarning`. It is
removed in 0.9.
"""

from __future__ import annotations

import warnings


def __getattr__(name: str):  # PEP 562 — fires only on attribute access
    warnings.warn(
        "lazybridge.external_tools.doc_skills moved to lazytools.skills in 0.8; "
        "install 'lazytoolkit' and import from there. This shim is removed in 0.9.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from lazytools import skills as _moved
    except ImportError as exc:
        raise ImportError(
            "lazybridge.external_tools.doc_skills now requires 'lazytoolkit' (pip install 'lazytoolkit')."
        ) from exc
    return getattr(_moved, name)
