"""Deprecated location. Moved to ``lazytools.documents`` (pip install lazytoolkit).

This shim keeps ``from lazybridge.external_tools.read_docs import read_docs_tools``
working with a :class:`DeprecationWarning`. It is removed in 0.9.
"""

from __future__ import annotations

import warnings


def __getattr__(name: str):  # PEP 562 — fires only on attribute access
    warnings.warn(
        "lazybridge.external_tools.read_docs moved to lazytools.documents in 0.8; "
        "install 'lazytoolkit' and import from there. This shim is removed in 0.9.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from lazytools import documents as _moved
    except ImportError as exc:
        raise ImportError(
            "lazybridge.external_tools.read_docs now requires 'lazytoolkit' (pip install 'lazytoolkit[docs]')."
        ) from exc
    return getattr(_moved, name)
