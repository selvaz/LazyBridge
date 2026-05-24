"""Deprecated location. Moved to ``lazytools.connectors.gateway`` (pip install lazytoolkit).

The external tool gateway (``ExternalToolProvider`` and friends — a connector
to Composio/Pipedream/Arcade-style HTTP tool catalogues) now lives with the
other connectors in ``lazytools``. This shim keeps
``from lazybridge.ext.gateway import ExternalToolProvider`` working with a
:class:`DeprecationWarning`. It is removed in 0.9.
"""

from __future__ import annotations

import warnings


def __getattr__(name: str):  # PEP 562 — fires only on attribute access
    warnings.warn(
        "lazybridge.ext.gateway moved to lazytools.connectors.gateway in 0.8; "
        "install 'lazytoolkit' and import from there. This shim is removed in 0.9.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from lazytools.connectors import gateway as _moved
    except ImportError as exc:
        raise ImportError("lazybridge.ext.gateway now requires 'lazytoolkit' (pip install 'lazytoolkit').") from exc
    return getattr(_moved, name)
