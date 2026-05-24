"""Deprecated location. Moved to ``lazytools.connectors.mcp`` (pip install lazytoolkit).

The MCP connector (``MCP``, ``MCPServer``) is a tool provider that connects to
external MCP servers, so it now lives with the other connectors in
``lazytools``. This shim keeps ``from lazybridge.ext.mcp import MCP`` working
with a :class:`DeprecationWarning`. It is removed in 0.9.

Install with::

    pip install lazytoolkit[mcp]
"""

from __future__ import annotations

import warnings


def __getattr__(name: str):  # PEP 562 — fires only on attribute access
    warnings.warn(
        "lazybridge.ext.mcp moved to lazytools.connectors.mcp in 0.8; "
        "install 'lazytoolkit' and import from there. This shim is removed in 0.9.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from lazytools.connectors import mcp as _moved
    except ImportError as exc:
        raise ImportError("lazybridge.ext.mcp now requires 'lazytoolkit' (pip install 'lazytoolkit[mcp]').") from exc
    return getattr(_moved, name)
