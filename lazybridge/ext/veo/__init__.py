"""lazybridge.ext.veo — Video generation via Google Veo API (domain example).

Domain example shipped with LazyBridge — not part of the framework
contract. Pin to a specific lazybridge release if you depend on it.
"""

#: Stability tag — see ``docs/guides/core-vs-ext.md``.
#: ``"domain"`` = worked example shipped with the framework; not part of
#: the LazyBridge framework contract and may be moved or removed.
from lazybridge.ext.veo.veo import VeoError, veo_tool

__all__ = ["veo_tool", "VeoError"]
