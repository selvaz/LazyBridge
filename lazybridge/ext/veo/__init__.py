"""lazybridge.ext.veo — Video generation via Google Veo API (domain example).

Domain example shipped with LazyBridge — not part of the framework
contract. Pin to a specific lazybridge release if you depend on it.
"""

#: Stability tag — see ``docs/guides/core-vs-ext.md``.
#: ``"domain"`` = worked example shipped with the framework; not part of
#: the LazyBridge framework contract and may be moved or removed.
__stability__ = "domain"
__lazybridge_min__ = "1.0.0"
from lazybridge.ext.veo.veo import VeoError, veo_tool

__all__ = ["veo_tool", "VeoError"]
