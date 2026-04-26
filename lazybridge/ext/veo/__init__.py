"""lazybridge.ext.veo — Video generation via Google Veo API."""

#: Stability tag — see ``docs/guides/core-vs-ext.md``.
__stability__ = "alpha"
__lazybridge_min__ = "1.0.0"
from lazybridge.ext.veo.veo import VeoError, veo_tool

__all__ = ["veo_tool", "VeoError"]
