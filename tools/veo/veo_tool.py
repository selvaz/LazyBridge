"""
Thin compatibility wrapper — re-exports everything from lazybridge.tools.veo.

Prefer the canonical import:
    from lazybridge.tools.veo import veo_tool

This file exists so that scripts dropped into tools/veo/ can do:
    from veo_tool import veo_tool
"""
from lazybridge.tools.veo import *  # noqa: F401, F403
from lazybridge.tools.veo import veo_tool, VeoError  # noqa: F401 — explicit for IDEs
