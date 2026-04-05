"""
Thin compatibility wrapper — re-exports everything from lazybridge.tools.read_docs.

Prefer the canonical import:
    from lazybridge.tools.read_docs import read_folder_docs

This file exists so that scripts with the old import path continue to work:
    from read_docs_tool import read_folder_docs
"""
from lazybridge.tools.read_docs import *  # noqa: F401, F403
from lazybridge.tools.read_docs import read_folder_docs  # noqa: F401 — explicit for IDEs
