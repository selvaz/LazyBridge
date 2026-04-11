"""
lazybridge.tools — Ready-made LazyBridge-compatible tools.

Each tool is a standalone module that can be used as a plain Python function
or wrapped as a LazyTool and passed to any agent or pipeline.

Available tools
---------------
    lazybridge.tools.doc_skills   BM25 local documentation skill runtime
    lazybridge.tools.read_docs    Multi-format document reader (.txt .md .pdf .docx .html)
    lazybridge.tools.veo          Google Veo video generation (text-to-video, image-to-video)

Usage
-----
    from lazybridge.tools.doc_skills import build_skill, skill_tool
    from lazybridge.tools.read_docs import read_folder_docs
    from lazybridge.tools.veo import veo_tool

Optional dependencies
---------------------
    read_docs requires:  pip install lazybridge[tools]
    doc_skills requires: no extra dependencies
    veo requires:        pip install lazybridge[google]
"""
