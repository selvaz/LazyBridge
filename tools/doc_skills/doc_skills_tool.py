"""
Thin compatibility wrapper — re-exports everything from lazybridge.ext.tools.doc_skills.

Prefer the canonical import:
    from lazybridge.ext.tools.doc_skills import build_skill, skill_tool, skill_pipeline

This file exists so that scripts with the old import path continue to work:
    from doc_skills_tool import build_skill, skill_tool, skill_pipeline
"""
from lazybridge.ext.tools.doc_skills import *  # noqa: F401, F403
from lazybridge.ext.tools.doc_skills import (  # noqa: F401 — explicit for IDEs
    build_skill,
    query_skill,
    skill_tool,
    skill_builder_tool,
    skill_pipeline,
    DocChunk,
    SkillManifest,
)
