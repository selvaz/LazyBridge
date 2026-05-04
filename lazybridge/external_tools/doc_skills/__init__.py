"""lazybridge.external_tools.doc_skills — BM25 local documentation skill runtime (domain example).

Domain example shipped with LazyBridge — not part of the framework
contract. Pin to a specific lazybridge release if you depend on it.

Index local documentation folders into a portable skill bundle, then expose
the bundle as a LazyBridge tool or pipeline that any agent can call.

Quick start::

    from lazybridge.external_tools.doc_skills import build_skill, skill_tools
    from lazybridge import Agent

    meta = build_skill(["./docs", "./reference"], "my-project")
    tools = skill_tools(skill_dir=meta["skill_dir"])
    resp = Agent("anthropic", tools=tools)("How does X work?")

No extra dependencies required beyond the standard library.
"""

from lazybridge.external_tools.doc_skills.doc_skills import (
    DocChunk,
    SkillManifest,
    build_skill,
    query_skill,
    skill_builder_tools,
    skill_pipeline,
    skill_tools,
)

__all__ = [
    "DocChunk",
    "SkillManifest",
    "build_skill",
    "query_skill",
    "skill_builder_tools",
    "skill_pipeline",
    "skill_tools",
]
