#: Stability tag — see ``docs/guides/core-vs-ext.md``.
__stability__ = "alpha"
__lazybridge_min__ = "1.0.0"

"""lazybridge.ext.doc_skills — BM25 local documentation skill runtime.

Index local documentation folders into a portable skill bundle, then expose
the bundle as a LazyBridge tool or pipeline that any agent can call.

Quick start::

    from lazybridge.ext.doc_skills import build_skill, skill_tool
    from lazybridge import Agent

    meta = build_skill(["./docs", "./reference"], "my-project")
    tool = skill_tool(meta["skill_dir"])
    resp = Agent("anthropic", tools=[tool])("How does X work?")

No extra dependencies required beyond the standard library.
"""

from lazybridge.ext.doc_skills.doc_skills import (
    DocChunk,
    SkillManifest,
    build_skill,
    query_skill,
    skill_builder_tool,
    skill_pipeline,
    skill_tool,
)

__all__ = [
    "DocChunk",
    "SkillManifest",
    "build_skill",
    "query_skill",
    "skill_builder_tool",
    "skill_pipeline",
    "skill_tool",
]
