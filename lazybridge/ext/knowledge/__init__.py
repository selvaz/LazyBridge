"""Durable, cross-session "lessons" a long-running agent records after
genuinely non-obvious success — reusable notes that outlive whichever
plan, task, or single conversation they were learned on.

This is an **extension** module under the LazyBridge core-vs-ext policy
(see ``docs/guides/core-vs-ext.md``). API may change between minor
releases; consult the per-module CHANGELOG before pinning.

Promoted from LazyCEO's ``lazyceo.lessons`` (a sibling project built on
this package) after live production use — see
:class:`~lazybridge.ext.knowledge.durable.DurableKnowledgeBase`'s own
module docstring for the full design rationale (why Store-backed and
CAS'd rather than file-based, why save is create-only, and how this
differs from both :class:`lazybridge.Memory` and
:class:`~lazybridge.ext.planners.DurableBlackboard`).

    from lazybridge import Store
    from lazybridge.ext.knowledge import DurableKnowledgeBase

    knowledge = DurableKnowledgeBase(Store(db="knowledge.sqlite"))
    knowledge.save_lesson(
        topic="windows worktree paths",
        what_worked="quote the path; backslashes break naive globbing",
    )
    print(knowledge.search_lessons("windows paths"))
"""

from lazybridge.ext.knowledge.durable import (
    DEFAULT_PREFIX,
    MAX_SLUG_LENGTH,
    STALE_AFTER_DAYS,
    DurableKnowledgeBase,
    LessonStatus,
    render_lesson_line,
    slugify,
)

__all__ = [
    "DEFAULT_PREFIX",
    "MAX_SLUG_LENGTH",
    "STALE_AFTER_DAYS",
    "DurableKnowledgeBase",
    "LessonStatus",
    "render_lesson_line",
    "slugify",
]
