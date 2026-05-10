"""Skill-doc tooling.

Holds the build/check entry point for ``lazybridge/skill/SKILL.md``
drift detection. Invoked from CI as::

    python -m lazybridge.skill_docs._build --check

A non-zero exit signals that ``SKILL.md`` is missing, empty, or out of
sync with ``lazybridge.__all__``.
"""
