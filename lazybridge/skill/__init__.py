"""Bundled Claude Skill for the LazyBridge framework.

The skill is shipped with the library so that Claude Code, Claude.ai, or any
Anthropic API caller that supports skills can load it directly from a
``pip install lazybridge`` checkout — no separate download, no drift between
the installed framework and the assistant guidance for it.

To make Claude Code pick it up::

    ln -s "$(python -c 'from lazybridge.skill import skill_path; print(skill_path())')" \\
          ~/.claude/skills/lazybridge

The full how-to (Claude API, Claude.ai zip upload, downloadable mirror) lives
at ``selvaz.github.io/LazyBridge/for-llms/claude-skill``.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__all__ = ["skill_path"]


def skill_path() -> Path:
    """Return the on-disk directory containing ``SKILL.md`` and friends."""
    return Path(str(files(__name__)))
