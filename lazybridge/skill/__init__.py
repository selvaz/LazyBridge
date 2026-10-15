"""Bundled Claude Skill for the LazyBridge framework.

The skill is shipped with the library so that Claude Code, Claude.ai, Codex,
or any Anthropic API caller that supports skills can load it directly from a
``pip install lazybridge`` checkout — no separate download, no drift between
the installed framework and the assistant guidance for it.

Layout (progressive disclosure)::

    lazybridge/skill/
        SKILL.md              # short index: mental model, golden rules, routing table
        references/*.md       # one file per area, read only when the task needs it

Install the whole directory, not just ``SKILL.md``: the index routes to the
reference files by relative path. To make Claude Code pick it up::

    ln -s "$(python -c 'from lazybridge.skill import skill_path; print(skill_path())')" \\
          ~/.claude/skills/lazybridge

The full how-to (Claude API, Claude.ai zip upload, downloadable mirror) lives
at ``core.lazybridge.com/for-llms/claude-skill``.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__all__ = ["reference_paths", "skill_path"]


def skill_path() -> Path:
    """Return the on-disk directory containing ``SKILL.md`` and ``references/``."""
    return Path(str(files(__name__)))


def reference_paths() -> list[Path]:
    """Return the bundled reference files (``references/*.md``), sorted by name."""
    ref_dir = skill_path() / "references"
    if not ref_dir.is_dir():
        return []
    return sorted(ref_dir.glob("*.md"))
