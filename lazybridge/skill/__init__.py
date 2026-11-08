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
reference files by relative path. To install it for Claude Code and Codex::

    python -m lazybridge.skill install

The full how-to (Claude API, Claude.ai zip upload, downloadable mirror) lives
at ``core.lazybridge.com/for-llms/claude-skill``.
"""

from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path

__all__ = ["MARKER", "install_skill", "reference_paths", "skill_path"]

#: File written into every folder this module installs, so a reinstall can
#: tell its own folder from one the user created.
MARKER = ".installed-by-lazybridge"


def skill_path() -> Path:
    """Return the on-disk directory containing ``SKILL.md`` and ``references/``."""
    return Path(str(files(__name__)))


def reference_paths() -> list[Path]:
    """Return the bundled reference files (``references/*.md``), sorted by name."""
    ref_dir = skill_path() / "references"
    if not ref_dir.is_dir():
        return []
    return sorted(ref_dir.glob("*.md"))


def install_skill(dest: Path, *, source: Path | None = None, force: bool = False) -> Path:
    """Copy a skill folder (default: this one) into ``dest`` and return ``dest``.

    A copy, not a symlink, so it works on Windows without admin rights. An
    earlier install made by this function is replaced; any other folder at
    ``dest`` raises ``FileExistsError`` unless ``force`` is true. An old
    symlink install is removed as a link, never by deleting its target.
    Other packages reuse this with their own ``source``.
    """
    from lazybridge import __version__

    src = Path(source) if source is not None else skill_path()
    dest = Path(dest)
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        if not (dest / MARKER).is_file() and not force:
            raise FileExistsError(f"{dest} exists and was not installed by LazyBridge; use --force to replace it")
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.py", "*.pyc"))
    (dest / MARKER).write_text(f"lazybridge {__version__}\n", encoding="utf-8")
    return dest
