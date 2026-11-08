"""Install the bundled LazyBridge skill for Claude Code and Codex.

    python -m lazybridge.skill install            # every assistant found
    python -m lazybridge.skill install --to claude
    python -m lazybridge.skill install --dest DIR # any folder

It copies the skill folder (no symlink, so it works on Windows without admin
rights) into ``~/.claude/skills/lazybridge`` and/or ``~/.codex/skills/lazybridge``.
A folder it installed before is replaced; anything else at that path is left
alone unless ``--force`` is given. Rerun it after upgrading LazyBridge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lazybridge.skill import MARKER, install_skill

HOMES = {"claude": Path.home() / ".claude", "codex": Path.home() / ".codex"}
__all__ = ["HOMES", "MARKER", "main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lazybridge.skill")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("install", help="install the skill for Claude Code and/or Codex")
    cmd.add_argument("--to", choices=["claude", "codex", "all"], help="default: every assistant found")
    cmd.add_argument("--dest", type=Path, help="install into this folder instead")
    cmd.add_argument("--force", action="store_true", help="replace a folder not installed by LazyBridge")
    args = parser.parse_args(argv)

    if args.dest:
        targets = [args.dest]
    else:
        names = (
            list(HOMES) if args.to == "all" else [args.to] if args.to else [n for n, h in HOMES.items() if h.is_dir()]
        )
        targets = [HOMES[n] / "skills" / "lazybridge" for n in names or ["claude"]]
    try:
        for dest in targets:
            print(f"installed {install_skill(dest, force=args.force)}")
    except FileExistsError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
