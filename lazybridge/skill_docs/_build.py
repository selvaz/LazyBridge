"""Skill-doc drift check.

Minimum-viable check around the hand-maintained
``lazybridge/skill/SKILL.md``.  Earlier alpha shipped a fragment-based
generator that wrote tier files (``01_basic.md`` … ``04_advanced.md``);
that pipeline was retired in 0.7.9 in favour of one curated SKILL.md.

Until then, ``--check`` enforces three light invariants that catch the
practical drift modes we care about:

1. ``lazybridge/skill/SKILL.md`` exists and is non-empty.
2. Every public symbol in ``lazybridge.__all__`` is mentioned somewhere
   in ``SKILL.md`` (case-sensitive substring match against the literal
   identifier — agent docs are written in code-fenced examples so the
   identifier appears verbatim).
3. ``SKILL.md`` does not still mention symbols that were deleted from
   the public API (catches stale references after a Phase-2 deletion).

Usage::

    python -m lazybridge.skill_docs._build         # report status, no-op
    python -m lazybridge.skill_docs._build --check # exit non-zero on drift
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Symbols that are intentionally *not* mentioned in SKILL.md even though
# they appear in ``__all__``.  Keep this list extremely tight — anything
# here is a documented escape from the drift check.  The list is the
# Phase-1 baseline (see CHANGELOG); Phase 4 either restores the
# fragment-based generator or shrinks this set to zero.
_SKILL_OPTIONAL: frozenset[str] = frozenset(
    {
        # Session plumbing — taught implicitly via Session() examples.
        "EventLog",
        "EventType",
        # Guard* — taught in the Mid tier guide, not the skill (yet).
        "Guard",
        "GuardAction",
        "GuardError",
        "ContentGuard",
        "DeduplicateGuard",
        "GuardChain",
        "LLMGuard",
        # Provider-routing introspection — surfaced in
        # ``docs/for-llms/codegen-contract.md`` / ``llms.json``, not in
        # SKILL.md (which teaches the model-string conventions narratively).
        "PROVIDER_ALIASES",
        # Engine errors — surfaced via test snippets, not the skill body.
        "ConcurrentPlanRunError",
        "PlanPaused",
        "PlanRuntimeError",
        "ToolTimeoutError",
        "StreamStallError",
        # Graph schema — advanced extension, not core LLM-codegen surface.
        "GraphSchema",
        # Exporter zoo — Session() is what users construct; specific
        # exporters are catalogue items.
        "EventExporter",
        "CallbackExporter",
        "ConsoleExporter",
        "FilteredExporter",
        "StructuredLogExporter",
        # Provider extension — advanced; covered by docs/reference.
        "BaseProvider",
        "UnsupportedFeatureError",
        "UnsupportedNativeToolError",
        # Multimodal types — taught via examples, not by name.
        "ImageContent",
        "AudioContent",
        # CacheConfig — kept in 0.7.9 (carries real semantic value), but
        # the canonical entry path is the bool ``cache=True`` flag, so
        # SKILL.md doesn't need to teach the dataclass directly.
        "CacheConfig",
        # Testing helper — referenced in tests, not the skill.
        "MockAgent",
    }
)


def _skill_path() -> Path:
    """Return the absolute path to ``lazybridge/skill/SKILL.md``."""
    return Path(__file__).resolve().parent.parent / "skill" / "SKILL.md"


def _public_symbols() -> list[str]:
    """Return the current ``lazybridge.__all__`` contents."""
    import lazybridge

    return list(getattr(lazybridge, "__all__", []))


def _missing_from_skill(skill_text: str, symbols: list[str]) -> list[str]:
    """Return public symbols that don't appear anywhere in ``skill_text``.

    Match is by word-boundary regex on the literal identifier so a symbol
    named ``Tool`` doesn't false-match ``Tools`` or ``ToolTimeoutError``.
    """
    missing: list[str] = []
    for sym in symbols:
        if sym in _SKILL_OPTIONAL:
            continue
        pattern = re.compile(rf"\b{re.escape(sym)}\b")
        if not pattern.search(skill_text):
            missing.append(sym)
    return missing


def _stale_in_skill(skill_text: str) -> list[str]:
    """Return symbols still referenced in ``SKILL.md`` that are no longer
    importable from ``lazybridge``.

    We don't try to be exhaustive — the goal is to catch *deletions*
    (Phase 2) where the skill kept advertising a symbol the user can
    no longer import.  We restrict ourselves to a curated whitelist of
    names the skill historically taught, so unrelated identifiers in
    code blocks don't false-trip.
    """
    import lazybridge

    # Phase-2 module-level deletion targets.  ``Agent.from_*`` factories
    # are class methods — their removal is caught by mypy / pytest, not
    # by this drift check.  Keep this list to symbols that genuinely
    # live at module scope and would leave broken references in
    # SKILL.md when removed.
    historical = (
        "AgentRuntimeConfig",
        "CacheConfig",
        "ObservabilityConfig",
        "ResilienceConfig",
        "_ParallelAgent",  # 0.7-era private name; renamed → ParallelAgent in 0.7.9
        "wrap_tool",
    )
    stale: list[str] = []
    for hist in historical:
        if hasattr(lazybridge, hist):
            continue  # still importable; not stale
        if re.search(rf"\b{re.escape(hist)}\b", skill_text):
            stale.append(hist)
    return stale


def build(check: bool) -> int:
    skill_path = _skill_path()

    if not skill_path.exists():
        print(f"SKILL.md missing at {skill_path}", file=sys.stderr)
        return 1
    skill_text = skill_path.read_text(encoding="utf-8")
    if not skill_text.strip():
        print(f"SKILL.md is empty at {skill_path}", file=sys.stderr)
        return 1

    symbols = _public_symbols()
    if not symbols:
        # Importing lazybridge succeeded but ``__all__`` is empty — bail
        # rather than report every API as "missing", which would be wrong
        # in the noisy direction.
        print("lazybridge.__all__ is empty; skipping drift check.", file=sys.stderr)
        return 0

    missing = _missing_from_skill(skill_text, symbols)
    stale = _stale_in_skill(skill_text)

    if not missing and not stale:
        if not check:
            print(f"SKILL.md OK ({len(symbols)} public symbols, no drift).")
        return 0

    if missing:
        print(
            f"SKILL.md missing references to {len(missing)} public symbol(s):",
            file=sys.stderr,
        )
        for s in missing:
            print(f"  {s}", file=sys.stderr)
    if stale:
        print(
            f"SKILL.md still references {len(stale)} deleted symbol(s):",
            file=sys.stderr,
        )
        for s in stale:
            print(f"  {s}", file=sys.stderr)

    if check:
        print(
            "Drift detected. Update lazybridge/skill/SKILL.md or add an "
            "intentional entry to _SKILL_OPTIONAL with a rationale.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if SKILL.md is missing, empty, or out of sync with lazybridge.__all__.",
    )
    args = p.parse_args(argv)
    return build(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
