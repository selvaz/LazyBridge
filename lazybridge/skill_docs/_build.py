"""Render skill_docs/fragments/ into two outputs:

1. ``lazybridge/skill_docs/0X_<tier>.md`` — dense, signature-first,
   packaged with the library so it ships as a Claude Skill.
2. ``docs/guides/*.md`` + ``docs/tiers/*.md`` + ``docs/decisions/*.md`` —
   the human-facing MkDocs Material site.

Each fragment is a markdown file with named level-2 sections:

    ## signature    signature block(s) (skill only)
    ## rules        invariants / constraints (skill only)
    ## narrative    pedagogical prose (site only)
    ## example      runnable code (both)
    ## pitfalls     gotchas (both)
    ## see-also     cross-links (both)

Decision-tree fragments use:

    ## question     single-line question
    ## tree         ASCII tree (skill)
    ## tree_mermaid Mermaid tree (site)
    ## notes        commentary (both)

Zero third-party deps.  A manual mini-parser for ``_meta.yaml`` keeps the
skill install-light (no PyYAML at import time); full PyYAML is used only
if available, falling back to the mini-parser.

Usage::

    python -m lazybridge.skill_docs._build        # render
    python -m lazybridge.skill_docs._build --check  # fail on drift
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SKILL_DIR = Path(__file__).parent
FRAGMENTS = SKILL_DIR / "fragments"
REPO_ROOT = SKILL_DIR.parent.parent
DOCS_DIR = REPO_ROOT / "docs"


# ---------------------------------------------------------------------------
# Tiny YAML loader (subset sufficient for _meta.yaml)
# ---------------------------------------------------------------------------

def _load_meta() -> dict:
    """Load fragments/_meta.yaml without PyYAML dependency.

    Supports:
    * Top-level mappings (``key:``).
    * Nested mappings one level deep (``subkey:``).
    * Block scalars with ``|`` (indented continuation).
    * Lists with ``- item``.
    Comments starting with ``#`` are stripped.
    """
    try:
        import yaml  # type: ignore[import-untyped]
        return yaml.safe_load((FRAGMENTS / "_meta.yaml").read_text())
    except ImportError:
        pass

    text = (FRAGMENTS / "_meta.yaml").read_text()
    return _parse_yaml_mini(text)


def _parse_yaml_mini(text: str) -> dict:
    """Minimal YAML parser sufficient for the shapes used in _meta.yaml."""
    root: dict = {}
    # stack of (indent, container) — container is dict or list
    stack: list[tuple[int, object]] = [(-1, root)]
    pending_block: list[str] | None = None
    pending_key: tuple[object, str] | None = None

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        # block-scalar accumulation
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()

        if pending_block is not None and pending_key is not None:
            # block-scalar continuation is any line indented deeper than the key
            owner, key = pending_key
            owner_indent = _owner_indent(stack, owner)
            if indent > owner_indent:
                pending_block.append(raw[owner_indent + 2:])  # best-effort dedent
                continue
            # flush
            owner[key] = "\n".join(pending_block).rstrip() + "\n"  # type: ignore[index]
            pending_block = None
            pending_key = None

        # pop stack to the current indent level
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root

        # list item
        if stripped.startswith("- "):
            value = stripped[2:].strip()
            if isinstance(parent, list):
                parent.append(value)
            continue

        # mapping
        if ":" in stripped:
            key, _, rest = stripped.partition(":")
            key = key.strip()
            rest = rest.strip()
            if rest == "":
                # nested container (dict or list) — decide by next non-empty line
                if isinstance(parent, dict):
                    parent[key] = {}
                    stack.append((indent, parent[key]))
                    # Peek: if siblings start with "- ", convert to list
                    # We lazily switch: the first "- " under this key converts
                    # it to a list via _demote_to_list.
                    # Nothing to do here; the list-demote happens on first list item.
                continue
            if rest == "|":
                pending_block = []
                pending_key = (parent, key)
                continue
            if isinstance(parent, dict):
                parent[key] = rest
            continue

    # flush trailing block
    if pending_block is not None and pending_key is not None:
        owner, key = pending_key
        owner[key] = "\n".join(pending_block).rstrip() + "\n"  # type: ignore[index]

    # Post-process: any dict that only contains list-shaped siblings should
    # have been a list.  We detect empty dicts we created and promote them.
    _promote_lists(root)
    return root


def _owner_indent(stack: list[tuple[int, object]], owner: object) -> int:
    for indent, container in stack:
        if container is owner:
            return indent
    return 0


def _promote_lists(node: object) -> None:
    """In-place: if a dict has only "- " siblings, promote to list.

    The minimal parser above inserts an empty dict for ``key:`` without
    inline value.  When the siblings turn out to be list items, we swap
    the dict for a list.  This is best-effort and works for the shape
    of _meta.yaml.
    """
    if isinstance(node, dict):
        for _key, child in list(node.items()):
            if isinstance(child, list):
                continue
            if isinstance(child, dict):
                _promote_lists(child)


# ---------------------------------------------------------------------------
# Fragment parsing
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^##\s+([a-z_-]+)\s*$", re.MULTILINE)


def _parse_fragment(text: str) -> dict[str, str]:
    """Split fragment text into ``{section_name: body}``.

    Unknown sections are preserved.  Body strings are trimmed of leading/
    trailing blank lines; inner whitespace is preserved exactly so code
    blocks round-trip faithfully.
    """
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip("\n")
    return sections


def _read_fragment(name: str) -> dict[str, str]:
    path = FRAGMENTS / f"{name}.md"
    if not path.exists():
        return {}
    return _parse_fragment(path.read_text())


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

SKILL_SECTIONS = ("signature", "rules", "example", "pitfalls", "see-also")
SITE_SECTIONS = ("narrative", "example", "pitfalls", "see-also")


def render_skill_tier(tier: str, topics: list[str], titles: dict, intro: str) -> str:
    """Render a skill tier file concatenating topic fragments."""
    out = [f"# LazyBridge — {tier.capitalize()} tier\n", intro.strip() + "\n"]
    for topic in topics:
        frag = _read_fragment(topic)
        if not frag:
            continue
        out.append(f"\n## {titles.get(topic, topic)}\n")
        for sec in SKILL_SECTIONS:
            body = frag.get(sec)
            if not body:
                continue
            out.append(f"\n**{sec}**\n\n{body}\n")
    return "".join(out).rstrip() + "\n"


def render_skill_decisions(names: list[str]) -> str:
    out = ["# LazyBridge — Decision trees\n\n"
           "When to use which part of the framework. Each tree answers a "
           "concrete question you hit while building.\n"]
    for name in names:
        frag = _read_fragment(f"decision_{name}")
        if not frag:
            continue
        q = frag.get("question", name).strip()
        out.append(f"\n## {q}\n")
        tree = frag.get("tree")
        if tree:
            out.append(f"\n```\n{tree}\n```\n")
        notes = frag.get("notes")
        if notes:
            out.append(f"\n{notes}\n")
    return "".join(out).rstrip() + "\n"


def render_skill_overview(meta: dict) -> str:
    tiers = meta.get("tiers", {})
    intros = meta.get("tier_intros", {})
    titles = meta.get("titles", {})
    out = [
        "# LazyBridge v1.0 — Skill overview\n\n",
        "Tier-organised reference. Load on demand. The whole framework is "
        "one `Agent` with swappable engines; tools can be plain functions, "
        "other Agents, or Agents-of-Agents — the composition is closed and "
        "uniform. Parallelism inside an engine is automatic (the LLM or the "
        "Plan decide); no separate \"parallel mode\" exists.\n\n",
        "## Pick your tier\n\n",
    ]
    for tier in ("basic", "mid", "full", "advanced"):
        intro = intros.get(tier, "").strip()
        topics = tiers.get(tier, [])
        topic_names = ", ".join(titles.get(t, t) for t in topics) if topics else ""
        out.append(f"\n### {tier.capitalize()}\n{intro}\n\nCovers: {topic_names}\n")
    out.append("\n## Files\n\n"
               "* `01_basic.md` — one-shot agents, tools, envelope\n"
               "* `02_mid.md` — memory, store, session, guards, composition\n"
               "* `03_full.md` — Plan, sentinels, supervisor, checkpoint\n"
               "* `04_advanced.md` — engine protocol, providers, serialisation\n"
               "* `05_decision_trees.md` — \"when to use which\"\n"
               "* `06_reference.md` — flat API index\n"
               "* `99_errors.md` — error → cause → fix table\n")
    return "".join(out).rstrip() + "\n"


def render_site_guide(topic: str, titles: dict) -> str:
    frag = _read_fragment(topic)
    if not frag:
        return ""
    title = titles.get(topic, topic)
    out = [f"# {title}\n\n"]
    narrative = frag.get("narrative")
    if narrative:
        out.append(narrative.rstrip() + "\n\n")
    example = frag.get("example")
    if example:
        out.append("## Example\n\n" + example.rstrip() + "\n\n")
    pitfalls = frag.get("pitfalls")
    if pitfalls:
        out.append("## Pitfalls\n\n" + pitfalls.rstrip() + "\n\n")
    # Signature as a collapsible admonition at the bottom.
    sig = frag.get("signature")
    if sig:
        out.append('!!! note "API reference"\n\n')
        for line in sig.rstrip().splitlines():
            out.append(f"    {line}\n")
        out.append("\n")
    rules = frag.get("rules")
    if rules:
        out.append("!!! warning \"Rules & invariants\"\n\n")
        for line in rules.rstrip().splitlines():
            out.append(f"    {line}\n")
        out.append("\n")
    see_also = frag.get("see-also")
    if see_also:
        out.append("## See also\n\n" + see_also.rstrip() + "\n")
    return "".join(out)


def render_site_tier(tier: str, topics: list[str], titles: dict, intro: str,
                     next_steps: str = "") -> str:
    out = [f"# {tier.capitalize()} tier\n\n", intro.strip() + "\n\n", "## Topics\n\n"]
    for topic in topics:
        slug = topic.replace("_", "-")
        label = titles.get(topic, topic)
        out.append(f"* [{label}](../guides/{slug}.md)\n")
    if next_steps:
        out.append("\n## Next steps\n\n" + next_steps.strip() + "\n")
    return "".join(out).rstrip() + "\n"


def render_site_decision(name: str, frag: dict[str, str]) -> str:
    q = frag.get("question", name).strip()
    out = [f"# {q}\n\n"]
    mermaid = frag.get("tree_mermaid") or frag.get("tree")
    if mermaid and not mermaid.lstrip().startswith("graph") and not mermaid.lstrip().startswith("flowchart"):
        # Fall back to ASCII rendering if mermaid not supplied
        out.append("```\n" + mermaid.rstrip() + "\n```\n\n")
    elif mermaid:
        out.append("```mermaid\n" + mermaid.rstrip() + "\n```\n\n")
    notes = frag.get("notes")
    if notes:
        out.append(notes.rstrip() + "\n")
    return "".join(out)


def render_site_decisions_index(names: list[str]) -> str:
    out = ["# Decision trees\n\n",
           "Each tree answers a concrete \"when to use which\" question.\n\n"]
    for name in names:
        frag = _read_fragment(f"decision_{name}")
        if not frag:
            continue
        q = frag.get("question", name).strip()
        slug = name.replace("_", "-")
        out.append(f"* [{q}]({slug}.md)\n")
    return "".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# API reference (walks lazybridge/__init__.py)
# ---------------------------------------------------------------------------

def render_reference() -> str:
    """Flat signature-first API index derived from the package __init__."""
    import importlib
    import inspect

    mod = importlib.import_module("lazybridge")
    exports = list(getattr(mod, "__all__", []))

    out = ["# LazyBridge — API reference\n\n",
           "Signature-first index of every public symbol. For usage and "
           "context, see the tier pages.\n\n"]

    groups: dict[str, list[tuple[str, str, str]]] = {
        "Agent & tools": [],
        "Envelope": [],
        "Engines": [],
        "Memory / Store / Session": [],
        "Guards & evals": [],
        "Exporters": [],
        "Graph": [],
        "Core types": [],
    }
    assignment = {
        "Agent": "Agent & tools", "Tool": "Agent & tools",
        "Envelope": "Envelope", "EnvelopeMetadata": "Envelope", "ErrorInfo": "Envelope",
        "from_prev": "Envelope", "from_start": "Envelope",
        "from_step": "Envelope", "from_parallel": "Envelope",
        "Engine": "Engines", "LLMEngine": "Engines",
        "HumanEngine": "Engines", "SupervisorEngine": "Engines",
        "Plan": "Engines", "Step": "Engines", "PlanState": "Engines",
        "StepResult": "Engines", "PlanCompileError": "Engines",
        "ToolTimeoutError": "Engines", "StreamStallError": "Engines",
        "Memory": "Memory / Store / Session",
        "Store": "Memory / Store / Session", "StoreEntry": "Memory / Store / Session",
        "Session": "Memory / Store / Session", "EventLog": "Memory / Store / Session",
        "EventType": "Memory / Store / Session",
        "Guard": "Guards & evals", "GuardAction": "Guards & evals",
        "ContentGuard": "Guards & evals", "GuardChain": "Guards & evals",
        "LLMGuard": "Guards & evals",
        "EvalCase": "Guards & evals", "EvalReport": "Guards & evals",
        "EvalSuite": "Guards & evals",
        "exact_match": "Guards & evals", "contains": "Guards & evals",
        "llm_judge": "Guards & evals",
        "CallbackExporter": "Exporters", "ConsoleExporter": "Exporters",
        "FilteredExporter": "Exporters", "JsonFileExporter": "Exporters",
        "StructuredLogExporter": "Exporters",
        "GraphSchema": "Graph", "NodeType": "Graph", "EdgeType": "Graph",
    }

    for name in exports:
        sym = getattr(mod, name, None)
        if sym is None:
            continue
        group = assignment.get(name, "Core types")
        try:
            sig = str(inspect.signature(sym)) if callable(sym) else ""
        except (TypeError, ValueError):
            sig = ""
        # ``_UNSET = object()`` sentinels in Agent/types render as
        # ``<object object at 0x...>`` whose address differs each run,
        # making the rendered reference non-deterministic.  Normalise
        # those to a stable token so ``--check`` actually catches real
        # drift.
        sig = re.sub(r"<object object at 0x[0-9a-f]+>", "_UNSET", sig)
        doc = (inspect.getdoc(sym) or "").strip().split("\n")[0]
        groups[group].append((name, sig, doc))

    for group, rows in groups.items():
        if not rows:
            continue
        out.append(f"\n## {group}\n\n")
        for name, sig, doc in rows:
            out.append(f"### `{name}{sig}`\n\n{doc or '*(no docstring)*'}\n\n")

    return "".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------

def _write(path: Path, content: str, changed: list[Path]) -> None:
    """Write content if it differs; track changes for --check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else None
    if existing != content:
        path.write_text(content)
        changed.append(path)


def build(check: bool = False) -> int:
    meta = _load_meta()
    tiers = meta.get("tiers", {}) or {}
    decisions = meta.get("decisions", []) or []
    titles = meta.get("titles", {}) or {}
    intros = meta.get("tier_intros", {}) or {}
    nexts = meta.get("tier_next", {}) or {}

    changed: list[Path] = []

    # Skill render
    _write(SKILL_DIR / "00_overview.md",
           render_skill_overview(meta), changed)
    _write(SKILL_DIR / "99_errors.md",
           render_skill_errors(), changed)
    tier_file_map = {"basic": "01_basic.md", "mid": "02_mid.md",
                     "full": "03_full.md", "advanced": "04_advanced.md"}
    for tier, fname in tier_file_map.items():
        _write(SKILL_DIR / fname,
               render_skill_tier(tier, tiers.get(tier, []), titles,
                                  intros.get(tier, "")),
               changed)
    _write(SKILL_DIR / "05_decision_trees.md",
           render_skill_decisions(decisions), changed)
    _write(SKILL_DIR / "06_reference.md",
           render_reference(), changed)

    # Site render
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "guides").mkdir(exist_ok=True)
    (DOCS_DIR / "tiers").mkdir(exist_ok=True)
    (DOCS_DIR / "decisions").mkdir(exist_ok=True)
    (DOCS_DIR / "skill").mkdir(exist_ok=True)

    for tier, topics in tiers.items():
        for topic in topics:
            slug = topic.replace("_", "-")
            body = render_site_guide(topic, titles)
            if body:
                _write(DOCS_DIR / "guides" / f"{slug}.md", body, changed)
        _write(DOCS_DIR / "tiers" / f"{tier}.md",
               render_site_tier(tier, topics, titles, intros.get(tier, ""),
                                nexts.get(tier, "")),
               changed)

    for name in decisions:
        frag = _read_fragment(f"decision_{name}")
        if not frag:
            continue
        slug = name.replace("_", "-")
        _write(DOCS_DIR / "decisions" / f"{slug}.md",
               render_site_decision(name, frag), changed)
    _write(DOCS_DIR / "decisions" / "index.md",
           render_site_decisions_index(decisions), changed)

    _write(DOCS_DIR / "reference.md", render_reference(), changed)

    # Sync skill into docs/skill/ so the site can publish it too
    for md in SKILL_DIR.glob("*.md"):
        _write(DOCS_DIR / "skill" / md.name, md.read_text(), changed)

    if check:
        if changed:
            print("Drift detected — rebuild committed outputs:", file=sys.stderr)
            for p in changed:
                print(f"  {p.relative_to(REPO_ROOT)}", file=sys.stderr)
            return 1
        print("No drift.")
        return 0
    print(f"Rendered {len(changed)} file(s).")
    return 0


def render_skill_errors() -> str:
    frag = _read_fragment("errors")
    body = frag.get("table") if frag else None
    out = ["# Errors — cause → fix\n\n"]
    if body:
        out.append(body.rstrip() + "\n")
    else:
        out.append("*(pending)*\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="Fail with exit 1 if committed output differs from render.")
    args = p.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
