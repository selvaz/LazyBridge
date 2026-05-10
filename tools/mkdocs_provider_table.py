"""MkDocs hook — replaces the ``<!-- PROVIDER_CAPABILITY_TABLE -->``
marker in any docs page with a freshly-rendered table generated from
``lazybridge.matrix.provider_capabilities()``.

Wired in ``mkdocs.yml`` under ``hooks:``.  Runs at build time so the
rendered docs always reflect the current provider ClassVars — pre-fix
the README had a hand-maintained "what each provider supports" table
that drifted on every provider change.

Marker syntax in the source ``.md`` file::

    <!-- PROVIDER_CAPABILITY_TABLE -->

When the hook detects the marker, the table is inserted in its place
during ``on_page_markdown``.  The original ``.md`` source on disk is
unchanged.
"""

from __future__ import annotations

from typing import Any

MARKER = "<!-- PROVIDER_CAPABILITY_TABLE -->"


def _build_table() -> str:
    from lazybridge.matrix import provider_capabilities

    caps = provider_capabilities()
    # Stable order: alphabetical by provider name.
    providers = sorted(caps.keys())

    # Union of every native-tool name seen across providers, sorted for
    # diff-friendliness.
    all_tools = sorted({t.value for c in caps.values() for t in c.native_tools})

    lines: list[str] = []
    # Header row.
    header = ["Provider", "Streaming", "Structured output", "Thinking"] + all_tools
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    def _y(b: bool) -> str:
        return "✓" if b else "—"

    for name in providers:
        c = caps[name]
        row = [
            f"`{name}`",
            _y(c.streaming),
            _y(c.structured_output),
            _y(c.thinking),
        ]
        # Per-native-tool cells.
        tool_names = {t.value for t in c.native_tools}
        for tool_label in all_tools:
            row.append(_y(tool_label in tool_names))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "*Generated from ``lazybridge.matrix.provider_capabilities()`` at docs "
        "build time — see ``tools/mkdocs_provider_table.py``.*"
    )
    return "\n".join(lines)


def on_page_markdown(markdown: str, *, page: Any, config: Any, files: Any) -> str:
    """MkDocs page hook — runs once per markdown file at build time.

    When the marker is present we substitute the generated table; in
    every other case we return the markdown unchanged so build time is
    not impacted on pages that don't opt in.
    """
    if MARKER not in markdown:
        return markdown
    return markdown.replace(MARKER, _build_table())
