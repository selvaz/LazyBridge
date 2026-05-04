"""Build the ``_quarto.yml`` project file the renderer consumes.

Quarto reads this YAML to decide which formats to emit, which theme to
apply, where the bibliography lives, and which CSL style to use.  We
generate it programmatically so users don't have to hand-write it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Bootswatch themes bundled with Quarto.  We default to ``cosmo`` which
# is clean, conservative, and renders well in both light + print.
_VALID_HTML_THEMES = {
    "cosmo", "flatly", "journal", "litera", "lumen", "lux", "materia",
    "minty", "morph", "pulse", "quartz", "sandstone", "simplex", "sketchy",
    "slate", "solar", "spacelab", "superhero", "united", "vapor", "yeti",
    "zephyr", "cyborg", "darkly",
}


def build_quarto_yml(
    *,
    formats: Iterable[str],
    theme: str = "cosmo",
    title: str | None = None,
    author: str | None = None,
    bibliography: str | Path | None = None,
    csl_style: str | None = None,
) -> str:
    """Return YAML content suitable for ``_quarto.yml`` in a project root.

    ``formats`` is a list of Quarto format names: ``html``, ``pdf``,
    ``docx``, ``revealjs``.  ``pdf`` defaults to the Typst engine — much
    faster than LaTeX and no TinyTeX install required.

    Returns the YAML as a string; the caller writes it to disk.
    """
    if theme not in _VALID_HTML_THEMES:
        # Don't reject — pass it through so users can supply custom themes;
        # just warn-via-comment in the file so a misspelling is debuggable.
        theme_line = f'    theme: {theme}  # NOTE: not in our known Bootswatch list'
    else:
        theme_line = f'    theme: {theme}'

    fmt_lines: list[str] = []
    if "html" in formats:
        fmt_lines.append("  html:")
        fmt_lines.append(theme_line)
        fmt_lines.append("    toc: true")
        fmt_lines.append("    toc-location: left")
        fmt_lines.append("    code-fold: false")
        fmt_lines.append("    fig-cap-location: bottom")
    if "pdf" in formats:
        # Typst is faster than LaTeX, no TinyTeX dep.  Falls back to LaTeX
        # only if the user opts in by editing the YAML themselves.
        fmt_lines.append("  typst:")
        fmt_lines.append("    toc: true")
        fmt_lines.append("    margin:")
        fmt_lines.append('      x: 1.25in')
        fmt_lines.append('      y: 1in')
    if "docx" in formats:
        fmt_lines.append("  docx:")
        fmt_lines.append("    toc: true")
        fmt_lines.append("    number-sections: false")
    if "revealjs" in formats:
        fmt_lines.append("  revealjs:")
        fmt_lines.append("    theme: simple")
        fmt_lines.append("    slide-level: 2")
        fmt_lines.append("    transition: fade")
        fmt_lines.append("    incremental: false")

    parts: list[str] = ["project:", "  type: default", ""]
    if title or author:
        # Project-level metadata is inherited by every format unless the
        # .qmd file overrides it.  We still set title in the .qmd front-
        # matter for clarity, but author belongs at the project level.
        if title:
            parts.append(f'title: "{title}"')
        if author:
            parts.append(f'author: "{author}"')
        parts.append("")

    parts.append("format:")
    parts.extend(fmt_lines)
    parts.append("")

    if bibliography:
        parts.append(f"bibliography: {bibliography}")
    if csl_style:
        parts.append(f"csl: {csl_style}")
    if bibliography or csl_style:
        parts.append("")

    parts.extend(["execute:", "  echo: false", "  warning: false"])
    parts.append("")
    return "\n".join(parts)
