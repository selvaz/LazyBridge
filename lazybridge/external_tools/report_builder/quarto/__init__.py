"""Quarto integration: detect the CLI, build project files, render fragments.

This package is the *Quarto-first* render path described in the plan.  It is
deliberately quiet at import time — the CLI probe runs only when an exporter
asks for it, so importing the package on a machine without Quarto is fine.
"""

from __future__ import annotations

from lazybridge.external_tools.report_builder.quarto.detect import (
    QuartoNotFoundError,
    find_quarto,
    quarto_version,
)
from lazybridge.external_tools.report_builder.quarto.project import build_quarto_yml
from lazybridge.external_tools.report_builder.quarto.qmd import render_fragment_to_qmd, render_report_to_qmd

__all__ = [
    "QuartoNotFoundError",
    "find_quarto",
    "quarto_version",
    "build_quarto_yml",
    "render_fragment_to_qmd",
    "render_report_to_qmd",
]
