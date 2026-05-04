"""Multi-format exporters: Quarto-first with a pure-Python fallback.

Each exporter implements :class:`Exporter` and accepts an
:class:`AssembledReport` plus an output directory.  They return a mapping
``{format: output_path}`` for every format they actually produced.

Resolution: :func:`resolve_exporter` picks the right backend.  ``"auto"``
detects Quarto on ``$PATH``; explicit ``"quarto"`` raises if absent;
explicit ``"weasyprint"`` skips Quarto entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal

from lazybridge.ext.report_builder.assemblers import AssembledReport

# Re-exported here so callers (and monkeypatch in tests) can override the
# Quarto-discovery path without reaching into the inner sub-package.
from lazybridge.ext.report_builder.quarto.detect import find_quarto


class Exporter(ABC):
    """Abstract base class for report exporters."""

    @abstractmethod
    def export(
        self,
        report: AssembledReport,
        output_dir: Path,
        *,
        formats: list[str],
        theme: str = "cosmo",
    ) -> dict[str, Path]: ...


def resolve_exporter(
    *,
    backend: Literal["quarto", "weasyprint", "auto"] = "auto",
    formats: Iterable[str] = (),
) -> Exporter:
    """Pick an exporter based on backend preference and Quarto availability.

    ``backend="auto"`` picks Quarto when it's on PATH, otherwise WeasyPrint.
    Explicit choices are honoured even if the chosen backend isn't installed
    — the exporter's own export() raises a clear error in that case.
    """
    from lazybridge.ext.report_builder.exporters.quarto import QuartoExporter
    from lazybridge.ext.report_builder.exporters.weasyprint import WeasyPrintExporter

    if backend == "quarto":
        return QuartoExporter()
    if backend == "weasyprint":
        return WeasyPrintExporter()
    # auto: detect Quarto on PATH at call time.  ``find_quarto`` is imported
    # at module level so tests can monkeypatch it.
    if find_quarto():
        return QuartoExporter()
    return WeasyPrintExporter()


__all__ = ["Exporter", "find_quarto", "resolve_exporter"]
