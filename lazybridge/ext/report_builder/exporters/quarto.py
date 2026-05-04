"""Quarto-backed exporter — the primary render path.

Pipeline:

1. Render the :class:`AssembledReport` to a `.qmd` document via
   :func:`render_report_to_qmd`.
2. Write a CSL-JSON bibliography (``refs.json``) and a `_quarto.yml`
   project file alongside it.
3. Pre-rasterize Plotly chart fragments to PNG so non-HTML formats embed
   correctly (Quarto handles Vega-Lite rasterization itself).
4. Shell out to ``quarto render report.qmd --to <fmt1>,<fmt2>,...``.
5. Collect the produced files and return them keyed by format.

If Quarto exits non-zero we surface its stderr in
:class:`QuartoRenderError`.  The bus' ``backend="auto"`` path catches the
``QuartoNotFoundError`` raised on construction and falls back to
:class:`WeasyPrintExporter` automatically.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from lazybridge.ext.report_builder.assemblers import AssembledReport
from lazybridge.ext.report_builder.citations import to_csl_json
from lazybridge.ext.report_builder.exporters import Exporter
from lazybridge.ext.report_builder.quarto.detect import require_quarto
from lazybridge.ext.report_builder.quarto.project import build_quarto_yml
from lazybridge.ext.report_builder.quarto.qmd import render_report_to_qmd

_FORMAT_TO_OUTPUT_SUFFIX = {
    "html": ".html",
    "pdf": ".pdf",
    "docx": ".docx",
    "revealjs": ".html",  # reveal.js outputs as standalone HTML
}

_FORMAT_TO_QUARTO_TARGET = {
    "html": "html",
    "pdf": "typst",  # we configure Typst as the PDF engine in _quarto.yml
    "docx": "docx",
    "revealjs": "revealjs",
}


class QuartoRenderError(RuntimeError):
    """Raised when ``quarto render`` exits non-zero."""

    def __init__(self, returncode: int, stderr: str) -> None:
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"quarto render failed (exit {returncode}): {stderr.strip()}")


class QuartoExporter(Exporter):
    """Render via the ``quarto`` CLI.

    The CLI is probed eagerly on construction so ``backend="auto"``
    failover works cleanly — if Quarto isn't installed,
    :class:`QuartoNotFoundError` fires here and the bus catches it.
    """

    def __init__(self) -> None:
        self._quarto = require_quarto()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def export(
        self,
        report: AssembledReport,
        output_dir: Path,
        *,
        formats: list[str],
        theme: str = "cosmo",
    ) -> dict[str, Path]:
        valid = [f for f in formats if f in _FORMAT_TO_OUTPUT_SUFFIX]
        if not valid:
            raise ValueError(f"No supported formats in {formats!r}")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Bibliography (only emit when there are citations).
        bib_path: Path | None = None
        if report.citations:
            bib_path = output_dir / "refs.json"
            bib_path.write_text(json.dumps(to_csl_json(report.citations), indent=2), encoding="utf-8")

        # 2. _quarto.yml — declares formats, theme, citations.
        yml = build_quarto_yml(
            formats=valid,
            theme=theme,
            title=report.title,
            author=report.metadata.get("author"),
            bibliography=bib_path.name if bib_path else None,
        )
        (output_dir / "_quarto.yml").write_text(yml, encoding="utf-8")

        # 3. report.qmd
        qmd_text = render_report_to_qmd(
            report,
            bibliography_path=bib_path.name if bib_path else None,
        )
        qmd_path = output_dir / "report.qmd"
        qmd_path.write_text(qmd_text, encoding="utf-8")

        # 4. Run quarto render once per format.  Doing one invocation per
        # format keeps stderr scoped, and Quarto's own batching has rough
        # edges (e.g. it can't always pick the right Typst engine when
        # mixed with HTML in a single call).
        produced: dict[str, Path] = {}
        for fmt in valid:
            target = _FORMAT_TO_QUARTO_TARGET[fmt]
            cmd = [self._quarto, "render", str(qmd_path), "--to", target]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, check=False, cwd=str(output_dir), timeout=300
                )
            except subprocess.TimeoutExpired as exc:
                raise QuartoRenderError(-1, f"timed out after 5 minutes: {exc}") from exc
            if proc.returncode != 0:
                raise QuartoRenderError(proc.returncode, proc.stderr)

            suffix = _FORMAT_TO_OUTPUT_SUFFIX[fmt]
            # Quarto names outputs after the input stem.  Reveal.js shares
            # the .html suffix with HTML; we rename it so both formats can
            # coexist in the same directory.
            default_out = qmd_path.with_suffix(suffix)
            if fmt == "revealjs":
                final_out = output_dir / "report.revealjs.html"
                if default_out.exists():
                    default_out.rename(final_out)
                produced[fmt] = final_out
            else:
                produced[fmt] = default_out

        return produced
