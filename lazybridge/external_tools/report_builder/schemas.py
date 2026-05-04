"""Pydantic schemas for report_builder."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Typed content blocks (structured input alternative to markdown_path)
# ---------------------------------------------------------------------------


class TextSection(BaseModel):
    """A narrative section with an optional heading and Markdown body."""

    type: Literal["text"] = "text"
    heading: str = Field(default="", description="Section heading rendered as <h2>. Leave empty to omit.")
    body: str = Field(description="Section body in Markdown format.")


class ChartSection(BaseModel):
    """A chart embedded from a pre-generated PNG file."""

    type: Literal["chart"] = "chart"
    path: str = Field(description="Path to an existing PNG chart file.")
    title: str = Field(description="Caption displayed under the chart.")
    heading: str = Field(default="", description="Optional <h3> heading shown above the chart figure.")


class TableSection(BaseModel):
    """A data table defined as headers + rows."""

    type: Literal["table"] = "table"
    caption: str = Field(default="", description="Optional caption shown above the table.")
    headers: list[str] = Field(description="Column headers.")
    rows: list[list[str]] = Field(description="Table rows — each a list of cell strings (same length as headers).")


ReportSection = Annotated[Union[TextSection, ChartSection, TableSection], Field(discriminator="type")]


class ChartRef(BaseModel):
    """Reference to an existing chart image file.

    The PNG file is produced by an upstream charting tool and
    passed to generate_report as a file path.  The report_builder reads the
    file, base64-encodes it, and embeds it inline in the HTML.

    Auto-placement: the ``name`` field is matched against section headings
    in the rendered Markdown.  The chart is inserted immediately after the
    best-matching heading.  Unmatched charts are appended at the end.
    """

    path: str = Field(description="Absolute or relative path to an existing PNG file")
    title: str = Field(description="Caption displayed under the chart image")
    name: str = Field(
        default="",
        description=(
            "Logical name used for heading-based auto-placement. "
            "Should match (or overlap with) a section heading in the Markdown. "
            "If empty, the chart title is used for matching."
        ),
    )

    @property
    def match_name(self) -> str:
        return self.name if self.name.strip() else self.title


class ReportResult(BaseModel):
    """Result returned by generate_report on success."""

    html_path: str | None = None
    pdf_path: str | None = None
    title: str
    charts_embedded: int
    theme: str
    template: str
