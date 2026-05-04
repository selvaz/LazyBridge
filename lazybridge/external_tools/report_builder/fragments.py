"""Fragment schema for the parallel-assembly reporting pipeline.

A *fragment* is one piece a Step contributes to a report.  Steps in a Plan
emit fragments via :func:`fragment_tools` (LLM-callable) or directly via
:meth:`FragmentBus.append` (Python-callable).  At the end of the pipeline
an :class:`Assembler` recombines the fragments into a single report tree
that the exporters render to HTML / PDF / DOCX / Reveal.js.

Design notes
------------

* One discriminated ``kind`` field keeps the assembler simple; everything
  else is an additional payload field on the same model.  Pydantic v2
  validates the union at construction time.
* Citations are carried per-fragment as full objects (not bare keys) so a
  single fragment is self-contained and can be merged into a CSL-JSON
  bibliography at assembly time without round-trips.  Inline references
  inside ``body_md`` use Pandoc's ``[@citation_key]`` syntax.
* :class:`Provenance` records *who produced this fragment*, mirroring
  :class:`lazybridge.envelope.EnvelopeMetadata` so the assembler can
  surface a per-fragment audit trail (model, tokens, cost, latency).
* Chart specs are stored as raw JSON dicts.  We never ``exec`` Python.
  Vega-Lite is the default; Plotly is opt-in.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A bibliographic source attached to a fragment.

    The ``key`` is a BibTeX-shaped identifier emitted into Markdown via
    Pandoc citeproc syntax, e.g. ``As shown in [@smith2024]``.  The CSL-JSON
    record (``csl``) is what Pandoc actually consumes — we generate it
    from the structured fields when missing, or accept a pre-enriched dict
    from :func:`enrich_from_url`.
    """

    key: str = Field(description="BibTeX-shaped citation key, e.g. 'smith2024'.")
    title: str
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    accessed: datetime | None = None
    csl: dict | None = Field(
        default=None,
        description=(
            "Full CSL-JSON record (Pandoc citeproc input).  Set automatically "
            "by enrich_from_url(); users rarely populate this directly."
        ),
    )

    def to_csl_json(self) -> dict:
        """Return a CSL-JSON dict suitable for a Pandoc bibliography file.

        If ``csl`` was pre-populated (e.g. by Crossref enrichment) we trust
        it and merely ensure ``id`` matches our key.  Otherwise we synthesise
        a minimal record from the structured fields.
        """
        if self.csl:
            from_csl: dict = dict(self.csl)
            from_csl["id"] = self.key
            return from_csl
        record: dict = {
            "id": self.key,
            "type": "webpage" if self.url and not self.doi else "article",
            "title": self.title,
        }
        if self.authors:
            record["author"] = [{"literal": a} for a in self.authors]
        if self.year is not None:
            record["issued"] = {"date-parts": [[self.year]]}
        if self.url:
            record["URL"] = self.url
        if self.doi:
            record["DOI"] = self.doi
        if self.accessed:
            record["accessed"] = {"date-parts": [[self.accessed.year, self.accessed.month, self.accessed.day]]}
        return record


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Audit trail for a single fragment — who/what produced it.

    Field shape mirrors :class:`lazybridge.envelope.EnvelopeMetadata` so the
    bus can stamp this directly from an Envelope when a step volunteers one.
    All fields are optional — Python-side append calls without an
    Envelope just get ``timestamp``.
    """

    step_name: str | None = None
    agent_name: str | None = None
    model: str | None = None
    provider: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Chart / Table payloads
# ---------------------------------------------------------------------------


class ChartSpec(BaseModel):
    """Engine-agnostic chart payload.

    ``spec`` is the raw JSON spec as the engine consumes it (Vega-Lite or
    Plotly figure).  ``data`` is optional inline tabular data that gets
    spliced into the spec at render time — convenient for LLMs that want to
    keep the spec generic and supply numbers separately.
    """

    engine: Literal["vega-lite", "plotly"] = "vega-lite"
    spec: dict = Field(description="Raw chart spec JSON (Vega-Lite v5 or Plotly figure).")
    data: list[dict] | None = Field(
        default=None,
        description="Optional inline data rows. If provided, replaces spec.data.values for Vega-Lite or fig.data[0].* for Plotly.",
    )
    title: str = ""


class TableSpec(BaseModel):
    """Simple tabular data payload — rendered as a Pandoc pipe table or HTML."""

    headers: list[str]
    rows: list[list[str]]
    caption: str = ""


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------


class Fragment(BaseModel):
    """One piece of a report contributed by a Step.

    The discriminator is :attr:`kind`.  The assembler reads the discriminator
    to decide which payload field to render; all other fields stay populated
    or ``None`` based on kind.  This keeps the wire format flat and keeps
    LLM tool schemas easy to author.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: Literal["text", "chart", "table", "callout"]

    # Where in the report this fragment belongs.  ``section`` is either a
    # free-form tag (BlackboardAssembler groups by it alphabetically) or a
    # dotted path like ``"1.2.intro"`` (OutlineAssembler builds a tree
    # from these).  ``order_hint`` orders within the same section.
    section: str | None = None
    heading: str | None = None
    order_hint: float = 0.0

    # Payloads — exactly one of (body_md / chart / table) is meaningful for
    # a given kind.  ``callout`` uses ``body_md`` and ``callout_style``.
    body_md: str | None = None
    chart: ChartSpec | None = None
    table: TableSpec | None = None
    callout_style: Literal["note", "tip", "important", "warning", "caution"] | None = None

    # Per-fragment metadata.
    citations: list[Citation] = Field(default_factory=list)
    provenance: Provenance | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _check_payload_matches_kind(self) -> Fragment:
        """Reject obviously-wrong combinations.

        We're permissive — extra fields are tolerated to keep the schema
        forgiving for LLM-produced JSON that includes spurious nulls — but
        we DO insist that the active payload matches the kind.
        """
        if self.kind in ("text", "callout") and not (self.body_md and self.body_md.strip()):
            raise ValueError(f"Fragment kind={self.kind!r} requires non-empty body_md")
        if self.kind == "chart" and self.chart is None:
            raise ValueError("Fragment kind='chart' requires a 'chart' field")
        if self.kind == "table" and self.table is None:
            raise ValueError("Fragment kind='table' requires a 'table' field")
        if self.kind == "callout" and self.callout_style is None:
            self.callout_style = "note"
        return self
