"""Fragment assembly strategies.

An :class:`Assembler` reduces a flat list of :class:`Fragment` objects into
a structured :class:`AssembledReport` that an :class:`Exporter` can render.

Two strategies ship by default:

* :class:`BlackboardAssembler` — emergent / free-form pipelines.  Groups
  fragments by their ``section`` string (alphabetically), or by
  ``created_at`` when ``section`` is unset.  Within each group, sorts by
  ``order_hint`` (ascending) then by ``created_at`` for ties.  This is the
  shape of the daily-news pipeline: regions don't know their structure
  upfront, the assembler discovers it from what the agents actually wrote.

* :class:`OutlineAssembler` — STORM-shaped structured pipelines.  Takes an
  outline ``{section_id: title}`` mapping at construction; expects
  fragments to carry the matching ``section`` ids; builds a tree.  Empty
  outline nodes get a placeholder so the exporter still emits the heading
  and the document structure survives a partial write.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field

from lazybridge.ext.report_builder.fragments import Citation, Fragment, Provenance


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class RenderedSection(BaseModel):
    """One section of the assembled report — may nest via ``children``.

    ``fragments`` holds the leaf fragments belonging directly to this
    section.  ``children`` are the nested sub-sections (Outline-shaped).
    Blackboard output has a flat tree (children always empty).
    """

    section_id: str | None = None  # the original Fragment.section value, if any
    heading: str | None = None  # rendered heading text
    level: int = 2  # h2 by default; OutlineAssembler may override
    fragments: list[Fragment] = Field(default_factory=list)
    children: list[RenderedSection] = Field(default_factory=list)


class AssembledReport(BaseModel):
    """The structured-report output of an Assembler — rendering input."""

    title: str
    sections: list[RenderedSection]
    citations: list[Citation] = Field(default_factory=list)
    provenance_log: list[Provenance] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Assembler protocol
# ---------------------------------------------------------------------------


class Assembler(Protocol):
    """Reduce a list of fragments to an :class:`AssembledReport`."""

    def assemble(self, fragments: list[Fragment], *, title: str) -> AssembledReport: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_citations(fragments: list[Fragment]) -> list[Citation]:
    """De-duplicate citations across fragments by their ``key``.

    First occurrence wins — this preserves any enriched CSL-JSON record on
    the first appearance and prevents a thinner downstream copy from
    overwriting it.
    """
    seen: dict[str, Citation] = {}
    for f in fragments:
        for c in f.citations:
            seen.setdefault(c.key, c)
    return list(seen.values())


def _aggregate_provenance(fragments: list[Fragment]) -> list[Provenance]:
    """Collect every fragment's provenance entry, preserving order."""
    return [f.provenance for f in fragments if f.provenance is not None]


def _summary_metadata(fragments: list[Fragment]) -> dict:
    """Roll up totals across the fragment population."""
    total_in = sum(f.provenance.tokens_in or 0 for f in fragments if f.provenance)
    total_out = sum(f.provenance.tokens_out or 0 for f in fragments if f.provenance)
    total_cost = sum(f.provenance.cost_usd or 0.0 for f in fragments if f.provenance)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fragment_count": len(fragments),
        "tokens_in_total": total_in,
        "tokens_out_total": total_out,
        "cost_usd_total": round(total_cost, 6),
    }


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------


class BlackboardAssembler:
    """Free-form: group by ``section`` string, sort within group by hint+time.

    Fragments without a section land in the unnamed default group at the
    top.  Section ids that look like dotted paths still get treated as flat
    strings here — use :class:`OutlineAssembler` for tree structure.
    """

    def assemble(self, fragments: list[Fragment], *, title: str = "Report") -> AssembledReport:
        groups: dict[str | None, list[Fragment]] = defaultdict(list)
        for f in fragments:
            groups[f.section].append(f)

        # Sort sections: unset (None) first, then alphabetical.
        ordered_keys: list[str | None] = []
        if None in groups:
            ordered_keys.append(None)
        ordered_keys.extend(sorted(k for k in groups if k is not None))

        sections: list[RenderedSection] = []
        for key in ordered_keys:
            group = groups[key]
            group.sort(key=lambda fr: (fr.order_hint, fr.created_at))
            heading = key  # use the section id verbatim — agents pick readable strings
            sections.append(
                RenderedSection(
                    section_id=key,
                    heading=heading,
                    level=2,
                    fragments=group,
                )
            )

        return AssembledReport(
            title=title,
            sections=sections,
            citations=_aggregate_citations(fragments),
            provenance_log=_aggregate_provenance(fragments),
            metadata={**_summary_metadata(fragments), "assembler": "blackboard"},
        )


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------


class OutlineAssembler:
    """Tree-shaped: dotted ``section`` ids fold into a hierarchy.

    The outline mapping orders top-level sections in declaration order
    (Python 3.7+ dict ordering).  Leaf sections are sorted by
    ``order_hint``.  Missing outline nodes get a placeholder so headings
    still render even when an agent wrote nothing for that node.
    """

    def __init__(self, outline: dict[str, str]) -> None:
        # ``outline`` is {dotted_id: heading}, e.g. {"1.intro": "Introduction"}.
        if not outline:
            raise ValueError("OutlineAssembler requires a non-empty outline")
        self._outline = dict(outline)

    @staticmethod
    def _depth(section_id: str) -> int:
        return section_id.count(".") + 1

    def assemble(self, fragments: list[Fragment], *, title: str = "Report") -> AssembledReport:
        # Bucket fragments by their declared section.  Unknown sections
        # (an agent invented a path the outline doesn't mention) go into a
        # synthetic ``__unrouted__`` bucket appended at the end so we never
        # silently lose content.
        buckets: dict[str, list[Fragment]] = defaultdict(list)
        unrouted: list[Fragment] = []
        for f in fragments:
            if f.section and f.section in self._outline:
                buckets[f.section].append(f)
            else:
                unrouted.append(f)

        # Build a flat list of RenderedSection in outline order (preserves the
        # caller's declared order).  Each section's ``level`` reflects depth.
        flat: list[RenderedSection] = []
        for sid, heading in self._outline.items():
            group = buckets.get(sid, [])
            group.sort(key=lambda fr: (fr.order_hint, fr.created_at))
            flat.append(
                RenderedSection(
                    section_id=sid,
                    heading=heading,
                    level=min(self._depth(sid) + 1, 6),  # h2 for top-level, h3 etc.
                    fragments=group,
                )
            )

        # Convert flat list to a tree by parent-prefix.  We rely on dotted
        # ids: "1" parents "1.1" parents "1.1.intro".  Anything whose parent
        # isn't in the outline becomes a top-level child — defensive against
        # malformed outlines.
        by_id: dict[str, RenderedSection] = {s.section_id: s for s in flat if s.section_id}
        roots: list[RenderedSection] = []
        for s in flat:
            sid = s.section_id or ""
            parent_id = sid.rsplit(".", 1)[0] if "." in sid else None
            if parent_id and parent_id in by_id:
                by_id[parent_id].children.append(s)
            else:
                roots.append(s)

        if unrouted:
            unrouted.sort(key=lambda fr: (fr.order_hint, fr.created_at))
            roots.append(
                RenderedSection(
                    section_id="__unrouted__",
                    heading="Other",
                    level=2,
                    fragments=unrouted,
                )
            )

        return AssembledReport(
            title=title,
            sections=roots,
            citations=_aggregate_citations(fragments),
            provenance_log=_aggregate_provenance(fragments),
            metadata={
                **_summary_metadata(fragments),
                "assembler": "outline",
                "outline_size": len(self._outline),
                "unrouted_fragments": len(unrouted),
            },
        )
