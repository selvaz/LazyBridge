"""Tests for BlackboardAssembler + OutlineAssembler."""

from __future__ import annotations

import pytest

from lazybridge.ext.report_builder.assemblers import (
    BlackboardAssembler,
    OutlineAssembler,
)
from lazybridge.ext.report_builder.fragments import Citation, Fragment, Provenance


def _f(body: str, **kwargs) -> Fragment:
    return Fragment(kind="text", body_md=body, **kwargs)


class TestBlackboard:
    def test_groups_by_section_alphabetically(self):
        a = BlackboardAssembler()
        out = a.assemble(
            [
                _f("a1", section="us"),
                _f("a2", section="cn"),
                _f("a3", section="us"),
                _f("a4", section="in"),
            ],
            title="t",
        )
        section_ids = [s.section_id for s in out.sections]
        # Alphabetical by section id (None first if any).
        assert section_ids == ["cn", "in", "us"]
        # us bucket has both us fragments.
        us = [s for s in out.sections if s.section_id == "us"][0]
        assert len(us.fragments) == 2

    def test_orders_within_section_by_hint_then_time(self):
        a = BlackboardAssembler()
        out = a.assemble(
            [
                _f("first", section="x", order_hint=2.0),
                _f("second", section="x", order_hint=1.0),
            ],
            title="t",
        )
        bodies = [f.body_md for f in out.sections[0].fragments]
        assert bodies == ["second", "first"]

    def test_unsectioned_at_top(self):
        a = BlackboardAssembler()
        out = a.assemble(
            [
                _f("top"),
                _f("z", section="z"),
                _f("a", section="a"),
            ],
            title="t",
        )
        # None first, then a, then z.
        section_ids = [s.section_id for s in out.sections]
        assert section_ids[0] is None
        assert section_ids[1:] == ["a", "z"]

    def test_aggregates_citations(self):
        a = BlackboardAssembler()
        f1 = _f("x", section="a", citations=[Citation(key="k1", title="T1")])
        f2 = _f("y", section="a", citations=[Citation(key="k1", title="T1-dup")])  # same key
        f3 = _f("z", section="b", citations=[Citation(key="k2", title="T2")])
        out = a.assemble([f1, f2, f3], title="t")
        keys = {c.key for c in out.citations}
        assert keys == {"k1", "k2"}
        # First-occurrence wins.
        k1 = [c for c in out.citations if c.key == "k1"][0]
        assert k1.title == "T1"

    def test_metadata_assembler_marker(self):
        out = BlackboardAssembler().assemble([_f("x")], title="t")
        assert out.metadata["assembler"] == "blackboard"
        assert out.metadata["fragment_count"] == 1


class TestOutline:
    def test_requires_non_empty_outline(self):
        with pytest.raises(ValueError):
            OutlineAssembler({})

    def test_outline_order_preserved(self):
        outline = {"1.intro": "Introduction", "2.body": "Body", "3.outro": "Outro"}
        a = OutlineAssembler(outline)
        out = a.assemble(
            [
                _f("body", section="2.body"),
                _f("intro", section="1.intro"),
                _f("outro", section="3.outro"),
            ],
            title="t",
        )
        # Top-level sections preserve outline declaration order.
        # ``__unrouted__`` might tail if anything is unrouted.
        section_ids = [s.section_id for s in out.sections if s.section_id != "__unrouted__"]
        assert section_ids == ["1.intro", "2.body", "3.outro"]

    def test_unknown_section_routed_to_unrouted(self):
        a = OutlineAssembler({"1.x": "X"})
        out = a.assemble(
            [
                _f("known", section="1.x"),
                _f("unknown1", section="9.foo"),
                _f("unknown2", section=None),
            ],
            title="t",
        )
        unrouted = [s for s in out.sections if s.section_id == "__unrouted__"]
        assert len(unrouted) == 1
        assert len(unrouted[0].fragments) == 2

    def test_outline_with_missing_node_renders_placeholder_section(self):
        outline = {"1.a": "A", "1.b": "B"}
        a = OutlineAssembler(outline)
        out = a.assemble(
            [_f("only-a", section="1.a")],
            title="t",
        )
        # Both top-level sections present even though only one had a fragment.
        ids = [s.section_id for s in out.sections]
        assert "1.a" in ids
        assert "1.b" in ids

    def test_nested_sections_become_children(self):
        outline = {"1": "Top", "1.a": "A", "1.b": "B"}
        a = OutlineAssembler(outline)
        out = a.assemble(
            [
                _f("top body", section="1"),
                _f("a body", section="1.a"),
                _f("b body", section="1.b"),
            ],
            title="t",
        )
        # Roots = top-level "1" + any unrouted (none here).
        roots = [s for s in out.sections if s.section_id == "1"]
        assert len(roots) == 1
        top = roots[0]
        child_ids = {c.section_id for c in top.children}
        assert child_ids == {"1.a", "1.b"}

    def test_metadata_includes_outline_info(self):
        outline = {"1.a": "A", "2.b": "B"}
        out = OutlineAssembler(outline).assemble(
            [_f("x", section="1.a"), _f("y", section="zzz")],
            title="t",
        )
        assert out.metadata["assembler"] == "outline"
        assert out.metadata["outline_size"] == 2
        assert out.metadata["unrouted_fragments"] == 1


class TestSummaryRollups:
    def test_provenance_aggregation_sums_costs(self):
        a = BlackboardAssembler()
        out = a.assemble(
            [
                _f("x", provenance=Provenance(tokens_in=10, tokens_out=20, cost_usd=0.001)),
                _f("y", provenance=Provenance(tokens_in=30, tokens_out=40, cost_usd=0.002)),
            ],
            title="t",
        )
        assert out.metadata["tokens_in_total"] == 40
        assert out.metadata["tokens_out_total"] == 60
        assert out.metadata["cost_usd_total"] == pytest.approx(0.003)
        assert len(out.provenance_log) == 2
