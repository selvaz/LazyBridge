"""Tests for fragment_tools — the LLM-facing append API."""

from __future__ import annotations

import pytest

from lazybridge.external_tools.report_builder import FragmentBus, fragment_tools


@pytest.fixture
def bus() -> FragmentBus:
    return FragmentBus("ft-test")


def _call(tool, **kwargs):
    """Invoke a Tool via its sync run path."""
    return tool.run_sync(**kwargs)


def _by_name(tools, name):
    return next(t for t in tools if t.name == name)


class TestToolList:
    def test_returns_six_tools(self, bus):
        tools = fragment_tools(bus=bus)
        names = sorted(t.name for t in tools)
        assert names == [
            "append_callout",
            "append_chart",
            "append_table",
            "append_text",
            "cite_url",
            "list_fragments",
        ]

    def test_rejects_non_bus_argument(self):
        with pytest.raises(TypeError):
            fragment_tools(bus="not a bus")  # type: ignore[arg-type]


class TestAppendText:
    def test_text_lands_in_bus(self, bus):
        tool = _by_name(fragment_tools(bus=bus, default_section="intro"), "append_text")
        result = _call(tool, heading="H", body_markdown="Body")
        assert "id" in result
        assert result["kind"] == "text"
        assert len(bus) == 1
        f = bus.fragments()[0]
        assert f.heading == "H"
        assert f.body_md == "Body"
        # default_section was applied because the call didn't specify section.
        assert f.section == "intro"

    def test_explicit_section_overrides_default(self, bus):
        tool = _by_name(fragment_tools(bus=bus, default_section="d"), "append_text")
        _call(tool, heading="x", body_markdown="y", section="elsewhere")
        assert bus.fragments()[0].section == "elsewhere"

    def test_provenance_step_name_stamped(self, bus):
        tool = _by_name(fragment_tools(bus=bus, step_name="research"), "append_text")
        _call(tool, heading="x", body_markdown="y")
        f = bus.fragments()[0]
        assert f.provenance is not None
        assert f.provenance.step_name == "research"

    def test_invalid_text_raises(self, bus):
        from pydantic import ValidationError

        tool = _by_name(fragment_tools(bus=bus), "append_text")
        with pytest.raises((ValueError, ValidationError)):
            _call(tool, heading="", body_markdown="")


class TestAppendChart:
    def test_vega_chart_lands(self, bus):
        tool = _by_name(fragment_tools(bus=bus), "append_chart")
        result = _call(
            tool,
            engine="vega-lite",
            spec={"mark": "bar"},
            title="T",
            data=[{"x": 1, "y": 2}],
        )
        assert result.get("kind") == "chart"
        assert len(bus) == 1
        f = bus.fragments()[0]
        assert f.chart.engine == "vega-lite"
        assert f.chart.title == "T"

    def test_invalid_engine_raises(self, bus):
        from pydantic import ValidationError

        tool = _by_name(fragment_tools(bus=bus), "append_chart")
        with pytest.raises((ValueError, ValidationError)):
            _call(tool, engine="bogus", spec={"mark": "bar"}, title="t")


class TestAppendTable:
    def test_table_lands(self, bus):
        tool = _by_name(fragment_tools(bus=bus), "append_table")
        result = _call(
            tool,
            headers=["a", "b"],
            rows=[["1", "2"], ["3", "4"]],
            caption="Cap",
        )
        assert result.get("kind") == "table"
        f = bus.fragments()[0]
        assert f.table.headers == ["a", "b"]
        assert f.table.caption == "Cap"

    def test_row_length_mismatch_raises(self, bus):
        tool = _by_name(fragment_tools(bus=bus), "append_table")
        with pytest.raises(ValueError, match="headers"):
            _call(tool, headers=["a", "b"], rows=[["1"]])


class TestAppendCallout:
    def test_callout_with_default_style(self, bus):
        tool = _by_name(fragment_tools(bus=bus), "append_callout")
        result = _call(tool, style="warning", body_markdown="Take care")
        assert result.get("kind") == "callout"
        f = bus.fragments()[0]
        assert f.callout_style == "warning"


class TestListFragments:
    def test_lists_all_when_no_filter(self, bus):
        tools = fragment_tools(bus=bus)
        text_tool = _by_name(tools, "append_text")
        list_tool = _by_name(tools, "list_fragments")

        _call(text_tool, heading="A", body_markdown="x", section="s1")
        _call(text_tool, heading="B", body_markdown="y", section="s2")

        all_frags = _call(list_tool)
        assert len(all_frags) == 2
        s1_only = _call(list_tool, section="s1")
        assert len(s1_only) == 1
        assert s1_only[0]["heading"] == "A"


class TestCiteUrlGracefulFallback:
    def test_returns_minimal_citation_when_no_network(self, bus):
        # We don't mock the network here — habanero/openalex calls get
        # caught by the broad except in citations.py and we fall back to a
        # minimal citation.  The test confirms the call produces *something*
        # usable rather than crashing.
        tools = fragment_tools(bus=bus)
        tool = _by_name(tools, "cite_url")
        result = _call(tool, url="https://example.com/no-such-paper-abcdef")
        # Either a Citation dict or a structured error — both shapes mean
        # the LLM can react sensibly.
        assert isinstance(result, dict)
        if "error" not in result:
            assert "key" in result
            assert "title" in result
