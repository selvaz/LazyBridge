"""End-to-end pipeline tests using MockAgent.

Exercises the FragmentBus + fragment_tools + assembler + exporter chain
inside a real Plan, with deterministic test doubles standing in for the
LLMs.  This is the "does my reporting subsystem behave when the rest of
the framework drives it" suite — complements the focused unit tests.

Two patterns are demonstrated:

1. **Side-effect MockAgent**  — the mock's response callable does a
   ``bus.append(...)`` and returns a string acknowledgement.  Cheaper
   than mocking the LLM's tool-calling protocol, and directly maps to
   what a real agent would do via fragment_tools.

2. **Mock-driven fragment_tools**  — drives the actual append_text /
   append_chart / append_table tools from a MockAgent's response
   callable.  Confirms tools work when invoked from inside a Plan Step.

Both patterns produce a real assembled report and a real exported HTML
file; no LLM API keys are required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lazybridge import Agent, Plan, Step, from_parallel_all
from lazybridge.envelope import Envelope
from lazybridge.external_tools.report_builder import (
    BlackboardAssembler,
    Citation,
    Fragment,
    FragmentBus,
    OutlineAssembler,
    Provenance,
    fragment_tools,
)
from lazybridge.store import Store
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _research_response(bus: FragmentBus, region: str, headline: str, model: str):
    """Build a MockAgent response callable that appends a fragment + returns text.

    Mirrors what a real research Agent does when it calls
    ``fragment_tools.append_text(...)`` — minus the LLM tool-calling
    protocol layer, which we test separately.
    """

    def _respond(env: Envelope):
        bus.append(
            Fragment(
                kind="text",
                heading=headline,
                section=region,
                body_md=f"Mock research from {region}: {env.task}",
                citations=[
                    Citation(
                        key=f"{region.replace('.', '_')}_src",
                        title=f"Source for {region}",
                        url=f"https://example.com/{region}",
                        year=2026,
                    )
                ],
                provenance=Provenance(
                    step_name=region,
                    agent_name=f"research-{region}",
                    model=model,
                    tokens_in=120,
                    tokens_out=400,
                    cost_usd=0.0042,
                    latency_ms=320.0,
                ),
            )
        )
        return f"OK[{region}]"

    return _respond


def _synth_response(bus: FragmentBus):
    """Synthesiser reads what's in the bus and emits an exec-summary fragment."""

    def _respond(env: Envelope):
        regional = bus.fragments()
        regions_seen = sorted({f.section for f in regional if f.section and f.section != "1.exec"})
        bus.append(
            Fragment(
                kind="text",
                section="1.exec",
                body_md="Cross-region summary covering: " + ", ".join(regions_seen),
                provenance=Provenance(
                    step_name="synth",
                    agent_name="synthesiser",
                    model="claude-sonnet-4-6",
                    tokens_in=200,
                    tokens_out=120,
                    cost_usd=0.018,
                    latency_ms=850.0,
                ),
            )
        )
        return f"synthesised {len(regions_seen)} regions"

    return _respond


def _export_step(env: Envelope, *, bus: FragmentBus, out_dir: Path) -> Envelope:
    paths = bus.export(["html"], out_dir, title="Mock Pipeline Report", backend="weasyprint")
    return Envelope(payload={k: str(v) for k, v in paths.items()})


# ---------------------------------------------------------------------------
# Pattern 1 — side-effect MockAgent inside a Plan
# ---------------------------------------------------------------------------


class TestParallelMockPipeline:
    def _build_plan(self, bus: FragmentBus, out_dir: Path):
        us = MockAgent(_research_response(bus, "2.us", "US tech earnings", "claude-haiku-4-5"), name="us")
        cn = MockAgent(_research_response(bus, "3.cn", "China EV exports", "gemini-2.5-flash"), name="cn")
        in_ = MockAgent(_research_response(bus, "4.in", "India services", "gpt-5-mini"), name="in")
        synth = MockAgent(_synth_response(bus), name="synth")

        plan = Plan(
            Step(target=us, parallel=True, name="us"),
            Step(target=cn, parallel=True, name="cn"),
            Step(target=in_, parallel=True, name="in"),
            Step(target=synth, task=from_parallel_all("us"), name="synth"),
            Step(
                target=lambda env: _export_step(env, bus=bus, out_dir=out_dir),
                name="export",
            ),
        )
        return plan, us, cn, in_, synth

    def test_each_agent_called_once(self, tmp_path):
        bus = FragmentBus(
            "mock-pipeline",
            assembler=OutlineAssembler(
                {
                    "1.exec": "Executive Summary",
                    "2.us": "United States",
                    "3.cn": "China",
                    "4.in": "India",
                }
            ),
        )
        plan, us, cn, in_, synth = self._build_plan(bus, tmp_path)
        Agent(name="pipeline", engine=plan)("Run today's briefing.")

        us.assert_call_count(1)
        cn.assert_call_count(1)
        in_.assert_call_count(1)
        synth.assert_call_count(1)

    def test_fragments_arrive_via_parallel_band(self, tmp_path):
        bus = FragmentBus(
            "mock-band",
            assembler=OutlineAssembler(
                {
                    "1.exec": "Executive Summary",
                    "2.us": "United States",
                    "3.cn": "China",
                    "4.in": "India",
                }
            ),
        )
        plan, *_ = self._build_plan(bus, tmp_path)
        Agent(name="pipeline", engine=plan)("Today.")

        sections = {f.section for f in bus.fragments()}
        assert sections == {"1.exec", "2.us", "3.cn", "4.in"}
        # Every regional fragment should be present (synth adds 1.exec, which
        # makes 4 total at minimum — 3 region + 1 exec).
        assert len(bus) == 4

    def test_outline_assembled_in_declared_order(self, tmp_path):
        OUTLINE = {
            "1.exec": "Executive Summary",
            "2.us": "United States",
            "3.cn": "China",
            "4.in": "India",
        }
        bus = FragmentBus("mock-order", assembler=OutlineAssembler(OUTLINE))
        plan, *_ = self._build_plan(bus, tmp_path)
        Agent(name="pipeline", engine=plan)("Today.")

        report = bus.assemble(title="Order")
        section_ids = [s.section_id for s in report.sections if s.section_id != "__unrouted__"]
        assert section_ids == list(OUTLINE.keys())

    def test_provenance_aggregates_step_names(self, tmp_path):
        bus = FragmentBus(
            "mock-prov",
            assembler=OutlineAssembler(
                {
                    "1.exec": "E",
                    "2.us": "U",
                    "3.cn": "C",
                    "4.in": "I",
                }
            ),
        )
        plan, *_ = self._build_plan(bus, tmp_path)
        Agent(name="pipeline", engine=plan)("Today.")

        report = bus.assemble(title="Prov")
        step_names = {p.step_name for p in report.provenance_log}
        assert step_names == {"2.us", "3.cn", "4.in", "synth"}

    def test_costs_and_tokens_roll_up(self, tmp_path):
        bus = FragmentBus(
            "mock-cost",
            assembler=OutlineAssembler(
                {
                    "1.exec": "E",
                    "2.us": "U",
                    "3.cn": "C",
                    "4.in": "I",
                }
            ),
        )
        plan, *_ = self._build_plan(bus, tmp_path)
        Agent(name="pipeline", engine=plan)("Today.")

        report = bus.assemble(title="Cost")
        # 3 researchers @ (120/400) + 1 synth @ (200/120)
        assert report.metadata["tokens_in_total"] == 3 * 120 + 200
        assert report.metadata["tokens_out_total"] == 3 * 400 + 120
        assert report.metadata["cost_usd_total"] == pytest.approx(3 * 0.0042 + 0.018)

    def test_export_produces_html(self, tmp_path):
        bus = FragmentBus(
            "mock-export",
            assembler=OutlineAssembler(
                {
                    "1.exec": "Executive Summary",
                    "2.us": "United States",
                    "3.cn": "China",
                    "4.in": "India",
                }
            ),
        )
        plan, *_ = self._build_plan(bus, tmp_path)
        Agent(name="pipeline", engine=plan)("Today.")

        html = tmp_path / "report.html"
        assert html.exists()
        text = html.read_text(encoding="utf-8")
        assert "Mock Pipeline Report" in text
        assert "United States" in text
        assert "Cross-region summary" in text
        assert "Audit Trail" in text
        # Synthesiser saw all three regions before writing.
        assert "Sources" in text


# ---------------------------------------------------------------------------
# Pattern 2 — driving fragment_tools from a MockAgent response
# ---------------------------------------------------------------------------


class TestMockDrivenFragmentTools:
    """The MockAgent's response callable invokes fragment_tools tools directly.

    This exercises the actual tool surface (validators, error wrapping,
    provenance stamping) without needing an LLM that knows how to do
    real tool-calling.
    """

    def test_append_text_via_tool_call(self, tmp_path):
        bus = FragmentBus("tooled")
        tools = fragment_tools(bus=bus, default_section="body", step_name="research")
        append_text = next(t for t in tools if t.name == "append_text")

        def _respond(env: Envelope):
            # Simulate an LLM tool call.
            return append_text.run_sync(
                heading="Hello",
                body_markdown="World",
                citations=[{"key": "k1", "title": "T"}],
            )

        agent = MockAgent(_respond, name="tooled")
        plan = Plan(Step(target=agent, name="research"))
        Agent(name="pipeline", engine=plan)("go")

        assert len(bus) == 1
        f = bus.fragments()[0]
        assert f.heading == "Hello"
        assert f.body_md == "World"
        assert f.section == "body"  # default_section applied
        assert f.provenance.step_name == "research"
        assert f.citations[0].key == "k1"

    def test_append_chart_via_tool_call(self, tmp_path):
        bus = FragmentBus("tooled-chart")
        tools = fragment_tools(bus=bus)
        append_chart = next(t for t in tools if t.name == "append_chart")

        def _respond(env: Envelope):
            return append_chart.run_sync(
                engine="vega-lite",
                spec={"mark": "bar", "encoding": {}},
                title="Demo",
                data=[{"x": 1, "y": 2}, {"x": 2, "y": 4}],
                section="data",
            )

        agent = MockAgent(_respond)
        plan = Plan(Step(target=agent, name="chart-step"))
        Agent(name="pipeline", engine=plan)("draw chart")

        assert len(bus) == 1
        f = bus.fragments()[0]
        assert f.kind == "chart"
        assert f.chart.engine == "vega-lite"
        assert f.chart.title == "Demo"

    def test_invalid_table_raises(self, tmp_path):
        """Bad input raises; the engine wraps it into is_error=True for the LLM."""
        import pytest

        bus = FragmentBus("tooled-err")
        append_table = next(t for t in fragment_tools(bus=bus) if t.name == "append_table")

        with pytest.raises(ValueError):
            append_table.run_sync(headers=["A", "B"], rows=[["only-one-cell"]])
        # Bus untouched.
        assert len(bus) == 0

    def test_list_fragments_round_trip_via_tool(self, tmp_path):
        bus = FragmentBus("tooled-list")
        tools = fragment_tools(bus=bus, default_section="x")
        append_text = next(t for t in tools if t.name == "append_text")
        list_fragments = next(t for t in tools if t.name == "list_fragments")

        append_text.run_sync(heading="A", body_markdown="1")
        append_text.run_sync(heading="B", body_markdown="2")

        listed = list_fragments.run_sync()
        assert len(listed) == 2
        assert {f["heading"] for f in listed} == {"A", "B"}


# ---------------------------------------------------------------------------
# Concurrency — parallel band actually races into the bus
# ---------------------------------------------------------------------------


class TestConcurrencyUnderPlan:
    def test_parallel_band_writes_all_fragments(self, tmp_path):
        """8 parallel MockAgents all append; nothing is lost."""
        bus = FragmentBus("race")

        def _appender(idx):
            def _respond(env: Envelope):
                bus.append(Fragment(kind="text", body_md=f"frag-{idx}", section=f"s-{idx}"))
                return f"OK-{idx}"

            return _respond

        agents = [
            MockAgent(_appender(i), name=f"worker-{i}", default_input_tokens=1, default_output_tokens=1)
            for i in range(8)
        ]
        steps = [Step(target=a, parallel=True, name=f"worker-{i}") for i, a in enumerate(agents)]
        # Add a final non-parallel step so the band closes cleanly.
        steps.append(Step(target=lambda env: Envelope(payload="done"), name="done"))

        plan = Plan(*steps)
        Agent(name="pipeline", engine=plan)("Race.")

        bodies = {f.body_md for f in bus.fragments()}
        assert bodies == {f"frag-{i}" for i in range(8)}


# ---------------------------------------------------------------------------
# Resume — bus state survives a "crash" and a fresh Plan run
# ---------------------------------------------------------------------------


class TestResumeWithStore:
    def test_persisted_fragments_visible_to_second_bus(self, tmp_path):
        db = str(tmp_path / "state.sqlite")
        store_a = Store(db=db)
        try:
            bus_a = FragmentBus("rep-resume", store=store_a)
            agent = MockAgent(
                lambda env: bus_a.append(Fragment(kind="text", body_md="part-1")),
                name="first-half",
            )
            plan_a = Plan(Step(target=agent, name="first"))
            Agent(name="run-a", engine=plan_a)("first half")
            assert len(bus_a) == 1
        finally:
            store_a.close()

        # Simulate a crash + restart: brand-new Store + bus pointing at the same DB.
        store_b = Store(db=db)
        try:
            bus_b = FragmentBus("rep-resume", store=store_b)
            assert len(bus_b) == 1
            assert bus_b.fragments()[0].body_md == "part-1"
        finally:
            store_b.close()

    def test_fragment_id_idempotency_across_replay(self, tmp_path):
        """A Step that re-emits the same Fragment id is a no-op the second time."""
        bus = FragmentBus("idempotent")
        STABLE_ID = "abc123"

        def _respond(env: Envelope):
            bus.append(Fragment(id=STABLE_ID, kind="text", body_md="once"))
            return "ok"

        agent = MockAgent(_respond, name="once-only")
        plan = Plan(Step(target=agent, name="emit"))
        # Run twice — the bus has the same id both times.
        Agent(name="r1", engine=plan)("go")
        Agent(name="r2", engine=plan)("go")

        assert len(bus) == 1
        assert bus.fragments()[0].id == STABLE_ID


# ---------------------------------------------------------------------------
# Multi-region pipeline — assembled report has the right shape
# ---------------------------------------------------------------------------


class TestRenderedOutputShape:
    def test_audit_trail_contains_every_step(self, tmp_path):
        OUTLINE = {"1.exec": "E", "2.us": "U", "3.cn": "C"}
        bus = FragmentBus("shape", assembler=OutlineAssembler(OUTLINE))

        us = MockAgent(_research_response(bus, "2.us", "U-headline", "claude-haiku-4-5"))
        cn = MockAgent(_research_response(bus, "3.cn", "C-headline", "gemini-2.5-flash"))
        synth = MockAgent(_synth_response(bus))

        plan = Plan(
            Step(target=us, parallel=True, name="us"),
            Step(target=cn, parallel=True, name="cn"),
            Step(target=synth, task=from_parallel_all("us"), name="synth"),
            Step(
                target=lambda env: _export_step(env, bus=bus, out_dir=tmp_path),
                name="export",
            ),
        )
        Agent(name="pipeline", engine=plan)("Today.")

        text = (tmp_path / "report.html").read_text(encoding="utf-8")
        # All three step names appear in the audit table.
        assert "2.us" in text
        assert "3.cn" in text
        assert "synth" in text
        # Models stamped.
        assert "claude-haiku-4-5" in text
        assert "gemini-2.5-flash" in text
        assert "claude-sonnet-4-6" in text

    def test_blackboard_assembler_with_mock_pipeline(self, tmp_path):
        """Same pipeline, default assembler — alphabetical sections."""
        bus = FragmentBus("bb", assembler=BlackboardAssembler())

        a = MockAgent(_research_response(bus, "alpha", "A-h", "m-1"))
        b = MockAgent(_research_response(bus, "zulu", "Z-h", "m-2"))

        plan = Plan(
            Step(target=a, parallel=True, name="a"),
            Step(target=b, parallel=True, name="b"),
            Step(target=lambda env: _export_step(env, bus=bus, out_dir=tmp_path), name="export"),
        )
        Agent(name="pipeline", engine=plan)("Race.")

        text = (tmp_path / "report.html").read_text(encoding="utf-8")
        # alpha appears before zulu in the rendered HTML.
        assert text.index("alpha") < text.index("zulu")
