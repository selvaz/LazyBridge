"""Regression tests for documentation examples.

Each test constructs the exact Python code shown in a docs page to catch
regressions where an example stops parsing or stops working after a
framework change.  These tests do NOT hit a real LLM — they only exercise
the construction path (kwargs, types, imports), which is where the bugs
this session fixed actually lived (``Agent(system=...)`` raising
TypeError, ``Memory("auto")`` failing keyword-only check, etc.).

The intent is a cheap guard: if a guide shows ``X(y=z)``, the matching
test below calls exactly that and fails loudly when ``X`` changes its
signature without the doc catching up.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Literal

import pytest
from pydantic import BaseModel

from lazybridge import (
    Agent,
    CallbackExporter,
    ContentGuard,
    EdgeType,
    EventType,
    FilteredExporter,
    GraphSchema,
    Guard,
    GuardAction,
    JsonFileExporter,
    LLMEngine,
    Memory,
    Plan,
    Session,
    Step,
    Store,
    from_prev,
    from_start,
    from_step,
)
from lazybridge.core.types import CacheConfig
from lazybridge.ext.evals import EvalCase, EvalSuite, contains, llm_judge
from lazybridge.testing import MockAgent, scripted_inputs

# ---------------------------------------------------------------------------
# guides/agent.md  — main tier example + reliability + sources= live view
# ---------------------------------------------------------------------------


class _DocsSummary(BaseModel):
    title: str
    bullets: list[str]


def test_agent_guide_main_example_constructs():
    """Tier-1 through Tier-4 agent constructions from guides/agent.md."""
    Agent("claude-opus-4-7")

    def search(query: str) -> str:
        """Search the web for ``query`` and return the top 3 hits."""
        return "..."

    Agent("claude-opus-4-7", tools=[search])
    Agent("claude-opus-4-7", output=_DocsSummary)

    researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
    Agent("claude-opus-4-7", tools=[researcher], name="editor")


def test_agent_guide_reliability_kwargs():
    """Reliability kwargs example — timeout, cache, retries, output_validator."""

    class _Hits(BaseModel):
        items: list[str]

    def my_validator(hits: _Hits) -> _Hits:
        return hits

    Agent(
        engine=LLMEngine(
            "claude-opus-4-7",
            thinking=True,
            max_turns=15,
            request_timeout=60.0,
        ),
        output=_Hits,
        output_validator=my_validator,
        max_output_retries=2,
        timeout=120.0,
        cache=CacheConfig(ttl="1h"),
        max_retries=3,
        retry_delay=1.0,
        fallback=Agent("claude-opus-4-7"),
    )


def test_agent_guide_sources_live_view():
    """sources=[Store, Memory, str] — live-view context injection."""
    store = Store()
    chat_memory = Memory(strategy="sliding", max_tokens=2000)
    Agent(
        engine=LLMEngine(
            "claude-opus-4-7",
            system="You monitor the blackboard.",
        ),
        sources=[store, chat_memory, "Today is 2026-04-22."],
        name="monitor",
    )


# ---------------------------------------------------------------------------
# guides/memory.md  — main + persistent + shared memory
# ---------------------------------------------------------------------------


def test_memory_guide_kwargonly_construction():
    """Memory is keyword-only — positional should fail, keyword should succeed."""
    mem = Memory(strategy="auto", max_tokens=3000)
    assert mem.strategy == "auto"
    with pytest.raises(TypeError):
        Memory("auto", max_tokens=3000)  # type: ignore[misc]


def test_memory_guide_shared_judge_example():
    """judge agent reading chat's memory via sources=[mem] (from guides/memory.md)."""
    mem = Memory(strategy="auto", max_tokens=3000)
    Agent("claude-opus-4-7", memory=mem, name="chat")
    Agent(
        engine=LLMEngine(
            "claude-opus-4-7",
            system="Grade the assistant's last reply on helpfulness 1-5.",
        ),
        name="judge",
        sources=[mem],
    )


def test_memory_guide_persistent_store():
    """Memory(store=...) persistence path — survives restarts via the Store."""
    store = Store()
    Memory(strategy="auto", max_tokens=4000, store=store)


def test_memory_guide_manual_methods():
    """add(), messages(), text(), clear() — explicit manual-use example."""
    mem = Memory(strategy="auto", max_tokens=3000)
    mem.add(user="Hi, I'm Marco", assistant="Nice to meet you, Marco.")
    mem.add(user="I work on energy grids", assistant="Interesting.")
    assert mem.messages()  # list[Message]
    assert isinstance(mem.text(), str)
    mem.clear()
    assert mem.text() == ""


# ---------------------------------------------------------------------------
# guides/store.md  — provenance + compare_and_swap + context manager
# ---------------------------------------------------------------------------


def test_store_guide_provenance_example():
    """read / read_entry / read_all / to_text(keys=) — provenance chain."""
    store = Store()
    store.write("hits", ["p1", "p2"], agent_id="researcher")
    store.write("ranked", ["p2", "p1"], agent_id="ranker")

    assert store.read("hits") == ["p1", "p2"]
    entry = store.read_entry("hits")
    assert entry.agent_id == "researcher"
    assert store.read_all() == {"hits": ["p1", "p2"], "ranked": ["p2", "p1"]}
    assert "hits" in store.to_text(keys=["hits"])


def test_store_guide_cas_increment_loop():
    """compare_and_swap — the increment-without-lost-update loop from guides/store.md."""
    store = Store()
    store.write("counter", 0)

    def increment(key: str) -> int:
        while True:
            current = store.read(key, default=0)
            new = current + 1
            if store.compare_and_swap(key, current, new):
                return new

    assert increment("counter") == 1
    assert increment("counter") == 2
    # expected=None semantics — only succeed when key is missing.
    assert store.compare_and_swap("fresh_key", expected=None, new="held") is True
    assert store.compare_and_swap("fresh_key", expected=None, new="again") is False


def test_store_guide_context_manager():
    """with Store(...) as s — context manager closes connections on exit."""
    with Store() as store:
        store.write("seed", 42)
        assert store.read("seed") == 42


# ---------------------------------------------------------------------------
# guides/session.md  — events.query + usage_summary + custom emit + ctx mgr
# ---------------------------------------------------------------------------


def test_session_guide_emit_and_query():
    """emit + events.query(event_type=) round-trip."""
    with Session() as sess:
        sess.emit(
            EventType.HIL_DECISION,
            {"agent_name": "test", "kind": "input", "command": "x"},
        )
        rows = sess.events.query(event_type=EventType.HIL_DECISION)
        assert len(rows) == 1
        assert rows[0]["payload"]["kind"] == "input"


def test_session_guide_redact_strict_mode():
    """Session(redact_on_error='strict') construction."""
    Session(redact=lambda p: p, redact_on_error="strict")
    Session(redact=lambda p: p, redact_on_error="fallback")


def test_session_guide_usage_summary_shape():
    """usage_summary() returns the {total, by_agent, by_run} three-level shape."""
    sess = Session()
    summary = sess.usage_summary()
    assert set(summary) == {"total", "by_agent", "by_run"}
    assert set(summary["total"]) == {"input_tokens", "output_tokens", "cost_usd"}


# ---------------------------------------------------------------------------
# guides/plan.md  — typed hand-offs + routing + on_concurrent=fork
# ---------------------------------------------------------------------------


def test_plan_guide_typed_handoffs():
    """Plan with input=/output= across steps compiles at Agent construction."""

    class Hits(BaseModel):
        items: list[str]
        next: Literal["rank", "empty"] = "rank"

    class Ranked(BaseModel):
        top: list[str]

    searcher = MockAgent(Hits(items=["p1"]), name="searcher", output=Hits)
    ranker = MockAgent(Ranked(top=["p1"]), name="ranker", output=Ranked)
    writer = MockAgent("draft", name="writer")
    apology = MockAgent("sorry", name="apology")

    plan = Plan(
        Step(searcher, name="search", output=Hits),
        Step(ranker, name="rank", input=Hits, output=Ranked),
        Step(writer, name="write", input=Ranked),
        Step(apology, name="empty"),
    )
    Agent.from_engine(plan)


def test_plan_guide_on_concurrent_fork():
    """Plan(on_concurrent='fork') construction (resume=True must not be set)."""
    store = Store()
    a = MockAgent("a", name="a")
    b = MockAgent("b", name="b")
    Plan(
        Step(a, name="search"),
        Step(b, name="write"),
        store=store,
        checkpoint_key="batch_job",
        on_concurrent="fork",
    )


def test_plan_guide_fork_rejects_resume():
    """on_concurrent='fork' + resume=True is rejected at construction."""
    store = Store()
    a = MockAgent("a", name="a")
    with pytest.raises(ValueError):
        Plan(
            Step(a, name="s"),
            store=store,
            checkpoint_key="k",
            on_concurrent="fork",
            resume=True,
        )


# ---------------------------------------------------------------------------
# guides/sentinels.md  — four-step plan with three sentinel kinds
# ---------------------------------------------------------------------------


def test_sentinels_guide_four_step_plan():
    """Sentinels example: from_prev (default), from_start, from_step (task+context)."""

    class Hits(BaseModel):
        items: list[str]

    researcher = MockAgent(Hits(items=["p1"]), name="researcher", output=Hits)
    fact_checker = MockAgent("checked", name="fact_checker")
    writer = MockAgent("draft", name="writer")
    editor = MockAgent("final", name="editor")

    Plan(
        Step(researcher, name="research", output=Hits),
        Step(fact_checker, name="check", task=from_prev),
        Step(writer, name="write", task=from_start),
        Step(
            editor,
            name="edit",
            task=from_step("write"),
            context=from_step("check"),
        ),
    )


# ---------------------------------------------------------------------------
# guides/guards.md  — modify pattern + async subclass + metadata
# ---------------------------------------------------------------------------


def test_guards_guide_modify_and_chain():
    """ContentGuard + GuardChain with modify() rewrites + metadata blocking."""
    import re

    _SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")

    def mask_secrets(text: str) -> GuardAction:
        masked, n = _SECRET_RE.subn("<redacted>", text)
        if n:
            return GuardAction.modify(masked, message=f"masked {n}")
        return GuardAction.allow()

    def trim_boilerplate(text: str) -> GuardAction:
        cleaned = text.split("\n\nDisclaimer:", 1)[0]
        if cleaned != text:
            return GuardAction.modify(cleaned)
        return GuardAction.allow()

    Agent(
        "claude-opus-4-7",
        guard=ContentGuard(input_fn=mask_secrets, output_fn=trim_boilerplate),
        name="scrubbed",
    )


def test_guards_guide_async_subclass_compiles():
    """Guard subclass overriding acheck_input / acheck_output."""

    class AllowlistGuard(Guard):
        async def acheck_input(self, text: str) -> GuardAction:
            return GuardAction.allow()

        async def acheck_output(self, text: str) -> GuardAction:
            return GuardAction.allow()

    Agent("claude-opus-4-7", guard=AllowlistGuard())


# ---------------------------------------------------------------------------
# guides/evals.md  — arun + two-arg checks + llm_judge
# ---------------------------------------------------------------------------


def test_evals_guide_arun_runs_concurrently():
    """suite.arun(agent) returns EvalReport after gathering all cases."""
    agent = MockAgent("4", name="agent")
    suite = EvalSuite(
        EvalCase("what is 2+2?", check=contains("4")),
        EvalCase("is 4 there?", check=contains("4")),
    )
    report = asyncio.run(suite.arun(agent))
    assert report.passed == report.total == 2


def test_evals_guide_two_arg_check_signature():
    """Two-arg check(output, expected) resolves via arity detection."""

    def jaccard_similar(output: str, expected: str) -> bool:
        a = set(output.lower().split())
        b = set(expected.lower().split())
        return len(a & b) / max(len(a | b), 1) >= 0.7

    suite = EvalSuite(
        EvalCase("q", expected="hello world foo", check=jaccard_similar),
    )
    # Construction alone — execution requires a real agent.
    assert suite


def test_evals_guide_llm_judge_factory():
    """llm_judge(agent, criteria) returns a check callable."""
    judge = MockAgent("approved", name="judge")
    check = llm_judge(judge, "must be helpful")
    assert callable(check)


# ---------------------------------------------------------------------------
# guides/exporters.md  — custom class + filtered compose
# ---------------------------------------------------------------------------


def test_exporters_guide_custom_class():
    """Custom BatchExporter with close() contract + Session usage."""

    class BatchExporter:
        def __init__(self, sink, batch_size=100):
            self._sink, self._buf, self._batch_size = sink, [], batch_size

        def export(self, event):
            self._buf.append(event)
            if len(self._buf) >= self._batch_size:
                self._flush()

        def close(self):
            self._flush()

        def _flush(self):
            if self._buf:
                self._sink(self._buf)
                self._buf.clear()

    collected = []
    sess = Session(exporters=[BatchExporter(collected.append, batch_size=5)])
    sess.emit(EventType.AGENT_START, {"agent_name": "x"})
    sess.close()
    # close() must flush even a partial batch.
    assert collected


def test_exporters_guide_filtered_compose():
    """FilteredExporter wraps CallbackExporter + JsonFileExporter side-by-side."""
    received = []

    def on_event(e):
        received.append(e)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        path = tmp.name
    try:
        sess = Session(
            exporters=[
                JsonFileExporter(path=path),
                FilteredExporter(
                    inner=CallbackExporter(fn=on_event),
                    event_types={EventType.TOOL_ERROR, EventType.AGENT_FINISH},
                ),
            ],
        )
        sess.emit(EventType.AGENT_START, {"agent_name": "x"})
        sess.emit(EventType.AGENT_FINISH, {"agent_name": "x"})
        sess.close()
        # Filter: AGENT_FINISH passes, AGENT_START doesn't.
        assert {e["event_type"] for e in received} == {str(EventType.AGENT_FINISH)}
    finally:
        import os

        os.unlink(path)


# ---------------------------------------------------------------------------
# guides/graph-schema.md  — edge kinds + router via duck type + roundtrip
# ---------------------------------------------------------------------------


def test_graph_schema_guide_edges_and_roundtrip(tmp_path):
    """Adding CONTEXT/ROUTER edges + save/from_file roundtrip."""
    sess = Session()
    researcher = Agent("claude-opus-4-7", name="researcher")
    writer = Agent("claude-opus-4-7", name="writer")
    monitor = Agent("claude-opus-4-7", name="monitor", session=sess)
    Agent(
        "claude-opus-4-7",
        name="orchestrator",
        tools=[researcher, writer],
        session=sess,
    )
    sess.graph.add_edge(monitor.name, "orchestrator", label="observes", kind=EdgeType.CONTEXT)
    sess.graph.add_edge("orchestrator", writer.name, label="if publish=true", kind=EdgeType.ROUTER)

    class _Router:
        def __init__(self, name):
            self.name = name

        def to_graph_node(self):
            return {
                "id": self.name,
                "name": self.name,
                "routes": {"bug": researcher.name, "feature": writer.name},
                "default": researcher.name,
            }

    sess.graph.add_router(_Router("intake_router"))

    path = str(tmp_path / "topology.json")
    sess.graph.save(path)
    reloaded = GraphSchema.from_file(path)
    assert len(reloaded.nodes()) >= 4

    sess.graph.clear()
    assert len(sess.graph.nodes()) == 0


# ---------------------------------------------------------------------------
# guides/testing.md  — MockAgent dict + scripted_inputs
# ---------------------------------------------------------------------------


def test_testing_guide_mockagent_dict_match():
    """Dict responses with substring-match keys + '*' catch-all."""
    m = MockAgent(
        {"weather": "sunny", "market": "bullish", "*": "no data"},
        name="researcher",
    )
    assert m("what about the weather today?").text() == "sunny"
    assert m("market up?").text() == "bullish"
    assert m("arbitrary").text() == "no data"


def test_testing_guide_scripted_inputs_callable():
    """scripted_inputs returns a sync Callable[[str], str]."""
    fn = scripted_inputs(["continue", "store policy"])
    assert fn("> ") == "continue"
    assert fn("> ") == "store policy"


# ---------------------------------------------------------------------------
# guides/litellm.md  — prefix routing + construction via stub
# ---------------------------------------------------------------------------


def test_litellm_guide_prefix_routes_to_bridge(monkeypatch):
    """litellm/ prefix routes through LiteLLMProvider, not a native adapter."""
    # Purge cached imports so a fresh stub is picked up.
    for mod in list(sys.modules):
        if mod == "litellm" or mod.endswith("providers.litellm"):
            sys.modules.pop(mod, None)

    stub = types.ModuleType("litellm")
    stub.completion = lambda **kw: None
    stub.acompletion = lambda **kw: None
    stub.drop_params = False
    stub.set_verbose = False
    monkeypatch.setitem(sys.modules, "litellm", stub)

    assert LLMEngine._infer_provider("litellm/groq/llama-3.3-70b") == "litellm"
    # Native routing stays intact.
    assert LLMEngine._infer_provider("claude-opus-4-7") == "anthropic"
