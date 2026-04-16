"""Unit tests for LazySession — mock agents, no API calls."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from lazybridge.lazy_session import LazySession
from lazybridge.lazy_store import LazyStore
from lazybridge.lazy_tool import LazyTool
from lazybridge.core.types import CompletionResponse, UsageStats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_response(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=UsageStats())


def _mock_agent(name: str, response_content: str) -> MagicMock:
    """Create a minimal mock LazyAgent-like object (standalone, no session)."""
    agent = MagicMock()
    agent.name = name
    agent._last_output = None
    agent.output_schema = None
    agent.tools = []
    agent.native_tools = []
    agent.session = None  # standalone — MagicMock would auto-generate a truthy value
    resp = _fake_response(response_content)

    def _chat(task, **kwargs):
        agent._last_output = resp.content
        return resp

    async def _achat(task, **kwargs):
        agent._last_output = resp.content
        return resp

    agent.chat = MagicMock(side_effect=_chat)
    agent.achat = MagicMock(side_effect=_achat)
    agent.json = MagicMock(return_value={"mocked": True})
    agent.ajson = AsyncMock(return_value={"mocked": True})
    # loop/aloop not set by default — tests that need them must set them explicitly
    agent.loop = MagicMock()
    agent.aloop = AsyncMock()
    return agent


def _mock_tool(name: str, return_value: str) -> MagicMock:
    """Create a minimal mock LazyTool-like object."""
    tool = MagicMock(spec=LazyTool)
    tool.name = name

    async def _arun(args):
        return return_value

    tool.run = MagicMock(return_value=return_value)
    tool.arun = MagicMock(side_effect=_arun)
    # LazyTool has chat attribute? No. Use hasattr to distinguish from agent.
    del tool.chat   # ensure hasattr(tool, "chat") is False
    del tool.achat
    tool._delegate = None  # spec=LazyTool would auto-generate _delegate as truthy MagicMock
    return tool


# ── T4.01 — as_tool: invalid combiner raises ValueError at creation time ──────

def test_as_tool_invalid_combiner():
    sess = LazySession()
    with patch("lazybridge.core.executor.Executor.execute"):
        a = _mock_agent("a", "output")
    with pytest.raises(ValueError, match="combiner"):
        sess.as_tool("t", "d", mode="parallel",
                     combiner="invalid", participants=[a])


# ── T4.02 — as_tool: empty participants raises ValueError ────────────────────

def test_as_tool_no_participants():
    sess = LazySession()
    with pytest.raises(ValueError, match="No participants"):
        sess.as_tool("t", "d", mode="parallel", participants=[])


# ── T4.03 — as_tool: invalid mode raises ValueError ──────────────────────────

def test_as_tool_invalid_mode():
    sess = LazySession()
    a = _mock_agent("a", "x")
    with pytest.raises(ValueError, match="Unknown mode"):
        sess.as_tool("t", "d", mode="flyby", participants=[a])


# ── T4.04 — _register_agent: agent appended to _agents, log set ──────────────

def test_register_agent():
    sess = LazySession()
    assert sess._agents == []

    # Patch executor so LazyAgent construction doesn't need an API key
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None), \
         patch("lazybridge.core.executor.Executor.execute"):
        from lazybridge.lazy_agent import LazyAgent
        agent = LazyAgent.__new__(LazyAgent)
        # Manually set minimum required state
        import uuid
        agent.id = str(uuid.uuid4())
        agent.name = "test_agent"
        agent.description = None
        agent.system = None
        agent.context = None
        agent.tools = []
        agent.native_tools = []
        agent.output_schema = None
        agent._last_output = None
        agent._executor = MagicMock()
        agent.session = sess
        agent._log = None
        sess._register_agent(agent)

    assert len(sess._agents) == 1
    assert sess._agents[0] is agent


# ── T4.05 — store shared between agents in the same session ───────────────────

def test_shared_store():
    sess = LazySession()
    assert isinstance(sess.store, LazyStore)
    # Two references to the same session share the same store
    sess.store.write("shared_key", "shared_value")
    assert sess.store.read("shared_key") == "shared_value"


# ── T4.06 — parallel mode: both agents called, result concatenated ────────────

def test_parallel_concat(fake_response):
    sess = LazySession()
    a = _mock_agent("alpha", "result_alpha")
    b = _mock_agent("beta",  "result_beta")

    tool = sess.as_tool("t", "d", mode="parallel",
                        participants=[a, b], combiner="concat")
    result = tool.run({"task": "do something"})

    assert "[alpha]" in result
    assert "result_alpha" in result
    assert "[beta]" in result
    assert "result_beta" in result


# ── T4.07 — parallel mode: combiner="last" returns only last result ───────────

def test_parallel_last(fake_response):
    sess = LazySession()
    a = _mock_agent("first",  "first_output")
    b = _mock_agent("second", "second_output")

    tool = sess.as_tool("t", "d", mode="parallel",
                        participants=[a, b], combiner="last")
    result = tool.run({"task": "task"})

    assert "second_output" in result
    assert "first_output" not in result


# ── T4.08 — chain mode: agent B receives agent A's output via context ─────────

def test_chain_agent_to_agent():
    sess = LazySession()
    a = _mock_agent("A", "A_output")
    b = _mock_agent("B", "B_output")

    tool = sess.as_tool("t", "d", mode="chain", participants=[a, b])
    result = tool.run({"task": "original_task"})

    # chain is async-under-the-hood — achat() is called, not chat()
    a.achat.assert_called_once()
    b.achat.assert_called_once()
    call_kwargs = b.achat.call_args[1]
    assert "context" in call_kwargs
    # Final output is B's output
    assert result == "B_output"


# ── T4.09 — chain mode: LazyTool output becomes next agent's task ─────────────

def test_chain_tool_to_agent():
    sess = LazySession()
    mock_tool = _mock_tool("data_fetcher", "fetched_data_from_tool")
    agent = _mock_agent("analyser", "analysis_output")

    tool = sess.as_tool("t", "d", mode="chain",
                        participants=[mock_tool, agent])
    result = tool.run({"task": "original"})

    # chain is async-under-the-hood — achat() is called, not chat()
    agent.achat.assert_called_once()
    call_args = agent.achat.call_args[0]
    assert call_args[0] == "fetched_data_from_tool"
    assert result == "analysis_output"


# ── T4.10 — chain mode: agent with output_schema → json() called ─────────────

def test_chain_output_schema_calls_json():
    from pydantic import BaseModel

    class MySchema(BaseModel):
        value: str

    sess = LazySession()
    agent = _mock_agent("structured_agent", "ignored")
    agent.output_schema = MySchema
    parsed = MySchema(value="structured_result")
    agent.ajson = AsyncMock(return_value=parsed)

    tool = sess.as_tool("t", "d", mode="chain", participants=[agent])
    tool.run({"task": "give me structure"})

    # chain is async-under-the-hood — ajson() is called, not json()
    agent.ajson.assert_called_once()
    agent.achat.assert_not_called()


# ── T4.11 — gather: one failing coroutine returns exception, others complete ──
#
# Audit finding: asyncio.gather() without return_exceptions=True causes other
# running coroutines to be silently abandoned when one raises. The fix is to
# use return_exceptions=True so all results (including exceptions) are returned.

# ── T4.12 — parallel tool: one failing agent must not crash the whole tool ────
#
# Bug: asyncio.gather(*coros) inside _run_parallel has no return_exceptions=True.
# If one agent raises, the entire parallel tool raises instead of returning
# partial results. LazySession.gather() was fixed in the first audit pass;
# as_tool(mode="parallel") was missed.

async def test_parallel_tool_partial_failure_does_not_raise():
    """as_tool(mode="parallel") must return partial results when one agent fails,
    not raise. The successful agent's output must appear in the result."""
    sess = LazySession()
    good = _mock_agent("good_agent", "good_output")

    bad = MagicMock()
    bad.name = "bad_agent"
    bad.output_schema = None
    bad.tools = []
    bad.native_tools = []
    bad.session = None  # standalone — prevent cross-session validation error

    async def _fail(task, **kwargs):
        raise RuntimeError("agent exploded")

    bad.achat = MagicMock(side_effect=_fail)

    tool = sess.as_tool("parallel_t", "d", mode="parallel",
                        participants=[good, bad], combiner="concat")

    # Must NOT raise — partial failure should be absorbed
    result = tool.run({"task": "run both"})
    assert "good_output" in result


# ── T4.13 — parallel tool: error marker in output when agent fails ────────────
#
# When an agent fails in parallel mode, the tool returns an [ERROR: ...] marker
# in the output string rather than raising.  This test documents the exact
# observable behaviour so a future refactor cannot silently change it.

async def test_parallel_tool_failure_produces_error_marker():
    """A failed agent produces an [ERROR: ...] string in the tool output."""
    sess = LazySession()
    bad = MagicMock()
    bad.name = "bad_agent"
    bad.output_schema = None
    bad.tools = []
    bad.native_tools = []
    bad.session = None  # standalone — prevent cross-session validation error

    async def _fail(task, **kwargs):
        raise ValueError("something went wrong")

    bad.achat = MagicMock(side_effect=_fail)

    tool = sess.as_tool("t", "d", mode="parallel",
                        participants=[bad], combiner="concat")
    result = tool.run({"task": "go"})

    assert "[ERROR:" in result
    assert "ValueError" in result
    assert "something went wrong" in result


async def test_gather_returns_exceptions_not_raises():
    """gather() must not raise when one coroutine fails; it must return the
    exception in the results list so callers can handle it explicitly."""
    import asyncio
    sess = LazySession()

    async def good():
        return "ok"

    async def bad():
        raise RuntimeError("boom")

    results = await sess.gather(good(), bad())

    # With return_exceptions=True: both complete, exception is in results
    assert len(results) == 2
    ok_results = [r for r in results if r == "ok"]
    err_results = [r for r in results if isinstance(r, Exception)]
    assert ok_results == ["ok"]
    assert len(err_results) == 1
    assert "boom" in str(err_results[0])


# ── T4.14 — TrackLevel.FULL accepted as valid tracking level ─────────────────

def test_tracklevel_full_accepted():
    # T4.14
    from lazybridge.lazy_session import TrackLevel
    sess = LazySession(tracking=TrackLevel.FULL)
    assert sess.events.level == TrackLevel.FULL


# ── T4.15 — TrackLevel.FULL is synonym for VERBOSE ───────────────────────────

def test_tracklevel_full_is_verbose_synonym():
    # T4.15
    from lazybridge.lazy_session import TrackLevel
    assert TrackLevel.FULL == "full"
    assert TrackLevel.VERBOSE == "verbose"
    assert TrackLevel.FULL != TrackLevel.VERBOSE  # different strings, same behaviour


# ── T4.16 — console=True on LazySession enables console on EventLog ──────────

def test_console_flag_sets_event_log_console():
    # T4.16
    sess = LazySession(tracking="basic", console=True)
    assert sess.events._console is True


# ── T4.17 — verbose=True on LazyAgent enables console on session EventLog ────

def test_verbose_agent_enables_session_console():
    # T4.17
    from lazybridge.lazy_agent import LazyAgent
    from unittest.mock import patch

    sess = LazySession(tracking="basic", console=False)
    assert sess.events._console is False

    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent("anthropic", name="verbose_agent", session=sess, verbose=True)

    # verbose=True on a session agent should flip console to True
    assert sess.events._console is True


# ── T4.18 — chain: last agent with output_schema returns Pydantic object ──────

def test_chain_last_agent_output_schema_returns_pydantic(tmp_path):
    """pipeline.run() must return the Pydantic object, not a JSON string."""
    from pydantic import BaseModel

    class Report(BaseModel):
        title: str
        body: str

    sess = LazySession()
    agent = _mock_agent("reporter", "ignored")
    agent.output_schema = Report
    expected = Report(title="AI Today", body="Some content")
    agent.ajson = AsyncMock(return_value=expected)  # chain is async-under-the-hood

    tool = sess.as_tool("t", "d", mode="chain", participants=[agent])
    result = tool.run({"task": "write a report"})

    # Must be the Pydantic object, not a string
    assert isinstance(result, Report)
    assert result.title == "AI Today"
    assert result.body == "Some content"


# ── T4.19 — chain: intermediate agent with output_schema, last without ────────

def test_chain_intermediate_schema_last_plain():
    """When only an intermediate agent has output_schema, return is a plain string."""
    from pydantic import BaseModel

    class Mid(BaseModel):
        data: str

    sess = LazySession()
    mid_agent = _mock_agent("mid", "ignored")
    mid_agent.output_schema = Mid
    mid_agent.ajson = AsyncMock(return_value=Mid(data="mid_result"))  # chain uses ajson

    last_agent = _mock_agent("last", "final_text")
    last_agent.output_schema = None

    tool = sess.as_tool("t", "d", mode="chain", participants=[mid_agent, last_agent])
    result = tool.run({"task": "go"})

    assert isinstance(result, str)
    assert result == "final_text"


# =============================================================================
# T_CH — _run_chain uses loop() for agents with tools (P1.3)
# =============================================================================

def test_chain_agent_with_tools_uses_loop():
    """T_CH.1: agent with self.tools in chain → aloop() called, not achat()."""
    sess = LazySession()

    agent = _mock_agent("worker", "tool result")
    agent.tools = [MagicMock()]  # non-empty tools list → has_tools = True

    resp_mock = MagicMock()
    resp_mock.content = "tool result"
    # chain is async-under-the-hood — uses aloop(), not loop()
    agent.aloop = AsyncMock(return_value=resp_mock)

    tool = sess.as_tool("t", "d", mode="chain", participants=[agent])
    tool.run({"task": "do something"})

    agent.aloop.assert_called_once()
    agent.achat.assert_not_called()


def test_chain_agent_without_tools_uses_chat():
    """T_CH.2: agent without tools in chain → achat() called (regression guard)."""
    sess = LazySession()

    agent = _mock_agent("worker", "plain result")
    # tools=[] already set by _mock_agent

    tool = sess.as_tool("t", "d", mode="chain", participants=[agent])
    tool.run({"task": "do something"})

    # chain is async-under-the-hood — uses achat(), not chat()
    agent.achat.assert_called_once()
    agent.aloop.assert_not_called()


# =============================================================================
# T_CS — _ChainState contract
# =============================================================================

def test_chain_state_ctx_none_means_tool_handoff():
    """T_CS.1: _ChainState with ctx=None signals tool→agent handoff (text as task)."""
    from lazybridge.lazy_session import _ChainState

    state = _ChainState(text="tool output", typed=None, ctx=None)
    assert state.ctx is None
    assert state.text == "tool output"
    assert state.typed is None


def test_chain_state_ctx_set_means_agent_handoff():
    """T_CS.2: _ChainState with ctx set signals agent→agent handoff (context injection)."""
    from lazybridge.lazy_session import _ChainState
    from lazybridge.lazy_context import LazyContext

    ctx = LazyContext.from_text("previous agent output")
    state = _ChainState(text="serialised text", typed=None, ctx=ctx)
    assert state.ctx is ctx
    assert state.text == "serialised text"


def test_chain_state_typed_carries_pydantic_object():
    """T_CS.3: _ChainState.typed carries the Pydantic object from a schema step."""
    from pydantic import BaseModel
    from lazybridge.lazy_session import _ChainState

    class Report(BaseModel):
        title: str

    report = Report(title="Q1 Analysis")
    state = _ChainState(text=report.model_dump_json(), typed=report, ctx=None)
    assert isinstance(state.typed, Report)
    assert state.typed.title == "Q1 Analysis"


# ── R-G: as_tool() guidance + _is_pipeline_tool consistency ───────────────────

def _mock_session_agent(name: str = "ag") -> MagicMock:
    import uuid as _uuid
    a = MagicMock()
    a.id = str(_uuid.uuid4())
    a.name = name
    a.description = None
    a.system = None
    a.context = None
    a.tools = []
    a.native_tools = []
    a.output_schema = None
    a._last_output = None
    a._executor = MagicMock()
    a._log = None
    return a


def _sess_with_agent(name: str = "ag") -> tuple:
    """Return (sess, agent) with agent registered in session."""
    sess = LazySession()
    ag = _mock_session_agent(name)
    ag.session = sess
    sess._register_agent(ag)
    return sess, ag


def test_as_tool_parallel_guidance_propagated():
    """R-G1: guidance kwarg is forwarded to the tool returned by as_tool(mode='parallel')."""
    sess, _ = _sess_with_agent()
    tool = sess.as_tool(mode="parallel", name="p", description="d", guidance="be concise")
    assert tool.guidance == "be concise"


def test_as_tool_chain_guidance_propagated():
    """R-G2: guidance kwarg is forwarded to the tool returned by as_tool(mode='chain')."""
    sess, _ = _sess_with_agent()
    tool = sess.as_tool(mode="chain", name="c", description="d", guidance="step by step")
    assert tool.guidance == "step by step"


def test_as_tool_parallel_is_pipeline_tool():
    """R-G3: tool._is_pipeline_tool is True for as_tool(mode='parallel')."""
    sess, _ = _sess_with_agent()
    tool = sess.as_tool(mode="parallel", name="p", description="d")
    assert tool._is_pipeline_tool is True


def test_as_tool_chain_is_pipeline_tool():
    """R-G4: tool._is_pipeline_tool is True for as_tool(mode='chain')."""
    sess, _ = _sess_with_agent()
    tool = sess.as_tool(mode="chain", name="c", description="d")
    assert tool._is_pipeline_tool is True


def test_as_tool_parallel_save_succeeds(tmp_path):
    """R-G5: save() produces a v2 pipeline file for parallel tools."""
    sess, _ = _sess_with_agent()
    tool = sess.as_tool(mode="parallel", name="p", description="d")
    out = str(tmp_path / "p.py")
    tool.save(out)
    content = open(out).read()
    assert "LAZYBRIDGE_GENERATED_TOOL v2" in content
    assert "LazyTool.parallel(" in content


def test_as_tool_chain_save_succeeds(tmp_path):
    """R-G6: save() produces a v2 pipeline file for chain tools."""
    sess, _ = _sess_with_agent()
    tool = sess.as_tool(mode="chain", name="c", description="d")
    out = str(tmp_path / "c.py")
    tool.save(out)
    content = open(out).read()
    assert "LAZYBRIDGE_GENERATED_TOOL v2" in content
    assert "LazyTool.chain(" in content


def test_as_tool_parallel_does_not_mutate_original_agent(fake_response):
    """R-G7: running as_tool(mode='parallel') must not mutate original agent._last_output."""
    from unittest.mock import AsyncMock
    sess = LazySession()
    ag = _mock_session_agent("writer")
    ag.session = sess
    ag.achat = AsyncMock(return_value=fake_response)
    sess._register_agent(ag)

    tool = sess.as_tool(mode="parallel", name="p", description="d")
    tool.run({"task": "test task"})

    assert ag._last_output is None  # original must be untouched


def test_as_tool_chain_does_not_mutate_original_agent(fake_response):
    """R-G8: running as_tool(mode='chain') must not mutate original agent._last_output."""
    sess = LazySession()
    ag = _mock_session_agent("writer")
    ag.session = sess
    # chain is async-under-the-hood — uses achat(), not chat()
    ag.achat = AsyncMock(return_value=fake_response)
    sess._register_agent(ag)

    tool = sess.as_tool(mode="chain", name="c", description="d")
    tool.run({"task": "test task"})

    assert ag._last_output is None  # original must be untouched


# ── R-EQ: semantic equivalence between as_tool() and LazyTool.parallel/chain ──

def test_as_tool_parallel_equivalent_to_lazytool_parallel_properties():
    """R-EQ1: as_tool(mode='parallel') and LazyTool.parallel() produce tools
    with identical observable properties (guidance, _is_pipeline_tool, flag)."""
    sess = LazySession()
    ag = _mock_session_agent("ag")
    ag.session = sess
    sess._register_agent(ag)

    via_session = sess.as_tool(mode="parallel", name="p", description="d", guidance="g")
    via_classmethod = LazyTool.parallel(ag, name="p", description="d", guidance="g")

    assert via_session.guidance == via_classmethod.guidance == "g"
    assert via_session._is_pipeline_tool is True
    assert via_classmethod._is_pipeline_tool is True
    assert via_session.name == via_classmethod.name
    assert via_session.description == via_classmethod.description


def test_as_tool_chain_equivalent_to_lazytool_chain_properties():
    """R-EQ2: as_tool(mode='chain') and LazyTool.chain() produce tools
    with identical observable properties."""
    sess = LazySession()
    ag = _mock_session_agent("ag")
    ag.session = sess
    sess._register_agent(ag)

    via_session = sess.as_tool(mode="chain", name="c", description="d", guidance="g")
    via_classmethod = LazyTool.chain(ag, name="c", description="d", guidance="g")

    assert via_session.guidance == via_classmethod.guidance == "g"
    assert via_session._is_pipeline_tool is True
    assert via_classmethod._is_pipeline_tool is True
    assert via_session.name == via_classmethod.name


def test_as_tool_parallel_equivalent_no_mutation(fake_response):
    """R-EQ3: both factories leave original agent._last_output untouched after run."""
    from unittest.mock import AsyncMock
    ag1 = _mock_session_agent("ag1")
    ag2 = _mock_session_agent("ag2")
    for ag in (ag1, ag2):
        ag.achat = AsyncMock(return_value=fake_response)

    sess = LazySession()
    ag1.session = sess
    sess._register_agent(ag1)

    via_session = sess.as_tool(mode="parallel", name="p", description="d")
    via_session.run({"task": "t"})
    assert ag1._last_output is None

    via_classmethod = LazyTool.parallel(ag2, name="p", description="d")
    via_classmethod.run({"task": "t"})
    assert ag2._last_output is None


def test_as_tool_save_succeeds_for_all_factories(tmp_path):
    """R-EQ4: save() succeeds for pipeline tools from both factories."""
    sess = LazySession()
    ag = _mock_session_agent()
    ag.session = sess
    sess._register_agent(ag)

    for tool in (
        sess.as_tool(mode="parallel", name="p", description="d"),
        LazyTool.parallel(ag, name="p2", description="d"),
        sess.as_tool(mode="chain", name="c", description="d"),
        LazyTool.chain(ag, name="c2", description="d"),
    ):
        out = str(tmp_path / f"{tool.name}.py")
        tool.save(out)
        content = open(out).read()
        assert "LAZYBRIDGE_GENERATED_TOOL v2" in content


# =============================================================================
# Checkpoint & Resume
# =============================================================================

def test_chain_checkpoint_and_resume():
    """Chain with store writes checkpoint after each step and resumes correctly."""
    store = LazyStore()

    # Three agents: A → B → C
    a = _mock_agent("a", "output_a")
    b = _mock_agent("b", "output_b")
    c = _mock_agent("c", "output_c")

    # Run full chain with checkpoint
    tool = LazyTool.chain(a, b, c, name="pipe", description="d", store=store, chain_id="test")
    result = tool.run({"task": "go"})
    assert result == "output_c"

    # Checkpoint should be cleared on success
    assert store.read("_ckpt:test") is None


def test_chain_resumes_from_checkpoint():
    """Chain skips already-completed steps when a checkpoint exists."""
    store = LazyStore()

    # Simulate a crash after step 1 by writing a checkpoint manually
    store.write("_ckpt:test", {"step": 1, "output": "output_from_step_1"})

    a = _mock_agent("a", "should_not_run")
    b = _mock_agent("b", "should_not_run")
    c = _mock_agent("c", "output_c")

    tool = LazyTool.chain(a, b, c, name="pipe", description="d", store=store, chain_id="test")
    result = tool.run({"task": "go"})

    assert result == "output_c"
    # Steps 0 and 1 were skipped
    a.chat.assert_not_called()
    b.chat.assert_not_called()
    # Step 2 ran
    assert c.chat.called or c.achat.called


def test_chain_no_store_no_checkpoint():
    """Chain without store works exactly as before — no checkpoint written."""
    a = _mock_agent("a", "out_a")
    b = _mock_agent("b", "out_b")

    tool = LazyTool.chain(a, b, name="pipe", description="d")
    result = tool.run({"task": "go"})
    assert result == "out_b"


def test_from_db_resumes_store(tmp_path):
    """LazySession.from_db() reconnects to an existing database with stored data."""
    db_path = str(tmp_path / "test.db")

    # Create session, write data, then discard
    sess1 = LazySession(db=db_path)
    sess1.store.write("key1", "value1")
    del sess1

    # Resume from the same db
    sess2 = LazySession.from_db(db_path)
    assert sess2.store.read("key1") == "value1"


def test_from_db_raises_on_missing_file():
    """LazySession.from_db() raises FileNotFoundError for nonexistent db."""
    with pytest.raises(FileNotFoundError):
        LazySession.from_db("/tmp/nonexistent_lazybridge.db")


def test_pydantic_model_in_store():
    """LazyStore.write() auto-converts Pydantic models via model_dump()."""
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str
        value: int

    store = LazyStore()
    store.write("item", Item(name="test", value=42))
    result = store.read("item")
    assert result == {"name": "test", "value": 42}


# ── Sync chain typed handoff ─────────────────────────────────────────────

def test_sync_chain_typed_handoff():
    """Sync build_chain_func passes model_dump() when previous step produced typed output."""
    from lazybridge.pipeline_builders import _ChainState, build_chain_func
    from pydantic import BaseModel

    class Params(BaseModel):
        x: int
        y: str

    calls = []

    class FakeAgent:
        """Mimics a LazyAgent that returns structured output."""
        output_schema = Params
        tools = None
        native_tools = None

        def json(self, task, schema, **kw):
            return Params(x=42, y="hello")

        def chat(self, task, **kw):
            pass

    class FakeTool:
        """Mimics a LazyTool that records its arguments."""
        def run(self, args):
            calls.append(args)
            return "done"

    parts = [FakeAgent(), FakeTool()]
    chain_fn = build_chain_func(parts, [])
    chain_fn("test task")

    # The tool should have received model_dump() output, not {"task": ...}
    assert len(calls) == 1
    assert calls[0] == {"x": 42, "y": "hello"}


# ── Checkpoint payload validation ─────────────────────────────────────────

def _make_echo_chain(store, chain_id="test"):
    """Helper: build a 1-step chain that echoes the task via a mock agent."""
    from lazybridge.pipeline_builders import build_chain_func

    calls = []

    class EchoAgent:
        tools = None
        native_tools = None
        output_schema = None
        _last_output = None

        def chat(self, task, **kw):
            calls.append(task)
            from lazybridge.core.types import CompletionResponse, UsageStats
            self._last_output = f"echo:{task}"
            return CompletionResponse(content=f"echo:{task}", usage=UsageStats())

    chain_fn = build_chain_func([EchoAgent()], [], store=store, chain_id=chain_id)
    return chain_fn, calls


def test_chain_ignores_malformed_checkpoint_missing_step():
    """Chain runs from step 0 when checkpoint is missing 'step' key."""
    store = LazyStore()
    store.write("_ckpt:test", {"output": "text"})
    chain_fn, calls = _make_echo_chain(store)
    chain_fn("hello")
    assert len(calls) == 1
    assert calls[0] == "hello"


def test_chain_ignores_malformed_checkpoint_wrong_type():
    """Chain runs from step 0 when checkpoint is not a dict."""
    store = LazyStore()
    store.write("_ckpt:test", "not_a_dict")
    chain_fn, calls = _make_echo_chain(store)
    chain_fn("hello")
    assert len(calls) == 1
    assert calls[0] == "hello"


def test_chain_ignores_malformed_checkpoint_step_not_int():
    """Chain runs from step 0 when checkpoint step is not an int."""
    store = LazyStore()
    store.write("_ckpt:test", {"step": "one", "output": "text"})
    chain_fn, calls = _make_echo_chain(store)
    chain_fn("hello")
    assert len(calls) == 1
    assert calls[0] == "hello"


# ── from_db() session_id restoration ──────────────────────────────────────

def test_from_db_restores_session_id(tmp_path):
    """from_db() restores the previous session_id so old events are visible."""
    db = str(tmp_path / "restore.db")
    sess1 = LazySession(db=db, tracking="verbose")
    original_id = sess1.id
    sess1.events.log("test_event", agent_id="a1", data="hello")
    del sess1

    sess2 = LazySession.from_db(db, tracking="verbose")
    assert sess2.id == original_id
    events = sess2.events.get(event_type="test_event")
    assert len(events) >= 1


def test_from_db_empty_db_keeps_fresh_id(tmp_path):
    """from_db() on a DB with no events keeps a fresh UUID (no crash)."""
    db = str(tmp_path / "empty.db")
    sess1 = LazySession(db=db)
    del sess1

    sess2 = LazySession.from_db(db)
    assert sess2.id  # has a valid UUID
    assert len(sess2.events.get()) == 0
