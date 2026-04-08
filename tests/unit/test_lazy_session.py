"""Unit tests for LazySession — mock agents, no API calls."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from lazybridge.lazy_session import LazySession
from lazybridge.lazy_store import LazyStore
from lazybridge.lazy_tool import LazyTool
from lazybridge.core.types import CompletionResponse, UsageStats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_response(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=UsageStats())


def _mock_agent(name: str, response_content: str) -> MagicMock:
    """Create a minimal mock LazyAgent-like object."""
    agent = MagicMock()
    agent.name = name
    agent._last_output = None
    agent.output_schema = None
    agent.tools = []
    agent.native_tools = []
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
    agent.ajson = MagicMock(return_value={"mocked": True})
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

    # A must have been called
    a.chat.assert_called_once()
    # B must have been called with context (kw contains context=)
    b.chat.assert_called_once()
    call_kwargs = b.chat.call_args[1]
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

    # The agent must have received the tool's output as its task
    agent.chat.assert_called_once()
    call_args = agent.chat.call_args[0]
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
    agent.json = MagicMock(return_value=parsed)

    tool = sess.as_tool("t", "d", mode="chain", participants=[agent])
    tool.run({"task": "give me structure"})

    agent.json.assert_called_once()
    # chat() must NOT have been called
    agent.chat.assert_not_called()


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
    agent.json = MagicMock(return_value=expected)

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
    mid_agent.json = MagicMock(return_value=Mid(data="mid_result"))

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
    """T_CH.1: agent with self.tools in chain → loop() called, not chat()."""
    sess = LazySession()

    agent = _mock_agent("worker", "tool result")
    agent.tools = [MagicMock()]  # non-empty tools list → _has_tools = True

    resp_mock = MagicMock()
    resp_mock.content = "tool result"
    agent.loop = MagicMock(return_value=resp_mock)

    tool = sess.as_tool("t", "d", mode="chain", participants=[agent])
    tool.run({"task": "do something"})

    agent.loop.assert_called_once()
    agent.chat.assert_not_called()


def test_chain_agent_without_tools_uses_chat():
    """T_CH.2: agent without tools in chain → chat() called (regression guard)."""
    sess = LazySession()

    agent = _mock_agent("worker", "plain result")
    # tools=[] already set by _mock_agent

    tool = sess.as_tool("t", "d", mode="chain", participants=[agent])
    tool.run({"task": "do something"})

    agent.chat.assert_called_once()
    agent.loop.assert_not_called()


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


def test_as_tool_parallel_save_raises(tmp_path):
    """R-G5: save() raises ValueError on pipeline tool created via as_tool(mode='parallel')."""
    sess, _ = _sess_with_agent()
    tool = sess.as_tool(mode="parallel", name="p", description="d")
    with pytest.raises(ValueError, match="pipeline tool"):
        tool.save(str(tmp_path / "p.json"))


def test_as_tool_chain_save_raises(tmp_path):
    """R-G6: save() raises ValueError on pipeline tool created via as_tool(mode='chain')."""
    sess, _ = _sess_with_agent()
    tool = sess.as_tool(mode="chain", name="c", description="d")
    with pytest.raises(ValueError, match="pipeline tool"):
        tool.save(str(tmp_path / "c.json"))
