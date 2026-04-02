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
