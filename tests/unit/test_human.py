"""Unit tests for HumanAgent and SupervisorAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lazybridge.core.types import CompletionResponse, UsageStats
from lazybridge.human import HumanAgent
from lazybridge.supervisor import SupervisorAgent

# ---------------------------------------------------------------------------
# HumanAgent — single mode
# ---------------------------------------------------------------------------


def test_human_chat_returns_response():
    human = HumanAgent(name="test", input_fn=lambda p: "my answer")
    resp = human.chat("What do you think?")
    assert isinstance(resp, CompletionResponse)
    assert resp.content == "my answer"


def test_human_text_returns_string():
    human = HumanAgent(name="test", input_fn=lambda p: "hello")
    result = human.text("Say something")
    assert result == "hello"
    assert isinstance(result, str)


def test_human_sets_last_output():
    human = HumanAgent(name="test", input_fn=lambda p: "output")
    human.chat("task")
    assert human._last_output == "output"
    assert human.result == "output"


def test_human_loop_delegates_to_chat():
    human = HumanAgent(name="test", input_fn=lambda p: "looped")
    resp = human.loop("task")
    assert resp.content == "looped"


def test_human_as_tool():
    human = HumanAgent(name="reviewer", input_fn=lambda p: "approved")
    tool = human.as_tool("review", "Human reviews output")
    assert tool.name == "review"
    assert tool._delegate is not None


def test_human_custom_prompt_template():
    received = []
    human = HumanAgent(
        name="test",
        input_fn=lambda p: (received.append(p), "ok")[1],
        prompt_template="Please review: {task}",
    )
    human.chat("the report")
    assert received[0] == "Please review: the report"


def test_human_timeout_with_default():
    import time

    def slow_input(p):
        time.sleep(10)
        return "too late"

    human = HumanAgent(name="test", input_fn=slow_input, timeout=0.1, default="default_answer")
    resp = human.chat("hurry")
    assert resp.content == "default_answer"


def test_human_timeout_without_default_raises():
    import time

    def slow_input(p):
        time.sleep(10)
        return "too late"

    human = HumanAgent(name="test", input_fn=slow_input, timeout=0.1)
    with pytest.raises(TimeoutError):
        human.chat("hurry")


# ---------------------------------------------------------------------------
# HumanAgent — dialogue mode
# ---------------------------------------------------------------------------


def test_human_dialogue_mode():
    responses = iter(["first thought", "second thought", "done"])
    human = HumanAgent(
        name="test",
        input_fn=lambda p: next(responses),
        mode="dialogue",
    )
    resp = human.chat("Review this")
    assert "first thought" in resp.content
    assert "second thought" in resp.content


# ---------------------------------------------------------------------------
# HumanAgent — async
# ---------------------------------------------------------------------------


async def test_human_achat_with_ainput():
    human = HumanAgent(
        name="test",
        ainput_fn=AsyncMock(return_value="async answer"),
    )
    resp = await human.achat("question")
    assert resp.content == "async answer"


async def test_human_achat_falls_back_to_thread():
    human = HumanAgent(name="test", input_fn=lambda p: "threaded")
    resp = await human.achat("question")
    assert resp.content == "threaded"


# ---------------------------------------------------------------------------
# HumanAgent — verify compatibility
# ---------------------------------------------------------------------------


def test_human_as_verifier():
    human = HumanAgent(name="judge", input_fn=lambda p: "approved: looks good")
    verdict = human.text("Question: X\nAnswer: Y")
    assert "approved" in verdict.lower()


# ---------------------------------------------------------------------------
# HumanAgent — duck-type attributes
# ---------------------------------------------------------------------------


def test_human_duck_type_attributes():
    human = HumanAgent(name="test")
    assert human.output_schema is None
    assert human.tools == []
    assert human.native_tools == []
    assert human._is_human is True
    assert human.name == "test"
    assert human.id == "human-test"


# ---------------------------------------------------------------------------
# HumanAgent — chain compatibility
# ---------------------------------------------------------------------------


def test_human_in_chain():
    from lazybridge.pipeline_builders import build_chain_func

    class FakeAgent:
        tools = None
        native_tools = None
        output_schema = None
        _last_output = None

        def chat(self, task, **kw):
            self._last_output = f"agent:{task[:20]}"
            return CompletionResponse(content=self._last_output, usage=UsageStats())

    human = HumanAgent(name="reviewer", input_fn=lambda p: "human approved")
    chain_fn = build_chain_func([FakeAgent(), human], [])
    result = chain_fn("start task")
    assert result == "human approved"


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------


def test_supervisor_continue():
    supervisor = SupervisorAgent(name="sup", input_fn=lambda p: "continue")
    resp = supervisor.chat("previous output")
    assert resp.content == "previous output"


def test_supervisor_continue_with_message():
    supervisor = SupervisorAgent(name="sup", input_fn=lambda p: "continue: my custom output")
    resp = supervisor.chat("previous output")
    assert resp.content == "my custom output"


def test_supervisor_tool_call():
    from lazybridge.lazy_tool import LazyTool

    def search(query: str) -> str:
        """Search."""
        return f"found: {query}"

    tool = LazyTool.from_function(search)
    inputs = iter(['search("AI safety")', "continue"])
    supervisor = SupervisorAgent(
        name="sup",
        tools=[tool],
        input_fn=lambda p: next(inputs),
    )
    resp = supervisor.chat("task")
    assert "found: AI safety" in resp.content


def test_supervisor_retry_agent():
    retry_calls = []

    class FakeAgent:
        name = "researcher"
        tools = None
        native_tools = None
        output_schema = None
        _last_output = None

        def chat(self, task, **kw):
            retry_calls.append(task)
            self._last_output = "retried output"
            return CompletionResponse(content="retried output", usage=UsageStats())

    inputs = iter(["retry researcher: be more specific", "continue"])
    supervisor = SupervisorAgent(
        name="sup",
        agents=[FakeAgent()],
        input_fn=lambda p: next(inputs),
    )
    resp = supervisor.chat("original task")
    assert len(retry_calls) == 1
    assert "Feedback: be more specific" in retry_calls[0]
    assert resp.content == "retried output"


def test_supervisor_store_access():
    from lazybridge.lazy_session import LazySession

    sess = LazySession()
    sess.store.write("key1", "value1")

    inputs = iter(["store key1", "continue"])
    supervisor = SupervisorAgent(
        name="sup",
        session=sess,
        input_fn=lambda p: next(inputs),
    )
    supervisor.chat("task")


def test_supervisor_duck_type():
    supervisor = SupervisorAgent(name="sup")
    assert supervisor._is_human is True
    assert supervisor.output_schema is None
    assert supervisor.name == "sup"


async def test_supervisor_achat():
    supervisor = SupervisorAgent(name="sup", input_fn=lambda p: "continue")
    resp = await supervisor.achat("task")
    assert resp.content == "task"


def test_supervisor_as_tool():
    supervisor = SupervisorAgent(name="sup", input_fn=lambda p: "continue")
    tool = supervisor.as_tool("supervise", "Human supervises")
    assert tool.name == "supervise"
