from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from lazybridge import Agent, Memory, Session, Tool
from lazybridge.core.types import AudioContent, ImageContent
from lazybridge.engines.codex.app_server import CodexRunResult
from lazybridge.engines.codex.engine import CodexEngine
from lazybridge.session import EventType


class FakeAppServer:
    """Stand-in for ``CodexAppServerClient`` — same shape, no subprocess."""

    def __init__(self, result: CodexRunResult | None = None, fail_times: int = 0, exc_factory=None):
        self.calls = 0
        self.prompts: list[str] = []
        self.dynamic_tools_seen: list[list[dict]] = []
        self.attachments_seen: list[list[dict]] = []
        self.effort_seen: list[str | None] = []
        self.result = result or CodexRunResult(
            text="AMZN is available", input_tokens=11, output_tokens=7, cost_usd=0.002
        )
        self.fail_times = fail_times
        self.exc_factory = exc_factory or (lambda: ConnectionError("reset"))

    async def run(
        self, *, prompt, model, cwd, dynamic_tools, on_tool_call, on_text=None, attachments=None, effort=None
    ):
        self.calls += 1
        self.prompts.append(prompt)
        self.dynamic_tools_seen.append(dynamic_tools)
        self.attachments_seen.append(attachments or [])
        self.effort_seen.append(effort)
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        if dynamic_tools:
            tool_result = await on_tool_call(dynamic_tools[0]["name"], {"symbol": "AMZN"})
            assert tool_result["success"] is True
        if on_text:
            for chunk in ("streamed ", "answer"):
                await on_text(chunk)
        return self.result


def get_quote(symbol: str) -> dict[str, str]:
    """Return a deterministic quote lookup."""
    return {"symbol": symbol}


def test_engine_works_through_a_standard_agent_and_updates_memory():
    memory = Memory()
    session = Session()
    client = FakeAppServer()
    agent = Agent(
        name="codex-prototype",
        engine=CodexEngine(client=client),
        tools=[get_quote],
        memory=memory,
        session=session,
    )

    result = agent("Find AMZN")

    assert result.ok
    assert result.text() == "AMZN is available"
    assert "Find AMZN" in memory.text()
    assert result.metadata.provider == "codex"
    assert result.metadata.input_tokens == 11
    assert result.metadata.output_tokens == 7
    assert result.metadata.cost_usd == 0.002
    assert client.dynamic_tools_seen[0][0]["name"] == "get_quote"


def test_engine_streams_incremental_chunks_and_records_memory():
    memory = Memory()
    client = FakeAppServer()
    agent = Agent(
        name="streaming",
        engine=CodexEngine(client=client),
        tools=[Tool.wrap(get_quote, name="get_quote")],
        memory=memory,
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in agent.stream("Stream this")])

    assert asyncio.run(collect()) == "streamed answer"
    assert "Stream this" in memory.text()


def test_transient_failures_are_retried_then_succeed():
    client = FakeAppServer(fail_times=2)
    agent = Agent(name="retrying", engine=CodexEngine(client=client, max_retries=3, retry_delay=0.01))

    assert agent("Find AMZN").ok
    assert client.calls == 3


def test_non_transient_failure_is_not_retried():
    client = FakeAppServer(fail_times=99, exc_factory=lambda: ValueError("bad model"))
    agent = Agent(name="failing", engine=CodexEngine(client=client, max_retries=3, retry_delay=0.01))

    result = agent("Find AMZN")

    assert not result.ok
    assert client.calls == 1


def test_structured_output_type_injects_json_schema_into_prompt():
    class Quote(BaseModel):
        symbol: str
        price: float

    client = FakeAppServer(result=CodexRunResult(text='{"symbol": "AMZN", "price": 123.45}'))
    agent = Agent(name="structured", engine=CodexEngine(client=client), output=Quote)

    result = agent("Get AMZN quote")

    assert result.ok
    assert isinstance(result.payload, Quote)
    assert "JSON schema" in client.prompts[0]


def test_images_are_forwarded_as_app_server_user_input_items():
    client = FakeAppServer()
    agent = Agent(name="viewer", engine=CodexEngine(client=client))

    result = agent(
        "What is in this image?",
        images=[
            ImageContent(base64_data="aGk=", media_type="image/png"),
            ImageContent(url="https://example.com/chart.png", media_type="image/png"),
        ],
    )

    assert result.ok
    # Inline bytes become a data: URL; a real URL passes through unchanged.
    assert client.attachments_seen[0] == [
        {"type": "image", "url": "data:image/png;base64,aGk="},
        {"type": "image", "url": "https://example.com/chart.png"},
    ]


def test_audio_is_dropped_with_a_warning_because_codex_cannot_read_it():
    client = FakeAppServer()
    agent = Agent(name="listener", engine=CodexEngine(client=client))

    with pytest.warns(UserWarning, match="does not forward Envelope.audio"):
        result = agent("Transcribe this", audio=AudioContent(base64_data="aGk=", media_type="audio/wav"))

    assert result.ok
    assert client.attachments_seen[0] == []


def test_agent_name_is_resolved_per_invocation_not_stamped_on_the_engine():
    """Regression guard for the ``_agent_name`` attribution bug (audit finding #1):

    a shared engine instance must attribute Session events to whichever
    Agent is actually running, not whichever Agent constructed it last.
    """
    shared_engine = CodexEngine(client=FakeAppServer())
    session = Session()
    first = Agent(name="first", engine=shared_engine, session=session)
    second = Agent(name="second", engine=shared_engine, session=session)

    first("task one")
    second("task two")

    starts = session.events.query(event_type=EventType.AGENT_START)
    agent_names = {row["payload"].get("agent_name") for row in starts}
    assert {"first", "second"}.issubset(agent_names)


def test_reasoning_effort_is_forwarded_to_turn_start():
    """The App Server takes a per-model ``effort`` on turn/start.

    ``ClaudeCodeEngine`` has had ``reasoning_effort`` since it merged; this
    is the Codex equivalent, passed through unvalidated because the accepted
    values are advertised per model via ``model/list``.
    """
    client = FakeAppServer()
    agent = Agent(name="thinker", engine=CodexEngine(client=client, reasoning_effort="high"))

    assert agent("Think hard").ok
    assert client.effort_seen == ["high"]


def test_no_reasoning_effort_leaves_the_account_default():
    client = FakeAppServer()
    agent = Agent(name="default-effort", engine=CodexEngine(client=client))

    assert agent("Think normally").ok
    assert client.effort_seen == [None]
