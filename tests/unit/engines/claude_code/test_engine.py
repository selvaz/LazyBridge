from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from lazybridge import Agent, Envelope, Memory, Session, Tool
from lazybridge.core.types import AudioContent, ImageContent
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions, ClaudeSdkResult, ClaudeSdkStreamEvent


@pytest.fixture
def fake_tag_session(monkeypatch):
    """Record ``tag_session`` calls instead of touching a real session file.

    ``_tag_new_session`` does ``from claude_agent_sdk import tag_session``
    fresh on every call (a deliberately lazy import), so this installs a
    STUB MODULE in ``sys.modules`` rather than patching a real package
    attribute. That is deliberate: the CI ``unit tests`` job installs
    ``.[anthropic,openai,google,test]`` and never the ``claude-code``
    extra, so the SDK is genuinely absent there — and these tests exercise
    the engine against fakes, so they must run anyway. (An earlier version
    imported the SDK at module level and would have broken collection for
    the whole file in CI: Codex review finding on 181f33e.)
    """
    import sys
    import types

    calls: list[tuple[str, str | None, str | None]] = []

    def fake(session_id, tag, *, directory=None):
        calls.append((session_id, tag, directory))

    module = sys.modules.get("claude_agent_sdk")
    if module is None:
        module = types.ModuleType("claude_agent_sdk")
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    monkeypatch.setattr(module, "tag_session", fake, raising=False)
    return calls


class FakeSdk:
    def __init__(self):
        self.options: list[ClaudeSdkOptions] = []
        self.prompts: list[str] = []
        self.attachments: list[tuple[dict, ...]] = []

    async def run(
        self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict, ...] = ()
    ) -> ClaudeSdkResult:
        self.options.append(options)
        self.prompts.append(prompt)
        self.attachments.append(attachments)
        assert options.builtin_tools == ("WebSearch", "WebFetch")
        assert len(options.mcp_tools) == 1
        tool = options.mcp_tools[0]
        arguments = {"symbol": "AMZN"} if tool.name == "get_quote" else {"task": "analyse AMZN"}
        tool_result = await tool.handler(arguments)
        assert tool_result["content"]
        return ClaudeSdkResult(text="AMZN is available", session_id="fake-session", input_tokens=11, output_tokens=7)

    async def stream(self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict, ...] = ()):
        self.options.append(options)
        self.prompts.append(prompt)
        self.attachments.append(attachments)
        yield ClaudeSdkStreamEvent(text="streamed ")
        yield ClaudeSdkStreamEvent(text="answer")
        yield ClaudeSdkStreamEvent(session_id="fake-stream-session", final=True)


def get_quote(symbol: str) -> dict[str, str]:
    """Return a deterministic quote lookup."""
    return {"symbol": symbol}


def test_engine_works_through_a_standard_agent_and_updates_memory():
    memory = Memory()
    session = Session()
    agent = Agent(
        name="claude-code-prototype",
        engine=ClaudeCodeEngine(client=FakeSdk()),
        # A plain callable is the normal, zero-ceremony LazyBridge input.
        # Agent normalises it before the engine receives its tool list.
        tools=[get_quote],
        memory=memory,
        session=session,
    )

    result = agent("Find AMZN")

    assert result.ok
    assert result.text() == "AMZN is available"
    assert "Find AMZN" in memory.text()
    assert result.metadata.provider == "claude-code"


def test_engine_streams_incremental_chunks_and_records_memory():
    memory = Memory()
    agent = Agent(
        name="streaming",
        engine=ClaudeCodeEngine(client=FakeSdk()),
        tools=[Tool.wrap(get_quote, name="get_quote")],
        memory=memory,
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in agent.stream("Stream this")])

    assert asyncio.run(collect()) == "streamed answer"
    assert "Stream this" in memory.text()


def test_runtime_session_is_kept_on_lazybridge_session_and_skips_parent_memory():
    client = FakeSdk()
    memory = Memory()
    session = Session()
    agent = Agent(
        name="persistent",
        engine=ClaudeCodeEngine(client=client, session_mode="runtime", session_name="fixed-name"),
        tools=[get_quote],
        memory=memory,
        session=session,
    )

    assert agent("First turn").ok
    assert agent("Second turn").ok

    assert client.options[0].resume is None
    assert client.options[1].resume == "fake-session"
    assert "First turn" not in client.prompts[1]
    assert memory.text()  # still available to explicit from_memory users


def test_runtime_session_streaming_resumes_without_reinjecting_memory():
    client = FakeSdk()
    memory = Memory()
    session = Session()
    agent = Agent(
        name="streaming-persistent",
        engine=ClaudeCodeEngine(client=client, session_mode="runtime"),
        tools=[get_quote],
        memory=memory,
        session=session,
    )

    async def collect(task: str) -> str:
        return "".join([chunk async for chunk in agent.stream(task)])

    assert asyncio.run(collect("First stream")) == "streamed answer"
    assert asyncio.run(collect("Second stream")) == "streamed answer"
    assert client.options[-2].resume is None
    assert client.options[-1].resume == "fake-stream-session"
    assert "First stream" not in client.prompts[-1]


class ChildEngine:
    async def run(self, env, *, tools, output_type, memory, session, store=None, plan_state=None):
        return Envelope(task=env.task, payload="specialist result")

    async def stream(self, env, *, tools, output_type, memory, session):
        yield "specialist result"


def test_another_lazybridge_agent_is_exposed_as_a_normal_mcp_tool():
    specialist = Agent(name="quote_specialist", engine=ChildEngine())
    parent = Agent(name="parent", engine=ClaudeCodeEngine(client=FakeSdk()), tools=[specialist])

    assert parent("Delegate quote analysis").ok


def test_standard_tool_provider_is_expanded_before_the_engine_runs():
    class QuoteProvider:
        _is_lazy_tool_provider = True

        def as_tools(self):
            return [Tool.wrap(get_quote, name="get_quote")]

    agent = Agent(
        name="provider-parent",
        engine=ClaudeCodeEngine(client=FakeSdk()),
        tools=[QuoteProvider()],
    )

    assert agent("Find AMZN through the provider").ok


class StructuredSdk:
    """Answers with the CLI's parsed ``structured_output``, not prose."""

    def __init__(self):
        self.options: list[ClaudeSdkOptions] = []
        self.prompts: list[str] = []
        self.attachments: list[tuple[dict, ...]] = []

    async def run(
        self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict, ...] = ()
    ) -> ClaudeSdkResult:
        self.options.append(options)
        self.prompts.append(prompt)
        self.attachments.append(attachments)
        return ClaudeSdkResult(text='{"symbol": "AMZN", "price": 123.45}', session_id="fake-session")

    async def stream(self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict, ...] = ()):
        self.options.append(options)
        self.attachments.append(attachments)
        yield ClaudeSdkStreamEvent(text='{"symbol": "AMZN", "price": 123.45}')
        yield ClaudeSdkStreamEvent(final=True)


class Quote(BaseModel):
    symbol: str
    price: float


def test_output_type_is_enforced_natively_not_asked_for_in_the_prompt():
    client = StructuredSdk()
    agent = Agent(name="quoter", engine=ClaudeCodeEngine(client=client), output=Quote)

    result = agent("AMZN trades at 123.45")

    assert result.ok
    assert result.payload == Quote(symbol="AMZN", price=123.45)
    # Constrained server-side by the SDK (--json-schema), so the schema must
    # reach the options — and must NOT be pasted into the prompt.
    assert client.options[-1].output_format == {"type": "json_schema", "schema": Quote.model_json_schema()}
    assert "json schema" not in client.prompts[-1].lower()


def test_plain_text_output_sets_no_output_format():
    client = FakeSdk()
    agent = Agent(name="plain", engine=ClaudeCodeEngine(client=client), tools=[get_quote])

    assert agent("Find AMZN").ok
    assert client.options[-1].output_format is None


def test_inline_images_become_content_blocks_and_url_images_warn():
    client = FakeSdk()
    agent = Agent(name="viewer", engine=ClaudeCodeEngine(client=client), tools=[get_quote])

    with pytest.warns(UserWarning, match="inline image bytes only"):
        result = agent(
            "What is in these images?",
            images=[
                ImageContent(base64_data="aGk=", media_type="image/png"),
                ImageContent(url="https://example.com/chart.png", media_type="image/png"),
            ],
        )

    assert result.ok
    # The CLI accepts a base64 source but rejects a url one, so only the
    # inline image survives — and it must not be dropped silently.
    assert client.attachments[-1] == (
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
    )


def test_audio_is_dropped_with_a_warning_because_claude_takes_no_audio():
    client = FakeSdk()
    agent = Agent(name="listener", engine=ClaudeCodeEngine(client=client), tools=[get_quote])

    with pytest.warns(UserWarning, match="does not forward Envelope.audio"):
        result = agent("Transcribe this", audio=AudioContent(base64_data="aGk=", media_type="audio/wav"))

    assert result.ok
    assert client.attachments[-1] == ()


class _PlainSdk:
    """Minimal client: records what it was asked, needs no tools."""

    def __init__(self, session_id: str = "sess-9"):
        self.options: list[ClaudeSdkOptions] = []
        self.prompts: list[str] = []
        self.session_id = session_id

    async def run(self, prompt, *, options, attachments=()):
        self.options.append(options)
        self.prompts.append(prompt)
        return ClaudeSdkResult(text="ok", session_id=self.session_id)

    @property
    def resumes(self):
        return [o.resume for o in self.options]


class TestDurableSessions:
    """``persist_session`` / ``session_id``: the mirror of Codex' durable
    threads — a handle the caller keeps, not state parked on a Session."""

    def test_default_starts_a_fresh_session_per_run(self):
        sdk = _PlainSdk()
        agent = Agent(ClaudeCodeEngine(client=sdk), name="a")

        agent("one")
        agent("two")

        assert sdk.resumes == [None, None]

    def test_the_session_id_is_kept_and_resumed(self, fake_tag_session):
        sdk = _PlainSdk()
        engine = ClaudeCodeEngine(client=sdk, persist_session=True)
        agent = Agent(engine, name="a")

        agent("first")
        agent("second")

        assert sdk.resumes == [None, "sess-9"]
        assert engine.session_id == "sess-9"
        # Tagged once, on the run that CREATED the session — not on the one
        # that resumed it (the SDK has no "already tagged" check of its own;
        # tagging twice would just append a second, redundant JSONL line).
        assert fake_tag_session == [("sess-9", "lazybridge", None)]

    def test_tag_is_overridable_and_can_be_disabled(self, fake_tag_session):
        sdk = _PlainSdk()
        Agent(ClaudeCodeEngine(client=sdk, persist_session=True, tag="approval-lab", cwd="C:/work"), name="a")("hi")
        assert fake_tag_session == [("sess-9", "approval-lab", "C:/work")]

        fake_tag_session.clear()
        sdk2 = _PlainSdk()
        Agent(ClaudeCodeEngine(client=sdk2, persist_session=True, tag=None), name="a")("hi")
        assert fake_tag_session == []

    def test_a_tagging_failure_warns_but_does_not_fail_the_run(self, monkeypatch):
        # Same stub-module approach as the fake_tag_session fixture: works
        # whether or not the claude-code extra is installed.
        import sys
        import types

        def broken(session_id, tag, *, directory=None):
            raise FileNotFoundError("no such session file")

        module = sys.modules.get("claude_agent_sdk")
        if module is None:
            module = types.ModuleType("claude_agent_sdk")
            monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
        monkeypatch.setattr(module, "tag_session", broken, raising=False)
        sdk = _PlainSdk()

        with pytest.warns(UserWarning, match="could not tag session"):
            result = Agent(ClaudeCodeEngine(client=sdk, persist_session=True), name="a")("hi")

        assert result.ok

    def test_a_supplied_session_id_resumes_immediately(self):
        sdk = _PlainSdk()
        agent = Agent(ClaudeCodeEngine(client=sdk, session_id="sess-7"), name="a", memory=Memory())

        agent("carry on")

        assert sdk.resumes == ["sess-7"]
        # No memory block: Claude owns the history on a resumed session.
        assert sdk.prompts == ["carry on"]

    def test_resuming_stops_re_sending_memory(self, fake_tag_session):
        sdk = _PlainSdk()
        agent = Agent(ClaudeCodeEngine(client=sdk, persist_session=True), name="a", memory=Memory())

        agent("first question")
        agent("second question")

        assert sdk.prompts[1] == "second question"

    def test_an_ordinary_engine_keeps_sending_memory(self):
        sdk = _PlainSdk()
        agent = Agent(ClaudeCodeEngine(client=sdk), name="a", memory=Memory())

        agent("first question")
        agent("second question")

        assert "Conversation context from LazyBridge" in sdk.prompts[1]

    def test_an_explicit_handle_beats_the_session_parked_one(self):
        sdk = _PlainSdk()
        engine = ClaudeCodeEngine(client=sdk, session_mode="runtime", session_id="explicit")
        agent = Agent(engine, name="a", session=Session())

        agent("one")
        agent("two")

        # First run resumes the handle given; the second follows the id the
        # SDK reported, which for a resumed session need not be the same one.
        assert sdk.resumes == ["explicit", "sess-9"]
        assert engine.session_id == "sess-9"

    def test_concurrent_first_runs_do_not_open_two_sessions(self, fake_tag_session):
        # Without a lock before the first id exists, both runs start their own
        # session and race to store the id: one conversation is orphaned.
        class Slow(_PlainSdk):
            async def run(self, prompt, *, options, attachments=()):
                await asyncio.sleep(0.01)
                return await super().run(prompt, options=options, attachments=attachments)

        sdk = Slow()
        agent = Agent(ClaudeCodeEngine(client=sdk, persist_session=True), name="a")

        async def both():
            await asyncio.gather(agent.run("one"), agent.run("two"))

        asyncio.run(both())

        assert sdk.resumes == [None, "sess-9"]


def test_policy_extra_tools_extend_the_builtin_set_deduplicated(tmp_path):
    # extra_tools is what lets a gated agent be granted Write/Edit/Bash at
    # the SDK level: the engine used to hardcode the read-only set, so no
    # approval gate could ever be asked about a write — the model simply
    # never had the tool. Names already derived (Read here) must not repeat.
    from lazybridge.engines.coding import ApprovalDecision, ClaudeCodePolicy, CodingAgentConfig

    class CapturingSdk(FakeSdk):
        async def run(self, prompt, *, options, attachments=()):
            self.options.append(options)
            return ClaudeSdkResult(text="ok", session_id="s", input_tokens=1, output_tokens=1)

    async def gate(request):  # Bash is unconfinable, so a gate is mandatory
        return ApprovalDecision.deny()

    sdk = CapturingSdk()
    engine = ClaudeCodeEngine(
        client=sdk,
        cwd=str(tmp_path),
        file_roots=[str(tmp_path)],
        web=False,
        config=CodingAgentConfig(
            claude=ClaudeCodePolicy(extra_tools=("Write", "Bash", "Read")),
            approval_gate=gate,
        ),
    )
    agent = Agent(name="writer-prototype", engine=engine)

    assert agent("do something").ok
    assert sdk.options[0].builtin_tools == ("Read", "Glob", "Grep", "Write", "Bash")


def test_unconfinable_extra_tools_require_an_approval_gate(tmp_path):
    # Bash has no file_roots sandbox (the confinement hook matches the file
    # tools only), so its sole boundary is the gate's policy — granting it
    # without a gate must fail at construction, not at the first escape.
    from lazybridge.engines.coding import ApprovalDecision, ClaudeCodePolicy, CodingAgentConfig

    with pytest.raises(ValueError, match="Bash"):
        ClaudeCodeEngine(
            client=FakeSdk(),
            cwd=str(tmp_path),
            file_roots=[str(tmp_path)],
            config=CodingAgentConfig(claude=ClaudeCodePolicy(extra_tools=("Write", "Bash"))),
        )

    # With a gate the same grant is accepted; Write alone never needed one.
    async def gate(request):
        return ApprovalDecision.deny()

    ClaudeCodeEngine(
        client=FakeSdk(),
        cwd=str(tmp_path),
        file_roots=[str(tmp_path)],
        config=CodingAgentConfig(claude=ClaudeCodePolicy(extra_tools=("Write", "Bash")), approval_gate=gate),
    )
    ClaudeCodeEngine(
        client=FakeSdk(),
        cwd=str(tmp_path),
        file_roots=[str(tmp_path)],
        config=CodingAgentConfig(claude=ClaudeCodePolicy(extra_tools=("Write", "Edit"))),
    )
