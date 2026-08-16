from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from lazybridge import Agent, Memory, Session, Tool
from lazybridge.core.types import AudioContent, ImageContent
from lazybridge.engines.codex.app_server import CodexRunResult
from lazybridge.engines.codex.engine import CodexEngine
from lazybridge.engines.coding import ApprovalDecision, CodexPolicy, CodingAgentConfig
from lazybridge.session import EventType


class FakeAppServer:
    """Stand-in for ``CodexAppServerClient`` — same shape, no subprocess."""

    def __init__(self, result: CodexRunResult | None = None, fail_times: int = 0, exc_factory=None):
        self.calls = 0
        self.prompts: list[str] = []
        self.dynamic_tools_seen: list[list[dict]] = []
        self.attachments_seen: list[list[dict]] = []
        self.effort_seen: list[str | None] = []
        self.developer_instructions_seen: list[str | None] = []
        self.thread_ids_seen: list[str | None] = []
        self.ephemeral_seen: list[bool] = []
        self.review_targets_seen: list[dict | None] = []
        self.result = result or CodexRunResult(
            text="AMZN is available", input_tokens=11, output_tokens=7, cost_usd=0.002
        )
        self.fail_times = fail_times
        self.exc_factory = exc_factory or (lambda: ConnectionError("reset"))
        self.tool_results: list[dict] = []

    async def run(
        self,
        *,
        prompt,
        model,
        cwd,
        dynamic_tools,
        on_tool_call,
        developer_instructions=None,
        on_text=None,
        attachments=None,
        effort=None,
        sandbox="read-only",
        approval_policy="never",
        approval_gate=None,
        thread_id=None,
        ephemeral=True,
        review_target=None,
        progress=None,
    ):
        if progress is not None:
            # What the real client publishes as it goes, so the engine can
            # tell "nothing sent yet" from "turn accepted, fate unknown".
            progress["thread_id"] = self.result.thread_id or "thread-live"
            progress["turn_sent"] = True
        self.calls += 1
        self.review_targets_seen.append(review_target)
        self.prompts.append(prompt)
        self.thread_ids_seen.append(thread_id)
        self.ephemeral_seen.append(ephemeral)
        self.dynamic_tools_seen.append(dynamic_tools)
        self.attachments_seen.append(attachments or [])
        self.effort_seen.append(effort)
        self.developer_instructions_seen.append(developer_instructions)
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        if dynamic_tools:
            # Recorded, not asserted: a gated run legitimately gets a
            # success=False payload back instead of a tool result.
            self.tool_results.append(await on_tool_call(dynamic_tools[0]["name"], {"symbol": "AMZN"}))
        if on_text:
            for chunk in ("streamed ", "answer"):
                await on_text(chunk)
        return self.result


def get_quote(symbol: str) -> dict[str, str]:
    """Return a deterministic quote lookup."""
    return {"symbol": symbol}


class TestDurableThreads:
    """``persist_thread`` / ``thread_id``: Codex owns the transcript."""

    def test_default_is_still_an_ephemeral_thread_per_run(self):
        fake = FakeAppServer()
        agent = Agent(CodexEngine(client=fake), name="a")

        agent("hello")

        assert fake.ephemeral_seen == [True]
        assert fake.thread_ids_seen == [None]

    def test_the_thread_id_is_kept_and_reused_across_runs(self):
        fake = FakeAppServer(result=CodexRunResult(text="ok", thread_id="thread-9"))
        engine = CodexEngine(client=fake, persist_thread=True)
        agent = Agent(engine, name="a")

        agent("first")
        agent("second")

        assert fake.ephemeral_seen == [False, False]
        # First run opens the thread, the second resumes it by id.
        assert fake.thread_ids_seen == [None, "thread-9"]
        assert engine.thread_id == "thread-9"

    def test_resuming_does_not_re_send_lazybridge_memory(self):
        # Codex already holds the history; prepending Memory would give the
        # model two chronologies of the same conversation.
        fake = FakeAppServer(result=CodexRunResult(text="ok", thread_id="thread-9"))
        agent = Agent(CodexEngine(client=fake, persist_thread=True), name="a", memory=Memory())

        agent("first question")
        agent("second question")

        # The second run resumes, so its prompt is the bare question — the
        # ephemeral engine sends a "LazyBridge conversation context" block here
        # (asserted in the next test).
        assert fake.prompts[1] == "second question"

    def test_an_ephemeral_engine_keeps_sending_memory(self):
        fake = FakeAppServer()
        agent = Agent(CodexEngine(client=fake), name="a", memory=Memory())

        agent("first question")
        agent("second question")

        assert "LazyBridge conversation context" in fake.prompts[1]

    def test_a_supplied_thread_id_resumes_from_the_first_run(self):
        fake = FakeAppServer(result=CodexRunResult(text="ok", thread_id="thread-7"))
        agent = Agent(CodexEngine(client=fake, thread_id="thread-7"), name="a", memory=Memory())

        agent("carry on")

        assert fake.thread_ids_seen == ["thread-7"]
        assert fake.ephemeral_seen == [False]
        assert fake.prompts == ["carry on"]  # no memory block on a resumed thread


def test_writer_profile_gates_dynamic_tools_before_dispatch():
    seen = []

    async def gate(request):
        seen.append(request)
        return ApprovalDecision.allow()

    client = FakeAppServer()
    agent = Agent(
        name="gated-writer",
        engine=CodexEngine(client=client, config=CodingAgentConfig.writer(gate)),
        tools=[get_quote],
    )

    result = agent("Find AMZN")

    assert result.ok
    assert seen[0].kind == "tool"
    assert seen[0].name == "get_quote"


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


def test_system_prompt_is_forwarded_as_developer_instructions_not_user_text_for_run_and_stream():
    client = FakeAppServer()
    agent = Agent(name="instructed", engine=CodexEngine(client=client, system="Never reveal secrets."))

    assert agent("Summarize this").ok

    async def collect() -> str:
        return "".join([chunk async for chunk in agent.stream("Stream this")])

    assert asyncio.run(collect()) == "streamed answer"

    assert client.developer_instructions_seen == ["Never reveal secrets.", "Never reveal secrets."]
    assert all("Never reveal secrets." not in prompt for prompt in client.prompts)


def test_gated_tools_are_approved_once_per_agent_across_runs():
    """End-to-end scope check through the engine, not just the helper."""
    asked: list[str] = []

    async def gate(request):
        asked.append(request.name)
        return ApprovalDecision.allow_for_session()

    client = FakeAppServer()
    session = Session()
    engine = CodexEngine(
        client=client,
        config=CodingAgentConfig(codex=CodexPolicy(preapprove_dynamic_tools=False), approval_gate=gate),
    )
    agent = Agent(name="analyst", engine=engine, tools=[get_quote], session=session)

    assert agent("Find AMZN").ok
    assert agent("Find AMZN again").ok

    assert asked == ["get_quote"]


def test_a_denied_tool_never_runs_and_reports_back_to_the_model():
    calls: list[str] = []

    def tracked_quote(symbol: str) -> dict[str, str]:
        """Return a deterministic quote lookup."""
        calls.append(symbol)
        return {"symbol": symbol}

    async def gate(request):
        return ApprovalDecision.deny("not allowed here")

    client = FakeAppServer()
    engine = CodexEngine(
        client=client,
        config=CodingAgentConfig(codex=CodexPolicy(preapprove_dynamic_tools=False), approval_gate=gate),
    )
    agent = Agent(name="blocked", engine=engine, tools=[tracked_quote])

    assert agent("Find AMZN").ok
    assert calls == []  # the tool body must never execute
    assert client.tool_results[-1] == {
        "success": False,
        "contentItems": [{"type": "inputText", "text": "not allowed here"}],
    }


class TestDurableThreadSerialisation:
    """Two turns must not be appended to one transcript at the same time."""

    def test_concurrent_first_runs_do_not_open_two_threads(self):
        # Without a lock before the first id exists, both runs start their own
        # durable thread and race to store the id: one conversation is
        # orphaned, and later turns resume only the survivor.
        import asyncio as aio

        class SlowFake(FakeAppServer):
            async def run(self, **kwargs):
                await aio.sleep(0.01)  # widen the window
                return await super().run(**kwargs)

        fake = SlowFake(result=CodexRunResult(text="ok", thread_id="thread-9"))
        agent = Agent(CodexEngine(client=fake, persist_thread=True), name="a")

        async def both():
            await aio.gather(agent.run("one"), agent.run("two"))

        aio.run(both())

        # Serialised: the first opens the thread, the second resumes it.
        assert fake.thread_ids_seen == [None, "thread-9"]


class TestDurableFailureHandling:
    """What a durable thread needs when a turn does *not* come back."""

    def test_a_timeout_after_the_turn_was_sent_is_not_retryable(self):
        # asyncio.wait_for cancels with CancelledError, a BaseException that
        # unwinds past the client's own uncertainty handling — so the engine is
        # the last place that can say "this may already be committed".
        import asyncio as aio

        class HangingMidTurn(FakeAppServer):
            async def run(self, *, progress=None, **kwargs):
                progress.update({"thread_id": "thread-live", "turn_sent": True})
                await aio.sleep(3600)

        engine = CodexEngine(client=HangingMidTurn(), persist_thread=True, request_timeout=0.05)
        result = Agent(engine, name="a")("go")

        assert not result.ok
        assert result.error.retryable is False
        assert "unknown" in result.error.message.lower()
        # ...and the handle survives, because inspecting the thread is the
        # only way to find out what happened.
        assert engine.thread_id == "thread-live"

    def test_a_timeout_before_anything_was_sent_stays_retryable(self):
        # A hang in startup/thread creation committed nothing: marking it
        # non-retryable would make every slow launch look like data loss.
        import asyncio as aio

        class HangingAtStartup(FakeAppServer):
            async def run(self, *, progress=None, **kwargs):
                await aio.sleep(3600)

        engine = CodexEngine(client=HangingAtStartup(), persist_thread=True, request_timeout=0.05)
        result = Agent(engine, name="a")("go")

        assert not result.ok
        assert result.error.retryable is True

    def test_an_ephemeral_timeout_stays_retryable(self):
        # Nothing survives the subprocess there, so a retry is a clean restart.
        import asyncio as aio

        class Hanging(FakeAppServer):
            async def run(self, **kwargs):
                await aio.sleep(3600)

        engine = CodexEngine(client=Hanging(), request_timeout=0.05, max_retries=0)
        result = Agent(engine, name="a")("go")

        assert not result.ok
        assert result.error.retryable is True

    def test_an_uncertain_first_turn_still_yields_the_thread_id(self):
        # Inspecting the thread is the documented recovery path; losing the id
        # would make the next call open a new thread instead.
        from lazybridge.engines.codex.app_server import CodexTurnUncertain

        class Uncertain(FakeAppServer):
            async def run(self, **kwargs):
                raise CodexTurnUncertain("lost", thread_id="thread-77", turn_id="turn-1")

        engine = CodexEngine(client=Uncertain(), persist_thread=True)
        result = Agent(engine, name="a")("go")

        assert not result.ok
        assert engine.thread_id == "thread-77"

    def test_a_review_with_no_deltas_still_streams_its_findings(self):
        # Inline native reviews emit no item/agentMessage/delta at all.
        class Silent(FakeAppServer):
            async def run(self, **kwargs):
                return CodexRunResult(text="- [P1] something", thread_id="t")

        agent = Agent(CodexEngine(client=Silent(), review_target={"type": "uncommittedChanges"}), name="a")

        async def collect() -> str:
            return "".join([chunk async for chunk in agent.stream("ignored")])

        assert asyncio.run(collect()) == "- [P1] something"


class TestLockScoping:
    """Locks must survive the way LazyBridge actually calls engines."""

    def test_a_durable_engine_works_across_separate_event_loops(self):
        # Every synchronous Agent.__call__ runs on a FRESH loop, and an
        # asyncio.Lock binds to the loop that first waits on it — a
        # process-wide lock cache therefore raised "bound to a different event
        # loop" on the second sync call (reproduced by the Claude reviewer).
        fake = FakeAppServer(result=CodexRunResult(text="ok", thread_id="thread-9"))
        engine = CodexEngine(client=fake, persist_thread=True)
        agent = Agent(engine, name="a")

        first = agent("one")  # loop A
        second = agent("two")  # loop B, same thread id

        assert first.ok and second.ok
        assert fake.thread_ids_seen == [None, "thread-9"]

    def test_a_run_starting_after_the_id_exists_still_queues(self):
        # The hand-off hole: a run that begins once thread_id is set would key
        # straight onto the shared lock and pass the run that created it.
        import asyncio as aio

        order: list[str] = []

        class Tracking(FakeAppServer):
            async def run(self, **kwargs):
                order.append(f"enter:{kwargs.get('thread_id')}")
                await aio.sleep(0.05)
                order.append(f"exit:{kwargs.get('thread_id')}")
                return self.result

        fake = Tracking(result=CodexRunResult(text="ok", thread_id="T"))
        agent = Agent(CodexEngine(client=fake, persist_thread=True), name="a")

        async def three():
            a = aio.create_task(agent.run("a"))
            b = aio.create_task(agent.run("b"))
            await aio.sleep(0.075)  # after a's turn opened T, while b waits
            c = aio.create_task(agent.run("c"))
            await aio.gather(a, b, c)

        aio.run(three())

        # Strict enter/exit alternation: never two turns inside at once.
        assert order == [
            "enter:None",
            "exit:None",
            "enter:T",
            "exit:T",
            "enter:T",
            "exit:T",
        ]


class TestHandleRetention:
    """The thread id is the recovery path: it must survive every exit."""

    def test_an_external_cancellation_still_leaves_the_handle(self):
        # CancelledError is a BaseException: it reaches none of run()'s
        # handlers, yet the durable thread it started is real.
        import asyncio as aio

        class SlowAfterStart(FakeAppServer):
            async def run(self, *, progress=None, **kwargs):
                progress.update({"thread_id": "thread-live", "turn_sent": True})
                await aio.sleep(3600)

        engine = CodexEngine(client=SlowAfterStart(), persist_thread=True, request_timeout=None)
        agent = Agent(engine, name="a")

        async def cancel_it():
            task = aio.create_task(agent.run("go"))
            await aio.sleep(0.05)
            task.cancel()
            with pytest.raises(aio.CancelledError):
                await task

        aio.run(cancel_it())

        assert engine.thread_id == "thread-live"

    def test_a_rejected_first_turn_keeps_the_id_but_not_the_history_flag(self):
        # The thread exists but is empty: withholding Memory from the next
        # call would starve a model that has never seen any.
        from lazybridge.engines.codex.app_server import CodexRequestRejected

        class Rejecting(FakeAppServer):
            async def run(self, *, progress=None, **kwargs):
                # Exactly what the real client records: the request WAS
                # transmitted, and then refused.
                progress.update({"thread_id": "thread-empty", "turn_sent": True, "rejected": True})
                raise CodexRequestRejected("invalid review target")

        engine = CodexEngine(client=Rejecting(), persist_thread=True)
        result = Agent(engine, name="a")("go")

        assert not result.ok
        assert engine.thread_id == "thread-empty"
        assert engine._resuming is False
