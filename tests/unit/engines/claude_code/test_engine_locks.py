"""Session-lock behaviour of ``ClaudeCodeEngine``: ``run()`` and ``stream()``.

Every test is ordered by explicit events (the fake SDK parks each turn on a
per-call gate the test opens), never by sleeping for a while: "B must still be
waiting" is asserted after a bounded number of event-loop yields, which is only
ever the *negative* half of an ordering check.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lazybridge import Envelope
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions, ClaudeSdkResult, ClaudeSdkStreamEvent
from lazybridge.engines.sessions import SessionRegistry


@pytest.fixture(autouse=True)
def _no_real_tagging(monkeypatch):
    import sys
    import types

    module = sys.modules.get("claude_agent_sdk")
    if module is None:
        module = types.ModuleType("claude_agent_sdk")
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    monkeypatch.setattr(module, "tag_session", lambda *a, **k: None, raising=False)


async def settle(cycles: int = 25) -> None:
    """Let every runnable task make progress (the 'still waiting' half)."""
    for _ in range(cycles):
        await asyncio.sleep(0)


class GatedSdk:
    """A fake SDK whose turns stay open until the test releases them.

    Call ``i`` (0-based, in arrival order) sets ``entered(i)``, optionally
    reports its session id early (as the real SDK's init message does), then
    parks on ``gate(i)``. Streamed turns yield one chunk before parking, so a
    consumer can be holding the turn open at a ``yield``.
    """

    def __init__(self, new_id: str = "sess-1", *, early: bool = False, fail: set[int] | None = None):
        self.new_id = new_id
        self.early = early
        self.fail = fail or set()
        self.options: list[ClaudeSdkOptions] = []
        self._entered: dict[int, asyncio.Event] = {}
        self._gates: dict[int, asyncio.Event] = {}
        self.torn_down: set[int] = set()
        #: for each call: which earlier calls had finished tearing down when it began
        self.torn_down_at_entry: list[frozenset[int]] = []

    def entered(self, i: int) -> asyncio.Event:
        return self._entered.setdefault(i, asyncio.Event())

    def gate(self, i: int) -> asyncio.Event:
        return self._gates.setdefault(i, asyncio.Event())

    @property
    def resumes(self) -> list[str | None]:
        return [o.resume for o in self.options]

    def _begin(self, options: ClaudeSdkOptions) -> tuple[int, str]:
        i = len(self.options)
        self.options.append(options)
        self.torn_down_at_entry.append(frozenset(self.torn_down))
        session_id = options.resume or self.new_id
        self.entered(i).set()
        if self.early and options.on_session_id is not None:
            options.on_session_id(session_id)
        return i, session_id

    async def run(self, prompt, *, options, attachments=()):
        i, session_id = self._begin(options)
        try:
            await self.gate(i).wait()
            if i in self.fail:
                raise RuntimeError(f"boom-{i}")
            return ClaudeSdkResult(text="ok", session_id=session_id)
        finally:
            self.torn_down.add(i)

    async def stream(self, prompt, *, options, attachments=()):
        i, session_id = self._begin(options)
        try:
            yield ClaudeSdkStreamEvent(text="hi")
            await self.gate(i).wait()
            if i in self.fail:
                raise RuntimeError(f"boom-{i}")
            yield ClaudeSdkStreamEvent(session_id=session_id, final=True)
        finally:
            self.torn_down.add(i)


def _call(engine: ClaudeCodeEngine, mode: str):
    """One whole turn against ``engine``, via ``run`` or ``stream``."""
    env = Envelope(task="go")
    kwargs: dict[str, Any] = {"tools": [], "output_type": str, "memory": None, "session": None}

    async def via_run() -> Any:
        return await engine.run(env, **kwargs)

    async def via_stream() -> Any:
        return [chunk async for chunk in engine.stream(env, **kwargs)]

    return via_run() if mode == "run" else via_stream()


def _locks_free(*engines: ClaudeCodeEngine) -> bool:
    return not any(e._own_lock.locked() for e in engines) and not any(
        lock.locked() for lock in [*ClaudeCodeEngine._session_locks.values(), *ClaudeCodeEngine._alias_locks.values()]
    )


@pytest.fixture(autouse=True)
def _fresh_class_locks():
    ClaudeCodeEngine._session_locks.clear()
    ClaudeCodeEngine._alias_locks.clear()
    yield
    ClaudeCodeEngine._session_locks.clear()
    ClaudeCodeEngine._alias_locks.clear()


class TestFirstTurns:
    def test_two_first_streams_on_one_engine_do_not_open_two_sessions(self):
        async def scenario() -> None:
            sdk = GatedSdk()
            engine = ClaudeCodeEngine(client=sdk, persist_session=True)
            first = asyncio.create_task(_call(engine, "stream"))
            await sdk.entered(0).wait()
            second = asyncio.create_task(_call(engine, "stream"))
            await settle()
            assert len(sdk.options) == 1, "the second stream entered the SDK while the first held the turn"
            sdk.gate(0).set()
            await sdk.entered(1).wait()
            sdk.gate(1).set()
            assert await first == ["hi"] and await second == ["hi"]
            assert sdk.resumes == [None, "sess-1"]
            assert _locks_free(engine)

        asyncio.run(scenario())

    @pytest.mark.parametrize("modes", [("run", "stream"), ("stream", "run")])
    def test_a_run_and_a_stream_on_one_engine_serialise(self, modes):
        async def scenario() -> None:
            sdk = GatedSdk()
            engine = ClaudeCodeEngine(client=sdk, persist_session=True)
            first = asyncio.create_task(_call(engine, modes[0]))
            await sdk.entered(0).wait()
            second = asyncio.create_task(_call(engine, modes[1]))
            await settle()
            assert len(sdk.options) == 1
            sdk.gate(0).set()
            await sdk.entered(1).wait()
            sdk.gate(1).set()
            await asyncio.gather(first, second)
            assert sdk.resumes == [None, "sess-1"]

        asyncio.run(scenario())

    def test_a_persistent_engine_reused_across_event_loops_still_serialises(self):
        # Contention in the first asyncio.run() must not bind the engine's own
        # lock to that loop: the id is known afterwards, yet the own lock is
        # still taken on every turn.
        sdk = GatedSdk()
        engine = ClaudeCodeEngine(client=sdk, persist_session=True)

        async def first_loop() -> None:
            first = asyncio.create_task(_call(engine, "run"))
            await sdk.entered(0).wait()
            second = asyncio.create_task(_call(engine, "run"))
            await settle()
            assert len(sdk.options) == 1
            sdk.gate(0).set()
            await sdk.entered(1).wait()
            sdk.gate(1).set()
            await asyncio.gather(first, second)

        async def second_loop() -> None:
            first = asyncio.create_task(_call(engine, "stream"))
            await sdk.entered(2).wait()
            second = asyncio.create_task(_call(engine, "run"))
            await settle()
            assert len(sdk.options) == 3
            sdk.gate(2).set()
            await sdk.entered(3).wait()
            sdk.gate(3).set()
            await asyncio.gather(first, second)

        asyncio.run(asyncio.wait_for(first_loop(), 10))
        ClaudeCodeEngine._session_locks.clear()  # the class-wide id locks are a known, separate limitation
        asyncio.run(asyncio.wait_for(second_loop(), 10))
        assert sdk.resumes == [None, "sess-1", "sess-1", "sess-1"]

    @pytest.mark.parametrize("modes", [("run", "run"), ("run", "stream"), ("stream", "run"), ("stream", "stream")])
    def test_own_lock_to_id_lock_handoff_with_three_calls(self, modes):
        # a creates the session, b queued behind it on the engine's own lock,
        # c arrives only after the id exists. c must NOT key straight onto the
        # (uncontended) id lock and run alongside b.
        async def scenario() -> None:
            sdk = GatedSdk()
            engine = ClaudeCodeEngine(client=sdk, persist_session=True)
            a = asyncio.create_task(_call(engine, modes[0]))
            await sdk.entered(0).wait()
            b = asyncio.create_task(_call(engine, modes[1]))
            await settle()
            sdk.gate(0).set()
            await a
            assert engine.session_id == "sess-1"
            await sdk.entered(1).wait()  # b is now inside, c has not been issued
            c = asyncio.create_task(_call(engine, modes[0]))
            await settle()
            assert len(sdk.options) == 2, "c ran alongside b: the id lock bypassed the own lock"
            sdk.gate(1).set()
            await sdk.entered(2).wait()
            sdk.gate(2).set()
            await asyncio.gather(b, c)
            assert sdk.resumes == [None, "sess-1", "sess-1"]
            assert _locks_free(engine)

        asyncio.run(scenario())


class TestSharedRawId:
    @pytest.mark.parametrize(
        "modes", [("stream", "stream"), ("run", "stream"), ("stream", "run"), ("run", "run")], ids="-".join
    )
    def test_two_engines_resuming_one_id_serialise(self, modes):
        async def scenario() -> None:
            sdk = GatedSdk()
            one = ClaudeCodeEngine(client=sdk, session_id="sess-R")
            two = ClaudeCodeEngine(client=sdk, session_id="sess-R")
            first = asyncio.create_task(_call(one, modes[0]))
            await sdk.entered(0).wait()
            second = asyncio.create_task(_call(two, modes[1]))
            await settle()
            assert len(sdk.options) == 1
            sdk.gate(0).set()
            await sdk.entered(1).wait()
            sdk.gate(1).set()
            await asyncio.gather(first, second)
            assert sdk.resumes == ["sess-R", "sess-R"]
            assert _locks_free(one, two)

        asyncio.run(scenario())

    def test_an_ephemeral_engine_still_takes_no_lock(self):
        async def scenario() -> None:
            sdk = GatedSdk()
            engine = ClaudeCodeEngine(client=sdk)
            first = asyncio.create_task(_call(engine, "stream"))
            second = asyncio.create_task(_call(engine, "stream"))
            await sdk.entered(0).wait()
            await sdk.entered(1).wait()  # both inside at once
            sdk.gate(0).set()
            sdk.gate(1).set()
            await asyncio.gather(first, second)

        asyncio.run(scenario())


class TestUnknownAlias:
    @pytest.fixture
    def registry(self, tmp_path):
        return SessionRegistry(tmp_path / "sessions.json")

    @pytest.mark.parametrize("modes", [("stream", "stream"), ("run", "stream"), ("stream", "run")])
    def test_a_second_engine_waits_for_the_creating_turn_then_resumes_its_id(self, registry, tmp_path, modes):
        async def scenario() -> None:
            sdk = GatedSdk("sess-A", early=True)

            def engine() -> ClaudeCodeEngine:
                return ClaudeCodeEngine(client=sdk, cwd=str(tmp_path), session_alias="probe", session_registry=registry)

            one, two = engine(), engine()
            first = asyncio.create_task(_call(one, modes[0]))
            await sdk.entered(0).wait()
            # The first turn has already reported (and bound) its id...
            assert registry.resolve("claude", tmp_path, "probe") == "sess-A"
            second = asyncio.create_task(_call(two, modes[1]))
            await settle()
            # ...yet the second still waits: the creating turn is not finished.
            assert len(sdk.options) == 1
            sdk.gate(0).set()
            await sdk.entered(1).wait()
            sdk.gate(1).set()
            await asyncio.gather(first, second)
            assert sdk.resumes == [None, "sess-A"]
            assert two.session_id == "sess-A"
            assert _locks_free(one, two)

        asyncio.run(scenario())


class TestTeardown:
    @staticmethod
    async def _successor_after(sdk: GatedSdk, engine: ClaudeCodeEngine, mode: str, index: int):
        task = asyncio.create_task(_call(engine, mode))
        await settle()
        assert len(sdk.options) == index, "the successor started before the first turn finished"
        return task

    def test_an_early_aclose_tears_the_sdk_down_before_the_lock_is_released(self):
        async def scenario() -> None:
            sdk = GatedSdk()
            engine = ClaudeCodeEngine(client=sdk, session_id="sess-E")
            env = Envelope(task="go")
            turn = engine.stream(env, tools=[], output_type=str, memory=None, session=None)
            assert await turn.__anext__() == "hi"  # the turn is parked at a yield
            successor = await self._successor_after(sdk, engine, "stream", 1)
            await turn.aclose()
            await sdk.entered(1).wait()
            assert 0 in sdk.torn_down_at_entry[1], "the lock was released before the SDK query was closed"
            sdk.gate(1).set()
            assert await successor == ["hi"]
            assert _locks_free(engine)

        asyncio.run(scenario())

    def test_cancelling_a_consumer_waiting_on_the_next_chunk_cleans_up(self):
        async def scenario() -> None:
            sdk = GatedSdk()
            engine = ClaudeCodeEngine(client=sdk, session_id="sess-C")
            env = Envelope(task="go")

            async def consume() -> None:
                async for _ in engine.stream(env, tools=[], output_type=str, memory=None, session=None):
                    pass

            consumer = asyncio.create_task(consume())
            await sdk.entered(0).wait()
            await settle()  # parked on gate(0), inside anext()
            successor = await self._successor_after(sdk, engine, "run", 1)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            await sdk.entered(1).wait()
            assert 0 in sdk.torn_down_at_entry[1]
            sdk.gate(1).set()
            await successor
            assert _locks_free(engine)

        asyncio.run(scenario())

    @pytest.mark.parametrize("mode", ["run", "stream"])
    def test_an_sdk_failure_releases_the_lock_for_the_queued_successor(self, mode):
        async def scenario() -> None:
            sdk = GatedSdk(fail={0})
            engine = ClaudeCodeEngine(client=sdk, session_id="sess-F", max_retries=0)
            failing = asyncio.create_task(_call(engine, mode))
            await sdk.entered(0).wait()
            successor = await self._successor_after(sdk, engine, mode, 1)
            sdk.gate(0).set()
            if mode == "stream":
                with pytest.raises(RuntimeError, match="boom-0"):
                    await failing
            else:
                assert not (await failing).ok  # run() reports the failure as an error envelope
            await sdk.entered(1).wait()
            assert 0 in sdk.torn_down_at_entry[1]
            sdk.gate(1).set()
            await successor
            assert _locks_free(engine)

        asyncio.run(scenario())

    @pytest.mark.parametrize("mode", ["run", "stream"])
    def test_cancelling_a_waiter_during_lock_acquisition_leaves_no_lock_held(self, mode):
        async def scenario() -> None:
            sdk = GatedSdk()
            engine = ClaudeCodeEngine(client=sdk, session_id="sess-W")
            holder = asyncio.create_task(_call(engine, mode))
            await sdk.entered(0).wait()
            waiter = asyncio.create_task(_call(engine, mode))
            await settle()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert len(sdk.options) == 1  # the cancelled waiter never reached the SDK
            sdk.gate(0).set()
            await holder
            assert _locks_free(engine)
            later = asyncio.create_task(_call(engine, mode))
            await sdk.entered(1).wait()
            sdk.gate(1).set()
            await later
            assert _locks_free(engine)

        asyncio.run(scenario())

    def test_an_unconsumed_stream_never_takes_the_lock(self):
        async def scenario() -> None:
            sdk = GatedSdk()
            engine = ClaudeCodeEngine(client=sdk, session_id="sess-U")
            engine.stream(Envelope(task="go"), tools=[], output_type=str, memory=None, session=None)
            assert _locks_free(engine)

        asyncio.run(scenario())
