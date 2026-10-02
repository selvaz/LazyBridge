"""Thread-lock behaviour of ``CodexEngine`` across engines sharing an alias.

Ordered by explicit events: the fake App Server parks each turn on a per-call
gate the test opens. "Still waiting" is asserted after a bounded number of
event-loop yields (the negative half of an ordering check), never by sleeping.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lazybridge import Envelope
from lazybridge.engines.codex.app_server import CodexRunResult
from lazybridge.engines.codex.engine import CodexEngine
from lazybridge.engines.sessions import SessionRegistry


async def settle(cycles: int = 25) -> None:
    for _ in range(cycles):
        await asyncio.sleep(0)


class GatedCodex:
    """Fake client: call ``i`` reports its thread id early, then parks on ``gate(i)``."""

    def __init__(self, new_thread: str = "thread-1", *, fail: set[int] | None = None):
        self.new_thread = new_thread
        self.fail = fail or set()
        self.thread_ids_seen: list[str | None] = []
        self._entered: dict[int, asyncio.Event] = {}
        self._gates: dict[int, asyncio.Event] = {}
        self.finished: set[int] = set()
        self.finished_at_entry: list[frozenset[int]] = []

    def entered(self, i: int) -> asyncio.Event:
        return self._entered.setdefault(i, asyncio.Event())

    def gate(self, i: int) -> asyncio.Event:
        return self._gates.setdefault(i, asyncio.Event())

    async def run(self, *, thread_id=None, progress=None, on_text=None, **_kwargs):
        i = len(self.thread_ids_seen)
        self.thread_ids_seen.append(thread_id)
        self.finished_at_entry.append(frozenset(self.finished))
        tid = thread_id or self.new_thread
        if progress is not None:
            progress["thread_id"] = tid
            progress["turn_sent"] = True
        self.entered(i).set()
        try:
            if on_text is not None:
                await on_text("hi")
            await self.gate(i).wait()
            if i in self.fail:
                raise RuntimeError(f"boom-{i}")
            return CodexRunResult(text="ok", thread_id=tid)
        finally:
            self.finished.add(i)


def _call(engine: CodexEngine, mode: str):
    env = Envelope(task="go")
    kwargs: dict[str, Any] = {"tools": [], "output_type": str, "memory": None, "session": None}

    async def via_run() -> Any:
        return await engine.run(env, **kwargs)

    async def via_stream() -> Any:
        return [chunk async for chunk in engine.stream(env, **kwargs)]

    return via_run() if mode == "run" else via_stream()


@pytest.fixture
def registry(tmp_path):
    return SessionRegistry(tmp_path / "sessions.json")


def _engine(client, registry, tmp_path, **kw) -> CodexEngine:
    return CodexEngine(client=client, cwd=str(tmp_path), session_alias="probe", session_registry=registry, **kw)


class TestUnknownAlias:
    @pytest.mark.parametrize("modes", [("run", "run"), ("stream", "stream"), ("run", "stream"), ("stream", "run")])
    def test_two_engines_on_one_unknown_alias_share_one_thread(self, registry, tmp_path, modes):
        async def scenario() -> None:
            codex = GatedCodex("thread-A")
            one, two = _engine(codex, registry, tmp_path), _engine(codex, registry, tmp_path)
            first = asyncio.create_task(_call(one, modes[0]))
            await codex.entered(0).wait()
            # The first turn has already reported its id through ``progress``...
            second = asyncio.create_task(_call(two, modes[1]))
            await settle()
            # ...yet the second must not open a thread of its own.
            assert len(codex.thread_ids_seen) == 1
            codex.gate(0).set()
            await codex.entered(1).wait()
            codex.gate(1).set()
            await asyncio.gather(first, second)
            assert codex.thread_ids_seen == [None, "thread-A"]
            assert two.thread_id == "thread-A"

        asyncio.run(scenario())

    def test_a_waiter_cancelled_during_lock_acquisition_leaves_the_alias_free(self, registry, tmp_path):
        async def scenario() -> None:
            codex = GatedCodex("thread-W")
            one, two = _engine(codex, registry, tmp_path), _engine(codex, registry, tmp_path)
            holder = asyncio.create_task(_call(one, "run"))
            await codex.entered(0).wait()
            waiter = asyncio.create_task(_call(two, "run"))
            await settle()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert len(codex.thread_ids_seen) == 1
            codex.gate(0).set()
            await holder
            later = asyncio.create_task(_call(two, "run"))
            await codex.entered(1).wait()
            codex.gate(1).set()
            await later
            assert codex.thread_ids_seen == [None, "thread-W"]

        asyncio.run(scenario())

    def test_an_early_aclose_lets_the_queued_engine_proceed_on_the_created_thread(self, registry, tmp_path):
        async def scenario() -> None:
            codex = GatedCodex("thread-E")
            one, two = _engine(codex, registry, tmp_path), _engine(codex, registry, tmp_path)
            turn = one.stream(Envelope(task="go"), tools=[], output_type=str, memory=None, session=None)
            assert await turn.__anext__() == "hi"
            successor = asyncio.create_task(_call(two, "run"))
            await settle()
            assert len(codex.thread_ids_seen) == 1
            await turn.aclose()
            await codex.entered(1).wait()
            assert 0 in codex.finished_at_entry[1], "the queued engine started before the first turn was torn down"
            codex.gate(1).set()
            await successor
            # The abandoned turn had already opened the thread; the alias queue
            # hands that id on rather than letting the successor open another.
            assert codex.thread_ids_seen == [None, "thread-E"]

        asyncio.run(scenario())

    def test_a_failed_creating_turn_still_lets_the_next_engine_resume_its_thread(self, registry, tmp_path):
        async def scenario() -> None:
            codex = GatedCodex("thread-F", fail={0})
            one, two = _engine(codex, registry, tmp_path), _engine(codex, registry, tmp_path)
            first = asyncio.create_task(_call(one, "run"))
            await codex.entered(0).wait()
            second = asyncio.create_task(_call(two, "run"))
            await settle()
            codex.gate(0).set()
            assert not (await first).ok
            await codex.entered(1).wait()
            codex.gate(1).set()
            await second
            assert codex.thread_ids_seen == [None, "thread-F"]

        asyncio.run(scenario())
