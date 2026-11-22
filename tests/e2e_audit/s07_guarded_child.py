"""Failed and guarded child results do not escape to an orchestrator."""

import asyncio
import sqlite3
import sys
from pathlib import Path

from _harness import scenario

from lazybridge import Agent, Envelope, Session
from lazybridge.engines.codex.dynamic_tools import dispatcher
from lazybridge.envelope import ErrorInfo
from lazybridge.guardrails import ContentGuard, GuardAction
from lazybridge.session import EventType

SECRET = "UNPUBLISHED-PRIVATE-DRAFT-4931"
PARTIAL = "FAILED-PARTIAL-DRAFT-8271"


async def check(directory):
    received = []

    class ParentEngine:
        async def run(self, env, *, tools, **kwargs):
            result = await dispatcher(tools)("writer", {"task": "write"})
            received.append(result)
            return Envelope(task=env.task, payload="handled")

    class Failed:
        async def run(self, env, **kwargs):
            return Envelope(payload=PARTIAL, error=ErrorInfo(type="Failure", message="worker failed"))

    class Draft:
        def __init__(self, text):
            self.text = text

        async def run(self, env, *, session, **kwargs):
            session.emit(EventType.MODEL_RESPONSE, {"content": self.text})
            return Envelope(task=env.task, payload=self.text)

        async def stream(self, env, *, session, **kwargs):
            session.emit(EventType.MODEL_RESPONSE, {"content": self.text})
            yield self.text

    class Review(ContentGuard):
        def __init__(self, allow):
            super().__init__()
            self.allow = allow
            self.entered = asyncio.Event()
            self.finish = asyncio.Event()

        async def acheck_output(self, text):
            self.entered.set()
            await self.finish.wait()
            return GuardAction.allow() if self.allow else GuardAction.block("review denied")

    with Session(db=str(directory / "session.sqlite")) as session:
        child = Agent(name="writer", engine=Failed(), session=session)
        parent = Agent(name="parent", engine=ParentEngine(), tools=[child], session=session)
        assert (await parent.run("delegate")).ok
        assert received[-1]["success"] is False and PARTIAL not in str(received[-1])
        denied = Agent(
            name="writer",
            engine=Draft(SECRET),
            session=session,
            guard=ContentGuard(output_fn=lambda text: GuardAction.block("denied")),
        )
        parent = Agent(name="parent", engine=ParentEngine(), tools=[denied], session=session)
        assert (await parent.run("delegate")).ok
        assert received[-1]["success"] is False and SECRET not in str(received[-1])
        for allowed in (False, True):
            review = Review(allowed)
            text = "APPROVED-PUBLIC-TEXT" if allowed else SECRET
            guarded = Agent(name="stream-writer", engine=Draft(text), session=session, guard=review)
            chunks = []

            async def consume(guarded=guarded, chunks=chunks, allowed=allowed):
                try:
                    async for chunk in guarded.stream("write"):
                        chunks.append(chunk)
                except ValueError:
                    if allowed:
                        raise

            pending = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(review.entered.wait(), 5)
                assert not chunks, "stream released before guard decision"
            finally:
                review.finish.set()
                await asyncio.wait_for(pending, 5)
            assert ("".join(chunks) == text) if allowed else not chunks
        rows = session.events.query()
        assert SECRET not in str(rows) and SECRET not in str(received)
    with sqlite3.connect(directory / "session.sqlite") as db:
        for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            quoted = '"' + table.replace('"', '""') + '"'
            assert SECRET not in str(db.execute(f"SELECT * FROM {quoted}").fetchall())
    return "Failed and blocked child became tool errors; denied draft stayed private; approved stream released after review."


if __name__ == "__main__":
    sys.exit(
        scenario(
            Path(__file__).stem,
            "A3 failed / guarded child",
            "Propagate tool errors and withhold denied drafts from parent, session and stream.",
            lambda d: asyncio.run(check(d)),
        )
    )
