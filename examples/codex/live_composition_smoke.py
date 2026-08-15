"""Live proof that ``CodexEngine`` is "an engine like the others".

Every LazyBridge composition pattern dispatches through ``Engine.run()`` /
``Engine.stream()`` / ``Agent._run_as_tool()`` and never special-cases the
engine type, so if these four pass against a real ``codex app-server`` the
engine composes exactly like ``LLMEngine`` and ``ClaudeCodeEngine``:

1. a Codex agent used as a *tool* by another Codex agent (multi-agent),
2. ``Agent.chain`` (Plan engine driving two Codex agents),
3. ``Agent.stream`` (token streaming through the App Server deltas),
4. ``output=<pydantic model>`` (structured output).

Requires a locally authenticated ``codex`` CLI; it runs ~6 real turns.

    .venv\\Scripts\\python.exe examples\\codex\\live_composition_smoke.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from lazybridge import Agent, CodexEngine, Memory, Session
from lazybridge.session import EventType


def lazybridge_probe() -> str:
    """Return a deterministic string for the Codex engine smoke test."""
    return "lazybridge-engine-ok"


def _engine(system: str | None = None) -> CodexEngine:
    return CodexEngine(system=system)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}", flush=True)
    return ok


def test_agent_as_tool() -> bool:
    """A Codex agent calling another Codex agent exposed as a tool."""
    session = Session()
    specialist = Agent(
        name="probe_specialist",
        description="Returns the LazyBridge probe value. Call it with an empty task.",
        engine=_engine("Call lazybridge_probe once and reply with only its result."),
        tools=[lazybridge_probe],
        session=session,
    )
    supervisor = Agent(
        name="supervisor",
        engine=_engine("Delegate to the probe_specialist tool; never answer from memory."),
        tools=[specialist],
        memory=Memory(),
        session=session,
    )
    result = supervisor("Ask probe_specialist for the probe value and reply with only that value.")
    session.flush()
    tool_calls = session.events.query(event_type=EventType.TOOL_CALL)
    usage = session.usage_summary()
    return check(
        "agent-as-tool (multi-agent)",
        result.ok and "lazybridge-engine-ok" in result.text() and len(tool_calls) >= 2,
        f"text={result.text()[:60]!r} tool_calls={[c.get('tool_name') or c.get('data') for c in tool_calls][:4]} usage_summary={usage}",
    )


def test_chain() -> bool:
    """Agent.chain: output of the first Codex agent feeds the second."""
    first = Agent(name="emitter", engine=_engine("Reply with only the word: alpha."), description="Emits a token")
    second = Agent(
        name="shouter",
        engine=_engine("Uppercase whatever you are given. Reply with only the uppercased text."),
        description="Uppercases its input",
    )
    chained = Agent.chain(first, second, name="emit-then-shout")
    result = chained("Start.")
    return check("Agent.chain", result.ok and "ALPHA" in result.text().upper(), f"text={result.text()[:60]!r}")


def test_stream() -> bool:
    """Engine.stream() over the App Server's item/agentMessage/delta events."""

    async def run() -> list[str]:
        agent = Agent(name="streamer", engine=_engine("Reply with only: one two three."))
        return [chunk async for chunk in agent.stream("Say it.")]

    chunks = asyncio.run(run())
    joined = "".join(chunks)
    return check(
        "Agent.stream",
        len(chunks) > 1 and "one" in joined.lower(),
        f"{len(chunks)} chunks, joined={joined[:60]!r}",
    )


class Quote(BaseModel):
    symbol: str
    price: float


def test_structured_output() -> bool:
    """output=<model>: the engine primes Codex to answer as JSON."""
    agent = Agent(name="quoter", engine=_engine(), output=Quote)
    result = agent("The symbol is AMZN and its price is 123.45. Report it.")
    payload = result.payload
    return check(
        "output=<pydantic model>",
        result.ok and isinstance(payload, Quote) and payload.symbol == "AMZN" and payload.price == 123.45,
        f"payload={payload!r}",
    )


def main() -> None:
    results = [
        test_agent_as_tool(),
        test_chain(),
        test_stream(),
        test_structured_output(),
    ]
    print(f"\n{sum(results)}/{len(results)} passed", flush=True)


if __name__ == "__main__":
    main()
