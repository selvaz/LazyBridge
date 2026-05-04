"""Test utilities for LazyBridge — scripted inputs, mock agents, helpers.

The name is intentionally module-level (``lazybridge.testing``) so test
modules can import stable public helpers instead of each recreating
private ``_scripted(...)`` closures.  Nothing here is used at runtime;
import only from tests / examples.

Public surface::

    from lazybridge.testing import (
        scripted_inputs,     # sync input_fn over a list / iterable
        scripted_ainputs,    # async variant (for SupervisorEngine.ainput_fn)
        MockAgent,           # deterministic test double for Agent
        MockCall,            # recorded invocation (task/context/envelopes)
    )
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo


def scripted_inputs(lines: Iterable[str]) -> Callable[[str], str]:
    """Return a sync ``input_fn`` that yields ``lines`` in order.

    Drop-in replacement for ``input()`` in HIL engine tests — lets the
    suite run deterministically without ``input()`` hangs.

    Raises ``StopIteration`` (caught by most REPL loops) when the
    scripted list is exhausted.  Supply an over-long list if you're
    unsure how many prompts the engine will issue.

    Example::

        from lazybridge import Agent
        from lazybridge.ext.hil import SupervisorEngine
        from lazybridge.testing import scripted_inputs

        sup = Agent(
            engine=SupervisorEngine(
                tools=[my_tool],
                input_fn=scripted_inputs(["my_tool(hello)", "continue"]),
            ),
            name="sup",
        )
        env = sup("task")
    """
    it = iter(lines)

    def _fn(_prompt: str) -> str:
        return next(it)

    return _fn


def scripted_ainputs(lines: Iterable[str]) -> Callable[[str], Awaitable[str]]:
    """Async counterpart of :func:`scripted_inputs`.

    Use when you want the supervisor's event-loop-native REPL path
    (pass as ``ainput_fn=``) — necessary to test cancellation /
    timeout semantics that don't exercise through the thread-pool
    fallback.
    """
    it = iter(lines)

    async def _fn(_prompt: str) -> str:
        return next(it)

    return _fn


# ---------------------------------------------------------------------------
# MockAgent — deterministic Agent test double for pipeline composition
# ---------------------------------------------------------------------------


#: Dict-response key meaning "match anything".  Put it last; first literal
#: substring match still wins so you can keep a catch-all default alongside
#: specific responses.
DEFAULT = "*"


@dataclass
class MockCall:
    """One recorded invocation of a :class:`MockAgent`.

    Captured per call so test suites can assert on what each agent was
    actually asked, what came back, and how long the call took.
    """

    task: str | None
    context: str | None
    env_in: Envelope
    env_out: Envelope
    elapsed_ms: float


class MockAgent:
    """Deterministic test double that quacks like :class:`lazybridge.Agent`.

    Designed for testing **pipeline composition and data transmission**
    without touching a real provider.  Drop-in compatible with:

    * ``Agent(..., tools=[mock])`` — wrapped via ``wrap_tool`` using the
      ``_is_lazy_agent`` duck-type marker; nested Envelope metadata rolls
      up through the tool boundary.
    * ``Plan(Step(target=mock, ...))`` — PlanEngine detects
      ``_is_lazy_agent`` and calls ``target.run(env)`` directly.
    * ``mock.as_tool()`` returns a standard :class:`Tool`.
    * ``Agent.chain(mock_a, mock_b)`` / ``Agent.parallel(mock_a, mock_b)``.

    **Response specification.**  The ``responses`` argument is resolved
    per call in this order:

    1. **Callable** ``fn(env) -> value`` (sync or async) — the incoming
       :class:`Envelope` is passed in.  Whatever the callable returns
       is re-fed through rules 2–4.
    2. **Dict** — keys are substrings checked against ``env.task``
       in insertion order; ``"*"`` is a catch-all default.  No match
       and no default raises :class:`RuntimeError`.
    3. **List** — one response per call, in order.  Exhausting the
       list raises :class:`RuntimeError` unless ``cycle=True``.
    4. **Scalar** — returned on every call.

    **Response value types** (after callable resolution):

    * :class:`Envelope` — returned verbatim; payload + metadata +
      error preserved.  Use this to test error-path propagation.
    * :class:`ErrorInfo` — returned as ``Envelope(error=...)``.
    * :class:`BaseException` instance — raised from ``run()`` so
      tests can assert on provider-style failures.
    * :class:`~pydantic.BaseModel` / str / dict / list / scalar —
      wrapped as the envelope payload with the mock's default
      metadata (tokens, cost, model, provider).

    **Recording.**  Every call is appended to :attr:`calls`.  Use
    :meth:`assert_called_with`, :meth:`assert_call_count`,
    :attr:`call_count`, :attr:`last_call` for assertions.
    :meth:`reset` clears both the call log and the list/cycle cursor.

    Example::

        from lazybridge import Agent, Plan, Step
        from lazybridge.testing import MockAgent

        researcher = MockAgent(
            {"weather": "sunny", "market": "bullish", "*": "no data"},
            name="researcher",
            default_input_tokens=50, default_output_tokens=30,
        )
        writer = MockAgent(
            lambda env: f"Report based on: {env.text()}",
            name="writer",
        )

        plan = Plan(
            Step(target=researcher, task="weather today"),
            Step(target=writer),   # from_prev by default
        )
        agent = Agent(engine=plan)
        env = agent("daily brief")

        assert "sunny" in env.text()
        assert researcher.call_count == 1
        assert writer.call_count == 1
        assert env.metadata.input_tokens + env.metadata.output_tokens > 0
    """

    _is_lazy_agent = True  # recognised by wrap_tool() and PlanEngine

    def __init__(
        self,
        responses: Any,
        *,
        name: str = "mock_agent",
        description: str | None = None,
        output: type = str,
        cycle: bool = False,
        delay_ms: float = 0.0,
        default_input_tokens: int = 10,
        default_output_tokens: int = 20,
        default_cost_usd: float = 0.0,
        default_latency_ms: float | None = None,
        default_model: str = "mock",
        default_provider: str = "mock",
    ) -> None:
        self._responses = responses
        self._cycle = cycle
        self._cursor = 0
        self.name = name
        self.description = description
        self.output = output
        # Mirror the Agent surface so test code that introspects agents
        # (session graph registration, verbose exporters, …) works
        # uniformly regardless of whether the agent is real or mocked.
        self.session: Any | None = None
        self.memory: Any | None = None
        self.sources: list[Any] = []
        self.guard: Any | None = None
        self.verify: Any | None = None
        self.timeout: float | None = None
        self.fallback: Any | None = None
        self._tools_raw: list[Any] = []
        self._tool_map: dict[str, Any] = {}

        self._delay_ms = float(delay_ms)
        self._default_input_tokens = int(default_input_tokens)
        self._default_output_tokens = int(default_output_tokens)
        self._default_cost_usd = float(default_cost_usd)
        self._default_latency_ms = default_latency_ms  # None = use measured elapsed
        self._default_model = default_model
        self._default_provider = default_provider

        self.calls: list[MockCall] = []

    # ------------------------------------------------------------------
    # Agent contract — run / __call__ / stream / as_tool
    # ------------------------------------------------------------------

    async def run(self, task: str | Envelope) -> Envelope:
        env_in = self._to_envelope(task)
        start = time.perf_counter()
        if self._delay_ms > 0:
            await asyncio.sleep(self._delay_ms / 1000.0)

        raw = await self._resolve(env_in)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Exceptions from the response spec are raised so the caller can
        # assert the failure mode directly (``pytest.raises(...)``) and
        # engines see a real exception just like they would from a
        # provider SDK.  Engines that wrap in error envelopes will do so.
        if isinstance(raw, BaseException):
            raise raw

        env_out = self._build_envelope(env_in, raw, elapsed_ms)
        self.calls.append(
            MockCall(
                task=env_in.task,
                context=env_in.context,
                env_in=env_in,
                env_out=env_out,
                elapsed_ms=elapsed_ms,
            )
        )
        return env_out

    def __call__(self, task: str | Envelope) -> Envelope:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(task))
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run(task)).result()

    async def stream(self, task: str | Envelope) -> AsyncGenerator[str, None]:
        """Minimal streaming surface: yields the final text in one chunk.

        Real streaming isn't meaningful for a deterministic mock, but
        tests that exercise the ``.stream()`` path still need something
        that behaves like an async generator.
        """
        env = await self.run(task)
        yield env.text()

    def as_tool(self, name: str | None = None, description: str | None = None) -> Any:
        from lazybridge.tools import Tool

        agent = self
        effective_name = name or self.name
        effective_desc = description or self.description or f"Run the {effective_name} mock agent."

        async def _run(task: str) -> Envelope:
            return await agent.run(task)

        _run.__name__ = effective_name
        _run.__doc__ = effective_desc
        return Tool(
            _run,
            name=effective_name,
            description=effective_desc,
            mode="signature",
            returns_envelope=True,
        )

    def definition(self) -> Any:
        """ToolDefinition for this mock — mirrors ``Agent.definition()``."""
        return self.as_tool().definition()

    # ------------------------------------------------------------------
    # Test-helper API
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> MockCall | None:
        return self.calls[-1] if self.calls else None

    def reset(self) -> None:
        """Clear recorded calls and rewind the list/cycle cursor."""
        self.calls.clear()
        self._cursor = 0

    def assert_called_with(
        self,
        task: str | None = None,
        *,
        contains: str | None = None,
    ) -> None:
        """Assert at least one call matched ``task=`` exactly or ``contains=``."""
        for c in self.calls:
            if task is not None and c.task == task:
                return
            if contains is not None and contains in (c.task or ""):
                return
        raise AssertionError(
            f"MockAgent({self.name!r}): no call matched task={task!r} "
            f"contains={contains!r}. Recorded tasks: "
            f"{[c.task for c in self.calls]}"
        )

    def assert_call_count(self, n: int) -> None:
        if self.call_count != n:
            raise AssertionError(f"MockAgent({self.name!r}): expected {n} calls, got {self.call_count}")

    def __repr__(self) -> str:
        return f"MockAgent({self.name!r}, calls={len(self.calls)})"

    # ------------------------------------------------------------------
    # Internal — response resolution + envelope construction
    # ------------------------------------------------------------------

    async def _resolve(self, env: Envelope) -> Any:
        r = self._responses

        # 1. Callable (but not a class — Pydantic models are callable).
        if callable(r) and not inspect.isclass(r):
            result = r(env)
            if inspect.isawaitable(result):
                result = await result
            return result

        # 2. Dict — substring match on task, with DEFAULT as catch-all.
        if isinstance(r, dict):
            task = env.task or ""
            for key, val in r.items():
                if key == DEFAULT:
                    continue
                if isinstance(key, str) and key in task:
                    return val
            if DEFAULT in r:
                return r[DEFAULT]
            raise RuntimeError(
                f"MockAgent({self.name!r}) got task {task!r} which matched "
                f"no dict key and there is no {DEFAULT!r} default"
            )

        # 3. List — one per call, optionally cycling.
        if isinstance(r, list):
            if not r:
                raise RuntimeError(f"MockAgent({self.name!r}) responses list is empty")
            if self._cursor >= len(r):
                if self._cycle:
                    self._cursor = 0
                else:
                    raise RuntimeError(
                        f"MockAgent({self.name!r}) exhausted after {len(r)} calls (pass cycle=True to loop)"
                    )
            val = r[self._cursor]
            self._cursor += 1
            return val

        # 4. Scalar — returned every call.
        return r

    def _to_envelope(self, task: str | Envelope) -> Envelope:
        if isinstance(task, Envelope):
            return task
        return Envelope.from_task(str(task))

    def _build_envelope(self, env_in: Envelope, raw: Any, elapsed_ms: float) -> Envelope:
        # Envelope pass-through: responder constructed a fully-formed
        # envelope, so honour it verbatim (error envelopes included).
        # Only backfill ``task`` / ``context`` if the responder omitted
        # them — preserves the "responder-owns-the-result" contract.
        if isinstance(raw, Envelope):
            updates: dict[str, Any] = {}
            if raw.task is None and env_in.task is not None:
                updates["task"] = env_in.task
            if raw.context is None and env_in.context is not None:
                updates["context"] = env_in.context
            return raw.model_copy(update=updates) if updates else raw

        # ErrorInfo short-circuit: no payload, no tokens, just the error.
        if isinstance(raw, ErrorInfo):
            return Envelope(
                task=env_in.task,
                context=env_in.context,
                error=raw,
                metadata=self._make_metadata(elapsed_ms, is_error=True),
            )

        # Everything else becomes the payload.
        return Envelope(
            task=env_in.task,
            context=env_in.context,
            payload=raw,
            metadata=self._make_metadata(elapsed_ms),
        )

    def _make_metadata(self, elapsed_ms: float, *, is_error: bool = False) -> EnvelopeMetadata:
        latency = self._default_latency_ms if self._default_latency_ms is not None else elapsed_ms
        return EnvelopeMetadata(
            input_tokens=0 if is_error else self._default_input_tokens,
            output_tokens=0 if is_error else self._default_output_tokens,
            cost_usd=0.0 if is_error else self._default_cost_usd,
            latency_ms=latency,
            model=self._default_model,
            provider=self._default_provider,
        )
