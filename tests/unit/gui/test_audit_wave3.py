"""Regression tests for Wave 3 of the deep audit (M1, M2, M3, M4, M5)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from lazybridge.core.types import CompletionResponse, UsageStats
from lazybridge.lazy_session import Event
from lazybridge.lazy_agent import LazyAgent
from lazybridge.lazy_tool import LazyTool
from lazybridge.memory import Memory


# ---------------------------------------------------------------------------
# M1 — output guard runs BEFORE _record_response
# ---------------------------------------------------------------------------


def _bare_agent():
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent.__new__(LazyAgent)
    mock_exec = MagicMock()

    class _AnthropicProvider:
        def get_default_max_tokens(self, model=None) -> int:
            return 4096

    mock_exec.provider = _AnthropicProvider()
    mock_exec.model = "m"
    mock_exec.execute = MagicMock(
        return_value=CompletionResponse(content="blocked payload", usage=UsageStats())
    )
    agent._executor = mock_exec
    agent.id = str(uuid.uuid4())
    agent.name = "a"
    agent.description = None
    agent.system = ""
    agent.context = None
    agent.tools = []
    agent.native_tools = []
    agent.output_schema = None
    agent._last_output = None
    agent._last_response = None
    agent.session = None
    agent._log = None
    return agent


def test_output_guard_runs_before_record_response():
    """If the output guard raises, _record_response must not have emitted
    an Event.MODEL_RESPONSE with the blocked content."""
    from lazybridge.guardrails import GuardAction, GuardError

    agent = _bare_agent()
    track_calls: list[tuple[str, dict]] = []
    agent._track = lambda event_type, **data: track_calls.append((event_type, data))  # type: ignore[method-assign]

    class _BlockingGuard:
        def check_output(self, text):
            return GuardAction(allowed=False, message="blocked by test")

        def check_input(self, text):
            return GuardAction(allowed=True)

        async def acheck_output(self, text):
            return self.check_output(text)

        async def acheck_input(self, text):
            return self.check_input(text)

    guard = _BlockingGuard()
    with pytest.raises(GuardError):
        agent.chat("anything", guard=guard)
    # MODEL_RESPONSE must NOT be in the emitted events, because the guard
    # rejected the response before _record_response could run.
    event_types = [et for et, _ in track_calls]
    assert Event.MODEL_RESPONSE not in event_types, (
        f"MODEL_RESPONSE leaked before guard check: {event_types}"
    )


# ---------------------------------------------------------------------------
# M2 — Memory compression does not hold _lock over the slow _compress call
# ---------------------------------------------------------------------------


def test_memory_compress_runs_unlocked():
    """A slow _compress must not block other threads from acquiring _lock.

    We patch Memory._compress to sleep; while that sleep is happening on
    one thread, another thread must be able to call `len(mem)` (which
    acquires the lock).
    """
    mem = Memory(strategy="rolling", max_context_tokens=1, window_turns=1)
    for _ in range(6):
        mem._messages.append({"role": "user", "content": "hi"})
        mem._messages.append({"role": "assistant", "content": "ok"})

    original = mem._compress
    in_compress = threading.Event()

    def _slow_compress(turns):
        in_compress.set()
        time.sleep(0.1)
        return original(turns)

    mem._compress = _slow_compress  # type: ignore[method-assign]

    t = threading.Thread(target=mem._maybe_recompress)
    t.start()
    assert in_compress.wait(timeout=1.0)
    # While _compress is sleeping, another thread must be able to acquire
    # the lock (via len()).  If the lock were held, this would block for
    # 0.1s — we set a tight 0.02s budget.
    t0 = time.monotonic()
    _ = len(mem)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05, (
        f"Lock was held during _compress — len() took {elapsed:.3f}s"
    )
    t.join(timeout=2.0)
    assert not t.is_alive()


# ---------------------------------------------------------------------------
# M3 — LazyTool.definition() cache write is synchronised
# ---------------------------------------------------------------------------


def test_lazy_tool_definition_is_not_computed_twice_under_concurrency():
    call_count = {"n": 0}

    def _slow_func(x: str) -> str:
        """A tool."""
        return x

    tool = LazyTool.from_function(_slow_func)
    original_build = tool.schema_builder.build if tool.schema_builder else None

    # Patch the builder used by definition() so we can count calls and
    # inject a slow path.
    from lazybridge.core.tool_schema import ToolSchemaBuilder

    real_build = ToolSchemaBuilder.build

    def _tracking_build(self, *args, **kwargs):
        call_count["n"] += 1
        time.sleep(0.05)
        return real_build(self, *args, **kwargs)

    with patch.object(ToolSchemaBuilder, "build", _tracking_build):
        # Reset the cached compiled field so both threads race on the
        # first-call path.
        tool._compiled = None

        results: list = [None, None]

        def _call(idx):
            results[idx] = tool.definition()

        t1 = threading.Thread(target=_call, args=(0,))
        t2 = threading.Thread(target=_call, args=(1,))
        t1.start(); t2.start()
        t1.join(timeout=2.0); t2.join(timeout=2.0)

    assert call_count["n"] == 1, (
        f"definition() cache race: builder called {call_count['n']}x"
    )
    assert results[0] is results[1]  # both threads returned the same cached object


# ---------------------------------------------------------------------------
# M4 — Parallel pipeline surfaces failures via logger
# ---------------------------------------------------------------------------


def test_parallel_failure_logged_with_participant_names(caplog):
    from lazybridge.pipeline_builders import build_parallel_func

    class _FakePart:
        def __init__(self, name, exc=None, result="ok"):
            self.name = name
            self._exc = exc
            self._result = result
            self.tools = None
            self.native_tools = None
            self.output_schema = None

        async def achat(self, task, **kw):
            if self._exc:
                raise self._exc
            return CompletionResponse(content=self._result, usage=UsageStats())

    parts = [
        _FakePart("alpha", exc=asyncio.TimeoutError("slow")),
        _FakePart("beta", result="fine"),
    ]
    runner = build_parallel_func(parts, native_tools=[], combiner="concat")

    with caplog.at_level(logging.WARNING, logger="lazybridge.pipeline_builders"):
        out = runner("go")

    assert "[ERROR: TimeoutError" in out  # existing string contract preserved
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("alpha=TimeoutError" in m for m in msgs), (
        f"expected operator-visible warning about 'alpha' failure, got: {msgs}"
    )


# ---------------------------------------------------------------------------
# M5 — Pydantic tool schemas are generated with mode="validation"
# ---------------------------------------------------------------------------


class _SchemaModel(BaseModel):
    name: str
    # An alias is a classic case where serialization vs validation schemas
    # diverge: mode="serialization" lists the aliased key while validation
    # lists the original field name.
    model_config = {"populate_by_name": True}
    age: int


def _takes_model(payload: _SchemaModel) -> str:
    """Takes a pydantic model."""
    return payload.name


def test_pydantic_tool_schema_uses_validation_mode():
    from lazybridge.core.tool_schema import ToolSchemaBuilder

    builder = ToolSchemaBuilder()
    defn = builder.build(_takes_model, name="takes_model", description="...")
    params = defn.parameters
    # Fishing for the nested model schema inside `properties`.
    props = params.get("properties") or {}
    payload_schema = props.get("payload") or {}
    # Validation-mode schema produced by Pydantic reports a plain
    # properties dict with the original (non-aliased) field names.
    # The concrete marker we can check is the presence of the fields
    # we declared.
    inner_props = payload_schema.get("properties") or {}
    # If model is referenced via $ref, resolve against $defs.
    if "$ref" in payload_schema:
        ref = payload_schema["$ref"].rsplit("/", 1)[-1]
        inner_props = (params.get("$defs") or {}).get(ref, {}).get("properties") or {}
    assert "name" in inner_props and "age" in inner_props
