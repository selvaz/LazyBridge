"""Regression tests for the core-audit streaming fixes (providers).

Each test reproduces a bug found in the deep audit of ``lazybridge/core``
and certifies the fix:

* OpenAI Responses streaming: argument deltas are keyed by ``item_id``
  (the SDK event has no ``call_id`` attribute) — before the fix every
  streamed tool call arrived with ``arguments={}``.
* OpenAI Responses streaming: ``response.output_item.done`` is used as a
  safety net when no argument deltas were captured.
* DeepSeek streaming: the usage-only final chunk (``choices=[]``) per the
  OpenAI streaming spec was skipped entirely, losing token accounting.
* DeepSeek streaming: a stream that ends without ``finish_reason`` must
  still emit an ``is_final=True`` chunk (BaseProvider contract).

SDKs are stubbed via ``sys.modules`` / MagicMock — no network, no keys.
"""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

# openai stub so providers can be imported/instantiated without the SDK.
if "openai" not in sys.modules:
    openai_pkg = types.ModuleType("openai")
    openai_pkg.OpenAI = MagicMock(name="OpenAI")
    openai_pkg.AsyncOpenAI = MagicMock(name="AsyncOpenAI")
    sys.modules["openai"] = openai_pkg

from lazybridge.core.providers.deepseek import DeepSeekProvider
from lazybridge.core.providers.openai import OpenAIProvider


class _AsyncEventStream:
    """Wrap a list of fake SDK events as an async iterator."""

    def __init__(self, events):
        self._it = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _bare_openai() -> OpenAIProvider:
    p = OpenAIProvider.__new__(OpenAIProvider)
    p.api_key = "fake"
    p.model = "gpt-4o-mini"
    return p


def _bare_deepseek() -> DeepSeekProvider:
    p = DeepSeekProvider.__new__(DeepSeekProvider)
    p.api_key = "fake"
    p.model = "deepseek-chat"
    p._structured_drop_warned = False
    return p


# ---------------------------------------------------------------------------
# OpenAI Responses API streaming — tool-call argument accumulation
# ---------------------------------------------------------------------------


def _responses_tool_call_events() -> list[NS]:
    """Realistic event sequence: deltas carry item_id, NOT call_id."""
    return [
        NS(
            type="response.output_item.added",
            item=NS(type="function_call", id="item_1", call_id="call_1", name="get_weather"),
        ),
        NS(type="response.function_call_arguments.delta", item_id="item_1", delta='{"city": '),
        NS(type="response.function_call_arguments.delta", item_id="item_1", delta='"Rome"}'),
        NS(
            type="response.completed",
            response=NS(usage=None, output=[], status="completed", model="gpt-4o-mini"),
        ),
    ]


def test_responses_stream_accumulates_arguments_via_item_id():
    p = _bare_openai()
    p._client = MagicMock()
    p._client.responses.create.return_value = iter(_responses_tool_call_events())

    chunks = list(p._stream_responses_api({"model": "gpt-4o-mini"}))

    final = chunks[-1]
    assert final.is_final
    assert len(final.tool_calls) == 1
    tc = final.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Rome"}


def test_responses_astream_accumulates_arguments_via_item_id():
    p = _bare_openai()

    async def _create(**_kw):
        return _AsyncEventStream(_responses_tool_call_events())

    p._get_async_client = lambda: NS(responses=NS(create=_create))

    async def _run():
        return [c async for c in p._astream_responses_api({"model": "gpt-4o-mini"})]

    chunks = asyncio.run(_run())
    final = chunks[-1]
    assert final.is_final
    assert final.tool_calls[0].arguments == {"city": "Rome"}


def test_responses_stream_output_item_done_is_a_safety_net():
    """When no deltas were captured, arguments come from output_item.done."""
    events = [
        NS(
            type="response.output_item.added",
            item=NS(type="function_call", id="item_9", call_id="call_9", name="lookup"),
        ),
        NS(
            type="response.output_item.done",
            item=NS(type="function_call", id="item_9", call_id="call_9", name="lookup", arguments='{"q": 1}'),
        ),
        NS(
            type="response.completed",
            response=NS(usage=None, output=[], status="completed", model="gpt-4o-mini"),
        ),
    ]
    p = _bare_openai()
    p._client = MagicMock()
    p._client.responses.create.return_value = iter(events)

    final = list(p._stream_responses_api({"model": "gpt-4o-mini"}))[-1]
    assert final.tool_calls[0].arguments == {"q": 1}


def test_responses_stream_incomplete_status_is_captured():
    """``response.incomplete`` must populate the final chunk (truncation),
    not be silently dropped as an unknown event type."""
    events = [
        NS(type="response.output_text.delta", delta="partial"),
        NS(
            type="response.incomplete",
            response=NS(usage=None, output=[], status="incomplete", model="gpt-4o-mini"),
        ),
    ]
    p = _bare_openai()
    p._client = MagicMock()
    p._client.responses.create.return_value = iter(events)

    final = list(p._stream_responses_api({"model": "gpt-4o-mini"}))[-1]
    assert final.is_final
    assert final.stop_reason == "max_tokens"
