"""Regression tests for the telemetry/robustness audit fixes (Anthropic + Google).

Each test locks one verified bug fix:

Anthropic (``lazybridge/core/providers/anthropic.py``):

* A1 — ``usage.input_tokens`` EXCLUDES the cache counters in the Anthropic
  API; cache reads (10% rate) and cache writes (125% rate) are ADDITIVE,
  not subsets of ``input_tokens``.  The old min/subtract logic
  under-reported cost.
* A2 — grounding sources live in nested ``web_search_tool_result`` blocks
  (``block.content`` holds the ``web_search_result`` items, or an error
  payload); the old code looked for top-level ``web_search_result`` blocks
  that the SDK never emits.
* A3 — ``_collect_streamed_response`` / ``_acollect_streamed_response``
  dropped ``grounding_sources`` / ``web_search_queries`` /
  ``search_entry_point`` from the final chunk.
* A4 — ``astream`` only read the legacy ``usage.reasoning_tokens`` field;
  it now checks ``usage.output_tokens_details.thinking_tokens`` first like
  ``stream`` and ``_parse_response``.
* A5 — ``_build_thinking`` clamps ``budget_tokens`` to
  ``effective_max_tokens - 1024`` (API requires budget < max_tokens) and
  honours an explicit ``budget_tokens=0`` instead of replacing it with the
  8000 default via ``or``.
* A6 — ``_should_force_streaming`` logged ``request.max_tokens`` with %d,
  crashing the log record when it is None (the common case).

Google (``lazybridge/core/providers/google.py``):

* G1 — Gemini bills thought tokens as output but excludes them from
  ``candidates_token_count``; cost now uses output + thinking tokens.
* G2 — ``get_default_max_tokens`` is applied in ``_build_config`` when the
  caller doesn't set ``max_tokens`` (contract parity with anthropic.py).
* G3 — ``_check_native_tools`` / ``_build_function_declarations`` were each
  invoked twice per request (duplicate warnings); now once.
* G4 — ``_init_client(**kwargs)`` forwards kwargs to ``genai.Client``.
* G5 — the ``tool_fn_name`` fallback on ``tool_use_id`` strips the
  synthetic ``-<N>`` suffix and warns to set ``tool_name``.

Conventions follow ``test_audit_core_streaming.py``: providers built via
``__new__`` (no SDK init), fake SDK objects as ``SimpleNamespace``.
"""

from __future__ import annotations

import asyncio
import logging
import warnings as _warnings_mod
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

from lazybridge.core.providers.anthropic import AnthropicProvider
from lazybridge.core.types import (
    CompletionRequest,
    GroundingSource,
    Message,
    NativeTool,
    Role,
    StreamChunk,
    ThinkingConfig,
    ToolDefinition,
    ToolResultContent,
)


def _bare_anthropic(model: str = "claude-sonnet-4-6") -> AnthropicProvider:
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.api_key = "fake"
    p.model = model
    p._user_model = model
    p.fallback_model = None
    p.strict_native_tools = False
    p._temperature_warned = False
    return p


def _bare_google():
    from lazybridge.core.providers.google import GoogleProvider

    p = GoogleProvider.__new__(GoogleProvider)
    p.api_key = "fake"
    p.model = "gemini-2.5-pro"
    p._user_model = "gemini-2.5-pro"
    p.fallback_model = None
    p.strict_native_tools = False
    return p


class _FakeAnthropicStream:
    """Sync stream context manager mimicking the Anthropic SDK."""

    def __init__(self, events, final):
        self._events = events
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


class _FakeAsyncAnthropicStream:
    """Async variant: ``async with`` + ``async for`` + awaitable final."""

    def __init__(self, events, final):
        self._events = events
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        self._it = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None

    async def get_final_message(self):
        return self._final


def _ws_tool_result_block():
    """Realistic SDK shape: nested results inside web_search_tool_result."""
    return NS(
        type="web_search_tool_result",
        tool_use_id="srvtoolu_1",
        content=[
            NS(type="web_search_result", url="https://example.com/a", title="A"),
            NS(type="web_search_result", url="https://example.com/b", title="B"),
        ],
    )


# ---------------------------------------------------------------------------
# A1 — Anthropic cache cost accounting: counters are additive, writes at 125%
# ---------------------------------------------------------------------------


def test_anthropic_cost_treats_cache_counters_as_additive():
    p = _bare_anthropic()
    # claude-sonnet-4-6 → ($3, $15) per 1M
    cost = p._compute_cost(
        "claude-sonnet-4-6",
        input_tokens=1_000,
        output_tokens=100,
        cached_input_tokens=9_000,
        cache_creation_tokens=2_000,
    )
    expected = (1_000 * 3.0 + 9_000 * 0.1 * 3.0 + 2_000 * 1.25 * 3.0 + 100 * 15.0) / 1_000_000
    assert cost == pytest.approx(expected)
    # Before the fix cached was clamped to min(9000, 1000) and SUBTRACTED
    # from input — yielding only 1000 * 0.3 / 1M.
    assert cost > (1_000 * 0.1 * 3.0) / 1_000_000


def test_anthropic_cost_back_compat_without_cache_kwargs():
    """4-positional-arg call sites (base.py contract) still work."""
    p = _bare_anthropic()
    cost = p._compute_cost("claude-sonnet-4-6", 1_000_000, 0)
    assert cost == pytest.approx(3.0)


def test_anthropic_parse_response_bills_cache_creation_tokens():
    p = _bare_anthropic()
    response = NS(
        content=[NS(type="text", text="hi")],
        usage=NS(
            input_tokens=100,
            output_tokens=10,
            cache_read_input_tokens=50,
            cache_creation_input_tokens=20,
        ),
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
    )
    resp = p._parse_response(response)
    assert resp.usage.cached_input_tokens == 50
    expected = (100 * 3.0 + 50 * 0.1 * 3.0 + 20 * 1.25 * 3.0 + 10 * 15.0) / 1_000_000
    assert resp.usage.cost_usd == pytest.approx(expected)


# ---------------------------------------------------------------------------
# A2 — grounding sources: nested web_search_tool_result extraction
# ---------------------------------------------------------------------------


def test_anthropic_parse_response_extracts_nested_web_search_results():
    p = _bare_anthropic()
    response = NS(
        content=[NS(type="text", text="answer"), _ws_tool_result_block()],
        usage=NS(input_tokens=1, output_tokens=1),
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
    )
    resp = p._parse_response(response)
    assert [(s.url, s.title) for s in resp.grounding_sources] == [
        ("https://example.com/a", "A"),
        ("https://example.com/b", "B"),
    ]


def test_anthropic_grounding_helper_tolerates_error_payload():
    """``.content`` can be an error object instead of a result list."""
    blocks = [
        NS(
            type="web_search_tool_result",
            content=NS(type="web_search_tool_result_error", error_code="max_uses_exceeded"),
        ),
        NS(type="text", text="x"),
    ]
    assert AnthropicProvider._extract_grounding_sources(blocks) == []


def test_anthropic_stream_final_chunk_extracts_nested_grounding():
    p = _bare_anthropic()
    events = [
        NS(type="content_block_delta", delta=NS(type="text_delta", text="hi")),
        NS(type="message_stop"),
    ]
    final = NS(
        content=[NS(type="text", text="hi"), _ws_tool_result_block()],
        usage=NS(input_tokens=10, output_tokens=5),
        stop_reason="end_turn",
        model="claude-sonnet-4-6",
    )
    p._client = MagicMock()
    p._client.messages.stream.return_value = _FakeAnthropicStream(events, final)

    req = CompletionRequest(messages=[Message.user("q")])
    chunks = list(p.stream(req))

    final_chunk = chunks[-1]
    assert final_chunk.is_final
    assert [s.url for s in final_chunk.grounding_sources] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_anthropic_astream_final_chunk_extracts_nested_grounding():
    p = _bare_anthropic()
    events = [NS(type="message_stop")]
    final = NS(
        content=[_ws_tool_result_block()],
        usage=NS(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
        model="claude-sonnet-4-6",
    )
    p._async_client = MagicMock()
    p._async_client.messages.stream.return_value = _FakeAsyncAnthropicStream(events, final)

    async def _run():
        return [c async for c in p.astream(CompletionRequest(messages=[Message.user("q")]))]

    final_chunk = asyncio.run(_run())[-1]
    assert final_chunk.is_final
    assert len(final_chunk.grounding_sources) == 2


# ---------------------------------------------------------------------------
# A3 — force-streamed collectors must copy grounding telemetry
# ---------------------------------------------------------------------------


def _grounded_final_chunk() -> StreamChunk:
    return StreamChunk(
        is_final=True,
        stop_reason="end_turn",
        grounding_sources=[GroundingSource(url="https://example.com", title="E")],
        web_search_queries=["query-1"],
        search_entry_point="<widget/>",
    )


def test_collect_streamed_response_copies_grounding_fields():
    p = _bare_anthropic()

    def fake_stream(_req):
        yield StreamChunk(delta="x")
        yield _grounded_final_chunk()

    p.stream = fake_stream
    resp = p._collect_streamed_response(CompletionRequest(messages=[Message.user("q")]))
    assert resp.grounding_sources[0].url == "https://example.com"
    assert resp.web_search_queries == ["query-1"]
    assert resp.search_entry_point == "<widget/>"


def test_acollect_streamed_response_copies_grounding_fields():
    p = _bare_anthropic()

    async def fake_astream(_req):
        yield StreamChunk(delta="x")
        yield _grounded_final_chunk()

    p.astream = fake_astream

    async def _run():
        return await p._acollect_streamed_response(CompletionRequest(messages=[Message.user("q")]))

    resp = asyncio.run(_run())
    assert resp.grounding_sources[0].url == "https://example.com"
    assert resp.web_search_queries == ["query-1"]
    assert resp.search_entry_point == "<widget/>"


# ---------------------------------------------------------------------------
# A4 — astream reads thinking tokens from output_tokens_details first
# ---------------------------------------------------------------------------


def test_anthropic_astream_reads_output_tokens_details_thinking_tokens():
    p = _bare_anthropic()
    final = NS(
        content=[],
        usage=NS(
            input_tokens=10,
            output_tokens=5,
            output_tokens_details=NS(thinking_tokens=3),
            # NOTE: no legacy ``reasoning_tokens`` attribute — before the
            # fix astream only looked there and reported 0.
        ),
        stop_reason="end_turn",
        model="claude-sonnet-4-6",
    )
    p._async_client = MagicMock()
    p._async_client.messages.stream.return_value = _FakeAsyncAnthropicStream([NS(type="message_stop")], final)

    async def _run():
        return [c async for c in p.astream(CompletionRequest(messages=[Message.user("q")]))]

    final_chunk = asyncio.run(_run())[-1]
    assert final_chunk.usage is not None
    assert final_chunk.usage.thinking_tokens == 3


# ---------------------------------------------------------------------------
# A5 — _build_thinking budget clamp (budget_tokens < max_tokens)
# ---------------------------------------------------------------------------


def test_build_thinking_clamps_budget_to_effective_max_minus_1024():
    p = _bare_anthropic()
    req = CompletionRequest(
        messages=[Message.user("q")],
        model="claude-3-5-sonnet",  # non-adaptive: explicit budget path
        max_tokens=10_000,
        thinking=ThinkingConfig(enabled=True, budget_tokens=50_000),
    )
    with pytest.warns(UserWarning, match="budget_tokens"):
        thinking = p._build_thinking(req)
    assert thinking == {"type": "enabled", "budget_tokens": 10_000 - 1024}


def test_build_thinking_default_budget_clamped_for_small_default_max():
    """claude-3-haiku default max_tokens is 4096 — the 8000 default budget
    would exceed it (API 400) without the clamp."""
    p = _bare_anthropic(model="claude-3-haiku")
    req = CompletionRequest(
        messages=[Message.user("q")],
        thinking=ThinkingConfig(enabled=True),  # budget_tokens=None → default 8000
    )
    with pytest.warns(UserWarning, match="budget_tokens"):
        thinking = p._build_thinking(req)
    assert thinking == {"type": "enabled", "budget_tokens": 4096 - 1024}


def test_build_thinking_zero_budget_is_not_replaced_by_default():
    """``0 or 8000`` silently turned an explicit 0 into 8000."""
    p = _bare_anthropic(model="claude-3-5-sonnet")
    req = CompletionRequest(
        messages=[Message.user("q")],
        max_tokens=10_000,
        thinking=ThinkingConfig(enabled=True, budget_tokens=0),
    )
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("error")  # no clamp warning expected
        thinking = p._build_thinking(req)
    assert thinking == {"type": "enabled", "budget_tokens": 0}


def test_build_thinking_in_range_budget_untouched():
    p = _bare_anthropic(model="claude-3-5-sonnet")
    req = CompletionRequest(
        messages=[Message.user("q")],
        max_tokens=10_000,
        thinking=ThinkingConfig(enabled=True, budget_tokens=4_000),
    )
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("error")
        thinking = p._build_thinking(req)
    assert thinking == {"type": "enabled", "budget_tokens": 4_000}


# ---------------------------------------------------------------------------
# A6 — _should_force_streaming logs the EFFECTIVE max_tokens (not None)
# ---------------------------------------------------------------------------


def test_should_force_streaming_log_formats_with_max_tokens_none(caplog):
    p = _bare_anthropic(model="claude-opus-4-6")  # default max_tokens = 128_000
    req = CompletionRequest(messages=[Message.user("q")])  # max_tokens=None
    with caplog.at_level(logging.DEBUG, logger="lazybridge.core.providers.anthropic"):
        assert p._should_force_streaming(req) is True
    assert caplog.records, "expected a debug record explaining the auto-forcing"
    # Before the fix %d on request.max_tokens=None raised TypeError at
    # format time; getMessage() must now succeed and show the real value.
    msg = caplog.records[-1].getMessage()
    assert "128000" in msg


# ---------------------------------------------------------------------------
# G1 — Gemini cost includes thought tokens (billed as output)
# ---------------------------------------------------------------------------


def test_google_parse_response_cost_includes_thinking_tokens():
    p = _bare_google()
    candidate = NS(
        content=NS(parts=[NS(text="hi", thought=None, function_call=None)]),
        finish_reason=None,
        grounding_metadata=None,
    )
    um = NS(prompt_token_count=100, candidates_token_count=10, thoughts_token_count=90)
    response = NS(candidates=[candidate], usage_metadata=um)

    resp = p._parse_response(response, "gemini-2.5-pro")  # ($1.25, $10) per 1M

    assert resp.usage.thinking_tokens == 90
    expected = (100 * 1.25 + (10 + 90) * 10.0) / 1_000_000
    assert resp.usage.cost_usd == pytest.approx(expected)


def test_google_stream_cost_includes_thinking_tokens():
    from lazybridge.core.providers import google as google_module

    p = _bare_google()
    um = NS(prompt_token_count=100, candidates_token_count=10, thoughts_token_count=90)
    chunk = NS(
        candidates=[
            NS(
                content=NS(parts=[NS(text="hi", thought=None, function_call=None)]),
                finish_reason=NS(name="STOP"),
                grounding_metadata=None,
            )
        ],
        usage_metadata=um,
    )
    p._client = MagicMock()
    p._client.models.generate_content_stream.return_value = iter([chunk])

    req = CompletionRequest(messages=[Message.user("q")])
    with patch.object(google_module, "_gtypes", MagicMock()):
        final = list(p.stream(req))[-1]

    assert final.usage is not None
    expected = (100 * 1.25 + (10 + 90) * 10.0) / 1_000_000
    assert final.usage.cost_usd == pytest.approx(expected)


# ---------------------------------------------------------------------------
# G2 — get_default_max_tokens applied when request.max_tokens is unset
# ---------------------------------------------------------------------------


def test_google_build_config_applies_default_max_tokens():
    from lazybridge.core.providers import google as google_module

    p = _bare_google()
    gtypes = MagicMock()
    with patch.object(google_module, "_gtypes", gtypes):
        p._build_config(CompletionRequest(messages=[Message.user("q")]))
    kwargs = gtypes.GenerateContentConfig.call_args.kwargs
    assert kwargs["max_output_tokens"] == 65_536  # gemini-2.5-pro default


def test_google_build_config_explicit_max_tokens_wins():
    from lazybridge.core.providers import google as google_module

    p = _bare_google()
    gtypes = MagicMock()
    with patch.object(google_module, "_gtypes", gtypes):
        p._build_config(CompletionRequest(messages=[Message.user("q")], max_tokens=123))
    kwargs = gtypes.GenerateContentConfig.call_args.kwargs
    assert kwargs["max_output_tokens"] == 123


# ---------------------------------------------------------------------------
# G3 — native-tool check / function declarations built once per request
# ---------------------------------------------------------------------------


def test_google_unsupported_native_tool_warns_exactly_once():
    from lazybridge.core.providers import google as google_module

    p = _bare_google()
    req = CompletionRequest(
        messages=[Message.user("q")],
        native_tools=[NativeTool.CODE_EXECUTION],  # not supported by Gemini
    )
    with patch.object(google_module, "_gtypes", MagicMock()):
        with _warnings_mod.catch_warnings(record=True) as rec:
            _warnings_mod.simplefilter("always")
            p._build_config(req)
    unsupported = [w for w in rec if "does not support native tool" in str(w.message)]
    assert len(unsupported) == 1  # was 2: _build_config AND _build_tools_config


def test_google_function_declarations_built_once_per_request():
    from lazybridge.core.providers import google as google_module

    p = _bare_google()
    calls: list[int] = []
    original = p._build_function_declarations

    def _counting(request):
        calls.append(1)
        return original(request)

    p._build_function_declarations = _counting
    tool = ToolDefinition(name="lookup", description="d", parameters={"type": "object", "properties": {}})
    with patch.object(google_module, "_gtypes", MagicMock()):
        p._build_config(CompletionRequest(messages=[Message.user("q")], tools=[tool]))
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# G4 — _init_client forwards kwargs to genai.Client
# ---------------------------------------------------------------------------


def test_google_init_client_forwards_kwargs():
    from lazybridge.core.providers import google as google_module
    from lazybridge.core.providers.google import GoogleProvider

    p = GoogleProvider.__new__(GoogleProvider)
    p.api_key = "fake"
    fake_genai = MagicMock()
    with patch.object(google_module, "_genai", fake_genai):
        p._init_client(http_options={"timeout": 5}, vertexai=True)
    kwargs = fake_genai.Client.call_args.kwargs
    assert kwargs["api_key"] == "fake"
    assert kwargs["http_options"] == {"timeout": 5}
    assert kwargs["vertexai"] is True


# ---------------------------------------------------------------------------
# G5 — tool_fn_name fallback strips the synthetic "-<N>" id suffix + warns
# ---------------------------------------------------------------------------


def test_google_function_response_fallback_strips_synthetic_suffix():
    from lazybridge.core.providers import google as google_module

    p = _bare_google()
    msg = Message(
        role=Role.TOOL,
        content=[ToolResultContent(tool_use_id="lookup-3", content="42")],  # no tool_name
    )
    gtypes = MagicMock()
    with patch.object(google_module, "_gtypes", gtypes):
        with pytest.warns(UserWarning, match="tool_name"):
            p._messages_to_gemini(CompletionRequest(messages=[msg]))
    assert gtypes.Part.from_function_response.call_args.kwargs["name"] == "lookup"


def test_google_function_response_with_tool_name_does_not_warn():
    from lazybridge.core.providers import google as google_module

    p = _bare_google()
    msg = Message(
        role=Role.TOOL,
        content=[ToolResultContent(tool_use_id="lookup-3", content="42", tool_name="real_fn")],
    )
    gtypes = MagicMock()
    with patch.object(google_module, "_gtypes", gtypes):
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("error")
            p._messages_to_gemini(CompletionRequest(messages=[msg]))
    assert gtypes.Part.from_function_response.call_args.kwargs["name"] == "real_fn"
