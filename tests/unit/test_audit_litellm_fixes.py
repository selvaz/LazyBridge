"""Regression tests for the audited LiteLLMProvider fixes.

Each test class targets one verified bug in
``lazybridge/core/providers/litellm.py`` and contains at least one test
that would have FAILED before the corresponding fix:

1. ``request.structured_output`` was ignored (no response_format, no
   ``apply_structured_validation`` in complete/stream/acomplete/astream)
   despite the inherited ``supports_structured_output=True``.
2. Streaming never opted into the OpenAI usage trailer
   (``stream_options={"include_usage": True}``) and a stream dying without
   a finish_reason produced a misleading ``stop_reason="end_turn"``.
3. ``_check_native_tools`` was never called — ``strict_native_tools=True``
   never raised ``UnsupportedNativeToolError`` as base.py promises.
4. A message mixing text and ``ToolResultContent`` emitted the text with
   ``role="tool"`` (no tool_call_id → HTTP 400) instead of ``role="user"``.
5. ``_init_client`` silently discarded unknown litellm config kwargs.
6. ``default_model`` was ``"litellm/gpt-4o-mini"`` — paid cloud providers
   must have ``default_model=None`` to force explicit model selection.
7. ``_extract_usage`` lacked the documented ``_compute_cost`` fallback and
   never extracted ``thinking_tokens`` / ``cached_input_tokens``.
8. ``_tool_choice_to_openai`` carried a dead ``tools`` parameter.

Style mirrors ``tests/unit/test_litellm_provider.py``: the ``litellm``
module is stubbed via ``sys.modules`` so no network and no real litellm
install are needed.
"""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# litellm stub + fakes (mirrors test_litellm_provider.py)
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message=None, finish_reason="stop", delta=None):
        self.message = message
        self.finish_reason = finish_reason
        self.delta = delta


class _FakeUsage:
    def __init__(
        self,
        prompt_tokens=0,
        completion_tokens=0,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.completion_tokens_details = completion_tokens_details
        self.prompt_tokens_details = prompt_tokens_details


class _FakeResponse:
    def __init__(self, choices, model="mock-model", usage=None, response_cost=None):
        self.choices = choices
        self.model = model
        self.usage = usage
        self._hidden_params = {"response_cost": response_cost} if response_cost is not None else {}


class _FakeToolCallFunction:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id="", name="", arguments="", index=0):
        self.id = id
        self.function = _FakeToolCallFunction(name, arguments)
        self.index = index


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _AIter:
    """Wrap a list in an async iterator for astream tests."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _install_stub_litellm(completion_impl=None, acompletion_impl=None):
    stub = types.ModuleType("litellm")
    stub.completion = completion_impl or MagicMock()
    stub.acompletion = acompletion_impl or MagicMock()
    # Known litellm module-level flags settable via _init_client kwargs:
    stub.drop_params = False
    stub.set_verbose = False
    sys.modules["litellm"] = stub
    return stub


@pytest.fixture(autouse=True)
def _clean_litellm_imports():
    """Ensure each test starts with a fresh litellm stub + re-imported provider."""
    for mod_name in list(sys.modules):
        if mod_name == "litellm" or mod_name.endswith("providers.litellm"):
            sys.modules.pop(mod_name, None)
    yield
    for mod_name in list(sys.modules):
        if mod_name == "litellm" or mod_name.endswith("providers.litellm"):
            sys.modules.pop(mod_name, None)


def _make_provider(completion_impl=None, acompletion_impl=None, **provider_kwargs):
    stub = _install_stub_litellm(completion_impl, acompletion_impl)
    from lazybridge.core.providers.litellm import LiteLLMProvider

    return LiteLLMProvider(**provider_kwargs), stub


def _basic_request(**overrides):
    from lazybridge.core.types import CompletionRequest, Message, Role

    defaults = dict(
        messages=[Message(role=Role.USER, content="hello")],
        model="litellm/groq/llama-3.3-70b",
        max_tokens=256,
    )
    defaults.update(overrides)
    return CompletionRequest(**defaults)


#: Raw JSON-schema dict — keeps tests independent of Pydantic and works
#: with the subset validator even when jsonschema is not installed.
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _structured_request(**overrides):
    from lazybridge.core.types import StructuredOutputConfig

    return _basic_request(structured_output=StructuredOutputConfig(schema=_SCHEMA), **overrides)


# ---------------------------------------------------------------------------
# Fix 1 — structured_output honoured end-to-end
# ---------------------------------------------------------------------------


def test_structured_output_sets_json_response_format():
    """_build_params must emit response_format json_object for SO requests."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage('{"answer": "42"}'))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    prov.complete(_structured_request())
    assert captured["response_format"] == {"type": "json_object"}


def test_complete_validates_structured_output_success():
    """complete() must run apply_structured_validation and set parsed/validated."""

    def fake_completion(**kwargs):
        return _FakeResponse([_FakeChoice(message=_FakeMessage('{"answer": "42"}'))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_structured_request())
    assert resp.validated is True
    assert resp.parsed == {"answer": "42"}
    assert resp.validation_error is None


def test_complete_records_validation_error_on_bad_json():
    """Invalid JSON → validated=False + validation_error, never an exception."""

    def fake_completion(**kwargs):
        return _FakeResponse([_FakeChoice(message=_FakeMessage("not json at all"))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_structured_request())
    assert resp.validated is False
    assert resp.validation_error
    assert resp.parsed is None


def test_complete_skips_validation_on_tool_call_turn():
    """Tool-call turns have empty content by design — no spurious validation."""

    def fake_completion(**kwargs):
        msg = _FakeMessage(content="", tool_calls=[_FakeToolCall(id="c1", name="search", arguments="{}")])
        return _FakeResponse([_FakeChoice(message=msg, finish_reason="tool_calls")], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_structured_request())
    assert resp.validated is None
    assert resp.validation_error is None


@pytest.mark.asyncio
async def test_acomplete_validates_structured_output():
    """acomplete() applies the same validation as complete()."""

    async def fake_acompletion(**kwargs):
        return _FakeResponse([_FakeChoice(message=_FakeMessage('{"answer": "ok"}'))], usage=_FakeUsage())

    prov, _ = _make_provider(None, fake_acompletion)
    resp = await prov.acomplete(_structured_request())
    assert resp.validated is True
    assert resp.parsed == {"answer": "ok"}


def test_stream_validates_accumulated_text_on_final_chunk():
    """stream() must accumulate deltas and validate on the is_final chunk."""
    chunks = [
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content='{"answer"'))]),
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content=': "yes"}'))]),
        _FakeResponse([_FakeChoice(delta=_FakeDelta(), finish_reason="stop")], usage=_FakeUsage(3, 2)),
    ]

    def fake_completion(**kwargs):
        return iter(chunks)

    prov, _ = _make_provider(fake_completion)
    emitted = list(prov.stream(_structured_request()))
    final = emitted[-1]
    assert final.is_final
    assert final.validated is True
    assert final.parsed == {"answer": "yes"}


def test_stream_skips_validation_on_tool_call_turn():
    """A streamed tool-call turn must not record a spurious validation_error."""
    chunks = [
        _FakeResponse(
            [
                _FakeChoice(
                    delta=_FakeDelta(tool_calls=[_FakeToolCall(id="c1", name="search", arguments="{}", index=0)])
                )
            ]
        ),
        _FakeResponse([_FakeChoice(delta=_FakeDelta(), finish_reason="tool_calls")], usage=_FakeUsage()),
    ]

    def fake_completion(**kwargs):
        return iter(chunks)

    prov, _ = _make_provider(fake_completion)
    final = list(prov.stream(_structured_request()))[-1]
    assert final.is_final
    assert final.tool_calls and final.tool_calls[0].name == "search"
    assert final.validated is None
    assert final.validation_error is None


@pytest.mark.asyncio
async def test_astream_validates_accumulated_text_on_final_chunk():
    """astream() applies the same final-chunk validation as stream()."""
    chunks = [
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content='{"ans'))]),
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content='wer": "a"}'))]),
        _FakeResponse([_FakeChoice(delta=_FakeDelta(), finish_reason="stop")], usage=_FakeUsage(1, 1)),
    ]

    async def fake_acompletion(**kwargs):
        return _AIter(chunks)

    prov, _ = _make_provider(None, fake_acompletion)
    emitted = []
    async for ch in prov.astream(_structured_request()):
        emitted.append(ch)
    final = emitted[-1]
    assert final.is_final
    assert final.validated is True
    assert final.parsed == {"answer": "a"}


# ---------------------------------------------------------------------------
# Fix 2 — streaming usage: include_usage opt-in, trailer chunk, incomplete stop
# ---------------------------------------------------------------------------


def test_stream_requests_include_usage():
    """stream() must opt into the OpenAI usage trailer via stream_options."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return iter([_FakeResponse([_FakeChoice(delta=_FakeDelta(), finish_reason="stop")])])

    prov, _ = _make_provider(fake_completion)
    list(prov.stream(_basic_request()))
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_astream_requests_include_usage():
    """astream() must also opt into the usage trailer."""
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _AIter([_FakeResponse([_FakeChoice(delta=_FakeDelta(), finish_reason="stop")])])

    prov, _ = _make_provider(None, fake_acompletion)
    async for _ in prov.astream(_basic_request()):
        pass
    assert captured["stream_options"] == {"include_usage": True}


def test_stream_reads_usage_from_choices_empty_trailer_chunk():
    """The OpenAI-spec trailer (choices=[]) carries usage — including details."""
    trailer_usage = _FakeUsage(
        prompt_tokens=11,
        completion_tokens=7,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
        prompt_tokens_details=SimpleNamespace(cached_tokens=6),
    )
    chunks = [
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content="hi"))]),
        _FakeResponse([_FakeChoice(delta=_FakeDelta(), finish_reason="stop")]),
        _FakeResponse([], usage=trailer_usage),  # choices=[] trailer
    ]

    def fake_completion(**kwargs):
        return iter(chunks)

    prov, _ = _make_provider(fake_completion)
    final = list(prov.stream(_basic_request()))[-1]
    assert final.is_final
    assert final.usage is not None
    assert final.usage.input_tokens == 11
    assert final.usage.output_tokens == 7
    assert final.usage.thinking_tokens == 4
    assert final.usage.cached_input_tokens == 6


def test_stream_dying_without_finish_reason_emits_incomplete_final_chunk():
    """A stream that ends with no finish_reason still yields is_final=True
    with stop_reason='incomplete' (not a misleading 'end_turn')."""
    chunks = [
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content="partial"), finish_reason=None)]),
        # ...stream dies here: no finish_reason, no usage trailer.
    ]

    def fake_completion(**kwargs):
        return iter(chunks)

    prov, _ = _make_provider(fake_completion)
    emitted = list(prov.stream(_basic_request()))
    final = emitted[-1]
    assert final.is_final
    assert final.stop_reason == "incomplete"


@pytest.mark.asyncio
async def test_astream_dying_without_finish_reason_emits_incomplete_final_chunk():
    """Async variant of the incomplete-stream guarantee."""

    async def fake_acompletion(**kwargs):
        return _AIter([_FakeResponse([_FakeChoice(delta=_FakeDelta(content="x"), finish_reason=None)])])

    prov, _ = _make_provider(None, fake_acompletion)
    emitted = []
    async for ch in prov.astream(_basic_request()):
        emitted.append(ch)
    assert emitted[-1].is_final
    assert emitted[-1].stop_reason == "incomplete"


# ---------------------------------------------------------------------------
# Fix 3 — _check_native_tools is actually called (strict mode raises)
# ---------------------------------------------------------------------------


def test_strict_native_tools_raises_unsupported_error():
    """strict_native_tools=True must raise per the base.py contract, not warn."""
    from lazybridge.core.providers.base import UnsupportedNativeToolError
    from lazybridge.core.types import NativeTool

    sent = {"called": False}

    def fake_completion(**kwargs):
        sent["called"] = True
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion, strict_native_tools=True)
    req = _basic_request(native_tools=[NativeTool.WEB_SEARCH])
    with pytest.raises(UnsupportedNativeToolError, match="web_search"):
        prov.complete(req)
    # The request never reached litellm.
    assert sent["called"] is False


def test_non_strict_native_tools_still_warn_and_drop():
    """Default (non-strict) mode keeps warn-and-drop, via the base helper."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    from lazybridge.core.types import NativeTool

    req = _basic_request(native_tools=[NativeTool.CODE_EXECUTION])
    with pytest.warns(UserWarning, match="does not support native tool"):
        prov.complete(req)
    assert "native_tools" not in captured


# ---------------------------------------------------------------------------
# Fix 4 — text alongside ToolResultContent is emitted as role=user
# ---------------------------------------------------------------------------


def test_text_next_to_tool_result_emitted_as_user_message():
    """Mixing text + ToolResultContent in a role=tool message must NOT emit the
    text with role='tool' (no tool_call_id → 400). It must be role='user'."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    from lazybridge.core.types import (
        CompletionRequest,
        Message,
        Role,
        TextContent,
        ToolResultContent,
    )

    req = CompletionRequest(
        messages=[
            Message(role=Role.USER, content="search"),
            Message(
                role=Role.TOOL,
                content=[
                    TextContent("operator note alongside the result"),
                    ToolResultContent(tool_use_id="c1", content="result payload"),
                ],
            ),
        ],
        model="litellm/x/y",
    )
    prov.complete(req)
    msgs = captured["messages"]
    text_msg, tool_msg = msgs[-2], msgs[-1]
    # The text part is demoted to user — never role=tool without tool_call_id.
    assert text_msg["role"] == "user"
    assert "tool_call_id" not in text_msg
    # The tool result itself keeps the proper role=tool shape.
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"] == "result payload"


# ---------------------------------------------------------------------------
# Fix 5 — _init_client warns on unknown kwargs instead of silently dropping
# ---------------------------------------------------------------------------


def test_init_client_warns_on_unknown_kwarg():
    """A typo'd config kwarg must raise a UserWarning, not vanish silently."""
    _install_stub_litellm()
    from lazybridge.core.providers.litellm import LiteLLMProvider

    with pytest.warns(UserWarning, match="drop_parms"):
        LiteLLMProvider(drop_parms=True)  # typo of drop_params


def test_init_client_sets_known_kwarg_without_warning():
    """Known module-level flags are still forwarded to the litellm module."""
    stub = _install_stub_litellm()
    from lazybridge.core.providers.litellm import LiteLLMProvider

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning → test failure
        LiteLLMProvider(drop_params=True)
    assert stub.drop_params is True


# ---------------------------------------------------------------------------
# Fix 6 — default_model is None (paid-cloud convention)
# ---------------------------------------------------------------------------


def test_default_model_is_none():
    """Paid cloud bridge: no silent default model."""
    _install_stub_litellm()
    from lazybridge.core.providers.litellm import LiteLLMProvider

    assert LiteLLMProvider.default_model is None


def test_missing_model_raises_helpful_value_error():
    """No request.model, no model=, no fallback → ValueError with fix options."""
    prov, _ = _make_provider()
    req = _basic_request(model=None)
    with pytest.raises(ValueError, match="no model configured"):
        prov.complete(req)


# ---------------------------------------------------------------------------
# Fix 7 — _extract_usage: _compute_cost fallback + token details
# ---------------------------------------------------------------------------


def test_extract_usage_falls_back_to_compute_cost():
    """When LiteLLM can't price the model, _compute_cost is consulted."""

    def fake_completion(**kwargs):
        return _FakeResponse(
            [_FakeChoice(message=_FakeMessage("ok"))],
            usage=_FakeUsage(prompt_tokens=100, completion_tokens=50),
            # no response_cost in _hidden_params
        )

    prov, _ = _make_provider(fake_completion)
    seen = {}

    def fake_compute_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
        seen.update(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )
        return 0.123

    prov._compute_cost = fake_compute_cost
    resp = prov.complete(_basic_request())
    assert resp.usage.cost_usd == 0.123
    assert seen["input_tokens"] == 100
    assert seen["output_tokens"] == 50


def test_extract_usage_prefers_litellm_response_cost_over_fallback():
    """_hidden_params.response_cost wins; _compute_cost is not consulted."""

    def fake_completion(**kwargs):
        return _FakeResponse(
            [_FakeChoice(message=_FakeMessage("ok"))],
            usage=_FakeUsage(10, 5),
            response_cost=0.00042,
        )

    prov, _ = _make_provider(fake_completion)
    prov._compute_cost = lambda *a, **kw: pytest.fail("fallback must not run when response_cost is present")
    resp = prov.complete(_basic_request())
    assert resp.usage.cost_usd == 0.00042


def test_extract_usage_without_cost_stays_none():
    """Base _compute_cost returns None → cost_usd stays None (old behaviour kept)."""

    def fake_completion(**kwargs):
        return _FakeResponse([_FakeChoice(message=_FakeMessage("ok"))], usage=_FakeUsage(1, 1))

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    assert resp.usage.cost_usd is None


def test_extract_usage_reads_thinking_and_cached_token_details():
    """reasoning_tokens / cached_tokens detail blocks land on UsageStats."""

    def fake_completion(**kwargs):
        return _FakeResponse(
            [_FakeChoice(message=_FakeMessage("ok"))],
            usage=_FakeUsage(
                prompt_tokens=20,
                completion_tokens=10,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=8),
                prompt_tokens_details=SimpleNamespace(cached_tokens=12),
            ),
        )

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    assert resp.usage.thinking_tokens == 8
    assert resp.usage.cached_input_tokens == 12


def test_extract_usage_handles_missing_detail_blocks():
    """Backends without the detail blocks keep the zero defaults."""

    def fake_completion(**kwargs):
        return _FakeResponse([_FakeChoice(message=_FakeMessage("ok"))], usage=_FakeUsage(2, 3))

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    assert resp.usage.thinking_tokens == 0
    assert resp.usage.cached_input_tokens == 0


# ---------------------------------------------------------------------------
# Fix 8 — _tool_choice_to_openai dead ``tools`` parameter removed
# ---------------------------------------------------------------------------


def test_tool_choice_to_openai_takes_single_parameter():
    """The dead ``tools`` parameter is gone — one positional arg suffices."""
    _install_stub_litellm()
    from lazybridge.core.providers.litellm import _tool_choice_to_openai

    assert len(inspect.signature(_tool_choice_to_openai).parameters) == 1
    assert _tool_choice_to_openai("auto") == "auto"
    assert _tool_choice_to_openai("any") == "required"
    assert _tool_choice_to_openai("my_tool") == {"type": "function", "function": {"name": "my_tool"}}


def test_tool_choice_still_forwarded_through_build_params():
    """Call-site sanity: tool_choice still reaches litellm after the signature change."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    from lazybridge.core.types import ToolDefinition

    req = _basic_request(
        tools=[ToolDefinition(name="search", description="", parameters={})],
        tool_choice="any",
    )
    prov.complete(req)
    assert captured["tool_choice"] == "required"
