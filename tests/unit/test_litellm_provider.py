"""Unit tests for LiteLLMProvider — the LazyBridge ↔ LiteLLM bridge.

Tests are self-contained: we stub the ``litellm`` module via
``sys.modules`` so these run without the real litellm package installed.
That mirrors production usage — everything behind ``lazybridge[litellm]``
is imported lazily, so the test environment only needs what LazyBridge
core needs.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# litellm stub — injected into sys.modules before LiteLLMProvider imports it.
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
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


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


def _install_stub_litellm(completion_impl=None, acompletion_impl=None):
    """Install a fake ``litellm`` module into sys.modules.

    Re-import LiteLLMProvider after calling this if you've already
    imported it in a previous test — or rely on ``_init_client`` to
    capture the fresh module via the re-imported provider class.
    """
    stub = types.ModuleType("litellm")
    stub.completion = completion_impl or MagicMock()
    stub.acompletion = acompletion_impl or MagicMock()
    # Known litellm module-level flags we might set via _init_client kwargs:
    stub.drop_params = False
    stub.set_verbose = False
    sys.modules["litellm"] = stub
    return stub


# ---------------------------------------------------------------------------
# Fixtures & import helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_litellm_imports(monkeypatch):
    """Ensure each test starts with a fresh litellm stub + re-imported provider."""
    # Drop any cached litellm/LiteLLMProvider imports so stubs propagate.
    for mod_name in list(sys.modules):
        if mod_name == "litellm" or mod_name.endswith("providers.litellm"):
            sys.modules.pop(mod_name, None)
    yield
    for mod_name in list(sys.modules):
        if mod_name == "litellm" or mod_name.endswith("providers.litellm"):
            sys.modules.pop(mod_name, None)


def _make_provider(completion_impl=None, acompletion_impl=None, **provider_kwargs):
    """Build a LiteLLMProvider instance backed by a fresh stub."""
    stub = _install_stub_litellm(completion_impl, acompletion_impl)
    from lazybridge.core.providers.litellm import LiteLLMProvider

    return LiteLLMProvider(**provider_kwargs), stub


# ---------------------------------------------------------------------------
# Import guard — missing litellm package
# ---------------------------------------------------------------------------


def test_init_raises_on_missing_litellm(monkeypatch):
    """Without litellm installed, construction must raise a helpful ImportError."""
    # Drop any cached stub so the import actually fails.
    sys.modules.pop("litellm", None)

    # Block import of litellm from the stubs we just flushed.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocker(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("No module named 'litellm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocker)

    from lazybridge.core.providers.litellm import LiteLLMProvider

    with pytest.raises(ImportError, match="lazybridge\\[litellm\\]"):
        LiteLLMProvider()


# ---------------------------------------------------------------------------
# Request translation — messages, tools, model prefix stripping
# ---------------------------------------------------------------------------


def _basic_request(**overrides):
    """Make a minimal CompletionRequest for translation tests."""
    from lazybridge.core.types import CompletionRequest, Message, Role

    defaults = dict(
        messages=[Message(role=Role.USER, content="hello")],
        model="litellm/groq/llama-3.3-70b",
        max_tokens=256,
    )
    defaults.update(overrides)
    return CompletionRequest(**defaults)


def test_prefix_stripped_from_model():
    """``litellm/`` prefix is removed before the string reaches LiteLLM."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(
            [_FakeChoice(message=_FakeMessage(content="hi"))],
            usage=_FakeUsage(1, 2),
        )

    prov, _ = _make_provider(fake_completion)
    prov.complete(_basic_request())
    assert captured["model"] == "groq/llama-3.3-70b"


def test_bare_model_passed_through_unchanged():
    """Model strings without the prefix pass through — LiteLLM's own inference handles them."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(
            [_FakeChoice(message=_FakeMessage(content=""))],
            usage=_FakeUsage(),
        )

    prov, _ = _make_provider(fake_completion)
    from lazybridge.core.types import CompletionRequest, Message, Role

    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        model="mistral/mistral-large-latest",
    )
    prov.complete(req)
    assert captured["model"] == "mistral/mistral-large-latest"


def test_system_prompt_prepended_as_system_message():
    """request.system becomes a leading {role:system,content} entry."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    req = _basic_request(system="you are concise")
    prov.complete(req)

    msgs = captured["messages"]
    assert msgs[0] == {"role": "system", "content": "you are concise"}
    assert msgs[1] == {"role": "user", "content": "hello"}


def test_tools_emitted_as_openai_function_shape():
    """ToolDefinition → OpenAI tools=[{type:function, function:{...}}] list."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    from lazybridge.core.types import ToolDefinition

    req = _basic_request(
        tools=[
            ToolDefinition(
                name="search",
                description="Search the web.",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                strict=True,
            ),
        ],
        tool_choice="search",
    )
    prov.complete(req)

    assert captured["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                "strict": True,
            },
        }
    ]
    # tool_choice names a specific tool → OpenAI shape
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "search"},
    }


def test_tool_choice_keywords_passed_through():
    """'auto' / 'required' / 'none' pass as plain strings."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    from lazybridge.core.types import ToolDefinition

    req = _basic_request(
        tools=[ToolDefinition(name="search", description="", parameters={})],
        tool_choice="auto",
    )
    prov.complete(req)
    assert captured["tool_choice"] == "auto"


def test_tool_choice_any_maps_to_required():
    """The LazyBridge keyword 'any' maps to OpenAI's 'required'."""
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


def test_extra_kwargs_forwarded():
    """CompletionRequest.extra is passed through to litellm.completion verbatim."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    req = _basic_request(extra={"top_p": 0.9, "seed": 42, "response_format": {"type": "json_object"}})
    prov.complete(req)
    assert captured["top_p"] == 0.9
    assert captured["seed"] == 42
    assert captured["response_format"] == {"type": "json_object"}


def test_native_tools_warn_and_drop():
    """native_tools raise a UserWarning and aren't forwarded."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    from lazybridge.core.types import NativeTool

    req = _basic_request(native_tools=[NativeTool.WEB_SEARCH])
    with pytest.warns(UserWarning, match="native_tools"):
        prov.complete(req)
    # Never forwarded.
    assert "native_tools" not in captured


def test_api_key_forwarded_when_set():
    """Explicit api_key= on construction flows through to litellm.completion."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion, api_key="sk-test-42")
    prov.complete(_basic_request())
    assert captured["api_key"] == "sk-test-42"


def test_no_api_key_omitted():
    """When no api_key is set, the kwarg isn't sent — LiteLLM falls back to env."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(message=_FakeMessage(""))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    prov.complete(_basic_request())
    assert "api_key" not in captured


# ---------------------------------------------------------------------------
# Response translation — content, tool_calls, usage, cost
# ---------------------------------------------------------------------------


def test_plain_text_response():
    """Content + stop_reason + usage round-trip into CompletionResponse."""

    def fake_completion(**kwargs):
        return _FakeResponse(
            [_FakeChoice(message=_FakeMessage(content="the answer is 42"))],
            model="groq/llama-3.3-70b",
            usage=_FakeUsage(prompt_tokens=10, completion_tokens=5),
        )

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    assert resp.content == "the answer is 42"
    assert resp.stop_reason == "stop"
    assert resp.model == "groq/llama-3.3-70b"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5


def test_tool_calls_parsed():
    """tool_calls on the response are converted into ToolCall dataclasses."""

    def fake_completion(**kwargs):
        msg = _FakeMessage(
            content="",
            tool_calls=[
                _FakeToolCall(
                    id="call_1",
                    name="search",
                    arguments='{"query": "AI news"}',
                )
            ],
        )
        return _FakeResponse(
            [_FakeChoice(message=msg, finish_reason="tool_calls")],
            usage=_FakeUsage(),
        )

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_1"
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].arguments == {"query": "AI news"}


def test_malformed_tool_call_arguments_fall_back_gracefully():
    """Bad JSON in arguments lands in ``_raw_arguments`` AND is tagged
    with ``_parse_error`` so the engine surfaces a structured TOOL_ERROR
    rather than letting the tool fail later (audit M-A)."""

    def fake_completion(**kwargs):
        msg = _FakeMessage(
            content="",
            tool_calls=[
                _FakeToolCall(
                    id="c1",
                    name="search",
                    arguments="not-json{{{",
                )
            ],
        )
        return _FakeResponse([_FakeChoice(message=msg)], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    args = resp.tool_calls[0].arguments
    assert args["_raw_arguments"] == "not-json{{{"
    assert args.get("_parse_error")


def test_cost_from_hidden_params():
    """LiteLLM's _hidden_params.response_cost becomes usage.cost_usd."""

    def fake_completion(**kwargs):
        return _FakeResponse(
            [_FakeChoice(message=_FakeMessage(content="ok"))],
            usage=_FakeUsage(20, 10),
            response_cost=0.00042,
        )

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    assert resp.usage.cost_usd == 0.00042


def test_missing_cost_stays_none():
    """When LiteLLM can't price the model, cost_usd is None."""

    def fake_completion(**kwargs):
        return _FakeResponse(
            [_FakeChoice(message=_FakeMessage(content="ok"))],
            usage=_FakeUsage(1, 1),
        )

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    assert resp.usage.cost_usd is None


def test_empty_choices_returns_empty_content():
    """Defensive path: provider hands back no choices."""

    def fake_completion(**kwargs):
        return _FakeResponse([], usage=_FakeUsage())

    prov, _ = _make_provider(fake_completion)
    resp = prov.complete(_basic_request())
    assert resp.content == ""
    assert resp.tool_calls == []


# ---------------------------------------------------------------------------
# Streaming — sync
# ---------------------------------------------------------------------------


def _stream_chunks(*chunks):
    """Build a LiteLLM-style stream generator from pre-built chunks."""

    def _gen():
        yield from chunks

    return _gen


def test_stream_yields_deltas_then_final_chunk():
    """Sync stream emits text deltas then a final chunk with stop_reason + usage."""
    chunks = [
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content="hello "))]),
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content="world"))]),
        _FakeResponse(
            [_FakeChoice(delta=_FakeDelta(content=None), finish_reason="stop")],
            usage=_FakeUsage(5, 2),
        ),
    ]

    def fake_completion(**kwargs):
        return _stream_chunks(*chunks)()

    prov, _ = _make_provider(fake_completion)
    emitted = list(prov.stream(_basic_request()))

    # Three yielded: "hello ", "world", final chunk.
    assert [c.delta for c in emitted[:2]] == ["hello ", "world"]
    assert emitted[-1].is_final
    assert emitted[-1].stop_reason == "stop"
    assert emitted[-1].usage.input_tokens == 5
    assert emitted[-1].usage.output_tokens == 2


def test_stream_accumulates_tool_calls_across_chunks():
    """Tool-call deltas from multiple chunks merge into one complete ToolCall on the final chunk."""
    chunks = [
        _FakeResponse(
            [
                _FakeChoice(
                    delta=_FakeDelta(
                        tool_calls=[_FakeToolCall(id="call_1", name="search", arguments='{"query"', index=0)],
                    )
                )
            ]
        ),
        _FakeResponse(
            [
                _FakeChoice(
                    delta=_FakeDelta(
                        tool_calls=[_FakeToolCall(arguments=': "AI"', index=0)],
                    )
                )
            ]
        ),
        _FakeResponse(
            [
                _FakeChoice(
                    delta=_FakeDelta(
                        tool_calls=[_FakeToolCall(arguments="}", index=0)],
                    )
                )
            ]
        ),
        _FakeResponse(
            [_FakeChoice(delta=_FakeDelta(), finish_reason="tool_calls")],
            usage=_FakeUsage(),
        ),
    ]

    def fake_completion(**kwargs):
        return _stream_chunks(*chunks)()

    prov, _ = _make_provider(fake_completion)
    emitted = list(prov.stream(_basic_request()))

    final = emitted[-1]
    assert final.is_final
    assert len(final.tool_calls) == 1
    tc = final.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "search"
    assert tc.arguments == {"query": "AI"}


# ---------------------------------------------------------------------------
# Async — acomplete and astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acomplete_calls_async_path():
    """acomplete() routes through litellm.acompletion, not sync completion."""
    called = {"sync": 0, "async": 0}

    def fake_sync(**kw):
        called["sync"] += 1
        return _FakeResponse([_FakeChoice(message=_FakeMessage(content="sync"))], usage=_FakeUsage())

    async def fake_async(**kw):
        called["async"] += 1
        return _FakeResponse([_FakeChoice(message=_FakeMessage(content="async"))], usage=_FakeUsage())

    prov, _ = _make_provider(fake_sync, fake_async)
    resp = await prov.acomplete(_basic_request())
    assert resp.content == "async"
    assert called == {"sync": 0, "async": 1}


@pytest.mark.asyncio
async def test_astream_yields_async_chunks():
    """astream() consumes async iterators and emits deltas + final chunk."""
    chunks = [
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content="a"))]),
        _FakeResponse([_FakeChoice(delta=_FakeDelta(content="b"))]),
        _FakeResponse(
            [_FakeChoice(delta=_FakeDelta(), finish_reason="stop")],
            usage=_FakeUsage(3, 2),
        ),
    ]

    class _AIter:
        def __init__(self, items):
            self._items = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    async def fake_acompletion(**kw):
        return _AIter(chunks)

    prov, _ = _make_provider(None, fake_acompletion)
    emitted = []
    async for ch in prov.astream(_basic_request()):
        emitted.append(ch)

    assert [c.delta for c in emitted[:2]] == ["a", "b"]
    assert emitted[-1].is_final
    assert emitted[-1].stop_reason == "stop"
    assert emitted[-1].usage.input_tokens == 3


# ---------------------------------------------------------------------------
# LLMEngine routing — the litellm/ prefix rule
# ---------------------------------------------------------------------------


def test_llmengine_routes_litellm_prefix_to_litellm_provider():
    """``Agent('litellm/...')`` must resolve to the litellm provider, not the default."""
    _install_stub_litellm()

    from lazybridge.engines.llm import LLMEngine

    assert LLMEngine._infer_provider("litellm/groq/llama") == "litellm"


def test_llmengine_keeps_native_claude_routing_intact():
    """Native provider routing must NOT be hijacked — claude-* still goes to anthropic."""
    _install_stub_litellm()

    from lazybridge.engines.llm import LLMEngine

    assert LLMEngine._infer_provider("claude-opus-4-7") == "anthropic"


def test_executor_resolves_litellm_provider_instance():
    """The provider registry returns a LiteLLMProvider when given 'litellm'."""
    _install_stub_litellm()

    from lazybridge.core.executor import _resolve_provider

    prov = _resolve_provider("litellm", model="litellm/groq/llama-3.3-70b")
    from lazybridge.core.providers.litellm import LiteLLMProvider

    assert isinstance(prov, LiteLLMProvider)


# ---------------------------------------------------------------------------
# Tool-result round-trip — Role.TOOL messages translate correctly
# ---------------------------------------------------------------------------


def test_tool_result_block_becomes_role_tool_message():
    """ToolResultContent inside a LazyBridge Message emits {role:tool, tool_call_id, content}."""
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
            Message(role=Role.ASSISTANT, content=[TextContent("calling tool")]),
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultContent(tool_use_id="c1", content="results here"),
                ],
            ),
        ],
        model="litellm/x/y",
    )
    prov.complete(req)
    msgs = captured["messages"]
    # Last message must be role=tool with the tool_call_id and content.
    last = msgs[-1]
    assert last["role"] == "tool"
    assert last["tool_call_id"] == "c1"
    assert last["content"] == "results here"
