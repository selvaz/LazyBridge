"""Unit tests for ``LMStudioProvider``.

Mocks the ``openai`` SDK at the module boundary (``OpenAI`` /
``AsyncOpenAI`` classes), so these tests run with no real LM Studio
server and no network traffic.

Coverage:
  * ``_init_client`` builds the OpenAI SDK client with the local base URL,
    honours ``LMSTUDIO_BASE_URL`` and explicit ``base_url=`` overrides,
    and forwards an explicit / env-supplied API key.
  * Routing: ``Agent("lmstudio")`` and ``Agent("lmstudio/<model>")``
    resolve to the LMStudio provider via :func:`_resolve_provider` and
    :meth:`LLMEngine._infer_provider`.
  * ``_resolve_model`` strips the optional ``lmstudio/`` prefix.
  * Tier aliases all collapse onto ``"local-model"`` (audit F2 invariant).
  * ``_use_responses_api`` is hard-disabled — LM Studio only implements
    Chat Completions.
  * ``_compute_cost`` always reports ``0.0`` (local inference is free).
  * Native tools requested by the caller emit a ``UserWarning`` and are
    dropped (LM Studio has no server-side tools).
  * ``complete`` round-trips a Chat Completions response through the
    inherited OpenAI parser.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lazybridge.core.providers.lmstudio import (
    _DEFAULT_BASE_URL,
    _PLACEHOLDER_API_KEY,
    LMStudioProvider,
)
from lazybridge.core.types import (
    CompletionRequest,
    Message,
    NativeTool,
    Role,
)


# ---------------------------------------------------------------------------
# Fakes — minimal OpenAI Chat Completions response shape
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message=None, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, choices, model="local-model", usage=None):
        self.choices = choices
        self.model = model
        self.usage = usage


def _basic_request(**overrides) -> CompletionRequest:
    defaults = dict(
        messages=[Message(role=Role.USER, content="hello")],
        max_tokens=64,
    )
    defaults.update(overrides)
    return CompletionRequest(**defaults)


# ---------------------------------------------------------------------------
# _init_client — base URL / API key resolution
# ---------------------------------------------------------------------------


def _patch_openai():
    """Patch the ``openai`` module's ``OpenAI`` and ``AsyncOpenAI`` classes."""
    return patch("openai.OpenAI"), patch("openai.AsyncOpenAI")


def test_init_client_uses_default_base_url(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
    sync_p, async_p = _patch_openai()
    with sync_p as sync_cls, async_p as async_cls:
        LMStudioProvider()
    sync_cls.assert_called_once()
    kwargs = sync_cls.call_args.kwargs
    assert kwargs["base_url"] == _DEFAULT_BASE_URL
    assert kwargs["api_key"] == _PLACEHOLDER_API_KEY
    async_cls.assert_called_once()
    assert async_cls.call_args.kwargs["base_url"] == _DEFAULT_BASE_URL


def test_init_client_explicit_base_url_wins(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://from-env:9999/v1")
    sync_p, async_p = _patch_openai()
    with sync_p as sync_cls, async_p:
        LMStudioProvider(base_url="http://explicit:1234/v1")
    assert sync_cls.call_args.kwargs["base_url"] == "http://explicit:1234/v1"


def test_init_client_env_base_url_overrides_default(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://lan-host:1234/v1")
    sync_p, async_p = _patch_openai()
    with sync_p as sync_cls, async_p:
        LMStudioProvider()
    assert sync_cls.call_args.kwargs["base_url"] == "http://lan-host:1234/v1"


def test_init_client_explicit_api_key_wins(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_API_KEY", "from-env")
    sync_p, async_p = _patch_openai()
    with sync_p as sync_cls, async_p:
        LMStudioProvider(api_key="explicit-key")
    assert sync_cls.call_args.kwargs["api_key"] == "explicit-key"


def test_init_client_env_api_key(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
    monkeypatch.setenv("LMSTUDIO_API_KEY", "proxy-token")
    sync_p, async_p = _patch_openai()
    with sync_p as sync_cls, async_p:
        LMStudioProvider()
    assert sync_cls.call_args.kwargs["api_key"] == "proxy-token"


# ---------------------------------------------------------------------------
# Class-level invariants
# ---------------------------------------------------------------------------


def test_default_model_is_local_model():
    assert LMStudioProvider.default_model == "local-model"


def test_no_native_tools_supported():
    assert LMStudioProvider.supported_native_tools == frozenset()


def test_all_five_tiers_defined_and_collapse_to_local_model():
    expected = {"top", "expensive", "medium", "cheap", "super_cheap"}
    assert set(LMStudioProvider._TIER_ALIASES) == expected
    for tier in expected:
        assert LMStudioProvider._TIER_ALIASES[tier] == "local-model"


# ---------------------------------------------------------------------------
# _resolve_model — prefix stripping + tier passthrough
# ---------------------------------------------------------------------------


def _bare_provider() -> LMStudioProvider:
    """A provider instance with no SDK client — just enough for resolver tests."""
    p = LMStudioProvider.__new__(LMStudioProvider)
    p.api_key = None
    p.model = LMStudioProvider.default_model
    return p


def test_resolve_model_strips_lmstudio_prefix():
    p = _bare_provider()
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="hi")],
        model="lmstudio/Qwen2.5-7B-Instruct",
    )
    assert p._resolve_model(req) == "Qwen2.5-7B-Instruct"


def test_resolve_model_passes_bare_model_through():
    p = _bare_provider()
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="hi")],
        model="my-finetune-v2",
    )
    assert p._resolve_model(req) == "my-finetune-v2"


def test_resolve_model_resolves_tier_alias_to_local_model():
    p = _bare_provider()
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="hi")],
        model="top",
    )
    assert p._resolve_model(req) == "local-model"


# ---------------------------------------------------------------------------
# _use_responses_api / _is_reasoning_model / _compute_cost / max_tokens
# ---------------------------------------------------------------------------


def test_responses_api_disabled_for_every_request():
    p = _bare_provider()
    req = CompletionRequest(messages=[Message(role=Role.USER, content="hi")])
    assert p._use_responses_api(req) is False


def test_is_reasoning_model_always_false():
    p = _bare_provider()
    # Even names that LOOK like OpenAI reasoning models stay False locally.
    assert p._is_reasoning_model("o3-mini") is False
    assert p._is_reasoning_model("gpt-5.5-pro") is False
    assert p._is_reasoning_model("local-model") is False


def test_compute_cost_is_zero():
    p = _bare_provider()
    assert p._compute_cost("local-model", 1_000_000, 1_000_000) == 0.0
    assert p._compute_cost("anything", 0, 0) == 0.0


def test_default_max_tokens():
    p = _bare_provider()
    assert p.get_default_max_tokens() == 4096
    assert p.get_default_max_tokens("Qwen2.5-7B-Instruct") == 4096


# ---------------------------------------------------------------------------
# Native tools — silently dropped with UserWarning
# ---------------------------------------------------------------------------


def test_native_tools_warn_and_are_filtered():
    p = _bare_provider()
    with pytest.warns(UserWarning, match="does not support native tool"):
        kept = p._check_native_tools([NativeTool.WEB_SEARCH, NativeTool.CODE_EXECUTION])
    assert kept == []


# ---------------------------------------------------------------------------
# Routing — registry + LLMEngine inference
# ---------------------------------------------------------------------------


def test_executor_resolves_lmstudio_alias_to_provider():
    sync_p, async_p = _patch_openai()
    with sync_p, async_p:
        from lazybridge.core.executor import _resolve_provider
        prov = _resolve_provider("lmstudio")
    assert isinstance(prov, LMStudioProvider)


@pytest.mark.parametrize("alias", ["lmstudio", "lm-studio", "lm_studio", "local"])
def test_executor_accepts_all_lmstudio_aliases(alias):
    sync_p, async_p = _patch_openai()
    with sync_p, async_p:
        from lazybridge.core.executor import _resolve_provider
        prov = _resolve_provider(alias)
    assert isinstance(prov, LMStudioProvider)


def test_llmengine_routes_lmstudio_alias():
    from lazybridge.engines.llm import LLMEngine
    assert LLMEngine._infer_provider("lmstudio") == "lmstudio"
    assert LLMEngine._infer_provider("lm-studio") == "lmstudio"
    assert LLMEngine._infer_provider("local") == "lmstudio"
    assert LLMEngine._infer_provider("local-model") == "lmstudio"


def test_llmengine_routes_lmstudio_prefix():
    from lazybridge.engines.llm import LLMEngine
    assert LLMEngine._infer_provider("lmstudio/Qwen2.5-7B-Instruct") == "lmstudio"


def test_llmengine_native_routing_unaffected():
    """Adding the lmstudio routes must not steal any other provider's traffic."""
    from lazybridge.engines.llm import LLMEngine
    assert LLMEngine._infer_provider("claude-opus-4-7") == "anthropic"
    assert LLMEngine._infer_provider("gpt-4o") == "openai"
    assert LLMEngine._infer_provider("deepseek-v4-flash") == "deepseek"


# ---------------------------------------------------------------------------
# complete() round-trip — Chat Completions path used (no Responses API)
# ---------------------------------------------------------------------------


def test_complete_uses_chat_completions_endpoint():
    """A real LM Studio server only exposes /v1/chat/completions — make sure
    the inherited OpenAI ``complete()`` takes that branch (not /v1/responses).
    """
    sync_client = MagicMock()
    sync_client.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice(message=_FakeMessage(content="hi from local"))],
        model="local-model",
        usage=_FakeUsage(prompt_tokens=4, completion_tokens=3),
    )
    sync_p, async_p = _patch_openai()
    with sync_p as sync_cls, async_p:
        sync_cls.return_value = sync_client
        prov = LMStudioProvider()
        resp = prov.complete(_basic_request())

    # Chat Completions path — NOT responses.create
    sync_client.chat.completions.create.assert_called_once()
    sync_client.responses.create.assert_not_called()

    assert resp.content == "hi from local"
    assert resp.usage.input_tokens == 4
    assert resp.usage.output_tokens == 3
    # Local inference is free.
    assert resp.usage.cost_usd == 0.0


def test_complete_strips_prefix_before_call():
    """The model name reaching the SDK must NOT carry the ``lmstudio/`` prefix."""
    sync_client = MagicMock()
    sync_client.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice(message=_FakeMessage(content=""))],
        usage=_FakeUsage(),
    )
    sync_p, async_p = _patch_openai()
    with sync_p as sync_cls, async_p:
        sync_cls.return_value = sync_client
        prov = LMStudioProvider(model="lmstudio/Qwen2.5-7B-Instruct")
        prov.complete(_basic_request())

    call_kwargs = sync_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "Qwen2.5-7B-Instruct"
