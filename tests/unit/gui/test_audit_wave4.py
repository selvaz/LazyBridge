"""Regression tests for Wave 4 of the deep audit (provider consistency)."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# M6 — Gemini grounding + structured output raises clearly
# ---------------------------------------------------------------------------


def test_gemini_grounding_plus_structured_output_raises():
    from lazybridge.core.providers.google import GoogleProvider
    from lazybridge.core.types import (
        CompletionRequest,
        Message,
        NativeTool,
        Role,
        StructuredOutputConfig,
    )

    provider = GoogleProvider.__new__(GoogleProvider)
    provider._default_model = "gemini-test"
    provider._client = MagicMock()
    provider._async_client = MagicMock()
    provider._api_key = "x"

    request = CompletionRequest(
        model="gemini-test",
        messages=[Message(role=Role.USER, content="hi")],
        native_tools=[NativeTool.GOOGLE_SEARCH],
        structured_output=StructuredOutputConfig(schema={"type": "object"}),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        provider._build_config(request)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# M7 — Anthropic warns (does not silently drop) when temperature is ignored
# ---------------------------------------------------------------------------


def test_anthropic_warns_when_temperature_dropped_on_opus_47():
    from lazybridge.core.providers.anthropic import _NO_SAMPLING_MODELS, AnthropicProvider
    from lazybridge.core.types import CompletionRequest, Message, Role

    # Pick the first no-sampling model alias (robust against table changes).
    no_sampling_model = next(iter(_NO_SAMPLING_MODELS))

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._default_model = no_sampling_model
    provider._api_key = "x"
    provider._client = MagicMock()
    provider._async_client = MagicMock()
    provider._beta_overrides = {}

    request = CompletionRequest(
        model=no_sampling_model,
        messages=[Message(role=Role.USER, content="hi")],
        temperature=0.7,
        max_tokens=128,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        params = provider._build_params(request)  # type: ignore[attr-defined]
    assert "temperature" not in params, "temperature must NOT be sent to a no-sampling model"
    assert any("does not support the temperature parameter" in str(w.message) for w in captured), captured


# ---------------------------------------------------------------------------
# M8 — DeepSeek no longer auto-switches; it raises
# ---------------------------------------------------------------------------


def test_deepseek_thinking_on_non_reasoner_raises():
    from lazybridge.core.providers.deepseek import DeepSeekProvider
    from lazybridge.core.types import (
        CompletionRequest,
        Message,
        Role,
        ThinkingConfig,
    )

    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider._default_model = "deepseek-chat"
    provider._api_key = "x"
    provider._client = MagicMock()
    provider._async_client = MagicMock()

    request = CompletionRequest(
        model="deepseek-chat",
        messages=[Message(role=Role.USER, content="hi")],
        thinking=ThinkingConfig(enabled=True),
    )
    with pytest.raises(ValueError, match="does not support reasoning"):
        provider._resolve_thinking(request)


def test_deepseek_thinking_on_reasoner_passes_through():
    from lazybridge.core.providers.deepseek import DeepSeekProvider
    from lazybridge.core.types import (
        CompletionRequest,
        Message,
        Role,
        ThinkingConfig,
    )

    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider._default_model = "deepseek-reasoner"
    provider._api_key = "x"

    request = CompletionRequest(
        model="deepseek-reasoner",
        messages=[Message(role=Role.USER, content="hi")],
        thinking=ThinkingConfig(enabled=True),
    )
    out = provider._resolve_thinking(request)
    assert out.model == "deepseek-reasoner"


# ---------------------------------------------------------------------------
# M9 — jsonschema is used when available and covers features the subset misses
# ---------------------------------------------------------------------------


def test_structured_validate_uses_jsonschema_when_installed():
    """If the `jsonschema` library is importable, `pattern` constraints
    should be enforced — the subset validator silently passed them."""
    try:
        import jsonschema  # noqa: F401
    except Exception:
        pytest.skip("jsonschema not installed; falls back to subset validator")
    from lazybridge.core.structured import _validate_schema

    schema = {"type": "string", "pattern": "^[a-z]+$"}
    # "abc" passes, "ABC" should fail under real jsonschema.
    assert _validate_schema("abc", schema) is None
    err = _validate_schema("ABC", schema)
    assert err is not None, "jsonschema didn't catch the pattern mismatch"


def test_structured_subset_validator_still_works_when_jsonschema_missing():
    """Smoke test: the subset path (called directly) still catches the
    cases it used to."""
    from lazybridge.core.structured import _validate_schema_subset

    schema = {"type": "object", "required": ["x"]}
    assert _validate_schema_subset({}, schema) is not None
    assert _validate_schema_subset({"x": 1}, schema) is None


# ---------------------------------------------------------------------------
# M13 — OpenAI streaming with a Pydantic schema warns
# ---------------------------------------------------------------------------


def test_openai_streaming_with_pydantic_schema_warns():
    from pydantic import BaseModel

    from lazybridge.core.providers.openai import OpenAIProvider
    from lazybridge.core.types import (
        CompletionRequest,
        Message,
        Role,
        StructuredOutputConfig,
    )

    class _Out(BaseModel):
        name: str

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._api_key = "x"
    provider._default_model = "gpt-test"
    provider._beta_overrides = {}

    # Force the Chat Completions path (we only patched that one).
    provider._use_responses_api = lambda r: False  # type: ignore[method-assign]

    # Stub client so the generator completes quickly without hitting the API.
    class _FakeChoice:
        def __init__(self):
            self.delta = type("d", (), {"content": None, "tool_calls": None})()
            self.finish_reason = "stop"

    class _FakeChunk:
        def __init__(self):
            self.usage = None
            self.choices = [_FakeChoice()]
            self.model = "gpt-test"

    def _empty_stream():
        yield _FakeChunk()

    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _empty_stream()

    request = CompletionRequest(
        model="gpt-test",
        messages=[Message(role=Role.USER, content="hi")],
        stream=True,
        structured_output=StructuredOutputConfig(schema=_Out),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        list(provider.stream(request))
    assert any("streaming with a Pydantic output_schema is best-effort" in str(w.message) for w in captured), captured


# ---------------------------------------------------------------------------
# L8 — Executor retry warning uses repr(exc) for type visibility
# ---------------------------------------------------------------------------


def test_executor_retry_warning_includes_repr():
    from lazybridge.core.executor import _RETRY_WARN

    assert "{exc!r}" in _RETRY_WARN, "retry warning lost the !r format spec"
