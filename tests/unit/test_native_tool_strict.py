"""Wave 5.1 — strict_native_tools mode + capability-aware errors.

Pre-W5.1 ``_check_native_tools`` only warned-and-dropped unsupported
NativeTool requests (e.g. asking DeepSeek for ``WEB_SEARCH``).  In
production this can mask a misconfiguration: the agent runs without
the requested capability and silently produces a non-grounded reply.

W5.1 adds:
* ``UnsupportedNativeToolError`` (subclasses ValueError so existing
  catch-all sites still match).
* ``strict_native_tools`` flag (class-level + per-instance).  When
  True, unsupported requests raise instead of warning-and-dropping.
* Improved warning / error message that lists the supported native
  tools so the user can see alternatives without hunting docs.
"""

from __future__ import annotations

import warnings

import pytest

from lazybridge.core.providers.base import BaseProvider, UnsupportedNativeToolError
from lazybridge.core.types import NativeTool


# ---------------------------------------------------------------------------
# UnsupportedNativeToolError type
# ---------------------------------------------------------------------------


def test_error_is_value_error_subclass():
    """Existing call sites that catch ValueError still match."""
    assert issubclass(UnsupportedNativeToolError, ValueError)


def test_error_is_distinct_class():
    """Production code can intercept the new type precisely."""
    assert UnsupportedNativeToolError is not ValueError


# ---------------------------------------------------------------------------
# Concrete test provider — supports WEB_SEARCH only
# ---------------------------------------------------------------------------


class _StubProvider(BaseProvider):
    default_model = "stub"
    supported_native_tools = frozenset({NativeTool.WEB_SEARCH})

    def _init_client(self, **kwargs) -> None:
        pass

    def complete(self, request):  # pragma: no cover
        raise NotImplementedError

    def stream(self, request):  # pragma: no cover
        raise NotImplementedError

    async def acomplete(self, request):  # pragma: no cover
        raise NotImplementedError

    async def astream(self, request):  # pragma: no cover
        raise NotImplementedError


class _NoNativeProvider(BaseProvider):
    """Mirrors DeepSeek / LMStudio — no native tools at all."""

    default_model = "no-native"
    supported_native_tools = frozenset()

    def _init_client(self, **kwargs) -> None:
        pass

    def complete(self, request):  # pragma: no cover
        raise NotImplementedError

    def stream(self, request):  # pragma: no cover
        raise NotImplementedError

    async def acomplete(self, request):  # pragma: no cover
        raise NotImplementedError

    async def astream(self, request):  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Default (non-strict) — warn-and-drop preserved
# ---------------------------------------------------------------------------


def test_default_mode_drops_unsupported_with_warning():
    p = _StubProvider()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kept = p._check_native_tools([NativeTool.WEB_SEARCH, NativeTool.CODE_EXECUTION])
    assert kept == [NativeTool.WEB_SEARCH]
    msgs = [str(x.message) for x in w]
    assert any("does not support native tool" in m and "code_execution" in m for m in msgs), msgs


def test_warning_lists_supported_tools_for_discoverability():
    """The new message format helps the user pick an alternative
    without grepping docs."""
    p = _StubProvider()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p._check_native_tools([NativeTool.CODE_EXECUTION])
    msg = str(w[0].message)
    assert "Supported:" in msg
    assert "web_search" in msg


def test_warning_for_provider_with_no_native_tools():
    """Provider with empty supported_native_tools still produces
    a useful message — explicitly states 'none'."""
    p = _NoNativeProvider()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p._check_native_tools([NativeTool.WEB_SEARCH])
    msg = str(w[0].message)
    assert "(none" in msg or "none —" in msg


# ---------------------------------------------------------------------------
# Strict mode — raises UnsupportedNativeToolError
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_unsupported():
    p = _StubProvider(strict_native_tools=True)
    with pytest.raises(UnsupportedNativeToolError, match="does not support native tool"):
        p._check_native_tools([NativeTool.CODE_EXECUTION])


def test_strict_mode_passes_supported_silently():
    p = _StubProvider(strict_native_tools=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kept = p._check_native_tools([NativeTool.WEB_SEARCH])
    assert kept == [NativeTool.WEB_SEARCH]
    # No warnings on the happy path even in strict mode.
    msgs = [str(x.message) for x in w]
    assert not any("does not support" in m for m in msgs)


def test_strict_mode_raises_for_provider_with_no_native_tools():
    p = _NoNativeProvider(strict_native_tools=True)
    with pytest.raises(UnsupportedNativeToolError):
        p._check_native_tools([NativeTool.WEB_SEARCH])


def test_strict_mode_caught_by_value_error_catch_all():
    """Existing user code that does ``except ValueError`` keeps working."""
    p = _StubProvider(strict_native_tools=True)
    with pytest.raises(ValueError):
        p._check_native_tools([NativeTool.CODE_EXECUTION])


# ---------------------------------------------------------------------------
# Per-instance vs class-level toggle
# ---------------------------------------------------------------------------


def test_strict_mode_per_instance_overrides_class_default():
    p_default = _StubProvider()
    p_strict = _StubProvider(strict_native_tools=True)
    assert p_default.strict_native_tools is False
    assert p_strict.strict_native_tools is True


def test_subclass_can_set_class_level_strict_default():
    """A user can build a strict-by-default provider via a subclass."""

    class _AlwaysStrict(_StubProvider):
        strict_native_tools = True

    p = _AlwaysStrict()
    with pytest.raises(UnsupportedNativeToolError):
        p._check_native_tools([NativeTool.CODE_EXECUTION])


# ---------------------------------------------------------------------------
# Built-in provider declarations — confirm nothing regressed
# ---------------------------------------------------------------------------


def test_builtin_anthropic_declares_supported_tools():
    from lazybridge.core.providers.anthropic import AnthropicProvider

    assert NativeTool.WEB_SEARCH in AnthropicProvider.supported_native_tools
    assert NativeTool.CODE_EXECUTION in AnthropicProvider.supported_native_tools


def test_builtin_openai_declares_supported_tools():
    from lazybridge.core.providers.openai import OpenAIProvider

    assert NativeTool.WEB_SEARCH in OpenAIProvider.supported_native_tools
    assert NativeTool.FILE_SEARCH in OpenAIProvider.supported_native_tools


def test_builtin_google_declares_supported_tools():
    from lazybridge.core.providers.google import GoogleProvider

    assert NativeTool.GOOGLE_SEARCH in GoogleProvider.supported_native_tools
    assert NativeTool.GOOGLE_MAPS in GoogleProvider.supported_native_tools


def test_builtin_deepseek_has_no_native_tools():
    """Documented as no native tools — confirm declaration matches docs."""
    from lazybridge.core.providers.deepseek import DeepSeekProvider

    assert DeepSeekProvider.supported_native_tools == frozenset()


def test_builtin_lmstudio_has_no_native_tools():
    from lazybridge.core.providers.lmstudio import LMStudioProvider

    assert LMStudioProvider.supported_native_tools == frozenset()
