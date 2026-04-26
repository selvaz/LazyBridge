"""LM Studio provider for LazyBridge — local LLM inference.

`LM Studio <https://lmstudio.ai/>`_ runs any GGUF / MLX model locally and
exposes them through an OpenAI-compatible REST API (the *local server*
feature, default ``http://localhost:1234/v1``).  Both Chat Completions
(``/v1/chat/completions``) and Responses (``/v1/responses``) are
supported — LM Studio added Responses API compatibility in v0.3.29
(2026), including streaming, reasoning effort, stateful chats via
``previous_response_id``, MCP tools, and token caching.

This provider is a thin subclass of
:class:`~lazybridge.core.providers.openai.OpenAIProvider` that:

* Points the underlying ``openai`` client at the local server.
* Lets traffic flow to either ``/v1/chat/completions`` or
  ``/v1/responses`` — the inherited routing decides per-request based
  on whether a Pydantic model schema is set.
* Reports zero cost (local inference is free).
* Drops native-tool support (web search, code interpreter etc. are
  cloud-only OpenAI features that have no analogue locally).  Remote
  MCP tools, when supported by the loaded model, flow through the
  standard ``tools=[...]`` path.

Configuration::

    from lazybridge import Agent
    from lazybridge.core.providers.lmstudio import LMStudioProvider

    # Use the LM Studio server's currently-loaded model (no name needed —
    # LM Studio routes "local-model" to whichever model the user loaded).
    agent = Agent(LMStudioProvider())

    # Or pass the exact identifier shown in the LM Studio "Local Server" tab:
    agent = Agent(LMStudioProvider(model="lmstudio-community/Qwen2.5-7B-Instruct-GGUF"))

    # Custom server (e.g. LM Studio running on another machine on the LAN):
    agent = Agent(LMStudioProvider(base_url="http://192.168.1.50:1234/v1"))

Environment overrides (no API key is required for a local server):

* ``LMSTUDIO_BASE_URL``  — default base URL when ``base_url=`` is unset.
* ``LMSTUDIO_API_KEY``   — only needed when LM Studio sits behind an
  authenticated reverse proxy.  Defaults to ``"lm-studio"`` (the value
  the official docs suggest, which the server itself ignores).

Function calling and JSON-mode structured output flow through the
standard OpenAI Chat Completions path inherited from
:class:`OpenAIProvider`.  Whether a particular call succeeds depends on
the loaded model — for example, Qwen-2.5-Instruct supports both, while
older Llama variants may only return text.  When the model can't honour
``tools=`` the request still succeeds and just returns a plain text
response.
"""

from __future__ import annotations

import os
from typing import Any

from lazybridge.core.providers.openai import OpenAIProvider
from lazybridge.core.types import (
    CompletionRequest,
    NativeTool,
)

#: Default base URL of the LM Studio "Local Server" feature.
_DEFAULT_BASE_URL = "http://localhost:1234/v1"

#: LM Studio ignores the API key when it is not behind an auth proxy, but
#: the underlying ``openai`` SDK rejects an empty string — so we send a
#: harmless placeholder that mirrors the value LM Studio's own docs use.
_PLACEHOLDER_API_KEY = "lm-studio"

#: Env var that can override the base URL without code changes.
_BASE_URL_ENV = "LMSTUDIO_BASE_URL"

#: Env var for an optional API key (only required when LM Studio is
#: fronted by an authenticated reverse proxy).
_API_KEY_ENV = "LMSTUDIO_API_KEY"

#: Model-string prefix the LLMEngine routing rule uses to opt in to this
#: provider (mirrors LiteLLM's ``litellm/<model>`` convention).  Stripped
#: from the model name before it reaches the LM Studio server, which
#: expects the bare loaded-model identifier.
_PREFIX = "lmstudio/"


class LMStudioProvider(OpenAIProvider):
    """Provider for `LM Studio <https://lmstudio.ai/>`_'s local OpenAI-compatible server.

    Inherits the OpenAI Chat Completions code path verbatim — all
    function-calling, streaming, and JSON-mode behaviour comes from
    :class:`OpenAIProvider`.  This class only customises the transport
    (local URL, no auth), the cost model (free), and the model catalogue
    (one logical "local-model" tier).
    """

    #: ``"local-model"`` is the conventional placeholder LM Studio
    #: accepts when the caller does not care which loaded model is used.
    #: Override per call (``Agent("lmstudio", model="...")``) or per
    #: instance (``LMStudioProvider(model="...")``) when you want to
    #: select among multiple loaded models.
    default_model = "local-model"

    #: All tier aliases collapse onto the single locally-loaded model —
    #: cost / capability tiers don't apply to a self-hosted server.
    _TIER_ALIASES = {
        "top": "local-model",
        "expensive": "local-model",
        "medium": "local-model",
        "cheap": "local-model",
        "super_cheap": "local-model",
    }

    #: No fallbacks — a local server has exactly one currently-loaded
    #: model.  Switching models is a user action inside LM Studio, not
    #: something the provider can do for you.
    _FALLBACKS: dict[str, list[str]] = {}

    #: Native server-side tools (web search, code interpreter, ...) are
    #: cloud-only OpenAI features — disable them so callers get a clear
    #: warning rather than a silent 404 from the local server.
    supported_native_tools: frozenset[NativeTool] = frozenset()

    def _init_client(self, **kwargs: Any) -> None:
        """Point the OpenAI SDK at the local LM Studio server.

        ``base_url`` / ``api_key`` precedence (highest first):

        1. Explicit ``base_url=`` / ``api_key=`` keyword args.
        2. ``LMSTUDIO_BASE_URL`` / ``LMSTUDIO_API_KEY`` environment vars.
        3. The hard-coded local defaults.
        """
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai") from None

        base_url = kwargs.pop("base_url", None) or os.environ.get(_BASE_URL_ENV) or _DEFAULT_BASE_URL
        key = self.api_key or os.environ.get(_API_KEY_ENV) or _PLACEHOLDER_API_KEY

        self._client = _openai.OpenAI(api_key=key, base_url=base_url, **kwargs)
        self._async_client = _openai.AsyncOpenAI(api_key=key, base_url=base_url, **kwargs)

    def _resolve_model(self, request: CompletionRequest) -> str:
        """Strip the optional ``lmstudio/`` routing prefix.

        ``Agent("lmstudio/Qwen2.5-7B-Instruct")`` flows through here as
        ``"lmstudio/Qwen2.5-7B-Instruct"``; LM Studio expects the bare
        ``"Qwen2.5-7B-Instruct"`` (the identifier shown in the Local
        Server tab), so we drop the prefix once before delegating to the
        inherited tier-alias / passthrough resolver.
        """
        resolved = super()._resolve_model(request)
        if resolved.startswith(_PREFIX):
            resolved = resolved[len(_PREFIX) :]
        return resolved

    # ``_use_responses_api`` is intentionally NOT overridden — LM Studio
    # supports both ``/v1/chat/completions`` and ``/v1/responses``, so
    # the inherited OpenAIProvider routing (Responses for dict-schema
    # structured output and tool-call flows; Chat Completions for
    # Pydantic-model schemas that need ``beta.parse``) is correct as-is.

    def _is_reasoning_model(self, model: str) -> bool:
        """Whether the loaded model accepts OpenAI's ``reasoning_effort`` knob.

        LM Studio's Responses API forwards ``reasoning.effort`` to
        models that support it (e.g. ``openai/gpt-oss-20b``).  Whether a
        given local model honours the parameter depends on which model
        is loaded; we conservatively return ``False`` so the inherited
        param-builder stays on ``max_tokens`` / ``temperature`` and
        callers wanting reasoning opt in explicitly via ``ThinkingConfig``.
        """
        return False

    def _compute_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Local inference is free — always report ``$0.00``.

        Returning ``0.0`` (rather than ``None``) lets cost-tracking code
        downstream still aggregate without special-casing local providers.
        """
        return 0.0

    def get_default_max_tokens(self, model: str | None = None) -> int:
        """Conservative default — most local models ship with ≤8K context.

        Callers running a long-context model (Qwen-2.5-128K, Llama-3.1-128K,
        ...) should set ``max_tokens=`` explicitly per call.
        """
        return 4096
