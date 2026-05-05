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

    #: Tier aliases map to RECOMMENDED open-weight models that fit
    #: comfortably on a ~16 GB-VRAM consumer GPU (e.g. AMD RX 9070 XT,
    #: NVIDIA 4070 Ti / 5070).  These strings are passed verbatim to
    #: LM Studio's ``/v1/chat/completions`` and ``/v1/responses``
    #: endpoints; LM Studio matches whichever model is currently loaded.
    #:
    #: Match the tier's identifier in LM Studio's "Local Server" tab —
    #: download the matching GGUF / MLX bundle there before calling
    #: the provider with the tier name.  Override per-call
    #: (``Agent("lmstudio", model="...")``) when you want a specific
    #: build (Q4_K_M vs Q5_K_M, MLX vs GGUF, etc.).
    #:
    #: Recommendations as of April 2026 — canonical LM Studio
    #: identifiers from ``lmstudio.ai/models``.  Mix of OpenAI
    #: open-weight (Apache 2.0), Google Gemma 4 (open weights), and
    #: Alibaba Qwen 3.5 (dense) so the tier ladder spans MoE and
    #: dense-small architectures:
    #:
    #: * ``top`` — gpt-oss-20b (OpenAI open-weight MoE; 21 B total,
    #:   3.6 B active; ~13.5 GB at Q4_K_M).  Best all-rounder for a
    #:   16 GB GPU; perfect-logic class.
    #: * ``expensive`` — gemma-4-26b-a4b (Google MoE; 26 B total,
    #:   4 B active; ~14–16 GB at Q4_K_M).  Strong reasoning + native
    #:   vision; tighter VRAM headroom than ``top`` but better at
    #:   long context.
    #: * ``medium`` — qwen3.5-14b (dense 14 B; ~10.7 GB Q4_K_M).
    #:   Strong general + tool-use, leaves headroom for KV cache.
    #: * ``cheap`` — qwen3.5-4b (dense 4 B; ~3 GB Q4_K_M).  Fast
    #:   everyday tasks, fits alongside other workloads on the GPU.
    #: * ``super_cheap`` — gemma-4-e2b (Google "edge" 2 B; ~1.5 GB at
    #:   Q4).  Tiny / always-on; routes to a long-context window.
    #:
    #: These are SUGGESTIONS — LM Studio passes the model string
    #: verbatim to its server; whichever build (GGUF / MLX, Q4 / Q5)
    #: the user downloaded is what answers.  Override per-call when
    #: you have a specific quant in mind:
    #: ``Agent("lmstudio", model="lmstudio-community/Qwen3.5-14B-GGUF")``.
    _TIER_ALIASES = {
        "top": "gpt-oss-20b",
        "expensive": "gemma-4-26b-a4b",
        "medium": "qwen3.5-14b",
        "cheap": "qwen3.5-4b",
        "super_cheap": "gemma-4-e2b",
    }

    #: No automatic fallback chain — a local server has exactly one
    #: currently-loaded model and the provider can't trigger LM Studio
    #: to swap models on the fly.  Use ``Agent(fallback=Agent(...))``
    #: at the framework level if you want a multi-model retry policy.
    _FALLBACKS: dict[str, list[str]] = {}

    #: Native server-side tools (web search, code interpreter, ...) are
    #: cloud-only OpenAI features — disable them so callers get a clear
    #: warning rather than a silent 404 from the local server.
    supported_native_tools: frozenset[NativeTool] = frozenset()

    # LM Studio routes any model the user has loaded — vision and audio
    # capability depend entirely on which weights are downloaded.  We
    # take the optimistic stance and report ``True`` so callers can pass
    # images / audio without LazyBridge stripping them client-side; if
    # the loaded model can't handle them the LM Studio server will
    # error or silently degrade and the caller sees that directly.
    # Override per-call by setting ``strict_multimodal=True`` and a
    # known model name to force a clear failure mode.

    @classmethod
    def supports_vision(cls, model: str | None = None) -> bool:
        return True

    @classmethod
    def supports_audio(cls, model: str | None = None) -> bool:
        return True

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
        """Per-model output-token defaults for the recommended catalogue.

        Output tokens, not context.  All recommended models ship with a
        128K+ context window; the cap here is what we'll request the
        model emit on a single call.  Callers running a different
        loaded model should set ``max_tokens=`` explicitly per call.
        """
        resolved = (model or self.model or self.default_model or "").lower()
        if "gpt-oss" in resolved:
            return 16_384  # 128K context, MoE, comfortable output budget
        if "gemma-4" in resolved and "26b" in resolved:
            return 32_768  # 256K context, MoE
        if "gemma-4-31b" in resolved:
            return 32_768  # 256K context, dense — only on 24GB+ GPUs
        if "gemma-4-e4b" in resolved or "gemma-4-e2b" in resolved:
            return 8_192  # 128K context, edge-class
        if "qwen3.5" in resolved or "qwen3.6" in resolved:
            return 32_768  # 262K context — generous output budget
        # Unknown / placeholder ``local-model`` — pick a safe middle ground.
        return 8_192
