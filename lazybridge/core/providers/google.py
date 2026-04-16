"""Google Gemini provider for LazyBridge.

Uses the ``google-genai`` SDK (not the legacy ``google-generativeai``).

Default model: gemini-3.1-pro-preview. Pass model= to override.

Native tools supported:
  - Google Search grounding  (NativeTool.GOOGLE_SEARCH / NativeTool.WEB_SEARCH)
  - Google Maps grounding    (NativeTool.GOOGLE_MAPS)

Grounding response fields populated in CompletionResponse / StreamChunk:
  - ``grounding_sources``   — list of GroundingSource(url, title)
  - ``web_search_queries``  — queries actually issued by the grounding tool
  - ``search_entry_point``  — rendered HTML attribution widget (required by
                               Google ToS when displaying grounded results)

Optional extra kwargs for chat()/achat():
  google_search_dynamic_threshold   float 0–1   — DynamicRetrievalConfig via
                                                   GoogleSearchRetrieval; Gemini 1.5 ONLY.
                                                   Gemini 2.0+ ignores it with a warning.
  google_search_exclude_domains     list[str]   — domains to exclude from results
  google_maps_lat / google_maps_lng float       — location hint; injected via
                                                   ToolConfig.RetrievalConfig.lat_lng
  google_maps_enable_widget         bool        — return Maps widget context token

Incompatibility:
  Google Search grounding + response_mime_type="application/json" raises 400
  INVALID_ARGUMENT.  When both are requested, the schema is skipped and a
  UserWarning is emitted.

Grounding chunks:
  Google Search chunks expose chunk.web (uri, title).
  Google Maps chunks expose chunk.maps (uri, title).
  Both are normalised into GroundingSource(url, title).
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    GroundingSource,
    NativeTool,
    Role,
    StreamChunk,
    ToolCall,
    UsageStats,
)

try:
    from google import genai as _genai
    from google.genai import types as _gtypes
except Exception:
    _genai = None  # type: ignore
    _gtypes = None  # type: ignore

_logger = logging.getLogger(__name__)

# Price per 1M tokens (input, output). Approximate; verify at ai.google.dev/gemini-api/docs/pricing.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gemini-3.1-pro": (1.25, 10.0),
    "gemini-3.1-flash": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.075, 0.30),
    "gemini-2.0-pro": (1.25, 10.0),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
}


class GoogleProvider(BaseProvider):
    """Google Gemini provider.

    Supports:
    - Thinking/reasoning via ThinkingConfig (budget=-1 for auto)
    - Structured output via response_schema + Pydantic or JSON schema
    - Function calling (automatic or manual)
    - Native Google Search grounding and Google Maps
    - Streaming
    """

    default_model = "gemini-3.1-pro-preview"

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        model_l = model.lower()
        for key, (in_price, out_price) in _PRICE_TABLE.items():
            if key in model_l:
                return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        return None

    def get_default_max_tokens(self, model: str | None = None) -> int:
        resolved = (model or self.model or self.default_model or "").lower()
        if "gemini-3" in resolved or "gemini-2.5" in resolved:
            return 65_536
        return 8_192

    supported_native_tools: frozenset[NativeTool] = frozenset(
        {
            NativeTool.WEB_SEARCH,
            NativeTool.GOOGLE_SEARCH,
            NativeTool.GOOGLE_MAPS,
        }
    )

    def _init_client(self, **kwargs) -> None:
        if _genai is None:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Google API key not found. Set the GOOGLE_API_KEY or GEMINI_API_KEY "
                "environment variable, or pass api_key= to the provider."
            )
        self._client = _genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _messages_to_gemini(self, request: CompletionRequest) -> list[Any]:
        """Convert unified Messages to Gemini Content objects.

        Serialization rules:
        - Role.USER     → Content(role="user")  with text/image/function_call parts
        - Role.ASSISTANT→ Content(role="model") with text/function_call parts
        - Role.TOOL     → Content(role="user")  with function_response parts only
        - Role.SYSTEM   → skipped (handled via system_instruction)

        Gemini requires that function_response Parts live in their own Content
        object, separate from text/image Parts. If a message mixes the two,
        they are split into consecutive Content objects.
        """
        from lazybridge.core.types import (
            ImageContent,
            TextContent,
            ThinkingContent,
            ToolResultContent,
            ToolUseContent,
        )

        contents = []
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                continue  # system_instruction handled separately

            # Gemini supports only two roles; validate suspicious combinations.
            if isinstance(msg.content, list):
                has_tool_result = any(isinstance(b, ToolResultContent) for b in msg.content)
                has_other = any(isinstance(b, (TextContent, ImageContent)) for b in msg.content)
                if msg.role == Role.TOOL and has_other:
                    warnings.warn(
                        "Role.TOOL message contains text/image blocks — only "
                        "ToolResultContent is valid here. Text/image parts will be skipped.",
                        UserWarning,
                        stacklevel=3,
                    )
                if msg.role == Role.USER and has_tool_result:
                    warnings.warn(
                        "Role.USER message contains ToolResultContent — use "
                        "Role.TOOL for tool results. "
                        "These will be emitted as a separate function_response Content.",
                        UserWarning,
                        stacklevel=3,
                    )

            # Gemini only supports "user" and "model".
            gemini_role = "user" if msg.role in (Role.USER, Role.TOOL) else "model"

            if isinstance(msg.content, str):
                contents.append(
                    _gtypes.Content(
                        role=gemini_role,
                        parts=[_gtypes.Part.from_text(text=msg.content)],
                    )
                )
                continue

            # Accumulate parts in two buckets:
            #   normal_parts           → text, image, function_call
            #   function_response_parts→ function_response (ToolResultContent)
            # Gemini cannot mix the two in the same Content object.
            normal_parts: list[Any] = []
            function_response_parts: list[Any] = []

            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    pass  # internal to Gemini — never re-inject

                elif isinstance(block, TextContent):
                    if msg.role != Role.TOOL:  # text skipped inside TOOL messages (warned above)
                        normal_parts.append(_gtypes.Part.from_text(text=block.text))

                elif isinstance(block, ImageContent):
                    if msg.role != Role.TOOL:  # same
                        if block.url:
                            normal_parts.append(
                                _gtypes.Part.from_uri(
                                    file_uri=block.url,
                                    mime_type=block.media_type,
                                )
                            )
                        elif block.base64_data:
                            import base64

                            data = base64.b64decode(block.base64_data)
                            normal_parts.append(
                                _gtypes.Part.from_bytes(
                                    data=data,
                                    mime_type=block.media_type,
                                )
                            )

                elif isinstance(block, ToolUseContent):
                    raw_part = getattr(block, "thought_signature", None)
                    if raw_part is not None and hasattr(raw_part, "function_call"):
                        # Re-emit the original SDK Part verbatim: this preserves any
                        # thought_signature that thinking models embed in function calls.
                        normal_parts.append(raw_part)
                    else:
                        normal_parts.append(
                            _gtypes.Part.from_function_call(
                                name=block.name,
                                args=block.input,
                            )
                        )

                elif isinstance(block, ToolResultContent):
                    content_str = block.content if isinstance(block.content, str) else json.dumps(block.content)
                    tool_fn_name = (
                        getattr(block, "tool_name", None) or getattr(block, "name", None) or block.tool_use_id
                    )
                    function_response_parts.append(
                        _gtypes.Part.from_function_response(
                            name=tool_fn_name,
                            response={"result": content_str, "error": block.is_error},
                        )
                    )

            # Emit text/image/function_call first (if any)
            if normal_parts:
                contents.append(_gtypes.Content(role=gemini_role, parts=normal_parts))

            # Emit function_response Parts in their own Content(role="user")
            # This split is required by the Gemini API.
            if function_response_parts:
                contents.append(_gtypes.Content(role="user", parts=function_response_parts))

        return contents

    def _get_system_instruction(self, request: CompletionRequest) -> str | None:
        if request.system:
            return request.system
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                return msg.to_text()
        return None

    def _build_function_declarations(self, request: CompletionRequest) -> list[Any]:
        """Convert ToolDefinitions to Gemini FunctionDeclarations."""
        decls = []
        for t in request.tools:
            decls.append(
                _gtypes.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
            )
        return decls

    def _build_tools_config(self, request: CompletionRequest) -> list[Any]:
        """Build Gemini tools list including native and function tools.

        Google Search:
          - Default: ``Tool(google_search=GoogleSearch())``, always grounds.
          - ``google_search_exclude_domains`` extra kwarg: list[str] of domains
            to exclude (Vertex AI only).
          - ``google_search_dynamic_threshold`` extra kwarg: float 0–1.
            **Only valid for Gemini 1.5 models.** Uses the legacy
            ``Tool(google_search_retrieval=GoogleSearchRetrieval(...))`` API so
            the model skips Search when its confidence is above the threshold.
            Passing this with Gemini 2.0+ models will raise ``400
            INVALID_ARGUMENT`` from the API — a warning is emitted.

        Google Maps:
          - ``Tool(google_maps=GoogleMaps())``.
          - ``google_maps_enable_widget`` extra kwarg: bool — requests the
            Maps widget context token in the response.
          - lat/lng are **not** passed here; they go into ``ToolConfig`` inside
            ``_build_config`` via ``google_maps_lat`` / ``google_maps_lng``.
        """
        tools = []
        native = self._check_native_tools(request.native_tools)

        google_search_requested = any(nt in (NativeTool.WEB_SEARCH, NativeTool.GOOGLE_SEARCH) for nt in native)
        google_maps_requested = NativeTool.GOOGLE_MAPS in native

        if google_search_requested:
            dynamic_threshold = request.extra.get("google_search_dynamic_threshold")
            exclude_domains: list[str] | None = request.extra.get("google_search_exclude_domains")

            if dynamic_threshold is not None:
                # DynamicRetrievalConfig is ONLY valid for Gemini 1.5 models.
                # The correct Tool key is google_search_retrieval= (not google_search=).
                model = self._resolve_model(request).lower()
                if "1.5" not in model:
                    warnings.warn(
                        "google_search_dynamic_threshold (DynamicRetrievalConfig) is only "
                        "supported on Gemini 1.5 models. On Gemini 2.0+ the API will return "
                        "400 INVALID_ARGUMENT. Falling back to plain GoogleSearch.",
                        UserWarning,
                        stacklevel=4,
                    )
                    kwargs: dict[str, Any] = {}
                    if exclude_domains:
                        kwargs["exclude_domains"] = exclude_domains
                    tools.append(_gtypes.Tool(google_search=_gtypes.GoogleSearch(**kwargs)))
                else:
                    try:
                        tools.append(
                            _gtypes.Tool(
                                google_search_retrieval=_gtypes.GoogleSearchRetrieval(
                                    dynamic_retrieval_config=_gtypes.DynamicRetrievalConfig(
                                        dynamic_threshold=float(dynamic_threshold),
                                    )
                                )
                            )
                        )
                    except AttributeError as exc:
                        _logger.debug(
                            "Google SDK DynamicRetrievalConfig not supported, falling back to plain GoogleSearch: %s",
                            exc,
                        )
                        # Older SDK version — fall back to plain GoogleSearch
                        tools.append(_gtypes.Tool(google_search=_gtypes.GoogleSearch()))
            else:
                gkwargs: dict[str, Any] = {}
                if exclude_domains:
                    gkwargs["exclude_domains"] = exclude_domains
                tools.append(_gtypes.Tool(google_search=_gtypes.GoogleSearch(**gkwargs)))

        if google_maps_requested:
            try:
                maps_kwargs: dict[str, Any] = {}
                enable_widget = request.extra.get("google_maps_enable_widget")
                if enable_widget is not None:
                    maps_kwargs["enable_widget"] = bool(enable_widget)
                tools.append(_gtypes.Tool(google_maps=_gtypes.GoogleMaps(**maps_kwargs)))
            except AttributeError:
                warnings.warn(
                    "NativeTool.GOOGLE_MAPS requires google-genai >= 1.0 with GoogleMaps "
                    "support. The tool will be skipped. Upgrade: pip install -U google-genai",
                    UserWarning,
                    stacklevel=3,
                )

        func_decls = self._build_function_declarations(request)
        if func_decls:
            tools.append(_gtypes.Tool(function_declarations=func_decls))

        return tools

    def _build_thinking_config(self, request: CompletionRequest) -> Any | None:
        if not request.thinking or not request.thinking.enabled:
            return None
        model = self._resolve_model(request).lower()
        if model.startswith("gemini-3"):
            level = {
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "high",
                "max": "high",
            }.get(request.thinking.effort, "high")
            return _gtypes.ThinkingConfig(thinking_level=level)
        # budget=-1 → model decides automatically; 0 → no thinking
        budget = request.thinking.budget_tokens if request.thinking.budget_tokens is not None else -1
        return _gtypes.ThinkingConfig(thinking_budget=budget)

    def _build_config(self, request: CompletionRequest) -> Any:
        kwargs: dict[str, Any] = {}

        system = self._get_system_instruction(request)
        if system:
            kwargs["system_instruction"] = system

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.max_tokens:
            kwargs["max_output_tokens"] = request.max_tokens

        native = self._check_native_tools(request.native_tools)
        google_search_active = any(nt in (NativeTool.WEB_SEARCH, NativeTool.GOOGLE_SEARCH) for nt in native)
        google_maps_active = NativeTool.GOOGLE_MAPS in native

        tools = self._build_tools_config(request)
        if tools:
            kwargs["tools"] = tools
            kwargs["automatic_function_calling"] = _gtypes.AutomaticFunctionCallingConfig(
                disable=True  # We handle tool calls ourselves for unified output
            )

        # Google Maps: lat/lng goes in ToolConfig.retrieval_config, not inside GoogleMaps()
        if google_maps_active:
            lat = request.extra.get("google_maps_lat")
            lng = request.extra.get("google_maps_lng")
            if lat is not None and lng is not None:
                try:
                    kwargs["tool_config"] = _gtypes.ToolConfig(
                        retrieval_config=_gtypes.RetrievalConfig(
                            lat_lng=_gtypes.LatLng(
                                latitude=float(lat),
                                longitude=float(lng),
                            )
                        )
                    )
                except AttributeError as exc:
                    _logger.debug("Google SDK RetrievalConfig not supported, skipping lat/lng: %s", exc)

        thinking = self._build_thinking_config(request)
        if thinking:
            kwargs["thinking_config"] = thinking

        # Structured output
        if request.structured_output:
            # The Gemini API does not support combining google_search grounding with
            # response_mime_type="application/json". Doing so raises 400 INVALID_ARGUMENT.
            if google_search_active:
                warnings.warn(
                    "Google Search grounding is incompatible with structured output "
                    "(response_mime_type='application/json'). The structured output "
                    "schema will be ignored for this request.",
                    UserWarning,
                    stacklevel=4,
                )
            else:
                schema = request.structured_output.schema
                kwargs["response_mime_type"] = "application/json"
                if isinstance(schema, dict):
                    kwargs["response_schema"] = schema
                else:
                    # Explicit model_json_schema() — don't rely on implicit SDK coercion
                    try:
                        kwargs["response_schema"] = schema.model_json_schema()  # type: ignore[attr-defined]
                    except AttributeError as exc:
                        _logger.debug(
                            "model_json_schema() not available on %r, passing schema object directly: %s", schema, exc
                        )
                        kwargs["response_schema"] = schema

        return _gtypes.GenerateContentConfig(**kwargs)

    def _extract_grounding_metadata(self, candidate: Any) -> tuple[list[GroundingSource], list[str], str | None]:
        """Extract grounding sources, search queries, and search entry point from a candidate.

        Returns:
            (grounding_sources, web_search_queries, search_entry_point_html)

        ``search_entry_point`` is the rendered HTML attribution widget that Google's
        Terms of Service require you to display when showing grounded results.
        ``web_search_queries`` are the actual queries the model issued to Google Search.
        """
        grounding_sources: list[GroundingSource] = []
        web_search_queries: list[str] = []
        search_entry_point: str | None = None

        gm = getattr(candidate, "grounding_metadata", None)
        if gm is None:
            return grounding_sources, web_search_queries, search_entry_point

        # grounding_chunks → GroundingSource (url + title)
        # Google Search chunks expose .web; Google Maps chunks expose .maps.
        chunks = getattr(gm, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            maps = getattr(chunk, "maps", None)
            source = web or maps
            if source:
                grounding_sources.append(
                    GroundingSource(
                        url=getattr(source, "uri", "") or "",
                        title=getattr(source, "title", None),
                    )
                )

        # web_search_queries — the actual queries issued by the grounding tool
        queries = getattr(gm, "web_search_queries", None)
        if queries:
            web_search_queries = list(queries)

        # search_entry_point.rendered_content — Google's HTML attribution widget.
        # Required by Google's Terms of Service when displaying grounded results.
        sep = getattr(gm, "search_entry_point", None)
        if sep:
            search_entry_point = getattr(sep, "rendered_content", None)

        return grounding_sources, web_search_queries, search_entry_point

    def _parse_response(self, response: Any, model: str) -> CompletionResponse:
        content = ""
        thinking = None
        tool_calls: list[ToolCall] = []

        if not response.candidates:
            _logger.warning("Google response has no candidates (safety block or API error).")
            usage = UsageStats()
            if response.usage_metadata:
                um = response.usage_metadata
                usage.input_tokens = getattr(um, "prompt_token_count", 0) or 0
                usage.output_tokens = getattr(um, "candidates_token_count", 0) or 0
            usage.cost_usd = self._compute_cost(model, usage.input_tokens, usage.output_tokens)
            return CompletionResponse(
                content="",
                tool_calls=[],
                stop_reason="error",
                model=model,
                usage=usage,
                raw=response,
            )

        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                content += part.text
            if hasattr(part, "thought") and part.thought:
                thinking = (thinking or "") + part.thought
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                # Store the raw Part so _messages_to_gemini can re-emit it unchanged.
                # Thinking models (e.g. gemini-3.1-*) embed an opaque thought_signature
                # inside the function_call Part; reconstructing the Part from scratch
                # loses that token and causes 400 INVALID_ARGUMENT on the next turn.
                tool_calls.append(
                    ToolCall(
                        id=getattr(fc, "id", fc.name),
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                        thought_signature=part,  # raw SDK Part — preserved verbatim
                    )
                )

        usage = UsageStats()
        if response.usage_metadata:
            um = response.usage_metadata
            usage.input_tokens = getattr(um, "prompt_token_count", 0) or 0
            usage.output_tokens = getattr(um, "candidates_token_count", 0) or 0
            usage.thinking_tokens = getattr(um, "thoughts_token_count", 0) or 0
        usage.cost_usd = self._compute_cost(model, usage.input_tokens, usage.output_tokens)

        grounding_sources, web_search_queries, search_entry_point = self._extract_grounding_metadata(candidate)

        return CompletionResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            stop_reason="end_turn",
            model=model,
            usage=usage,
            raw=response,
            grounding_sources=grounding_sources,
            web_search_queries=web_search_queries,
            search_entry_point=search_entry_point,
        )

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = self._resolve_model(request)
        contents = self._messages_to_gemini(request)
        config = self._build_config(request)

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        resp = self._parse_response(response, model)

        if request.structured_output:
            from lazybridge.core.structured import apply_structured_validation

            apply_structured_validation(resp, resp.content, request.structured_output.schema)

        return resp

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        model = self._resolve_model(request)
        contents = self._messages_to_gemini(request)
        config = self._build_config(request)

        last_chunk = None
        text_accum = ""
        tool_call_accum: dict[str, dict] = {}  # call_id → dict; deduplicates repeated parts
        for chunk in self._client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            last_chunk = chunk
            if not chunk.candidates:
                continue
            for part in chunk.candidates[0].content.parts:
                if hasattr(part, "thought") and part.thought:
                    yield StreamChunk(thinking_delta=part.thought)
                elif hasattr(part, "text") and part.text:
                    text_accum += part.text
                    yield StreamChunk(delta=part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    call_id = getattr(fc, "id", None) or fc.name
                    tool_call_accum[call_id] = {
                        "id": call_id,
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                        "thought_signature": part,  # raw Part preserved verbatim
                    }

        # Final chunk: usage + accumulated tool calls + grounding metadata
        usage = None
        if last_chunk is not None and last_chunk.usage_metadata:
            um = last_chunk.usage_metadata
            usage = UsageStats(
                input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                thinking_tokens=getattr(um, "thoughts_token_count", 0) or 0,
            )
            usage.cost_usd = self._compute_cost(model, usage.input_tokens, usage.output_tokens)
        tool_calls = [
            ToolCall(id=d["id"], name=d["name"], arguments=d["args"], thought_signature=d.get("thought_signature"))
            for d in tool_call_accum.values()
        ]
        grounding_sources: list[GroundingSource] = []
        web_search_queries: list[str] = []
        search_entry_point: str | None = None
        if last_chunk is not None and getattr(last_chunk, "candidates", None):
            grounding_sources, web_search_queries, search_entry_point = self._extract_grounding_metadata(
                last_chunk.candidates[0]
            )
        final_chunk = StreamChunk(
            stop_reason="end_turn",
            tool_calls=tool_calls,
            usage=usage,
            is_final=True,
            grounding_sources=grounding_sources,
            web_search_queries=web_search_queries,
            search_entry_point=search_entry_point,
        )
        if request.structured_output:
            from lazybridge.core.structured import apply_structured_validation

            apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
        yield final_chunk

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        model = self._resolve_model(request)
        contents = self._messages_to_gemini(request)
        config = self._build_config(request)

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        resp = self._parse_response(response, model)

        if request.structured_output:
            from lazybridge.core.structured import apply_structured_validation

            apply_structured_validation(resp, resp.content, request.structured_output.schema)

        return resp

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        model = self._resolve_model(request)
        contents = self._messages_to_gemini(request)
        config = self._build_config(request)

        last_chunk = None
        text_accum = ""
        tool_call_accum: dict[str, dict] = {}  # call_id → dict; deduplicates repeated parts
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            last_chunk = chunk
            if not chunk.candidates:
                continue
            for part in chunk.candidates[0].content.parts:
                if hasattr(part, "thought") and part.thought:
                    yield StreamChunk(thinking_delta=part.thought)
                elif hasattr(part, "text") and part.text:
                    text_accum += part.text
                    yield StreamChunk(delta=part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    call_id = getattr(fc, "id", None) or fc.name
                    tool_call_accum[call_id] = {
                        "id": call_id,
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                        "thought_signature": part,  # raw Part preserved verbatim
                    }

        usage = None
        if last_chunk is not None and last_chunk.usage_metadata:
            um = last_chunk.usage_metadata
            usage = UsageStats(
                input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                thinking_tokens=getattr(um, "thoughts_token_count", 0) or 0,
            )
            usage.cost_usd = self._compute_cost(model, usage.input_tokens, usage.output_tokens)
        tool_calls = [
            ToolCall(id=d["id"], name=d["name"], arguments=d["args"], thought_signature=d.get("thought_signature"))
            for d in tool_call_accum.values()
        ]
        grounding_sources: list[GroundingSource] = []
        web_search_queries: list[str] = []
        search_entry_point: str | None = None
        if last_chunk is not None and getattr(last_chunk, "candidates", None):
            grounding_sources, web_search_queries, search_entry_point = self._extract_grounding_metadata(
                last_chunk.candidates[0]
            )
        final_chunk = StreamChunk(
            stop_reason="end_turn",
            tool_calls=tool_calls,
            usage=usage,
            is_final=True,
            grounding_sources=grounding_sources,
            web_search_queries=web_search_queries,
            search_entry_point=search_entry_point,
        )
        if request.structured_output:
            from lazybridge.core.structured import apply_structured_validation

            apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
        yield final_chunk
