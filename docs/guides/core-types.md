# core.types

`core.types` is the vocabulary engines and providers share. When you
write a new `BaseProvider` you translate `CompletionRequest` →
provider SDK and provider SDK → `CompletionResponse`. When you write a
new `Engine` you build `CompletionRequest`s and read
`CompletionResponse`s.

Never import these in application code. An app uses `Agent`,
`Envelope`, `Tool`, `Memory`, `Session`. The only exception is
`NativeTool` — users pass it to `LLMEngine(native_tools=[...])` to
enable provider-native server-side tools (web search, code exec),
which is why it lives under `core.types` but is re-exported from
`lazybridge`.

## Example

```python
# Engine author — building a CompletionRequest.
from lazybridge.core.types import (
    CompletionRequest, Message, Role,
    StructuredOutputConfig, ThinkingConfig,
)

req = CompletionRequest(
    messages=[Message(role=Role.USER, content="summarise the Iliad")],
    system="Be concise.",
    max_tokens=1024,
    thinking=ThinkingConfig(enabled=True, effort="high"),
    structured_output=StructuredOutputConfig(schema=MyModel),
)
resp = await provider.acomplete(req)

# Provider author — yielding StreamChunks.
from lazybridge.core.types import StreamChunk, UsageStats

async def astream(self, request):
    async for sdk_chunk in self._client.chat.stream(...):
        yield StreamChunk(
            delta=sdk_chunk.delta.text,
            usage=UsageStats(input_tokens=..., output_tokens=...),
            is_final=sdk_chunk.is_final,
        )
```

## Pitfalls

- Forgetting that ``CompletionResponse.content`` is always a string:
  structured output lives in ``.parsed`` (the validated model instance)
  alongside ``.content`` (the raw JSON text).
- ``tool_choice`` on ``CompletionRequest`` is the provider-level knob
  ("auto" / "any" / specific tool); different from user-facing
  ``LLMEngine(tool_choice=...)`` which was "auto" / "any" post-v1.
- ``NativeTool`` entries are enums, not capabilities — the provider
  decides whether to honour each one; unsupported combinations raise
  at ``complete`` time.

!!! note "API reference"

    # lazybridge.core.types — internal types exposed for Engine / Provider authors.
    
    Role (StrEnum):   USER, ASSISTANT, SYSTEM, TOOL
    Message:          role + content (str or list[ContentBlock])
      Message.user(text)  Message.assistant(text)  Message.system(text)
    
    ContentBlock (union):
      TextContent(text)
      ImageContent(source, mime_type)
      ToolUseContent(id, name, input)
      ToolResultContent(tool_use_id, content, is_error)
      ThinkingContent(text)
    
    ToolDefinition(name, description, parameters: dict, strict: bool = False)
    ToolCall(id, name, arguments, thought_signature=None)
    
    CompletionRequest(
        messages: list[Message],
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        tools: list[ToolDefinition] = [],
        tool_choice: str | None = None,
        native_tools: list[NativeTool] = [],
        structured_output: StructuredOutputConfig | None = None,
        thinking: ThinkingConfig | None = None,
        skills: SkillsConfig | None = None,
        stream: bool = False,
        extra: dict = {},
    )
    
    CompletionResponse(
        content: str,
        thinking: str | None = None,
        tool_calls: list[ToolCall] = [],
        stop_reason: str = "end_turn",
        model: str | None = None,
        usage: UsageStats = UsageStats(),
        parsed: Any = None,                  # set when structured_output produced a model
        validated: bool | None = None,
        grounding_sources: list = [],
        # …plus provider-specific fields
    )
    
    UsageStats(input_tokens, output_tokens, thinking_tokens, cost_usd)
    
    StructuredOutputConfig(schema, strict=True)
    ThinkingConfig(enabled=True, effort="high", budget_tokens=None, display=None)
    SkillsConfig(skills: list[str])
    NativeTool (StrEnum):  WEB_SEARCH, CODE_EXECUTION, FILE_SEARCH, COMPUTER_USE,
                           GOOGLE_SEARCH, GOOGLE_MAPS
    StreamChunk(delta, thinking_delta, tool_calls, usage, is_final, ...)

    # --- Agent-facing config objects (re-exported from lazybridge) ---
    # Pass these to Agent() to share policy across a fleet; flat kwargs on
    # Agent still win per-field.  See guides/agent.md "Shared config objects".

    ResilienceConfig(
        timeout: float | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache: bool | CacheConfig = False,
        max_output_retries: int = 2,
        output_validator: Callable[[Any], Any] | None = None,
        fallback: Agent | None = None,
    )

    ObservabilityConfig(
        verbose: bool = False,
        session: Session | None = None,
        name: str | None = None,
        description: str | None = None,
    )

    AgentRuntimeConfig(
        resilience: ResilienceConfig | None = None,
        observability: ObservabilityConfig | None = None,
    )

!!! warning "Rules & invariants"

    - These types are the bridge between Engines and Providers. End users
      should NOT construct them directly — use ``Agent`` / ``Tool`` /
      ``Envelope`` instead.
    - ``CompletionRequest`` / ``CompletionResponse`` are the provider-
      neutral lingua franca. Every ``BaseProvider`` translates them to/from
      the provider's native SDK types.
    - ``Envelope`` (``lazybridge.envelope``) is the USER-facing data type;
      ``CompletionResponse`` is the PROVIDER-facing one. Engines glue them.
    - ``StreamChunk`` carries incremental updates; providers yield them
      during ``stream`` / ``astream``.

## See also

[base_provider](base-provider.md), [engine_protocol](engine-protocol.md),
[envelope](envelope.md)
