## signature
class BaseProvider(ABC):
    default_model: str = ...
    _TIER_ALIASES: dict[str, str] = {}     # "top" / "expensive" / "medium" / "cheap" / "super_cheap"

    @abstractmethod
    def _init_client(self, **kwargs) -> None: ...

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
    @abstractmethod
    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]: ...
    async def acomplete(self, request: CompletionRequest) -> CompletionResponse: ...
    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]: ...

    # Shared helper:
    def resolve_model_alias(self, model: str) -> str | None: ...

## rules
- Subclass ``BaseProvider`` to integrate a new LLM backend with
  LazyBridge.
- ``default_model`` is used when the user passes only the provider name
  (``Agent("anthropic")``); ``_TIER_ALIASES`` maps the five tier
  strings ("top" / "expensive" / "medium" / "cheap" / "super_cheap")
  to concrete model names.
- Implement the four request/response methods. ``acomplete`` / ``astream``
  default to off-the-main-thread wrappers around ``complete`` /
  ``stream`` — override for native async for lower latency.
- Register the provider with ``LLMEngine.register_provider_alias`` or
  ``register_provider_rule`` so ``Agent("mymodel-foo")`` routes to it.

## narrative
`BaseProvider` is the extension point for new LLM backends. Subclass
it, implement four methods, and register the routing rule — you're
wired.

The contract is deliberately narrow: translate between LazyBridge's
`CompletionRequest` / `CompletionResponse` types and the provider's
native SDK. Everything above (tool loops, memory, structured output,
session events) lives in `LLMEngine`, not in the provider adapter.

Tier aliases (`_TIER_ALIASES`) decouple model names from the rest of
the stack. A user who writes `Agent.from_provider("myllm", tier="top")`
will get whatever you currently rank "top" in that provider — preview
models, date-pinned snapshots — without their code changing when you
refresh the lineup.

Implement `resolve_model_alias` via the `BaseProvider` shared helper
if you only need standard tier lookup; override it for custom routing
logic.

## example
```python
from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import CompletionRequest, CompletionResponse
from lazybridge import LLMEngine, Agent

class MistralProvider(BaseProvider):
    default_model = "mistral-large-latest"
    _TIER_ALIASES = {
        "top":        "mistral-large-latest",
        "medium":     "mistral-small-latest",
        "cheap":      "mistral-small-latest",
        "super_cheap": "codestral-mamba-latest",
    }

    def _init_client(self, **kwargs):
        from mistralai import Mistral
        self._client = Mistral(api_key=self.api_key, **kwargs)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        # Translate our CompletionRequest → Mistral's API format → back.
        raw = self._client.chat.complete(
            model=request.model,
            messages=[...],
            tools=[...],
        )
        return CompletionResponse(content=raw.choices[0].message.content, ...)

    def stream(self, request): ...
    async def acomplete(self, request): ...
    async def astream(self, request): ...

# Register + use.
LLMEngine.register_provider_alias("mistral", "mistral")
LLMEngine.register_provider_rule("mistral-", "mistral", kind="startswith")

Agent.from_provider("mistral", tier="top")("hello").text()
```

## pitfalls
- Skipping ``acomplete`` / ``astream`` falls back to thread-pool
  wrappers. Acceptable for dev, suboptimal for latency-sensitive
  servers.
- Tool schema translation is the fiddly part — provider-native strict
  mode, parameter validation, and native tools (web search, code exec)
  each have gotchas. Study ``core/providers/anthropic.py`` and
  ``openai.py`` before shipping.
- Don't hard-code API keys; honour ``os.environ`` the same way the
  built-ins do for consistency.

## see-also
[register_provider](register-provider.md),
[engine_protocol](engine-protocol.md),
[core_types](core-types.md)
