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
The contract is deliberately narrow: translate between
`CompletionRequest` / `CompletionResponse` and the provider's native
SDK. Tool loops, memory, structured output, and session events live in
`LLMEngine`, not in the provider adapter.

`_TIER_ALIASES` decouples model names from application code — users
who write `Agent.from_provider("myllm", tier="top")` get whatever you
rank "top" without changing their code when you update the lineup.

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

