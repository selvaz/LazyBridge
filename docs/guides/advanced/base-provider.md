# BaseProvider

The stable extension point for integrating any LLM backend with
LazyBridge. Subclass `BaseProvider`, implement four request /
response methods, declare your tier aliases, and register the class
with the provider registry. Once registered, `Agent("your-model")`
routes to it like any built-in.

## Signature

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    NativeTool,
    StreamChunk,
)


class BaseProvider(ABC):
    # Class-level configuration
    default_model: str = ""
    supported_native_tools: frozenset[NativeTool] = frozenset()
    strict_native_tools: bool = False
    _TIER_ALIASES: dict[str, str] = {}                 # "top" / "expensive" / "medium" / "cheap" / "super_cheap"
    _FALLBACKS: dict[str, list[str]] = {}              # alternative models per primary
    _VISION_CAPABLE_MODEL_PATTERNS: frozenset[str] = frozenset()
    _AUDIO_CAPABLE_MODEL_PATTERNS: frozenset[str] = frozenset()

    # Construction
    def __init__(self, api_key=None, model=None, *, strict_native_tools=None, **kwargs): ...

    # MUST implement
    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
    @abstractmethod
    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]: ...
    @abstractmethod
    async def acomplete(self, request: CompletionRequest) -> CompletionResponse: ...
    @abstractmethod
    def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]: ...

    # SHOULD override
    def _init_client(self, **kwargs) -> None: ...
    def _compute_cost(self, model, input_tokens, output_tokens) -> float | None: ...
    def get_default_max_tokens(self, model=None) -> int: ...

    # MAY override
    def is_retryable(self, exc) -> bool | None: ...
    @classmethod
    def supports_vision(cls, model=None) -> bool: ...
    @classmethod
    def supports_audio(cls, model=None) -> bool: ...

    # Stable helpers (callable from subclasses)
    def _resolve_model(self, request) -> str: ...
    def _check_native_tools(self, tools) -> list[NativeTool]: ...


# Provider-side error types.
class UnsupportedNativeToolError(ValueError): ...     # subclass of ValueError
class UnsupportedFeatureError(ValueError): ...        # multimodal-capability mismatch
```

For the registry surface
(`LLMEngine.register_provider_alias` / `register_provider_rule`),
see the [Provider registry](#provider-registry) section below.

## Synopsis

A `BaseProvider` is a translator between LazyBridge's neutral
`CompletionRequest` / `CompletionResponse` types and a specific LLM
SDK's API. Tool loops, memory, structured output, retry policy, and
session events all live in `LLMEngine`, not in the provider —
keeping the provider surface narrow.

The contract is **stable**. Method signatures and the seven
helpers listed above don't break across minor versions; bumps to
either rename or remove anything follow a deprecation cycle and a
minor-version increment.

`_TIER_ALIASES` decouples model names from application code. A
user who writes `Agent.from_provider("myllm", tier="top")` gets
whatever you currently rank "top"; you update the lineup by
editing the alias table, not by asking every caller to change their
code.

`supported_native_tools` declares which provider-hosted tools
(`NativeTool.WEB_SEARCH`, `NativeTool.CODE_EXECUTION`, …) the
backend implements. Unsupported tools requested by the user are
filtered with a `UserWarning` by default; setting
`strict_native_tools=True` raises `UnsupportedNativeToolError`
instead — opt into strict mode in production so misconfiguration
fails loud.

## When to use it

- **A provider exists that LazyBridge doesn't ship support for.**
  Mistral, Cohere, Bedrock, Ollama, your team's internal model —
  subclass `BaseProvider` once and the rest of the framework picks
  up the new backend automatically.
- **You want native-tool routing for a custom provider.** Declare
  `supported_native_tools` so users can pass
  `Agent(native_tools=[NativeTool.WEB_SEARCH])` and have the
  framework reject (or warn) for unsupported combinations at
  request time.
- **You want cost tracking.** Override `_compute_cost(model,
  input_tokens, output_tokens)` to populate
  `Envelope.metadata.cost_usd` from your pricing table.

## When NOT to use it

- **The model you want is already routable through an existing
  provider.** Most OpenAI-compatible APIs (DeepSeek, LMStudio, your
  fine-tune endpoint) work via the existing OpenAI provider — set
  the `OPENAI_BASE_URL` env var or use `LiteLLMProvider` first.
- **You want to add an engine, not a provider.** Engines are the
  layer above; see [Engine protocol](engine-protocol.md).
- **You're tweaking request shape for an existing provider.**
  `LLMEngine` accepts `system=`, `temperature=`, `max_turns=`,
  `thinking=`, etc. without subclassing — most prompt-shape
  customisation happens there, not in the provider.

## Example

```python
from lazybridge import Agent, LLMEngine
from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    UsageStats,
)


class MistralProvider(BaseProvider):
    default_model = "mistral-large-latest"
    _TIER_ALIASES = {
        "top":         "mistral-large-latest",
        "expensive":   "mistral-large-latest",
        "medium":      "mistral-medium-latest",
        "cheap":       "mistral-small-latest",
        "super_cheap": "codestral-mamba-latest",
    }
    _PRICES = {
        # ($/1M input, $/1M output)
        "mistral-large-latest": (3.00, 9.00),
        "mistral-medium-latest": (2.70, 8.10),
        "mistral-small-latest":  (0.20, 0.60),
    }

    def _init_client(self, **kwargs) -> None:
        from mistralai import Mistral
        self._client = Mistral(api_key=self.api_key, **kwargs)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = self._resolve_model(request)
        raw = self._client.chat.complete(
            model=model,
            messages=...,                          # convert request.messages
            tools=...,                             # convert request.tools
        )
        return CompletionResponse(
            content=raw.choices[0].message.content,
            usage=UsageStats(
                input_tokens=raw.usage.prompt_tokens,
                output_tokens=raw.usage.completion_tokens,
                cost_usd=self._compute_cost(
                    model,
                    raw.usage.prompt_tokens,
                    raw.usage.completion_tokens,
                ) or 0.0,
            ),
            model=model,
        )

    def stream(self, request: CompletionRequest):
        for raw_chunk in self._client.chat.stream(...):
            yield StreamChunk(delta=raw_chunk.text)
        yield StreamChunk(stop_reason="end_turn", is_final=True)

    async def acomplete(self, request):
        # Use the SDK's async client when available.
        ...

    async def astream(self, request):
        async for raw_chunk in self._client.chat.astream(...):
            yield StreamChunk(delta=raw_chunk.text)
        yield StreamChunk(stop_reason="end_turn", is_final=True)

    def _compute_cost(self, model, input_tokens, output_tokens):
        for key, (in_rate, out_rate) in self._PRICES.items():
            if key in model:
                return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
        return None


# Register so Agent("mistral-…") routes here.
LLMEngine.register_provider_alias("mistral", "mistral")
LLMEngine.register_provider_rule("mistral-", "mistral", kind="startswith")


# Use exactly like a built-in.
agent = Agent(
    engine=LLMEngine("mistral-large-latest"),
)
print(agent("hello").text())
```

## Provider registry

`LLMEngine` exposes two `@classmethod`s that mutate class-level
tables to route a model string to a registered provider:

```python
LLMEngine.register_provider_alias(alias: str, provider: str) -> None
LLMEngine.register_provider_rule(
    pattern: str,
    provider: str,
    *,
    kind: Literal["contains", "startswith"] = "contains",
) -> None
```

- **`register_provider_alias`** — exact-match routing (case-
  insensitive). `Agent("mistral")` resolves to the registered
  provider.
- **`register_provider_rule`** — substring or prefix match
  (case-insensitive). New rules **prepend** the rule list, so user
  rules take priority over built-ins. A `register_provider_rule(
  "claude-opus-5", "my-proxy")` call wins over the built-in
  `claude` catch-all.

The internal tables are class-level, so they're shared across all
`LLMEngine` instances in the process:

| Table | Purpose |
|---|---|
| `LLMEngine._PROVIDER_ALIASES` | Exact-match model string → provider |
| `LLMEngine._PROVIDER_RULES` | List of `(kind, pattern, provider)` tuples |
| `LLMEngine._PROVIDER_DEFAULT` | Fallback when no rule matches |

```python
# Example registrations.
LLMEngine.register_provider_alias("mistral", "mistral")
LLMEngine.register_provider_rule("mistral-", "mistral", kind="startswith")
LLMEngine.register_provider_rule("bedrock/", "bedrock", kind="startswith")

# Override a built-in: send all "claude-*" calls through a local proxy.
LLMEngine.register_provider_rule("claude", "my-proxy")
```

For tests that register rules, snapshot and restore the tables in
a fixture so state doesn't leak across tests:

```python
import pytest
from lazybridge import LLMEngine


@pytest.fixture
def restore_provider_rules():
    aliases = dict(LLMEngine._PROVIDER_ALIASES)
    rules = list(LLMEngine._PROVIDER_RULES)
    yield
    LLMEngine._PROVIDER_ALIASES = aliases
    LLMEngine._PROVIDER_RULES = rules
```

## Pitfalls

- **Skipping `acomplete` / `astream` is acceptable but slower.**
  Default implementations on `BaseProvider` aren't auto-generated;
  you must implement all four. If your SDK has only sync APIs,
  wrap them in `asyncio.get_event_loop().run_in_executor(...)`
  inside `acomplete` / `astream` — but the documented preferred
  path is native async, which gives lower latency under load.
- **Tool schema translation is the fiddly part.** Provider-native
  strict mode, parameter validation, native-tools enabling — each
  has provider-specific quirks. Read
  `lazybridge/core/providers/anthropic.py` and `openai.py` before
  shipping a custom provider; they encode hard-won lessons.
- **Don't hard-code API keys.** The base class already accepts
  `api_key=None` and expects `_init_client` to fall back to env
  vars (the pattern every built-in follows). Mirror that
  convention so users can swap providers without changing how
  they manage secrets.
- **`request` is read-only.** Two callers may share the same
  request object across retries; mutating it inside `complete`
  silently corrupts the next attempt. Build the SDK-shaped
  payload in a local variable.
- **Don't block the event loop in `acomplete`/`astream`.** Use
  `await` for SDK calls or `loop.run_in_executor` for blocking
  ones. A blocking call in async path stalls every concurrent
  agent run sharing the loop.
- **`register_provider_rule` PREPENDS.** Your rule wins over
  earlier registrations, including the built-ins. If you need to
  append (rare — typically when you want a catch-all that runs
  after everything else), mutate `LLMEngine._PROVIDER_RULES`
  directly with `append(...)` — there's no public method for it.
- **Aliases without a registered subclass succeed silently** at
  registration time; they fail at `Executor` resolution when
  `Agent("mistral")` is actually constructed. Always register the
  alias *after* the subclass is importable in the resolution
  path.
- **`UnsupportedNativeToolError` subclasses `ValueError`.**
  Existing call sites catching `ValueError` still match it; add a
  more precise `except UnsupportedNativeToolError` only when you
  want to fail-over to a different provider rather than just
  surface the error.

## See also

- [Providers](providers.md) — the built-in catalogue (Anthropic,
  OpenAI, Google, DeepSeek) with their tier-alias tables and
  per-provider quirks.
- [Engine protocol](engine-protocol.md) — the layer above
  `BaseProvider` (custom decision-making mechanisms).
- [Native tools](../basic/native-tools.md) — what you declare in
  `supported_native_tools` to enable provider-hosted tools.
- [LLMEngine](../full/plan.md) — uses the provider you register;
  see its constructor for the full set of knobs that don't require
  a custom provider.
