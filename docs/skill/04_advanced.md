# LazyBridge — Advanced tier
**Use this when** you are extending the framework itself: adding a new LLM provider, writing a custom execution engine, or serialising Plans across processes.

**Skip this tier** if you're building apps — Basic/Mid/Full cover everything user-facing.

## Engine protocol

**signature**

# Protocol contract every engine implements.

@runtime_checkable
class Engine(Protocol):
    async def run(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> Envelope: ...

    async def stream(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> AsyncIterator[str]: ...

# Built-ins implementing this:
#   LLMEngine, HumanEngine, SupervisorEngine, Plan

**rules**

- ``run`` is the primary entry point: receives an Envelope, returns an
  Envelope. It must not raise; wrap exceptions in
  ``Envelope.error_envelope(exc)``.
- ``stream`` is optional for non-streaming engines; yield the final
  text as a single chunk if no incremental output is available.
- Engines receive an already-wrapped ``list[Tool]`` (Agent calls
  ``build_tool_map`` / ``wrap_tool`` before invoking the engine). You do
  NOT need to handle raw functions / Agents in the engine body.
- Agents set ``engine._agent_name`` before invocation. Use it when
  emitting events for observability.

**example**

```python
from lazybridge import Agent, Envelope
from lazybridge.session import EventType
from lazybridge.engines.base import Engine
from typing import AsyncIterator

class EchoEngine:
    """Trivial engine that returns the task prefixed with a tag."""

    async def run(self, env, *, tools, output_type, memory, session):
        if session:
            session.emit(EventType.AGENT_START,
                         {"agent_name": getattr(self, "_agent_name", "echo"),
                          "task": env.task})
        return Envelope(task=env.task, payload=f"echo:{env.task}")

    async def stream(self, env, *, tools, output_type, memory, session) -> AsyncIterator[str]:
        out = await self.run(env, tools=tools, output_type=output_type,
                             memory=memory, session=session)
        yield out.text()

# Runtime-check: EchoEngine satisfies the Engine Protocol.
assert isinstance(EchoEngine(), Engine)

# Plug into Agent — same surface as any built-in engine.
print(Agent(engine=EchoEngine())("hello").text())
```

**pitfalls**

- Skipping ``stream`` entirely breaks ``agent.stream(...)``. Implement
  it to at least yield the final text once, per the pattern above.
- Not emitting session events makes your engine invisible in tracing.
  At minimum emit AGENT_START + AGENT_FINISH.
- The engine receives ``tools``, not ``agent._tool_map``. Treat it as
  a flat list; don't assume the Agent's internal structure.

## BaseProvider

**signature**

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

**rules**

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

**example**

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

**pitfalls**

- Skipping ``acomplete`` / ``astream`` falls back to thread-pool
  wrappers. Acceptable for dev, suboptimal for latency-sensitive
  servers.
- Tool schema translation is the fiddly part — provider-native strict
  mode, parameter validation, and native tools (web search, code exec)
  each have gotchas. Study ``core/providers/anthropic.py`` and
  ``openai.py`` before shipping.
- Don't hard-code API keys; honour ``os.environ`` the same way the
  built-ins do for consistency.

## Plan serialization

**signature**

Plan.to_dict() -> dict
Plan.from_dict(data: dict, *, registry: dict[str, Any] | None = None) -> Plan

# Persisted shape (v=1):
# {
#   "version": 1,
#   "max_iterations": int,
#   "steps": [
#     {
#       "name": str,
#       "target": {"kind": "tool"|"agent"|"callable", "name": str},
#       "task":     {"kind": "from_prev"|"from_start"|"from_step"|"from_parallel"|"literal", ...},
#       "context":  {... single ref}  OR  [{...}, {...}]  OR  null,    # single OR list
#       "parallel": bool,
#       "writes": str | null,
#     },
#     ...
#   ]
# }

**rules**

- Only topology is serialised: step names, sentinels, ``writes``,
  ``parallel`` flag, ``max_iterations``. Callables / Agents are
  serialised by **name only**.
- Rebind by passing ``registry={name: callable_or_agent}`` to
  ``from_dict``. Unknown names raise ``KeyError`` with the offending
  entry — the load fails loud rather than producing a silently-broken
  Plan.
- ``target.kind == "tool"`` (a string target) survives round-trip
  without a registry entry — the tool is resolved at run time from the
  Agent's tool map.

**example**

```python
from lazybridge import Plan, Step, Agent, from_step
import json

def fetch(task: str) -> str: ...
def rank(task: str) -> str: ...

plan = Plan(
    Step(fetch, name="fetch", writes="hits"),
    Step(rank,  name="rank",  task=from_step("fetch"), writes="ranked"),
)

# Persist.
with open("plan.json", "w") as f:
    json.dump(plan.to_dict(), f, indent=2)

# Later / elsewhere:
with open("plan.json") as f:
    saved = json.load(f)

plan_reloaded = Plan.from_dict(saved, registry={
    "fetch": fetch,   # rebind to live functions
    "rank":  rank,
})

Agent.from_engine(plan_reloaded)("AI trends")
```

**pitfalls**

- The registry is a positional contract: every non-tool target must be
  in the registry or ``from_dict`` raises ``KeyError``. Keep target
  names stable across versions.
- Tool-name targets (``target=str``) survive without a registry — they
  are resolved by the outer Agent's ``tools=[...]`` at run time.
- The JSON shape is versioned (``version: 1``). Breaking changes will
  bump the number and ``from_dict`` will refuse older shapes; migrate
  explicitly rather than silently.

## Provider registry

**signature**

LLMEngine.register_provider_alias(alias: str, provider: str) -> None
LLMEngine.register_provider_rule(
    pattern: str,
    provider: str,
    *,
    kind: Literal["contains", "startswith"] = "contains",
) -> None

# Internal tables (user-extendable at runtime):
#   LLMEngine._PROVIDER_ALIASES  — exact-match model string → provider
#   LLMEngine._PROVIDER_RULES    — [(kind, pattern, provider), ...]
#   LLMEngine._PROVIDER_DEFAULT  — fallback when no rule matches

**rules**

- ``register_provider_alias`` adds exact-match routing: model string
  equal to ``alias`` (case-insensitive) resolves to ``provider``.
- ``register_provider_rule`` adds substring / prefix routing: if the
  rule matches, the provider is used. New rules PREPEND — so user
  rules take priority over built-ins. A newer "claude-opus-5-foo"
  rule wins over the built-in "claude" catch-all.
- Matching is case-insensitive (both pattern and model are lower-cased).
- Both methods are ``@classmethod``; they mutate class-level tables.
  Tests should snapshot/restore these tables if they register rules
  (see ``tests/unit/test_v1_refinements.py:restore_provider_rules``).

**example**

```python
import pytest
from lazybridge import Agent, LLMEngine

# Route all model strings starting with "bedrock/" to a custom
# AWS Bedrock provider that the user has subclassed from BaseProvider.
LLMEngine.register_provider_rule("bedrock/", "bedrock", kind="startswith")

# Override a built-in: send all "claude-*" calls through a local proxy.
LLMEngine.register_provider_rule("claude", "my-proxy")

# Exact-match alias for a new provider.
LLMEngine.register_provider_alias("mistral", "mistral")

# All of these now resolve via the registry.
Agent("bedrock/claude-opus-5")
Agent("claude-opus-4-7")    # routed to "my-proxy" because user rule takes priority
Agent("mistral")

# Test hygiene — snapshot+restore in a fixture.
@pytest.fixture
def restore_provider_rules():
    aliases = dict(LLMEngine._PROVIDER_ALIASES)
    rules = list(LLMEngine._PROVIDER_RULES)
    yield
    LLMEngine._PROVIDER_ALIASES = aliases
    LLMEngine._PROVIDER_RULES = rules
```

**pitfalls**

- Order matters: ``register_provider_rule`` PREPENDS. If you need to
  append instead (rare), mutate ``_PROVIDER_RULES`` directly.
- Registering an alias without a matching subclassed ``BaseProvider``
  in ``core/providers/`` will succeed but ``Agent(...)`` calls will
  fail at Executor resolution time.
- Tests that register rules leak state into subsequent tests unless
  you use the restore fixture pattern.

## core.types

**signature**

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

UsageStats(input_tokens, output_tokens, thinking_tokens, cached_input_tokens, cost_usd)

StructuredOutputConfig(schema, strict=True)
ThinkingConfig(enabled=True, effort="high", budget_tokens=None, display=None)
SkillsConfig(skills: list[str])
NativeTool (StrEnum):  WEB_SEARCH, CODE_EXECUTION, FILE_SEARCH, COMPUTER_USE,
                       GOOGLE_SEARCH, GOOGLE_MAPS
StreamChunk(delta, thinking_delta, tool_calls, usage, is_final, ...)

**rules**

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

**example**

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

**pitfalls**

- Forgetting that ``CompletionResponse.content`` is always a string:
  structured output lives in ``.parsed`` (the validated model instance)
  alongside ``.content`` (the raw JSON text).
- ``tool_choice`` on ``CompletionRequest`` is the provider-level knob
  ("auto" / "any" / specific tool); different from user-facing
  ``LLMEngine(tool_choice=...)`` which was "auto" / "any" post-v1.
- ``NativeTool`` entries are enums, not capabilities — the provider
  decides whether to honour each one; unsupported combinations raise
  at ``complete`` time.
