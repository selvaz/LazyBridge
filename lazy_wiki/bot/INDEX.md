# LazyBridgeFramework — LLM Reference Index

## What each class does

| Class | File | Role | Patterns |
|---|---|---|---|
| LazyAgent | 01_lazyagent.md | Single LLM agent: chat, loop, tools, context | A, B, C |
| LazySession | 02_lazysession.md | Multi-agent container: store, events, graph, concurrency | B, C |
| LazyTool | 03_lazytool.md | Tool abstraction: function or agent-backed, auto-schema | A, B, C |
| LazyContext | 04_lazycontext.md | Lazily-evaluated context injection into system prompt | A, B, C |
| LazyStore | 05_lazystore.md | Shared key-value blackboard for cross-agent state | B, C |
| LazyRouter | 06_lazyrouter.md | Conditional branching: condition → agent selection | A |
| GraphSchema | 07_graphschema.md | Serializable pipeline topology for GUI | all |
| TrackLevel/Event | 02_lazysession.md | Tracking levels and event types | all |
| ToolSchemaBuilder | 10_tool_schema.md | Schema pipeline: type mapping, LLM modes, artifact caching | all |

## Communication rules (critical)

1. Agents communicate via **return values**, NOT via LazyStore.
2. LazyStore is a **blackboard for persistent state**, not a message bus.
3. LazyContext injects strings into system prompts — it is **NOT a communication channel**.
4. `tool_runner` fallback only for tools without a callable (`ToolDefinition`/`dict`).

## Pipeline patterns

- **Pattern A — Hierarchy**: orchestrator calls sub-agents as `LazyTool` instances via `loop()`
- **Pattern B — Parallel**: multiple agents run concurrently via `sess.gather()` + `aloop()`
- **Pattern C — Network**: agents share state via `LazyStore`, context via `LazyContext.from_store` / `LazyContext.from_agent`

## Reading order

| Goal | Files |
|---|---|
| Quick pipeline | `00_quickref.md` → `01_lazyagent.md` → `03_lazytool.md` |
| Multi-agent | add `02_lazysession.md` → `04_lazycontext.md` → `05_lazystore.md` |
| Routing | `06_lazyrouter.md` |
| GUI integration | `07_graphschema.md` |
| All patterns with full code | `08_patterns.md` |
| Advanced features | `09_advanced.md` |
| Schema generation (type mapping, LLM modes, artifacts) | `10_tool_schema.md` |

## Providers supported

| Alias | Provider class | Default model |
|---|---|---|
| `anthropic` / `claude` | `AnthropicProvider` | `claude-sonnet-4-6` |
| `openai` / `gpt` | `OpenAIProvider` | — |
| `google` / `gemini` | `GoogleProvider` | — |
| `deepseek` | `DeepSeekProvider` | — |

Env vars: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`

## Import

```python
from lazybridgeframework import (
    LazyAgent, LazySession, LazyTool, LazyContext,
    LazyStore, LazyRouter, GraphSchema,
    TrackLevel, Event,
)
# Advanced types:
from lazybridgeframework.core.types import (
    CompletionResponse, CompletionRequest, StreamChunk,
    NativeTool, ThinkingConfig, SkillsConfig,
    StructuredOutputConfig, Message, Role,
)
```
