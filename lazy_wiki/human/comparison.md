# LazyBridgeFramework vs Raw SDK — Comparison

Side-by-side comparison of the same tasks done with the raw provider SDK and with LazyBridgeFramework.

---

## Task 1 — Single LLM call

### Raw Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
answer = message.content[0].text
print(answer)
```

**8 lines, 3 concepts** (client, messages format, content extraction)

### LazyBridgeFramework

```python
from lazybridgeframework import LazyAgent

answer = LazyAgent("anthropic").text("What is the capital of France?")
print(answer)
```

**2 lines, 1 concept**

Key benefits:
- No client instantiation boilerplate
- No messages array formatting
- No content extraction
- Identical code works with OpenAI, Google, DeepSeek (change `"anthropic"`)

---

## Task 2 — Tool loop (function calling)

### Raw OpenAI SDK (Responses API)

```python
import json
import openai

client = openai.OpenAI()

# 1. Define schema manually (Responses API flattened format)
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"],
        },
    }
]

def get_weather(city: str) -> str:
    return f"Weather in {city}: 22°C, sunny"

input_messages = [{"role": "user", "content": "What's the weather in Rome and Paris?"}]

# 2. Manual tool-call loop
while True:
    response = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        tools=tools,
    )
    # Check for function calls in output items
    function_calls = [item for item in response.output if item.type == "function_call"]

    if function_calls:
        # Append assistant output items to conversation
        input_messages.extend([item.model_dump() for item in response.output])
        for fc in function_calls:
            args = json.loads(fc.arguments)
            result = get_weather(**args)
            input_messages.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": result,
            })
    else:
        # Extract text from message output items
        text_items = [item for item in response.output if item.type == "message"]
        print(text_items[0].content[0].text)
        break
```

**~45 lines, 7+ concepts** (flattened schema format, output item types, call_id vs tool_call_id, model_dump for history, function_call_output format, output item iteration, content block extraction)

### LazyBridgeFramework

```python
from lazybridgeframework import LazyAgent, LazyTool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°C, sunny"

result = LazyAgent("openai").loop(
    "What's the weather in Rome and Paris?",
    tools=[LazyTool.from_function(get_weather)],
)
print(result.content)
```

**6 lines, 2 concepts** (define function, call loop)

Key benefits:
- Schema generated automatically from type hints + docstring
- Tool-call loop handled internally
- Message accumulation handled internally
- Error handling built in
- Same code works on any provider

---

## Task 3 — Multi-provider pipeline

### Raw SDKs

```python
import anthropic
import openai

# Two separate clients, two completely different APIs
anthropic_client = anthropic.Anthropic()
openai_client    = openai.OpenAI()

# Researcher (Anthropic — Messages API)
research_resp = anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    messages=[{"role": "user", "content": "Find AI news this week"}]
)
research_output = research_resp.content[0].text  # Anthropic: .content[0].text

# Writer (OpenAI — Responses API, new default)
writer_resp = openai_client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "system", "content": f"Use this research:\n{research_output}"},
        {"role": "user",   "content": "Write a newsletter section"},
    ]
)
# OpenAI Responses API: iterate output items, find message, extract text block
message_item = next(item for item in writer_resp.output if item.type == "message")
print(message_item.content[0].text)
```

**~22 lines, 5 concepts** (two clients, two different APIs, two different response formats, manual output bridging, two different content extraction patterns)

### LazyBridgeFramework

```python
from lazybridgeframework import LazyAgent, LazyContext

researcher = LazyAgent("anthropic", name="researcher")
writer     = LazyAgent("openai",    name="writer")

researcher.loop("Find AI news this week")
result = writer.chat(
    "Write a newsletter section",
    context=LazyContext.from_agent(researcher),
)
print(result.content)
```

**6 lines, 2 concepts** (create agents, pipe output)

Key benefits:
- Single unified API across providers
- Output bridging handled by `LazyContext.from_agent`
- No format translation between SDK response types
- Adding a third provider = adding one line

---

## Task 4 — Structured JSON output

### Raw Anthropic SDK

```python
import json
import anthropic
from pydantic import BaseModel, ValidationError

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

client = anthropic.Anthropic()

for attempt in range(3):   # manual retry loop
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Generate article ideas about AI safety. Return JSON."
        }],
    )
    raw = resp.content[0].text
    try:
        data = json.loads(raw)
        article = Article(**data)
        print(article.title)
        break
    except (json.JSONDecodeError, ValidationError) as e:
        if attempt == 2:
            raise
        # retry
```

**~25 lines, 5 concepts** (manual JSON mode, json.loads, try/except, Pydantic validation, retry loop)

### LazyBridgeFramework

```python
from lazybridgeframework import LazyAgent
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

article = LazyAgent("anthropic").json("Generate article ideas about AI safety.", Article)
print(article.title)
```

**5 lines, 1 concept** (pass schema, get typed object)

Key benefits:
- No JSON mode configuration
- No manual `json.loads`
- No try/except
- Retry on parse failure built in
- Returns typed Pydantic instance directly

---

## Task 5 — Concurrent agents

### Raw asyncio + OpenAI (Responses API)

```python
import asyncio
import openai

client = openai.AsyncOpenAI()

async def run_agent(task: str) -> str:
    resp = await client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": task}]
    )
    # Responses API: iterate output items to extract text
    message_item = next(item for item in resp.output if item.type == "message")
    return message_item.content[0].text

async def main():
    tasks = [
        "Summarise AI news from the US",
        "Summarise AI news from Europe",
        "Summarise AI news from Asia",
    ]
    results = await asyncio.gather(*[run_agent(t) for t in tasks])
    combined = "\n\n".join(results)

    summary_resp = await client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": combined},
            {"role": "user", "content": "Write a global digest"},
        ]
    )
    message_item = next(item for item in summary_resp.output if item.type == "message")
    print(message_item.content[0].text)

asyncio.run(main())
```

**~30 lines** (each response requires output-item iteration; no shared state, event tracking, or graph topology)

### LazyBridgeFramework

```python
import asyncio
from lazybridgeframework import LazyAgent, LazySession, LazyContext

sess = LazySession()
agents = [LazyAgent("openai", name=n, session=sess) for n in ["us", "eu", "asia"]]
tasks  = ["US AI news", "Europe AI news", "Asia AI news"]

async def main():
    results = await sess.gather(*[a.aloop(t) for a, t in zip(agents, tasks)])
    for agent, result in zip(agents, results):
        sess.store.write(agent.name, result.content)

    editor = LazyAgent("openai", context=LazyContext.from_store(sess.store), session=sess)
    digest = await editor.achat("Write a global AI digest")
    print(digest.content)

asyncio.run(main())
```

**~12 lines** — and gains event tracking, graph visualization, and a shared store for free.

---

## Summary

| Task | Raw SDK lines | LazyBridge lines | Reduction |
|------|:---:|:---:|:---:|
| Single call | 8 | 2 | **75%** |
| Tool loop (Responses API) | 45 | 6 | **87%** |
| Multi-provider pipeline | 22 | 6 | **73%** |
| Structured output | 25 | 5 | **80%** |
| Concurrent agents (Responses API) | 30 | 12 | **60%** |

Note: raw OpenAI examples use the **Responses API** (OpenAI's recommended default since 2025). It introduces a different response format (`response.output` item iteration instead of `response.choices[0].message`) and a different tool result format (`function_call_output` with `call_id`). LazyBridge handles all provider-specific formats transparently.

Beyond line count:

| Concern | Raw SDK | LazyBridgeFramework |
|---------|---------|---------------------|
| Provider switch | Rewrite client, format, extraction | Change one string |
| Tool schema | Write JSON dict manually | Type hints → automatic |
| Output bridging | Manual string passing | `LazyContext.from_agent` |
| Concurrent execution | Manual asyncio + gather | `sess.gather()` |
| Event tracking | Build your own logging | Built-in `sess.events` |
| Pipeline topology | Not captured | `sess.graph.to_json()` for GUI |
| Testing context | Baked into prompt strings | `ctx()` → testable string |
| Retry logic | Write your own | `max_retries=3` |
| Structured output retry | Write your own | Built-in with `output_schema=` |

The core design principle: **write what you want to happen, not how to make it happen**.

---

## Framework Comparison — LazyBridgeFramework vs LLM Orchestration Frameworks

### TL;DR

LazyBridgeFramework sits between raw SDKs (too much boilerplate) and heavy orchestration frameworks (too much abstraction). Its differentiator is a minimal, composable API with production-grade features (Memory, LazyContext, LazyStore, EventLog, native tools) that doesn't impose a paradigm on the developer.

---

### vs LangChain

LangChain is the most widely used framework (~100k+ LOC, hundreds of integrations). It solves everything but at the cost of complexity: LCEL chain DSL, multiple abstraction layers, frequent breaking changes, and a steep learning curve even for simple use cases.

| Concern | LangChain | LazyBridgeFramework |
|---------|-----------|---------------------|
| Simple call boilerplate | ~10 lines (ChatOpenAI, HumanMessage, invoke) | 2 lines |
| Tool loop | Chain + AgentExecutor + tool wrappers | `agent.loop(task, tools=[...])` |
| Multi-provider | Different ChatXxx classes, same interface | Single `LazyAgent(provider)` string |
| Memory | ConversationBufferMemory + chain wiring | `mem = Memory(); agent.chat(..., memory=mem)` |
| Output bridging | Manual chain composition | `LazyContext.from_agent(agent)` |
| Shared state | Manual dict passing | `LazyStore` |
| Native tools (web/code) | Wrapped tools, provider-specific config | `native_tools=[NativeTool.WEB_SEARCH]` |
| Ecosystem (RAG, vector DBs) | ✅ Extensive | ❌ Not a focus |
| Stability | ⚠️ Frequent breaking changes | ✅ Minimal surface area |

**When to choose LangChain:** You need deep RAG pipelines, document loaders, or specific vector store integrations not worth building yourself.

**When to choose LazyBridge:** You want agentic loops, multi-provider pipelines, and structured output without spending a day reading docs.

---

### vs PydanticAI

The closest competitor in philosophy (launched late 2024). PydanticAI is type-safe, Python-first, with clean dependency injection. Supports OpenAI, Anthropic, Google, Groq, Mistral.

| Concern | PydanticAI | LazyBridgeFramework |
|---------|------------|---------------------|
| Type safety | ✅ Strong (typed agents, typed deps) | ✅ Good (Pydantic structured output, typed responses) |
| Multi-provider | ✅ | ✅ |
| Stateful memory | ⚠️ Manual message history | ✅ `Memory` object |
| Dynamic context injection | ❌ | ✅ `LazyContext` (lazy, composable, testable) |
| Shared state across agents | ❌ | ✅ `LazyStore` |
| Event observability | ❌ | ✅ `LazySession.events` |
| Pipeline topology | ❌ | ✅ `sess.graph.to_json()` |
| Native tools (web/code) | ❌ | ✅ `NativeTool` enum, all providers |
| Maturity | ⚠️ ~6 months old | ⚠️ Comparable |

**When to choose PydanticAI:** You want maximum type-safety and dependency injection as a core pattern.

**When to choose LazyBridge:** You want the full pipeline ecosystem (Memory, LazyContext, LazyStore, EventLog, native tools) in a single composable API.

---

### vs CrewAI

CrewAI popularised the "crew of agents" metaphor: you declare Agent roles, Task descriptions, and a Process (sequential or hierarchical). Very readable for structured multi-agent workflows.

| Concern | CrewAI | LazyBridgeFramework |
|---------|--------|---------------------|
| Multi-agent | ✅ First-class (Crew, Task, Process) | ✅ Composable (LazySession, LazyRouter) |
| Flexibility | ❌ Crew metaphor required | ✅ No imposed paradigm |
| Single-agent usage | ⚠️ Crew overhead | ✅ `LazyAgent("openai").text(...)` |
| Provider switching | ⚠️ Per-agent config | ✅ One string |
| Streaming | ⚠️ Limited | ✅ Built-in sync/async |
| Native tools | ❌ | ✅ |
| Memory | ✅ (entity/short/long) | ✅ (explicit `Memory` object) |

**When to choose CrewAI:** You want a declarative, role-based multi-agent pipeline with minimal code.

**When to choose LazyBridge:** You want full control of execution flow, streaming, and provider switching without role/task boilerplate.

---

### vs Autogen / AG2

Microsoft's framework models agents as conversational participants that message each other. Powerful for human-in-the-loop and complex agent debates. Complex to configure for straightforward pipelines.

| Concern | Autogen | LazyBridgeFramework |
|---------|---------|---------------------|
| Human-in-the-loop | ✅ First-class | ⚠️ Manual |
| Simple pipeline | ❌ ConversableAgent + GroupChat overhead | ✅ |
| Provider support | ⚠️ OpenAI-first | ✅ Anthropic, OpenAI, Google, DeepSeek |
| Streaming | ⚠️ | ✅ |

**When to choose Autogen:** Human-in-the-loop conversations, agent debates, or research scenarios requiring agents to negotiate.

**When to choose LazyBridge:** Automated pipelines where you want clean code, not a conversation engine.

---

### Summary Matrix

| | Raw SDK | LangChain | PydanticAI | CrewAI | **LazyBridge** |
|---|:---:|:---:|:---:|:---:|:---:|
| Boilerplate | ❌ High | ⚠️ Medium | ✅ Low | ⚠️ Medium | ✅ Minimal |
| Multi-provider | ❌ | ⚠️ | ✅ | ⚠️ | ✅ |
| Tool loop | ❌ Manual | ✅ | ✅ | ✅ | ✅ |
| Stateful memory | ❌ Manual | ⚠️ Wired | ⚠️ Manual | ✅ | ✅ |
| Pipeline composability | ❌ | ⚠️ | ❌ | ⚠️ | ✅ |
| Dynamic context injection | ❌ | ❌ | ❌ | ❌ | ✅ |
| Observability built-in | ❌ | ❌ | ❌ | ⚠️ | ✅ |
| Native tools (web/code) | ❌ Manual | ⚠️ | ❌ | ❌ | ✅ |
| Ecosystem / integrations | ✅ | ✅ Best | ⚠️ | ⚠️ | ❌ Minimal |
| Learning curve | Low (SDK) | ❌ High | ✅ Low | ✅ Low | ✅ Low |

LazyBridgeFramework's unique position: **the only framework that combines a minimal API with a full pipeline ecosystem (Memory + LazyContext + LazyStore + EventLog + native tools) without imposing a metaphor**.

The gap to close: external integrations (vector stores, document loaders). Everything else is a strength.
