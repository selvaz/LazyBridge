# Module 11: Advanced Features & Production Patterns

Professional patterns for production deployments. This module covers every advanced feature not in the basic course.

---

## Part A — Advanced Agent Features

### Thinking / Reasoning Models

Enable chain-of-thought reasoning for complex tasks:

```python
from lazybridge import LazyAgent
from lazybridge.core.types import ThinkingConfig

# Simple — just enable thinking
ai = LazyAgent("anthropic")
resp = ai.chat("Solve: If 3x + 7 = 22, what is x?", thinking=True)
print(resp.content)    # "x = 5"
print(resp.thinking)   # "Let me solve step by step... 3x = 22 - 7 = 15, x = 15/3 = 5"

# Advanced — configure thinking behavior
config = ThinkingConfig(
    effort="high",          # "low" | "medium" | "high" | "xhigh"
    display="omitted",      # Anthropic: hide thinking in streaming
)
resp = ai.chat("Complex reasoning task...", thinking=config)

# OpenAI reasoning models (o1, o3, o4-mini) use effort automatically
ai_o = LazyAgent("openai", model="o4-mini")
resp = ai_o.chat("Solve this math olympiad problem...", thinking=True)

# Thinking works in loops too — the thinking blocks are preserved
# in the conversation so the model maintains its reasoning chain
resp = ai.loop("Research and analyze this dataset", tools=my_tools, thinking=True)
```

### Anthropic Skills

Server-side file processing without uploading or managing context:

```python
# Skills let Anthropic process PDFs, Excel files, etc. natively
ai = LazyAgent("anthropic")
resp = ai.chat(
    "Summarize the key findings in this PDF",
    skills=["pdf"],
)

# Multiple skills
resp = ai.chat(
    "Compare the data in this spreadsheet with the report",
    skills=["pdf", "excel"],
)

# Available skills: "pdf", "excel", "powerpoint", "word"
```

### Self-Verifying Loops

Add a quality gate that retries if the output doesn't pass review:

```python
from lazybridge import LazyAgent

worker = LazyAgent("anthropic")
judge = LazyAgent("openai", model="gpt-4o-mini")

# Verify with another agent — judge returns "APPROVED..." or feedback
resp = worker.loop(
    "Write a Python function to sort a list using quicksort",
    verify=judge,       # judge.text() called with "Question: ... Answer: ..."
    max_verify=3,       # retry up to 3 times
)
print(resp.verify_log)  # ["Missing docstring", "APPROVED: looks good"]

# Verify with a custom function
def check_has_docstring(question: str, answer: str) -> str:
    if '"""' in answer or "'''" in answer:
        return "APPROVED"
    return "Missing docstring — add one and try again"

resp = worker.loop(
    "Write a Python function to calculate fibonacci",
    verify=check_has_docstring,
    max_verify=3,
)

# Implement the Verifier protocol for reusable judges
from lazybridge import Verifier

class CodeReviewer:
    """Custom verifier that checks code quality."""
    def text(self, messages: str) -> str:
        # messages contains "Question: ...\nAnswer: ..."
        if "def " not in messages:
            return "No function definition found"
        if '"""' not in messages:
            return "Missing docstring"
        return "APPROVED: code looks good"

resp = worker.loop("Write a sorting function", verify=CodeReviewer())
```

### Tool Choice Control

Fine-grained control over when and which tools are used:

```python
# Force the agent to use at least one tool
resp = ai.loop("Hello", tools=my_tools, tool_choice="required")

# Prevent tool use entirely (useful for final formatting pass)
resp = ai.chat("Summarize this", tools=my_tools, tool_choice="none")

# Force a specific tool by name
resp = ai.loop("Get data", tools=[search, calculator], tool_choice="search")
```

### tool_choice on as_tool()

Force tool use when an agent is wrapped as a tool — the orchestrator doesn't need to know:

```python
# Researcher ALWAYS searches before answering
researcher = LazyAgent("anthropic", tools=[web_search, arxiv])
research_tool = researcher.as_tool("research", "Find information", tool_choice="required")

# Orchestrator just calls the tool — doesn't know about tool_choice
orchestrator = LazyAgent("anthropic")
orchestrator.loop("Find AI safety papers", tools=[research_tool])
```

### Parallel Tool Execution

When the model returns multiple tool calls in one step, run them concurrently:

```python
# Sequential (default) — tools run one after another
resp = ai.loop("Get data for 5 cities", tools=[weather])

# Parallel — all tool calls in a step run at the same time
# Async uses asyncio.gather() for true concurrency
resp = await ai.aloop("Get data for 5 cities", tools=[weather], tool_choice="parallel")
```

Best for I/O-bound tools (API calls, web requests). Not useful for CPU-bound or dependent tools.

---

## Part B — Tool System Internals

### Schema Generation Modes

Control how tool schemas are built:

```python
from lazybridge import LazyTool
from lazybridge.core.tool_schema import ToolSchemaMode

# SIGNATURE (default) — fast, deterministic, from type hints
tool = LazyTool.from_function(my_func, schema_mode=ToolSchemaMode.SIGNATURE)

# LLM — an LLM generates the entire schema (for untyped legacy functions)
tool = LazyTool.from_function(my_func, schema_mode=ToolSchemaMode.LLM)

# HYBRID — type hints for types, LLM for descriptions
tool = LazyTool.from_function(my_func, schema_mode=ToolSchemaMode.HYBRID)
```

When to use each:

| Mode | Speed | Best for |
|------|-------|----------|
| SIGNATURE | Instant | Functions with good type hints + docstrings |
| LLM | Slow (API call) | Legacy functions without type hints |
| HYBRID | Slow (API call) | Functions with types but poor docstrings |

### Tool Persistence — Save & Load

Save tools to disk as Python files, load them later:

```python
# Save a tool
def analyze_data(data: str, method: str = "mean") -> str:
    """Analyze data using the specified statistical method."""
    return f"Analysis ({method}): processed"

tool = LazyTool.from_function(analyze_data)
tool.save("tools/analyze_data.py")

# Load it back (even in a different process)
loaded = LazyTool.load("tools/analyze_data.py")
result = loaded.run({"data": "my dataset", "method": "median"})

# Save agent-as-tool
researcher = LazyAgent("anthropic", name="researcher")
research_tool = researcher.as_tool("research", "Research a topic")
research_tool.save("tools/researcher.py")

# Save pipeline tools
pipeline = LazyTool.chain(agent_a, agent_b, name="pipe", description="A then B")
pipeline.save("tools/pipeline.py")

# Security: restrict load paths
tool = LazyTool.load("tools/my_tool.py", base_dir="tools/")
```

### Shorthand tool definition

Quick tool creation without a function:

```python
# From a params dict — useful for dynamic tool creation
tool = LazyTool(
    name="search",
    description="Search the web",
    params={"query": str, "max_results": int},
)
# You'll need a tool_runner in loop() to handle execution:
resp = ai.loop("Search for AI news", tools=[tool], tool_runner=my_runner)
```

---

## Part C — Custom Extensions

### Custom Provider

Add support for any LLM backend:

```python
from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest, CompletionResponse, StreamChunk, UsageStats,
)
from collections.abc import Iterator, AsyncIterator

class MyLLMProvider(BaseProvider):
    default_model = "my-model-v1"

    def _init_client(self, **kwargs) -> None:
        import my_sdk
        key = self.api_key or os.environ.get("MY_LLM_KEY")
        if not key:
            raise ValueError("MY_LLM_KEY not set")
        self._client = my_sdk.Client(api_key=key, **kwargs)

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        return (input_tokens * 1.0 + output_tokens * 3.0) / 1_000_000

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = self._resolve_model(request)
        messages = [{"role": m.role.value, "content": m.to_text()} for m in request.messages]
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})

        resp = self._client.generate(model=model, messages=messages, max_tokens=request.max_tokens)
        usage = UsageStats(
            input_tokens=resp.usage.input,
            output_tokens=resp.usage.output,
            cost_usd=self._compute_cost(model, resp.usage.input, resp.usage.output),
        )
        return CompletionResponse(content=resp.text, usage=usage, model=model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        model = self._resolve_model(request)
        for chunk in self._client.stream(model=model, messages=[...]):
            yield StreamChunk(delta=chunk.text)
        yield StreamChunk(delta="", is_final=True, stop_reason="end_turn")

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        # async version of complete()
        ...

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        # async version of stream()
        ...

# Use it like any built-in provider
from lazybridge import LazyAgent
ai = LazyAgent(MyLLMProvider(api_key="..."))
print(ai.text("Hello from custom provider!"))
```

### Custom Event Exporter

Forward events to your own systems:

```python
from lazybridge.exporters import EventExporter

class SlackExporter:
    """Send tool errors to Slack."""

    def __init__(self, webhook_url: str):
        self._url = webhook_url

    def export(self, event: dict) -> None:
        if event.get("event_type") == "tool_error":
            data = event.get("data", {})
            msg = f"Tool error in {event.get('agent_name')}: {data.get('error')}"
            # requests.post(self._url, json={"text": msg})
            print(f"[Slack] {msg}")

# Register it
from lazybridge import LazySession
sess = LazySession(exporters=[SlackExporter("https://hooks.slack.com/...")])
```

The `EventExporter` protocol only requires one method: `export(event: dict) -> None`.

### Custom Stateful Guard

Build guards that maintain state across calls:

```python
from lazybridge.guardrails import GuardAction

class RateLimitGuard:
    """Block after N requests per minute."""

    def __init__(self, max_per_minute: int = 10):
        self._max = max_per_minute
        self._timestamps: list[float] = []

    def _clean(self):
        import time
        cutoff = time.time() - 60
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def check_input(self, text: str) -> GuardAction:
        import time
        self._clean()
        if len(self._timestamps) >= self._max:
            return GuardAction.block(f"Rate limit: {self._max}/min exceeded")
        self._timestamps.append(time.time())
        return GuardAction.allow()

    def check_output(self, text: str) -> GuardAction:
        return GuardAction.allow()

    async def acheck_input(self, text: str) -> GuardAction:
        return self.check_input(text)

    async def acheck_output(self, text: str) -> GuardAction:
        return GuardAction.allow()

# Use it
guard = RateLimitGuard(max_per_minute=5)
resp = ai.chat("hello", guard=guard)
```

---

## Part D — Production Patterns

### Multi-Provider Fallback

Try the primary provider; fall back on failure:

```python
from lazybridge import LazyAgent

def resilient_call(prompt: str, **kwargs) -> str:
    providers = [
        ("anthropic", "claude-sonnet-4-6"),
        ("openai", "gpt-4o"),
        ("google", "gemini-2.5-flash"),
    ]
    for provider, model in providers:
        try:
            ai = LazyAgent(provider, model=model, max_retries=2)
            return ai.text(prompt, **kwargs)
        except Exception as e:
            print(f"{provider} failed: {e}")
            continue
    raise RuntimeError("All providers failed")

result = resilient_call("What is the capital of France?")
```

### Concurrency Control

Limit parallel execution and add timeouts:

```python
from lazybridge import LazyTool, LazyAgent

agents = [LazyAgent("anthropic", name=f"worker-{i}") for i in range(10)]

# Limit to 3 concurrent API calls (respect rate limits)
panel = LazyTool.parallel(
    *agents,
    name="panel",
    description="10 workers with rate limiting",
    concurrency_limit=3,  # max 3 running at once
)

# Add per-step timeout for chains
pipeline = LazyTool.chain(
    researcher, writer, editor,
    name="pipeline",
    description="Full article pipeline",
    step_timeout=30.0,  # each step gets 30 seconds max
)
```

### Full Session Lifecycle

Persist and resume sessions across processes:

```python
from lazybridge import LazyAgent, LazySession

# --- Process 1: create and run ---
sess = LazySession(db="project.db", tracking="verbose")
researcher = LazyAgent("anthropic", name="researcher", session=sess)
researcher.chat("Find AI safety papers from 2025")
sess.store.write("phase", "research_complete")

# Serialize the graph topology
graph_json = sess.to_json()
with open("pipeline_graph.json", "w") as f:
    f.write(graph_json)

# --- Process 2: resume ---
sess2 = LazySession.from_db("project.db", tracking="verbose")
print(sess2.store.read("phase"))  # "research_complete"
print(sess2.events.get(event_type="model_response", limit=5))

# Resume a specific session (multi-session DB)
sess3 = LazySession.from_db("project.db", session_id="abc-123-...")

# Reconstruct graph from JSON
with open("pipeline_graph.json") as f:
    sess4 = LazySession.from_json(f.read(), db="project.db")
print(sess4.graph.nodes())  # AgentNode descriptors (not live agents)
```

### Native Tools Deep Dive

Provider-specific capabilities:

```python
from lazybridge import LazyAgent, NativeTool

# --- Anthropic ---
ai = LazyAgent("anthropic")

# Web search — grounded answers with citations
resp = ai.loop("Latest news on AI regulation", native_tools=[NativeTool.WEB_SEARCH])

# --- OpenAI ---
ai = LazyAgent("openai")

# Code interpreter — runs Python server-side
resp = ai.loop("Generate a plot of sin(x) from 0 to 2pi", native_tools=[NativeTool.CODE_EXECUTION])

# File search — search uploaded files
resp = ai.loop("Find references to 'quantum' in my files", native_tools=[NativeTool.FILE_SEARCH])

# Computer use (preview) — GUI interaction
resp = ai.loop("Open the calculator app", native_tools=[NativeTool.COMPUTER_USE])

# --- Google ---
ai = LazyAgent("google")

# Grounded search with citations
resp = ai.loop("What is the current population of Tokyo?", native_tools=[NativeTool.WEB_SEARCH])
print(resp.grounding_sources)  # list of GroundingSource with urls/titles

# Code execution
resp = ai.loop("Calculate the prime factors of 1234567", native_tools=[NativeTool.CODE_EXECUTION])

# Combine native + custom tools
resp = ai.loop(
    "Search the web for data, then analyze it",
    tools=[my_analysis_tool],
    native_tools=[NativeTool.WEB_SEARCH],
)
```

Provider support matrix:

| Tool | Anthropic | OpenAI | Google | DeepSeek |
|------|-----------|--------|--------|----------|
| WEB_SEARCH | Yes | Yes | Yes | No |
| CODE_EXECUTION | No | Yes | Yes | No |
| FILE_SEARCH | No | Yes | No | No |
| COMPUTER_USE | No | Yes (preview) | No | No |

---

## Part E — Complete Production Example

Everything together: multi-agent pipeline with guardrails, structured output, checkpoints, observability, verify loop, and evals.

```python
"""Production investment research pipeline.

Combines: chain pipeline + guardrails + structured output + verify loop +
OTel observability + checkpoints + evals.
"""
from pydantic import BaseModel, Field
from lazybridge import (
    LazyAgent, LazySession, LazyTool, LazyStore, LazyContext,
    ContentGuard, GuardChain, GuardAction,
    OTelExporter, CallbackExporter,
)
from lazybridge.evals import EvalSuite, EvalCase, contains, not_contains, min_length

# --- 1. Define output schema ---

class ResearchReport(BaseModel):
    title: str
    summary: str = Field(description="2-3 sentence executive summary")
    key_findings: list[str] = Field(min_length=3)
    risks: list[str]
    recommendation: str = Field(description="Buy / Hold / Sell with reasoning")

# --- 2. Set up guardrails ---

def no_financial_advice_disclaimer(text: str) -> GuardAction:
    """Ensure outputs include a disclaimer."""
    if len(text) > 200 and "not financial advice" not in text.lower():
        return GuardAction.block("Missing disclaimer: 'not financial advice'")
    return GuardAction.allow()

def no_pii(text: str) -> GuardAction:
    import re
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):  # SSN pattern
        return GuardAction.block("PII detected")
    return GuardAction.allow()

guard = GuardChain([
    ContentGuard(input_fn=no_pii),
    ContentGuard(output_fn=no_financial_advice_disclaimer),
])

# --- 3. Set up session with observability ---

store = LazyStore(db="research.db")
sess = LazySession(
    db="research.db",
    tracking="verbose",
    exporters=[
        OTelExporter(service_name="investment-research"),
        CallbackExporter(lambda e: None),  # custom logging
    ],
)

# --- 4. Create specialized agents ---

researcher = LazyAgent(
    "anthropic",
    name="researcher",
    session=sess,
    system="You are a senior financial analyst. Always cite your sources.",
    native_tools=["web_search"],
)

analyst = LazyAgent(
    "anthropic",
    name="analyst",
    session=sess,
    system="You are a risk analyst. Be thorough and conservative.",
    output_schema=ResearchReport,
)

# --- 5. Build pipeline with checkpoints ---

pipeline = LazyTool.chain(
    researcher, analyst,
    name="research_pipeline",
    description="Research a company then produce structured analysis",
    store=store,
    chain_id="investment-research",
)

# --- 6. Create verified orchestrator ---

judge = LazyAgent("openai", model="gpt-4o-mini")
orchestrator = LazyAgent(
    "anthropic",
    name="orchestrator",
    system="You coordinate investment research. Use the research_pipeline tool.",
)

result = orchestrator.loop(
    "Analyze Tesla's current market position and outlook for 2026",
    tools=[pipeline],
    verify=judge,
    max_verify=2,
    guard=guard,
)

# --- 7. Check results ---

print(f"Result: {result.content[:200]}...")
print(f"Cost: ${sess.usage_summary()['total']['cost_usd']:.4f}")
print(f"Verify log: {result.verify_log}")

# --- 8. Run evals to validate quality ---

suite = EvalSuite(cases=[
    EvalCase(
        "Analyze Apple stock",
        check=contains("Apple", "AAPL"),
        tags=["accuracy"],
    ),
    EvalCase(
        "Analyze a nonexistent company XYZ123",
        check=not_contains("strong buy", "definitely invest"),
        tags=["safety"],
    ),
    EvalCase(
        "Analyze Microsoft",
        check=min_length(200),
        tags=["completeness"],
    ),
])

# Run evals against the orchestrator
report = suite.run(orchestrator)
print(f"\nEval results: {report}")
for failure in report.failures:
    print(f"  FAILED: {failure.case.name}: {failure.output[:100]}")
```

---

## Quick Reference — All Advanced APIs

```python
# Thinking
ai.chat("...", thinking=True)
ai.chat("...", thinking=ThinkingConfig(effort="high"))

# Skills (Anthropic)
ai.chat("...", skills=["pdf", "excel"])

# Verify
ai.loop("...", verify=judge_agent, max_verify=3)
ai.loop("...", verify=my_callable, max_verify=3)
resp.verify_log  # list of rejection messages

# Tool schema modes
LazyTool.from_function(fn, schema_mode=ToolSchemaMode.SIGNATURE)
LazyTool.from_function(fn, schema_mode=ToolSchemaMode.LLM)
LazyTool.from_function(fn, schema_mode=ToolSchemaMode.HYBRID)

# Tool persistence
tool.save("path.py")
tool = LazyTool.load("path.py", base_dir="tools/")

# Pipeline controls
LazyTool.parallel(..., concurrency_limit=3, step_timeout=30.0)
LazyTool.chain(..., store=store, chain_id="id", run_id="run-1")

# Session lifecycle
sess = LazySession.from_db("db.db", session_id="specific-id")
sess = LazySession.from_json(json_str)
json_str = sess.to_json()

# Native tools
ai.loop("...", native_tools=[NativeTool.WEB_SEARCH, NativeTool.CODE_EXECUTION])

# Custom provider
ai = LazyAgent(MyProvider(api_key="..."))

# Smart Memory
Memory()                                           # auto compression (default)
Memory(strategy="full")                            # never compress
Memory(strategy="rolling", window_turns=10)        # always window + compress
Memory(compressor=LazyAgent("openai", model="gpt-4o-mini"))  # LLM compression
mem.history                                        # full raw history
mem.summary                                        # current compressed block
```

---

**You've completed the full LazyBridge course.** You now know every feature in the framework — from `ai.text("hello")` to production multi-agent pipelines with guardrails, evals, and observability.
