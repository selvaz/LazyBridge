# Module 2: Tools & Functions

Tools let agents call your Python functions. The agent decides *when* to call a tool and *what arguments* to pass — LazyBridge handles the schema, execution, and result formatting automatically.

## Your first tool

```python
from lazybridge import LazyAgent, LazyTool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 22°C, sunny"

tool = LazyTool.from_function(get_weather)
ai = LazyAgent("anthropic")
resp = ai.loop("What's the weather in Paris?", tools=[tool])
print(resp.content)
# The weather in Paris is 22°C and sunny.
```

**What happened:**

1. LazyBridge read the type hints and docstring to build a JSON schema
2. The agent decided to call `get_weather` with `city="Paris"`
3. LazyBridge executed your function and returned the result to the agent
4. The agent formatted a natural-language response

## `chat()` vs `loop()`

- `chat()` — single turn. The agent may *request* tool calls, but won't execute them
- `loop()` — agentic loop. Executes tool calls, feeds results back, repeats until done

**Always use `loop()` when you provide tools.**

## How tool schemas work

LazyBridge generates schemas automatically from type hints:

```python
def search(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: The search query
        max_results: Maximum results to return (1-10)
    """
    return f"Results for '{query}': ..."

tool = LazyTool.from_function(search)
print(tool.definition.input_schema)
# {
#   "type": "object",
#   "properties": {
#     "query": {"type": "string", "description": "The search query"},
#     "max_results": {"type": "integer", "description": "Maximum results to return (1-10)"}
#   },
#   "required": ["query"]
# }
```

**Rules for tool functions:**

- Must have type hints on all parameters
- Must have a docstring (used as the tool description)
- Return type should be `str` (or something that converts to string)
- Cannot be a lambda (no type hints or docstring)

## Multiple tools

```python
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))  # simplified — use a safe evaluator in production
    except Exception as e:
        return f"Error: {e}"

def current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().isoformat()

tools = [
    LazyTool.from_function(calculator),
    LazyTool.from_function(current_time),
]
ai = LazyAgent("anthropic")
resp = ai.loop("What is 147 * 23, and what time is it?", tools=tools)
print(resp.content)
```

The agent will call both tools and combine the results.

## Tool guidance

Add extra instructions that the agent sees when the tool is available:

```python
tool = LazyTool.from_function(
    search,
    guidance="Use this tool when the user asks about current events. "
             "Always search before answering factual questions.",
)
```

Guidance is injected into the system prompt when the tool is present.

## Controlling tool execution

### max_steps

Limit how many tool-call rounds the loop runs:

```python
resp = ai.loop("research this topic", tools=tools, max_steps=3)
# Stops after 3 rounds even if the agent wants more
```

### tool_choice

Control whether the agent must use a tool:

```python
# Force at least one tool call
resp = ai.loop("hello", tools=tools, tool_choice="required")

# Prevent tool use
resp = ai.chat("hello", tools=tools, tool_choice="none")

# Force a specific tool
resp = ai.loop("calculate 2+2", tools=tools, tool_choice="calculator")
```

### on_event callback

Watch what happens during the loop:

```python
def on_event(event_name, payload):
    if event_name == "tool_call":
        print(f"  Calling: {payload.name}({payload.arguments})")
    elif event_name == "tool_result":
        print(f"  Result: {payload['result']}")

resp = ai.loop("What's 42 * 17?", tools=tools, on_event=on_event)
```

Events: `"step"`, `"tool_call"`, `"tool_result"`, `"done"`, `"verify_rejected"`

## Native provider tools

Some providers have built-in tools (web search, code execution). Use them without writing any code:

```python
from lazybridge import NativeTool

# Anthropic — web search
ai = LazyAgent("anthropic")
resp = ai.loop("What happened in tech news today?", native_tools=[NativeTool.WEB_SEARCH])

# OpenAI — code interpreter
ai = LazyAgent("openai")
resp = ai.loop("Calculate the fibonacci sequence to 20", native_tools=[NativeTool.CODE_EXECUTION])

# Google — grounded search
ai = LazyAgent("google")
resp = ai.loop("Latest research on fusion energy", native_tools=[NativeTool.WEB_SEARCH])
```

## Agents as tools

Wrap one agent as a tool for another:

```python
researcher = LazyAgent("anthropic", name="researcher", system="You are a research assistant.")
writer = LazyAgent("openai", name="writer")

research_tool = researcher.as_tool(
    name="research",
    description="Research a topic and return findings",
)

resp = writer.loop(
    "Write a blog post about quantum computing",
    tools=[research_tool],
)
```

The writer agent can call the researcher agent as a tool. We'll explore this more in Module 6.

---

## Exercise

1. Create a tool that looks up information in a dictionary (e.g., word definitions)
2. Create a second tool that counts the characters in a string
3. Give both tools to an agent and ask: "How many characters are in the definition of 'serendipity'?"
4. Add an `on_event` callback to see the tool calls happening

**Next:** [Module 3: Structured Output](03-structured-output.md) — get typed Pydantic objects back from LLMs.
