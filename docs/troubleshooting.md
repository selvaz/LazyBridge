# Troubleshooting

Common errors and how to fix them.

---

## API key not found

```
ValueError: AnthropicProvider requires an API key.
Set the ANTHROPIC_API_KEY environment variable, or pass api_key= to LazyAgent/AnthropicProvider.
```

**Fix:** Set the environment variable for your provider:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

Or pass the key directly:

```python
ai = LazyAgent("anthropic", api_key="sk-ant-...")
```

---

## Stream return type confusion

```python
# This returns a UNION type — hard to use
resp = agent.chat("hello", stream=True)  # CompletionResponse | Iterator[StreamChunk]
```

**Fix:** Use the dedicated streaming methods instead:

```python
# Clear return type — always Iterator[StreamChunk]
for chunk in agent.chat_stream("hello"):
    print(chunk.delta, end="")

# Async
async for chunk in await agent.achat_stream("hello"):
    print(chunk.delta, end="")
```

---

## `_last_output` is None after pipeline

When you use `LazyTool.chain()` or `LazyTool.parallel()`, the pipeline builders **clone** agents internally. The original agent's `_last_output` won't be updated.

**Fix:** Read the return value of `tool.run()` instead:

```python
tool = LazyTool.chain(researcher, writer, name="pipe", description="d")
result = tool.run({"task": "analyze trends"})  # <-- use this
print(result)

# NOT: print(researcher._last_output)  # may be None
```

For the canonical accessor on a standalone agent, use `agent.result`:

```python
agent.chat("hello")
print(agent.result)  # typed if output_schema was set, text otherwise
```

---

## StructuredOutputParseError vs StructuredOutputValidationError

When using `agent.json()` or `output_schema=`, two error types can occur:

| Error | Meaning | Fix |
|-------|---------|-----|
| `StructuredOutputParseError` | Model returned invalid JSON | Retry, or simplify your schema |
| `StructuredOutputValidationError` | JSON is valid but doesn't match your Pydantic model | Check field types/names in your schema |

```python
from lazybridge import LazyAgent
from lazybridge.core.types import StructuredOutputParseError, StructuredOutputValidationError

try:
    result = agent.json("generate data", MyModel)
except StructuredOutputParseError as e:
    print(f"Invalid JSON: {e.raw}")
except StructuredOutputValidationError as e:
    print(f"Wrong shape: {e}")
```

Both inherit from `StructuredOutputError`, so you can catch either specifically or broadly.

---

## Tool schema errors

### Lambda functions

```
ToolSchemaBuildError: Cannot introspect lambda — use a named function
```

**Fix:** Tool schema generation needs type hints and a docstring. Lambdas have neither.

```python
# Bad
tool = LazyTool.from_function(lambda x: x * 2)

# Good
def double(x: int) -> int:
    """Double a number."""
    return x * 2
tool = LazyTool.from_function(double)
```

### Missing type hints

```
ToolSchemaBuildError: Parameter 'query' has no type annotation
```

**Fix:** Add type hints to all parameters:

```python
# Bad
def search(query):
    ...

# Good
def search(query: str) -> str:
    """Search the web."""
    ...
```

---

## Memory + stream incompatibility

```
TypeError: stream=True is not compatible with memory=.
Consume the stream manually and call memory._record() yourself.
```

The `Memory` object needs the full response text to record history. Streaming delivers text incrementally, so they can't be combined automatically.

**Fix:** Either drop streaming or manage history manually:

```python
# Option 1: no streaming (simplest)
resp = agent.chat("hello", memory=mem)

# Option 2: stream without Memory, manage history yourself
chunks = list(agent.chat_stream("hello"))
full_text = "".join(c.delta for c in chunks if c.delta)
```

---

## usage_summary() returns empty

`usage_summary()` aggregates `model_response` events, which are only emitted at **verbose** tracking level.

**Fix:**

```python
# Use verbose tracking
sess = LazySession(tracking="verbose")
```

---

## Checkpoint not resuming

Possible causes:

1. **No store configured:** Checkpoints require `store=` on the chain:
   ```python
   tool = LazyTool.chain(a, b, store=my_store, chain_id="pipe")
   ```

2. **Different chain_id:** The resume key is `_ckpt:{chain_id}`. If `chain_id` changed, the old checkpoint won't be found.

3. **Chain length changed:** If you add/remove agents from the chain after a checkpoint was saved, the saved step may be out of range. LazyBridge detects this and restarts from step 0 with a warning.

4. **Concurrent collision:** Two runs with the same `chain_id` overwrite each other's checkpoints. Use `run_id` for isolation:
   ```python
   tool = LazyTool.chain(a, b, store=store, chain_id="pipe", run_id="run-1")
   ```
