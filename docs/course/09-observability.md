# Module 9: Observability & Cost

Track what your agents are doing, how many tokens they use, and how much they cost.

## Event tracking

Every LazySession records events automatically:

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession(tracking="verbose")
ai = LazyAgent("anthropic", name="assistant", session=sess)
ai.chat("What is 2+2?")
ai.chat("And 3+3?")

for event in sess.events.get():
    print(f"{event['event_type']:20s} {event.get('agent_name', '')}")
# model_request        assistant
# model_response       assistant
# model_request        assistant
# model_response       assistant
```

### Tracking levels

```python
LazySession(tracking="off")      # no events
LazySession(tracking="basic")    # model_request, tool_call, agent_start/finish
LazySession(tracking="verbose")  # + model_response, messages, system_context
```

## Usage summary

Aggregate tokens and costs across all agents:

```python
sess = LazySession(tracking="verbose")
researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer = LazyAgent("openai", name="writer", session=sess)

researcher.chat("Find AI news")
writer.chat("Write a summary")

summary = sess.usage_summary()
print(f"Total cost: ${summary['total']['cost_usd']:.4f}")
print(f"Input tokens: {summary['total']['input_tokens']}")
print(f"Output tokens: {summary['total']['output_tokens']}")

for name, usage in summary["by_agent"].items():
    print(f"  {name}: ${usage['cost_usd']:.4f} ({usage['input_tokens']} in, {usage['output_tokens']} out)")
```

**Note:** `usage_summary()` requires `tracking="verbose"` because `model_response` events (which contain token counts) are only emitted at verbose level.

## Exporters

Forward events to external systems:

### Callback exporter — simplest

```python
from lazybridge import CallbackExporter

events = []
sess = LazySession(exporters=[CallbackExporter(events.append)])
# events list fills up as agents run
```

### Filtered exporter — only specific events

```python
from lazybridge import FilteredExporter, CallbackExporter

tool_only = FilteredExporter(
    CallbackExporter(print),
    event_types={"tool_call", "tool_result"},
)
sess = LazySession(exporters=[tool_only])
```

### JSON file exporter — append to file

```python
from lazybridge import JsonFileExporter

with JsonFileExporter("events.jsonl") as exporter:
    sess = LazySession(exporters=[exporter])
    # run agents...
# File contains one JSON object per line
```

### Structured log exporter — stdlib logging

```python
from lazybridge import StructuredLogExporter
import logging

logging.basicConfig(level=logging.INFO)
sess = LazySession(exporters=[StructuredLogExporter()])
# Events appear as structured JSON in your log output
```

### OpenTelemetry exporter

```bash
pip install lazybridge[otel]
```

```python
from lazybridge import OTelExporter

sess = LazySession(exporters=[OTelExporter(service_name="my-pipeline")])
# Events become OpenTelemetry spans:
#   agent_start/finish → agent span
#   model_request/response → LLM span with token attributes
#   tool_call/result → tool span
```

Spans include attributes:

- `llm.model`, `llm.input_tokens`, `llm.output_tokens`, `llm.cost_usd`
- `lazybridge.agent.name`, `lazybridge.tool.name`

Connect to Jaeger, Grafana Tempo, Datadog, or any OTEL-compatible backend.

## Console output (verbose mode)

Quick debugging — print events to console:

```python
# On the session
sess = LazySession(tracking="basic", console=True)

# Or on a single agent
ai = LazyAgent("anthropic", verbose=True)
ai.chat("Hello")
# Prints: timestamp agent_name >> model_request model=claude-sonnet-4-6
# Prints: timestamp agent_name << model_response tokens=12/45
```

## Adding/removing exporters at runtime

```python
exporter = CallbackExporter(my_handler)
sess.add_exporter(exporter)
# ... run agents ...
sess.remove_exporter(exporter)
```

## SQLite persistence

Use a database for durable event storage:

```python
sess = LazySession(db="pipeline.db", tracking="verbose")
# Events are stored in SQLite — survive process restarts

# Later, query them
events = sess.events.get(event_type="tool_call", limit=50)
```

---

## Exercise

1. Create a session with verbose tracking and two agents
2. Run both agents and print the usage summary
3. Add a JsonFileExporter and inspect the output file
4. If you have OTEL set up, try the OTelExporter with a Jaeger instance

**Next:** [Module 10: Evals & Testing](10-evals.md) — systematically test agent quality.
