# LazyBridge v1.0

Zero-boilerplate Python agent framework. One `Agent`, one `Envelope`,
one contract: **tool is tool**. Compose functions, Agents, and
Agents-of-Agents uniformly. Parallelism is automatic when the engine
decides; declared when you do.

## Two lines

```python
from lazybridge import Agent
print(Agent("claude-opus-4-7")("hello").text())
```

## Pick your tier

LazyBridge grows with you. Start low — every tier is additive, no
rewrite needed when you move up.

<div class="grid cards" markdown>

-   **[Basic →](tiers/basic.md)**

    One-shot or tool-calling agents. Functions-as-tools (auto-schema),
    native tools (web search, code execution), text or structured output.
    No memory, no pipeline.
    `Agent` · `Tool` · `NativeTool` · `Envelope`

-   **[Mid →](tiers/mid.md)**

    Realistic apps. Conversation memory, shared state, tracing,
    guardrails, simple chain / parallel composition, basic HIL, evals.
    `Memory` · `Store` · `Session` · `Guards` · `chain` · `parallel`
    · `as_tool` · `HumanEngine` · `EvalSuite`

-   **[Full →](tiers/full.md)**

    Production pipelines. Declared workflows with typed hand-offs,
    conditional routing, resume after crashes, OTel export, tool-level
    verifiers.
    `Plan` · `Step` · Sentinels · `SupervisorEngine` ·
    `checkpoint` · Exporters · `verify=`

-   **[Advanced →](tiers/advanced.md)**

    Framework extension. New providers, new engines, cross-process
    plan serialisation, `core.types`.
    `Engine` · `BaseProvider` · `Plan.to_dict` · `register_provider_*`

</div>

## Or skip straight to the tool you need

* [Quickstart — 5 minutes to first working agent](quickstart.md)
* [Decision trees — "when to use which"](decisions/index.md)
* [API reference — signature-first index](reference.md)
* [Errors — cause → fix table](skill/99_errors.md)
* [Claude Skill — for Claude Code / LLM assistants](skill/SKILL.md)

## What LazyBridge is for

If you want to build multi-agent systems in Python without picking a
side in the LangGraph / Pydantic-AI / CrewAI religious wars, and you
want your pipelines validated at construction time (not at the first
production failure), start with the **Full** tier's [Plan
guide](guides/plan.md).

If you just want to call an LLM with some tools and a Pydantic output
schema, stay at **Basic**. `Agent("claude-opus-4-7", output=MyModel)`
and you're done.
