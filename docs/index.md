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

## Choose your path

**New here?**
→ [Quickstart (5 min)](quickstart.md) → [Basic tier](tiers/basic.md) → [Decision trees](decisions/index.md)

**Building a real app?**
→ [Mid tier](tiers/mid.md) — [Memory](guides/memory.md) · [Session & tracing](guides/session.md) · [Guards](guides/guards.md) · [chain / parallel](guides/chain.md)

**Production pipeline?**
→ [Full tier](tiers/full.md) — [Plan](guides/plan.md) · [Checkpoint & resume](guides/checkpoint.md) · [Exporters](guides/exporters.md)

**Extending the framework?**
→ [Advanced tier](tiers/advanced.md) — [Engine protocol](guides/engine-protocol.md) · [BaseProvider](guides/base-provider.md)

## Top tasks

* [Tool calling end-to-end](recipes/tool-calling.md)
* [Structured output with Pydantic](recipes/structured-output.md)
* [Pipeline with typed steps and crash resume](recipes/plan-with-resume.md)
* [Human-in-the-loop: approval gates and REPL](recipes/human-in-the-loop.md)
* [Decision trees — "when to use which"](decisions/index.md)

## Reference

* [API reference](reference.md) · [Errors table](skill/99_errors.md) · [Claude Skill](skill/SKILL.md)
