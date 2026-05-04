# LazyBridge

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
    guardrails, simple chain / parallel composition, MCP servers,
    basic HIL, evals.
    `Memory` · `Store` · `Session` · `Guards` · `chain` · `parallel`
    · `as_tool` · `MCP` · `HumanEngine` · `EvalSuite`

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

## Documentation tracks

LazyBridge maintains **two parallel documentation surfaces** so the
content fits the reader:

* **You're a human reading this site.** You're already in the right
  place — start with the [Quickstart](quickstart.md), pick a tier, or
  look up a [decision tree](decisions/index.md).
* **You're an LLM assistant.** Load
  [`SKILL.md`](skill/SKILL.md) — it's signature-first, dense, and
  predictably structured. Or fetch [`llms.txt`](https://github.com/selvaz/LazyBridge/blob/main/llms.txt)
  for an indexed pointer.

Both tracks render from the same source: fragments under
`lazybridge/skill_docs/fragments/` build into the skill **and** the
site via `python -m lazybridge.skill_docs._build`.

## Choose your path

**New here?**
→ [Quickstart (5 min)](quickstart.md) → [Basic tier](tiers/basic.md) → [Decision trees](decisions/index.md)

**Building a real app?**
→ [Mid tier](tiers/mid.md) — [Memory](guides/memory.md) · [Session & tracing](guides/session.md) · [Guards](guides/guards.md) · [chain / parallel](guides/chain.md) · [MCP](recipes/mcp.md)

**Production pipeline?**
→ [Full tier](tiers/full.md) — [Plan](guides/plan.md) · [Checkpoint & resume](guides/checkpoint.md) · [Exporters](guides/exporters.md)

**Shipping to production?**
→ [Operations checklist](guides/operations.md) — back-pressure · OTel GenAI · `timeout` / `cache` / `fallback` · resume policy · CI hardening

**Extending the framework?**
→ [Advanced tier](tiers/advanced.md) — [Engine protocol](guides/engine-protocol.md) · [BaseProvider](guides/base-provider.md)

## Top tasks

* [Tool calling end-to-end](recipes/tool-calling.md)
* [Structured output with Pydantic](recipes/structured-output.md)
* [Pipeline with typed steps and crash resume](recipes/plan-with-resume.md)
* [Parallel report pipeline (multi-agent → HTML/PDF/Reveal.js)](recipes/parallel-report.md)
* [Human-in-the-loop: approval gates and REPL](recipes/human-in-the-loop.md)
* [MCP integration](recipes/mcp.md)
* [Orchestration tools — chain / parallel / plan as tools](recipes/orchestration-tools.md)
* [Decision trees — "when to use which"](decisions/index.md)

## Reference

* [API reference](reference.md) · [Errors table](skill/99_errors.md) · [Claude Skill](skill/SKILL.md)
