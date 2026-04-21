# verify= at Agent level, tool level, or Plan step level?

```mermaid
flowchart TD
    A[Where does the judge sit?] --> B{Scope}
    B -->|one agent, final output| C[Agent verify equals judge]
    B -->|specific sub-agent when used as tool| D[as_tool verify equals judge]
    B -->|specific Plan step| E[Step with agent verify equals judge]
    B -->|every tool invocation filter| F[not verify, use a Guard]
```

`verify=` is LLM-as-judge on an agent's output; it retries with
feedback. Three placements because three scopes:

* **Agent-level** — broadest. Use when you don't trust an agent's
  final output by default.
* **Tool-level (Option B)** — surgical. Use when one sub-agent is
  risky and the rest of the run is fine. Put the judge on the
  `as_tool(...)` wrapper.
* **Plan step-level** — same mechanism, scoped to one step in a
  declared workflow.

If you need to gate *every tool call* (e.g. "block any search with
PII"), `verify=` is the wrong tool — that's a **Guard**
(`GuardChain`, `ContentGuard`, `LLMGuard`), which intercepts call
inputs and outputs directly.
