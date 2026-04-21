## question
verify= at Agent level, tool level, or Plan step level?

## tree
Gating the final output of a single agent?
    → Agent("model", verify=judge, max_verify=3)

One sub-agent inside a tool list is the risky one;
rest of the run is fine?
    → parent.as_tool(verify=judge, max_verify=2)   # Option B

One step of a declared Plan needs a judge; other steps don't?
    → Step(Agent(..., verify=judge), ...)

Want a judge on every tool call emitted by the model?
    → This isn't what verify= does — it's LLM-as-judge on output.
      For call-time filtering use Guards (guards.md).

## tree_mermaid
flowchart TD
    A[Where does the judge sit?] --> B{Scope}
    B -->|one agent, final output| C[Agent verify equals judge]
    B -->|specific sub-agent when used as tool| D[as_tool verify equals judge]
    B -->|specific Plan step| E[Step with agent verify equals judge]
    B -->|every tool invocation filter| F[not verify, use a Guard]

## notes
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
