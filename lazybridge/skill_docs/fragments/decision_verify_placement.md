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
`verify=` retries with judge feedback. Use agent-level for broad
output gates, tool-level (`as_tool(verify=...)`) when one sub-agent
is risky, Plan step-level when one step needs a gate. For filtering
every tool invocation, use a Guard instead — `verify=` gates output,
not individual calls.
