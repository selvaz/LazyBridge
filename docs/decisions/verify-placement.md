# verify= at Agent level, tool level, or Plan step level?

```mermaid
flowchart TD
    A[Where does the judge sit?] --> B{Scope}
    B -->|one agent, final output| C[Agent verify equals judge]
    B -->|specific sub-agent when used as tool| D[as_tool verify equals judge]
    B -->|specific Plan step| E[Step with agent verify equals judge]
    B -->|every tool invocation filter| F[not verify, use a Guard]
```

`verify=` retries with judge feedback. Use agent-level for broad
output gates, tool-level (`as_tool(verify=...)`) when one sub-agent
is risky, Plan step-level when one step needs a gate. For filtering
every tool invocation, use a Guard instead — `verify=` gates output,
not individual calls.
