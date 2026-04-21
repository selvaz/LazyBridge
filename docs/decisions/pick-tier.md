# Where do I start: Basic, Mid, Full, or Advanced?

```mermaid
flowchart TD
    A[What are you building?] -->|one-shot call| B[Basic]
    A -->|composing, memory, tracing| C[Mid]
    A -->|declared DAG, resume, verify| D[Full]
    A -->|extending the framework| E[Advanced]
    B --> B1[Agent / Tool / Envelope]
    C --> C1[Memory / Store / Session / Guards /<br/>chain / parallel / as_tool / HumanEngine / Evals]
    D --> D1[Plan / Sentinels / SupervisorEngine /<br/>checkpoint / exporters / verify=]
    E --> E1[Engine protocol / BaseProvider /<br/>Plan serialisation / core.types]
```

Start as low as possible. Moving up a tier is additive — no code you
wrote in Basic needs to change when you later add Memory (Mid) or wrap
the agent in a Plan (Full). Most users live in Basic and Mid; Full is
for production-grade declared pipelines; Advanced only applies if you're
writing framework code.
