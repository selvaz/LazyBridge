# Human-in-the-loop: HumanEngine or SupervisorEngine?

```mermaid
flowchart TD
    A[Human in the loop?] --> B{Role}
    B -->|type answer / approve / fill form| C[HumanEngine]
    B -->|operator with tools and agent retry| D[SupervisorEngine]
    B -->|no human, LLM judge| E[verify equals judge_agent]
```

`HumanEngine` = one prompt, one answer. `SupervisorEngine` = full REPL
(continue / retry / store / tool commands). Use `input_fn=` in tests.
`verify=` is an automated LLM judge, not human-in-the-loop.
