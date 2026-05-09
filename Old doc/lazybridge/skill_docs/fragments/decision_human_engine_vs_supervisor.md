## question
Human-in-the-loop: HumanEngine or SupervisorEngine?

## tree
Simple "wait for input / approve / fill a form"?
    → HumanEngine                 # terminal prompt, optional Pydantic form

Interactive REPL where the human can call tools, retry agents
with feedback, and inspect the store?
    → SupervisorEngine            # commands: continue, retry <agent>: <fb>,
                                  #           store <key>, <tool>(<args>)

Automated (no human) verification at runtime?
    → verify=judge_agent          # NOT a HIL paradigm — it's an LLM judge

## tree_mermaid
flowchart TD
    A[Human in the loop?] --> B{Role}
    B -->|type answer / approve / fill form| C[HumanEngine]
    B -->|operator with tools and agent retry| D[SupervisorEngine]
    B -->|no human, LLM judge| E[verify equals judge_agent]

## notes
`HumanEngine` = one prompt, one answer. `SupervisorEngine` = full REPL
(continue / retry / store / tool commands). Use `input_fn=` in tests.
`verify=` is an automated LLM judge, not human-in-the-loop.
