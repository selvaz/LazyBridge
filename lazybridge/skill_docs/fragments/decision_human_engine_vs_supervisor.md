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
`HumanEngine` is the minimum: one prompt, one answer. Good for
approvals, reviewers, light annotation, Pydantic forms. Blocks on
`input()` or a custom UI adapter.

`SupervisorEngine` is a full REPL. The human sees the previous output
and can:

* `continue` — accept and return to the pipeline,
* `retry <agent>: <feedback>` — re-run a registered agent with feedback,
* `store <key>` — inspect the shared Store,
* `<tool>(<args>)` — invoke a registered tool directly.

Use it when the human is an operator in the pipeline, not just a
gate. Pass `input_fn=_scripted(...)` in tests to keep the loop
non-interactive.

`verify=` is **not** HIL at all — the judge is an Agent, automated.
Listed here because users often conflate "verification" with "human
review".
