# Sentinels (from_prev / from_start / from_step / from_parallel)

Sentinels are how Plan steps declare "where does my input come from?".
Without them you'd thread arguments manually at every step; with them,
the data flow is a 1-liner per step.

Four sentinels, four semantic slots:

* `from_prev` — the workhorse. In a straight chain, every step reads
  the one before it. This is the default.
* `from_start` — "I don't care what the previous step said; I want the
  original user task." Useful for verification steps ("does this draft
  answer the user's actual question?") or for branches that skip
  intermediate processing.
* `from_step("name")` — "I need step X specifically, even though it ran
  three steps ago." Sentinels validate against known step names at
  plan compile time; a typo is caught before any LLM call.
* `from_parallel("name")` — same mechanic as `from_step`, but reads
  better at the call site when the referenced step is a parallel
  branch.

Sentinels are also valid on `context=` — use them to inject prior
context into a step without overriding its task.

## Example

```python
from lazybridge import Plan, Step, from_prev, from_start, from_step

plan = Plan(
    Step(researcher,    name="research",  output=Hits),
    # Each step has an explicit task instruction; the upstream data
    # flows through ``context=`` (the idiomatic shape).
    Step(fact_checker,  name="check",
         task="Score each item for factual correctness; list any rejects.",
         context=from_prev),                                  # check researcher's output
    Step(writer,        name="write",
         task="Draft a 200-word answer to the user's task.",
         context=from_start),                                  # writer sees ORIGINAL user task
    Step(editor,        name="edit",
         task="Polish the draft; remove items the fact-checker flagged.",
         context=[from_step("write"), from_step("check")]),   # multi-source via list-context
)

# context= can carry MANY sources at once (no combiner step needed).
plan2 = Plan(
    Step(researcher,    name="research"),
    Step(policy_loader, name="policy"),
    Step(synthesiser,   name="synth",
         task="Draft a brief that cites both sources and follows the style note.",
         context=[
             from_step("research"),
             from_step("policy"),
             "Style: neutral, third person, no superlatives.",
         ]),
)
```

## Pitfalls

- ``from_prev`` after a parallel branch returns the join step's output,
  not one of the branches. Use ``from_parallel("<branch-name>")`` for a
  specific branch.
- Sentinels are module-level imports; don't shadow them with local
  variables of the same name.
- When passing a ``str`` as ``task=``, it's treated as a LITERAL, not a
  sentinel. Don't write ``task="from_prev"`` expecting the sentinel.

!!! note "API reference"

    from_prev                # singleton — previous step's output (default)
    from_start               # singleton — original user task
    from_step(name: str)     # named prior step's output
    from_parallel(name: str) # named parallel branch's output

    # Used on Step(..., task=<sentinel>) or Step(..., context=<sentinel>).

!!! warning "Rules & invariants"

    - ``from_prev`` (default): the previous step's output becomes the next
      step's task. This is real chain semantics — each step sees what its
      predecessor produced, not the original user input.
    - ``from_start``: explicit reference to the initial envelope. Use it
      when you want a step to operate on the original user request
      regardless of what preceded it.
    - ``from_step("n")``: reach back to a specific prior step's result.
      PlanCompiler verifies ``"n"`` names an earlier step, else raises.
    - ``from_parallel("n")``: alias for ``from_step`` intended for parallel
      branch joins. Indicates to readers that the step being referred to
      ran concurrently with siblings.
    - A plain string passed as ``task=`` is used verbatim — useful for
      hard-coded prompts at intermediate steps.
    - ``context=`` accepts a single sentinel/string OR a **list** of them.
      Each list item resolves independently; the parts join with
      blank-line separators in the step's ``Envelope.context``.  Mix
      sentinels with literal strings to inject fixed boilerplate alongside
      upstream data without an intermediate combiner step.

## See also

- [Plan](plan.md) — the engine that interprets sentinels.
- [Parallel plan steps](parallel-steps.md) — `from_parallel_all` aggregation.
- [Envelope](envelope.md) — the object sentinels carry between steps.
