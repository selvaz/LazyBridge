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

Below is a four-step plan that uses three of the four sentinels.  Read
it top-to-bottom: every step's data flow is declared in one line, and
the plan compiler checks at `Agent(...)` construction time that every
referenced step actually exists (a typo in `from_step("wirter")` is
caught before a single token is spent).

```python
from lazybridge import Plan, Step, from_prev, from_start, from_step

plan = Plan(
    # 1. researcher sees the user's original task (there's nothing
    #    before it). output=Hits switches the step into structured
    #    output so downstream steps can read fields, not re-parse text.
    Step(researcher,    name="research",  output=Hits),

    # 2. from_prev is the default; spelled out here for clarity.
    #    The fact_checker receives the researcher's Envelope.payload
    #    (the Hits instance, rendered to text at the tool boundary).
    Step(fact_checker,  name="check",     task=from_prev),

    # 3. from_start skips over the intermediate steps and hands the
    #    writer the ORIGINAL user task. Useful when a later step
    #    should answer the user's question directly rather than
    #    transform an upstream intermediate.
    Step(writer,        name="write",     task=from_start),

    # 4. Reach back to named steps: the editor's TASK is the writer's
    #    output, and its CONTEXT is the fact_checker's output. Two
    #    independent sentinels can compose on a single step.
    Step(editor,        name="edit",      task=from_step("write"),
                                          context=from_step("check")),
)
```

What you just declared: "research → check, write from the original
task, then edit the write output using the check output as context".
The plan compiler validates that `"write"` and `"check"` both exist
and precede `"edit"` — any dangling reference fails loud at
`Agent(engine=plan)` time with a `PlanCompileError`.

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

## See also

[plan](plan.md), [parallel_steps](parallel-steps.md),
decision tree: [composition](../decisions/composition.md)
