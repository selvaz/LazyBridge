# Full tier

**Use this when** you need a declared, multi-step pipeline: typed hand-offs, conditional routing, crash recovery via checkpoint/resume, or OTel/JSON observability.

**Stay at Mid** if your pipeline is a straight line with no typed models between steps and you don't need resume semantics.

## Walkthrough

Full introduces **declared pipelines**.  A `Plan` is a list of
`Step`s validated at construction time — `PlanCompileError` fires
before any LLM call on a misspelled step name, a forward
`from_step`, or a parallel-band misuse.

**Sentinels** (`from_prev`, `from_start`, `from_step("name")`,
`from_parallel`, `from_parallel_all`) are how each step declares
where its input and context come from — `task=` for the prompt
instruction, `context=` for the data (which can be a list of
sentinels for multi-source synthesis without a combiner step).

`parallel=True` marks a step as a member of a concurrent band.
`writes="key"` persists the step's payload to the `Store` so a
downstream agent (or a future resume) can read it.  Pair `Plan`
with `Store` + `checkpoint_key=` + `resume=True` to get
CAS-protected crash-resume; pair with `on_concurrent="fork"` for
fan-out workflows.

`SupervisorEngine` is the HIL counterpart: a step that hands
control to a human REPL.  `verify=` adds a judge/retry loop at
either agent-level or tool-level.  `Exporters` and `GraphSchema`
surface the run as OTel spans + a topology graph for dashboards.

Before shipping, read the
[Operations checklist](../guides/operations.md): back-pressure,
OTel GenAI conventions, ``Memory.summarizer_timeout``, MCP cache
TTL, and the production knobs on `Agent` (`timeout`, `cache`,
`fallback`).

## Topics

* [Plan](../guides/plan.md)
* [Sentinels (from_prev / from_start / from_step / from_parallel)](../guides/sentinels.md)
* [Parallel plan steps](../guides/parallel-steps.md)
* [SupervisorEngine (alpha — ext.hil)](../guides/supervisor.md)
* [Checkpoint & resume](../guides/checkpoint.md)
* [Exporters](../guides/exporters.md)
* [GraphSchema](../guides/graph-schema.md)
* [verify=](../guides/verify.md)

## Also see

* [Operations checklist](../guides/operations.md) — production deployment knobs

## Next steps

* Recipe: [Plan with typed steps and crash resume](../recipes/plan-with-resume.md)
* **Production**: [Operations checklist](../guides/operations.md) — back-pressure, OTel GenAI, resume, reliability knobs
* [Advanced tier →](advanced.md) if you need a custom provider or engine

**By the end of Full you can:**

- declare a typed multi-step pipeline that fails-fast at construction (`PlanCompileError`)
- route conditionally via `output.next: Literal[...]`
- run parallel bands and aggregate them via `from_parallel_all` or `context=[...]`
- resume from the failed step with CAS-protected checkpoints
- run N copies of the same pipeline concurrently with `on_concurrent="fork"`
- swap in a `SupervisorEngine` to give a human a tool-calling REPL mid-pipeline
- add a `verify=` judge at agent-level or tool-level
