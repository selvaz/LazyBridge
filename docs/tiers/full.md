# Full tier

**Use this when** you need a declared, multi-step pipeline: typed hand-offs, conditional routing, crash recovery via checkpoint/resume, or OTel/JSON observability.

**Stay at Mid** if your pipeline is a straight line with no typed models between steps and you don't need resume semantics.

## Topics

* [Plan](../guides/plan.md)
* [Sentinels (from_prev / from_start / from_step / from_parallel)](../guides/sentinels.md)
* [Parallel plan steps](../guides/parallel-steps.md)
* [SupervisorEngine](../guides/supervisor.md)
* [Checkpoint & resume](../guides/checkpoint.md)
* [Exporters](../guides/exporters.md)
* [GraphSchema](../guides/graph-schema.md)
* [verify=](../guides/verify.md)

## Next steps

* Recipe: [Plan with typed steps and crash resume](../recipes/plan-with-resume.md)
* [Advanced tier →](advanced.md) if you need a custom provider or engine
