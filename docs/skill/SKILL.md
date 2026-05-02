{
  "name": "lazybridge",
  "version": "1.0.0",
  "description": "LazyBridge agent framework — zero-boilerplate, tool-is-tool composition, compile-time-validated Plans. Tier-organised reference: load 00_overview first, then the tier you need.",
  "trigger": "User mentions LazyBridge, Agent, Envelope, Plan, PlanCompileError, Tool, NativeTool, SupervisorEngine, HumanEngine, LLMEngine, Step, from_prev/from_step/from_parallel, OTelExporter, MCP/MCPServer, or is writing Python code that imports from lazybridge or lazybridge.ext.*.",
  "entry": "00_overview.md",
  "files": [
    "00_overview.md",
    "01_basic.md",
    "02_mid.md",
    "03_full.md",
    "04_advanced.md",
    "05_decision_trees.md",
    "06_reference.md",
    "99_errors.md"
  ],
  "conventions": {
    "fragment_sections": [
      "signature — function/class signature(s)",
      "rules — invariants and constraints",
      "example — runnable code",
      "pitfalls — known gotchas",
      "see-also — cross-links"
    ],
    "extension_modules": [
      "lazybridge.ext.hil — HumanEngine, SupervisorEngine",
      "lazybridge.ext.mcp — Model Context Protocol integration",
      "lazybridge.ext.otel — OpenTelemetry exporter (GenAI conventions)",
      "lazybridge.ext.evals — EvalSuite, EvalCase, llm_judge",
      "lazybridge.ext.planners — make_planner, make_blackboard_planner (alpha)"
    ]
  }
}
