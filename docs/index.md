---
hide:
  - toc
title: LazyBridge Docs
---

<div class="lb-page">

<!-- ═══ DOCS HERO ══════════════════════════════════════════════════════════ -->
<section class="lb-docs-hero">
  <div class="lb-docs-hero__copy">
    <div class="lb-pill">0.8.0 Alpha &middot; Apache-2.0</div>
    <h1>LazyBridge <span class="accent">documentation</span></h1>
    <p class="lb-subhead">
      Zero-boilerplate multi-provider LLM agent framework.
      Engine + Tools + State — everything is a tool.
    </p>
    <div class="lb-cta-row">
      <a href="quickstart/" class="lb-btn lb-btn--primary">Quickstart &rarr;</a>
      <a href="why/" class="lb-btn lb-btn--ghost">Why LazyBridge</a>
    </div>
  </div>
</section>

<!-- ═══ NAV CARDS ══════════════════════════════════════════════════════════ -->
<section class="lb-nav-cards">
  <a href="quickstart/" class="lb-nav-card">
    <div class="lb-nav-card__icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
    </div>
    <div>
      <strong>Quickstart</strong>
      <p>First agent in 5 minutes</p>
    </div>
  </a>
  <a href="concepts/mental-model/" class="lb-nav-card">
    <div class="lb-nav-card__icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4m0-4h.01"/></svg>
    </div>
    <div>
      <strong>Concepts</strong>
      <p>Mental model, composition, progressive complexity</p>
    </div>
  </a>
  <a href="guides/basic/agent/" class="lb-nav-card">
    <div class="lb-nav-card__icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
    </div>
    <div>
      <strong>Guides</strong>
      <p>Agent, Tool, Plan, Session, and more</p>
    </div>
  </a>
  <a href="recipes/" class="lb-nav-card">
    <div class="lb-nav-card__icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 10c-.83 0-1.5-.67-1.5-1.5v-5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5z"/><path d="M20.5 10H19V8.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/><path d="M9.5 14c.83 0 1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5S8 21.33 8 20.5v-5c0-.83.67-1.5 1.5-1.5z"/><path d="M3.5 14H5v1.5c0 .83-.67 1.5-1.5 1.5S2 16.33 2 15.5 2.67 14 3.5 14z"/><path d="M14 14.5c0-.83.67-1.5 1.5-1.5h5c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5h-5c-.83 0-1.5-.67-1.5-1.5z"/><path d="M15.5 19H14v1.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5-.67-1.5-1.5-1.5z"/><path d="M10 9.5C10 8.67 9.33 8 8.5 8h-5C2.67 8 2 8.67 2 9.5S2.67 11 3.5 11h5c.83 0 1.5-.67 1.5-1.5z"/><path d="M8.5 5H10V3.5C10 2.67 9.33 2 8.5 2S7 2.67 7 3.5 7.67 5 8.5 5z"/></svg>
    </div>
    <div>
      <strong>Recipes</strong>
      <p>Copy-paste patterns for common workflows</p>
    </div>
  </a>
  <a href="reference/" class="lb-nav-card">
    <div class="lb-nav-card__icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
    </div>
    <div>
      <strong>Reference</strong>
      <p>Full API — Agent, Tool, Plan, Envelope</p>
    </div>
  </a>
  <a href="decisions/" class="lb-nav-card">
    <div class="lb-nav-card__icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
    </div>
    <div>
      <strong>Decisions</strong>
      <p>Pick tier, composition style, state layer</p>
    </div>
  </a>
</section>

<!-- ═══ CODE EXAMPLE ═══════════════════════════════════════════════════════ -->
<div class="lb-code-header lb-code-header--standalone">
  <span class="lb-code-caption">Simple stays simple. Complex is possible without changing the mental model.</span>
  <span class="lb-code-lang">Python</span>
</div>

<div class="lb-code-tabs">

=== "Single agent"

    ```python
    from lazybridge import Agent, LLMEngine

    agent = Agent(engine=LLMEngine("claude-sonnet-4-6"))
    print(agent("Explain LazyBridge in one sentence.").text())
    ```

=== "With tools"

    ```python
    from lazybridge import Agent, LLMEngine, Tool

    agent = Agent(
        engine=LLMEngine("claude-sonnet-4-6"),
        tools=[Tool.wrap(get_weather, name="get_weather")],
    )
    print(agent("What's the weather in Paris?").text())
    ```

=== "Multi-agent pipeline"

    ```python
    from lazybridge import Agent, LLMEngine, Plan, Step, from_step

    search    = Agent(engine=LLMEngine("gpt-5.4-mini"),      name="search")
    summarise = Agent(engine=LLMEngine("gemini-2.5-pro"),    name="summarise")
    writer    = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="write")

    research = Agent(
        engine=Plan(Step("search"), Step("summarise")),
        tools=[search, summarise], name="research",
    )
    article = Agent(
        engine=Plan(Step("research"),
                    Step("write", context=from_step("research"))),
        tools=[research, writer],
    )
    print(article("AI agents in 2026").text())
    ```

</div>

</div>

---

??? note "Maturity — 0.8.0 (Alpha)"

    LazyBridge 0.8.0 is on PyPI as **Alpha** (`lazybridge.__stability__ = "alpha"`).
    Breaking changes go through [migration guides](migrations/0.7-to-0.79.md).

    | Subsystem | Status | Notes |
    |---|---|---|
    | `Agent`, `LLMEngine`, `Tool`, `Envelope` | **Stable** | Public surface, exercised by every test path. |
    | `Plan`, `Step`, sentinels, routing | **Stable** | Compiler validates at construction; serialisation supported. |
    | `Memory`, `Store` (in-memory + SQLite) | **Stable** | API frozen; encrypted store adapter is also stable. |
    | `Session`, `EventLog`, exporters, `GraphSchema` | **Stable** | Default secret redaction enabled. |
    | Provider adapters (Anthropic / OpenAI / Google / DeepSeek / LiteLLM / LM Studio) | **Stable** | Adapters stable; model/price tables drift with providers. |
    | MCP / external tool gateway | **Moved** | Migrated to `lazytoolkit` in 0.8 — see [tools.lazybridge.com](https://tools.lazybridge.com). |
    | Native tools (`NativeTool`) | **Alpha** | Surface area changes when providers add new tools. |
    | `Checkpoint` / `resume` | **Alpha** | Atomic across parallel bands; external side-effect rollback not implemented. |
    | Guardrails (`Guard`, `ContentGuard`, `LLMGuard`, `GuardChain`) | **Alpha** | Behaviour stable; default rule libraries growing. |
    | `HumanEngine`, `SupervisorEngine` | **Alpha** | Public API stable; UX polish continues. |
    | Evals (`lazybridge.ext.evals`) | **Experimental** | Runner API may consolidate before 1.0. |
    | Visualizer (`lazybridge.ext.viz`) | **Experimental** | Useful for debugging; not on the runtime path. |
    | Provider model fallback chains | **Planned** | Data tables exist; retry path not yet implemented. |
    | Automatic PII redaction | **Planned** | Default redactor masks credential shapes only. |
