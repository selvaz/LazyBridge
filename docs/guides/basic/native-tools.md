# Native tools

Provider-hosted capabilities you turn on by passing an enum, not by
writing code. Web search, code execution, file search, and a few
others run server-side at the LLM provider — the model decides when
to invoke them just like any other tool, but your application
doesn't host or implement them.

## Signature

```python
from lazybridge import Agent, LLMEngine, NativeTool, Tool

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    native_tools=[NativeTool.WEB_SEARCH, NativeTool.CODE_EXECUTION],
    tools=[Tool.wrap(my_function, name="my_function")],   # native + custom coexist
)
```

`NativeTool` is a `StrEnum` — string aliases also work:
`native_tools=["web_search", "code_execution"]`.

### Available values

| Value | Provider(s) | What it does |
|---|---|---|
| `NativeTool.WEB_SEARCH` | Anthropic, OpenAI, Google | General web search |
| `NativeTool.CODE_EXECUTION` | Anthropic, OpenAI | Sandboxed Python / JavaScript execution |
| `NativeTool.FILE_SEARCH` | OpenAI | Search across uploaded files |
| `NativeTool.COMPUTER_USE` | Anthropic | Screen control (beta) |
| `NativeTool.IMAGE_GENERATION` | OpenAI Responses API | Inline image generation (gpt-image-2 family) |
| `NativeTool.GOOGLE_SEARCH` | Google | Gemini grounded search |
| `NativeTool.GOOGLE_MAPS` | Google | Gemini Maps grounding |

The `native_tools=` argument is a shortcut on `Agent` equivalent to
`Agent(engine=LLMEngine(..., native_tools=[...]))` — the engine is
where the values are actually consumed; `Agent` forwards them through.

## Synopsis

Native tools complete the "everything is a tool" picture from the
provider side. They behave identically to your own `tools=[...]` from
the agent's perspective — the model emits a tool call, the framework
routes it, and the result returns to the loop. The difference is who
runs the implementation: with a regular `Tool`, your code runs; with
a `NativeTool`, the provider's infrastructure runs.

You can mix freely. The model may use a native tool in one turn and
your custom function in the next, or both in parallel within a single
turn (engines emit parallel tool calls automatically).

## When to use native tools

- **The provider already hosts the capability.** Web search, code
  execution, file search — implementing these yourself is more work
  than passing an enum.
- **Your data doesn't need to leave the provider's environment.**
  Native tools execute server-side at the provider; if that's
  acceptable, opt in.
- **You want grounded answers** with citations the provider returns
  natively (web search, Google grounding). Provider-side grounding
  often produces better source attribution than rolling your own.
- **You want minimal supply chain.** No additional dependencies, no
  auth secrets to manage, no infrastructure to keep up.

## When NOT to use native tools

- **You need full control of the implementation** — custom auth
  headers, rate-limit handling, response post-processing, mocking
  for tests. Write a regular `Tool` instead.
- **You want provider portability.** Native tools tie the agent to
  a provider that supports the capability. A custom `Tool` runs the
  same against any provider.
- **The provider doesn't support the tool.** `NativeTool.GOOGLE_SEARCH`
  on Anthropic raises at the provider layer (see Pitfalls).
- **You need offline / air-gapped operation.** Native tools call out
  to provider infrastructure; if the agent must work without that,
  don't use them.

## Example

```python
from lazybridge import Agent, LLMEngine, NativeTool


# 1) Web search — one line of opt-in.
search_agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    native_tools=[NativeTool.WEB_SEARCH],
)
result = search_agent("what happened in AI news in April 2026?")
print(result.text())


# 2) Native + custom tools coexist freely.
def read_report(path: str) -> str:
    """Read and return the contents of a local markdown file."""
    return open(path).read()

analyst = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    native_tools=[NativeTool.WEB_SEARCH, NativeTool.CODE_EXECUTION],
    tools=[read_report],
    allow_dangerous_native_tools=True,   # required opt-in for CODE_EXECUTION
)
analyst("cross-reference report.md against current web consensus")


# 3) String aliases — same effect as the enum.
gpt_search = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    native_tools=["web_search"],
)
gpt_search("latest stable Python release?")


# 4) Provider-specific tools — match the model.
gemini_grounded = Agent(
    engine=LLMEngine("gemini-2.5-pro"),
    native_tools=[NativeTool.GOOGLE_SEARCH],
)
```

## Security gate: `allow_dangerous_native_tools`

`NativeTool.CODE_EXECUTION` and `NativeTool.COMPUTER_USE` give the
provider broad access — sandboxed code execution at the provider's
side, screen control on the user's machine. Both require explicit
opt-in via `allow_dangerous_native_tools=True` on either the `Agent`
or the `LLMEngine`:

```python
agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    native_tools=[NativeTool.CODE_EXECUTION],
    allow_dangerous_native_tools=True,   # without this: ValueError at construction
)
```

The gate runs at **each construction site** that introduces native
tools — `LLMEngine(native_tools=...)` validates against its own
`allow_dangerous_native_tools=`, and `Agent(native_tools=...)` (whose
list is merged into the engine) validates against the Agent's flag.
Specify your native tools at the construction site that owns the
flag; an already-configured `engine.native_tools` on a pre-built
engine is **not** re-checked when you wrap it in `Agent(engine=...)`,
so don't rely on `Agent(allow_dangerous_native_tools=False)` to
"undo" a permissive engine. `WEB_SEARCH`, `FILE_SEARCH`,
`IMAGE_GENERATION`, `GOOGLE_SEARCH`, `GOOGLE_MAPS` are NOT gated by
this flag — only the two genuinely dangerous tools.

The default (`allow_dangerous_native_tools=False`) raises
`ValueError` at construction with a message naming the offending
native tool. Catch it explicitly if you want to fall back to
non-dangerous alternatives.

## Pitfalls

- **Provider / tool mismatch fails at run time, not construction.**
  Mixing `NativeTool.GOOGLE_SEARCH` with an Anthropic model raises
  at the `complete` call — Agent construction is happy. Match the
  enum to the provider before you ship; cover this with one
  integration test per provider you support.
- **`COMPUTER_USE` requires extra setup** — the Anthropic API needs
  the right beta flag and additional permissions, and the model
  needs the resolution / tool definitions you're targeting. Read
  the provider's current docs before turning it on.
- **Billing.** Native tool calls are billed by the provider —
  search queries, code-execution time, image generation. Where the
  provider reports usage, the cost shows up in
  `Envelope.metadata.cost_usd` alongside the model's own tokens;
  where it doesn't, you may need to consult the provider dashboard
  for full attribution.
- **`UnsupportedNativeToolError`** is raised at provider time when
  the provider's native-tool support is incomplete or strict mode
  is on. Catch it explicitly when you want a graceful fallback to
  a custom `Tool`.
- **Native tools don't appear in your code path.** `tools=[search]`
  shows up in stack traces and event logs as `search`; native tools
  show up as `web_search` (or whatever alias the provider uses) and
  the implementation lives off-process. Don't expect to set a
  breakpoint on a native tool's execution.
- **Grounded responses expose sources via the raw provider
  payload.** When you need attribution, read
  `CompletionResponse.grounding_sources` (when the provider returns
  them) rather than parsing them out of the model's text.

## See also

- [Tool](tool.md) — write your own when a native tool is not a fit.
- [Agent](agent.md) — `native_tools=` is the kwarg that activates
  these; everything else still composes the same way.
- [Envelope](envelope.md) — provider-reported costs land in
  `metadata.cost_usd` together with model-call costs.
- *Guides → Full → Providers* (coming in Phase 3) — which native
  tools each provider supports and the strict-mode behaviour.
