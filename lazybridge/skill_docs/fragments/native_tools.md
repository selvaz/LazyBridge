## signature
from lazybridge import Agent, NativeTool

Agent("model", native_tools=[NativeTool.WEB_SEARCH, ...]) -> Agent

# Accepted values (NativeTool enum):
#   WEB_SEARCH       — web search (Anthropic, OpenAI, Google)
#   CODE_EXECUTION   — sandboxed Python/JS (Anthropic, OpenAI)
#   FILE_SEARCH      — file search over uploaded content (OpenAI)
#   COMPUTER_USE     — screen control (Anthropic)
#   GOOGLE_SEARCH    — Google grounded search (Google)
#   GOOGLE_MAPS      — Google Maps grounding (Google)

# String aliases also accepted:  native_tools=["web_search", "code_execution"]

## rules
- Native tools run server-side at the provider. You don't write or host
  them; you opt in by passing the enum.
- Not every tool is supported by every provider. Using
  ``NativeTool.GOOGLE_SEARCH`` on Anthropic raises at ``complete`` time.
  Match the tool to the model's provider.
- Native tools **coexist** with regular ``tools=[...]`` — the model may
  choose to call a native tool, one of your functions, both in the same
  turn, or neither.
- ``native_tools=`` is a shortcut on Agent equivalent to
  ``Agent(engine=LLMEngine(..., native_tools=[...]))``.
- Grounded responses from search tools expose sources via
  ``Envelope.metadata`` (``model``, ``provider``) and, where providers
  return them, via raw ``CompletionResponse.grounding_sources``.

## narrative
Native tools are the simplest kind of tool in LazyBridge — and often
the first ones you need. You don't write anything: just pass an enum
value and the provider does the work server-side.

Think of them as capabilities you unlock on a model, not as code you
ship. Want the model to search the web? `native_tools=[WEB_SEARCH]`.
Want it to execute Python in a sandbox? `native_tools=[CODE_EXECUTION]`.
No schema, no function, no loop.

Native tools compose with your own functions. A researcher agent can
have `native_tools=[WEB_SEARCH]` **and** `tools=[read_local_file]` at
the same time — the model picks which to call per step, and parallel
execution across them is automatic.

The tradeoff is portability: each native tool is provider-specific.
Web search works on Anthropic, OpenAI, and Google, but only
`GOOGLE_SEARCH` gives you Google's grounded results. If you need
provider-agnostic behaviour, write your own function and put it in
`tools=[...]`.

## example
```python
from lazybridge import Agent, NativeTool

# Web search is one line.
search = Agent("claude-opus-4-7", native_tools=[NativeTool.WEB_SEARCH])
print(search("what happened in AI news April 2026?").text())

# Native + custom tools coexist.
def read_report(path: str) -> str:
    """Read a local markdown file."""
    return open(path).read()

analyst = Agent(
    "claude-opus-4-7",
    native_tools=[NativeTool.WEB_SEARCH, NativeTool.CODE_EXECUTION],
    tools=[read_report],
)
analyst("cross-reference my report.md with current web consensus on the topic").text()

# Strings work too — equivalent to the enum.
Agent("gpt-4o", native_tools=["web_search"])("latest Python release?")
```

## pitfalls
- Mixing ``NativeTool.GOOGLE_SEARCH`` with an Anthropic model fails at
  provider time, not at Agent construction. Match the enum to the
  provider before you ship.
- Some native tools (``COMPUTER_USE``) require additional setup or beta
  flags on the provider's API. Check the provider's current docs.
- Cost: native tool calls are billed by the provider (search queries,
  code execution time). They appear in ``Envelope.metadata.cost_usd``
  when the provider reports them.

## see-also
[tool](tool.md), [agent](agent.md),
[core_types](core-types.md)
