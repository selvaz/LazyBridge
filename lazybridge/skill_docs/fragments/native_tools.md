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
**Use a native tool** when the provider already hosts the capability
(web search, code execution, file search) and your data doesn't need to
leave the provider's environment.  No code, no schema — pass an enum.

**Don't use a native tool** when you need full control over the
implementation, custom auth, or you want to swap providers without
re-testing the tool surface.  Write a regular `Tool` instead.

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
- [Tool](tool.md) — write your own when native isn't a fit.
- [Providers](providers.md) — which native tools each provider supports.
