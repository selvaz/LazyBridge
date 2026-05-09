# LiteLLM bridge

`LiteLLMProvider` is the catch-all provider that extends LazyBridge's
native lineup (Anthropic, OpenAI, Google, DeepSeek) with LiteLLM's
100+ model catalog — Mistral, Cohere, Groq, Bedrock, Vertex AI,
Ollama, HuggingFace, Together, Fireworks, xAI, local servers, and the
rest.  You opt in **per model string** via the `litellm/` prefix, so
your existing `Agent("claude-opus-4-7")` and `Agent("gpt-5")` calls
keep routing to the native providers.

Install:

```bash
pip install "lazybridge[litellm]"
```

Set the API key the target provider expects as an env var — LiteLLM
reads them by name (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GROQ_API_KEY`, `MISTRAL_API_KEY`, etc.).  One env var per provider
you route to.

## Quick start

```python
# What this shows: three different LiteLLM-backed providers through
# a single LazyBridge call surface. The model string follows LiteLLM's
# native "provider/model" syntax, prefixed with "litellm/" so
# LazyBridge's router sends it through the bridge instead of trying
# to infer the provider from the raw name.
# Why the prefix: without it, something like "mistral-large-latest"
# would not match any built-in rule and would warn-and-fall-back to
# Anthropic. The explicit prefix removes the ambiguity.

from lazybridge import Agent

# Groq — fast Llama inference
fast = Agent("litellm/groq/llama-3.3-70b-versatile", name="fast")
print(fast("hello").text())

# Mistral's hosted API
mistral = Agent("litellm/mistral/mistral-large-latest", name="mistral")
print(mistral("hello").text())

# Ollama on localhost — no API key needed
local = Agent("litellm/ollama/llama3", name="local")
print(local("hello").text())
```

After the `litellm/` prefix is stripped, the rest of the model string
is whatever LiteLLM accepts natively.  See the
[LiteLLM providers page](https://docs.litellm.ai/docs/providers) for
the full catalog and each provider's model-naming conventions.

## Tools and structured output

Function-calling passes through unchanged — LazyBridge emits
OpenAI-shaped tool schemas, and LiteLLM forwards them to the
underlying provider with whatever translation is needed.

```python
# What this shows: a plain tool-using Agent backed by a non-native
# provider via the bridge. Same tools= surface as any other Agent —
# the bridge is transparent.
# Why structured output still works: LazyBridge falls back to its
# JSON-prompting + Pydantic validation path when the provider doesn't
# natively enforce schemas. LiteLLM-routed models get the same loop.

from lazybridge import Agent
from pydantic import BaseModel

def search(query: str) -> str:
    """Search the web for query."""
    return f"hits for {query}"

class Summary(BaseModel):
    title: str
    bullets: list[str]

agent = Agent(
    "litellm/groq/llama-3.3-70b-versatile",
    tools=[search],
    output=Summary,
    name="groq_agent",
)
env = agent("summarise AI news")
print(env.payload.title)
```

## What you give up through the bridge

LiteLLM normalises every provider to the OpenAI wire shape.  That's
what gives it the broad coverage, but it also means provider-specific
capabilities that don't fit OpenAI's schema don't round-trip.  Use
LazyBridge's native adapters for these:

| Feature | Native adapter | LiteLLM bridge |
|---|---|---|
| Extended thinking / reasoning display (`thinking=True`) | ✅ Anthropic, OpenAI | ❌ silently dropped |
| Prompt caching (`cache=CacheConfig(ttl="1h")`) | ✅ Anthropic | ❌ silently dropped |
| Native tools (`NativeTool.WEB_SEARCH`, `CODE_EXECUTION`, …) | ✅ per provider | ⚠️ ignored with a `UserWarning` |
| Anthropic Skills (`skills=SkillsConfig(...)`) | ✅ Anthropic | ❌ silently dropped |
| Grounding sources returned from search | ✅ Google, Anthropic | ❌ not surfaced |

Function-calling, streaming, async, max_tokens / temperature, image
content, tool results, and cost tracking all DO forward cleanly.
Cost comes through when LiteLLM can price the model
(`response._hidden_params["response_cost"]`); otherwise
`env.metadata.cost_usd` is `None`.

## Default routing, explicit rules

The built-in router recognises the `litellm/` prefix automatically.
If you want a raw pattern (say every `groq/*` string) to flow through
the bridge without the prefix, register a rule once at startup:

```python
# What this shows: opting a whole vendor into the bridge without
# forcing users to type the litellm/ prefix every time.
# Why this is a registration call, not config: the registry is
# intentionally explicit — accidental routing of `anthropic/claude-3`
# through LiteLLM instead of the native Anthropic provider would
# silently disable thinking/caching.

from lazybridge import LLMEngine

LLMEngine.register_provider_rule("groq/", "litellm", kind="startswith")
# Now both of these route through LiteLLMProvider:
#   Agent("groq/llama-3.3-70b-versatile")
#   Agent("litellm/groq/llama-3.3-70b-versatile")
```

## Pitfalls

- Requesting a model whose provider's SDK isn't configured fails
  inside `litellm.completion()` with LiteLLM's own error — LazyBridge
  doesn't wrap it.  Check the LiteLLM stack trace; the fix is almost
  always "set the right env var".
- Passing `native_tools=[NativeTool.WEB_SEARCH]` on a LiteLLM-routed
  Agent emits a `UserWarning` and drops the tool.  Switch to the
  native provider that supports it.
- `max_turns` on `LLMEngine` (default 10) still caps the tool-call
  loop even when the underlying model is LiteLLM-routed.  Raise it if
  you're running deep tool loops on a long-context model.
- `request.extra` on `CompletionRequest` is forwarded verbatim to
  `litellm.completion()` — handy for `top_p`, `presence_penalty`,
  `seed`, `response_format={"type": "json_object"}`, etc.  That's
  also where you'd pass LiteLLM-specific kwargs the bridge doesn't
  model (`mock_response=`, `metadata=`, etc.).

!!! note "API reference"

    # Implicit — prefix-based routing:
    Agent("litellm/<litellm-model-string>")

    # Explicit — engine construction:
    from lazybridge import Agent, LLMEngine
    Agent(engine=LLMEngine("litellm/groq/llama-3.3-70b-versatile"))

    # Manual registration for a prefix-less routing rule:
    LLMEngine.register_provider_rule("groq/", "litellm", kind="startswith")

!!! warning "Rules & invariants"

    - ``litellm/`` is stripped before the model string reaches
      ``litellm.completion()``.  Everything after the prefix is passed
      through unchanged.
    - Native LazyBridge providers take priority over the bridge
      whenever a model string matches both.  ``"claude-opus-4-7"``
      still routes to the native Anthropic adapter — only
      ``"litellm/claude-opus-4-7"`` forces the bridge path.
    - ``LiteLLMProvider`` depends on ``pip install 'lazybridge[litellm]'``;
      the import is lazy, so users who never route through the bridge
      don't pay for LiteLLM's dependency tree.
    - ``api_key=`` on ``Agent(engine=LLMEngine(...))`` is forwarded to
      ``litellm.completion(api_key=...)``.  Without it, LiteLLM reads
      the key from the appropriate env var for the target provider.
    - Async is first-class: the bridge calls ``litellm.acompletion()``
      directly, no thread-pool hop.

## See also

[base_provider](base-provider.md), [register_provider](register-provider.md),
[agent](agent.md)
