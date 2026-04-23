# Provider registry

Provider discovery is a runtime registry, not a code edit. When a new
Claude or OpenAI model ships, you extend the registry in one line
rather than patching `_infer_provider` in the framework:

```python
LLMEngine.register_provider_rule("claude-opus-5", "anthropic")
Agent("claude-opus-5-20260701-preview")("hello")   # routed correctly
```

Two surfaces cover the two common needs:

* **Alias** for exact matches — `Agent("mistral")` → the mistral
  provider, no questions asked.
* **Rule** for model-name patterns — `claude-*` → anthropic, `gpt-*`
  / `o1-*` / `o3-*` → openai.

User-registered rules beat built-ins by list order. Restoration is
trivial in tests: snapshot the two dicts and restore them in a
fixture.

## Example

```python
import pytest
from lazybridge import Agent, LLMEngine

# Route all model strings starting with "bedrock/" to a custom
# AWS Bedrock provider that the user has subclassed from BaseProvider.
LLMEngine.register_provider_rule("bedrock/", "bedrock", kind="startswith")

# Override a built-in: send all "claude-*" calls through a local proxy.
LLMEngine.register_provider_rule("claude", "my-proxy")

# Exact-match alias for a new provider.
LLMEngine.register_provider_alias("mistral", "mistral")

# All of these now resolve via the registry.
Agent("bedrock/claude-opus-5")
Agent("claude-opus-4-7")    # routed to "my-proxy" because user rule takes priority
Agent("mistral")

# Test hygiene — snapshot+restore in a fixture.
@pytest.fixture
def restore_provider_rules():
    aliases = dict(LLMEngine._PROVIDER_ALIASES)
    rules = list(LLMEngine._PROVIDER_RULES)
    yield
    LLMEngine._PROVIDER_ALIASES = aliases
    LLMEngine._PROVIDER_RULES = rules
```

## Pitfalls

- Order matters: ``register_provider_rule`` PREPENDS. If you need to
  append instead (rare), mutate ``_PROVIDER_RULES`` directly.
- Registering an alias without a matching subclassed ``BaseProvider``
  in ``core/providers/`` will succeed but ``Agent(...)`` calls will
  fail at Executor resolution time.
- Tests that register rules leak state into subsequent tests unless
  you use the restore fixture pattern.

!!! note "API reference"

    LLMEngine.register_provider_alias(alias: str, provider: str) -> None
    LLMEngine.register_provider_rule(
        pattern: str,
        provider: str,
        *,
        kind: Literal["contains", "startswith"] = "contains",
    ) -> None

    # Strict routing: raise on unknown models rather than silently
    # falling back to the default provider (Anthropic out of the box).
    # Recommended for production — unknown-model bugs surface at
    # construction time instead of several RTTs into a doomed API call.
    LLMEngine.set_default_provider(None)

    # Or redirect the safety-net to a different built-in provider:
    LLMEngine.set_default_provider("openai")
    
    # Internal tables (user-extendable at runtime):
    #   LLMEngine._PROVIDER_ALIASES  — exact-match model string → provider
    #   LLMEngine._PROVIDER_RULES    — [(kind, pattern, provider), ...]
    #   LLMEngine._PROVIDER_DEFAULT  — fallback when no rule matches

!!! warning "Rules & invariants"

    - ``register_provider_alias`` adds exact-match routing: model string
      equal to ``alias`` (case-insensitive) resolves to ``provider``.
    - ``register_provider_rule`` adds substring / prefix routing: if the
      rule matches, the provider is used. New rules PREPEND — so user
      rules take priority over built-ins. A newer "claude-opus-5-foo"
      rule wins over the built-in "claude" catch-all.
    - Matching is case-insensitive (both pattern and model are lower-cased).
    - Both methods are ``@classmethod``; they mutate class-level tables.
      Tests should snapshot/restore these tables if they register rules
      (see ``tests/unit/test_v1_refinements.py:restore_provider_rules``).

## See also

[base_provider](base-provider.md), [engine_protocol](engine-protocol.md)
