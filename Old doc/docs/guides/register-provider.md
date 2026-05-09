# Provider registry

**Use the provider registry** to make `Agent("my-model")` route to a
custom provider without forking the framework.  Two granularities:

* `register_provider_alias("mistral", "mistral")` — exact-match alias.
* `register_provider_rule("claude-opus-5", "anthropic")` — substring or
  prefix rule (later registrations win).
* `set_default_provider(None)` — disable the safety-net fallback so an
  unrecognised model raises `ValueError` at construction instead of
  routing somewhere unintended.

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

- [BaseProvider](base-provider.md) — what you implement for a brand-new provider.
- [Providers](providers.md) — the built-in catalogue.
