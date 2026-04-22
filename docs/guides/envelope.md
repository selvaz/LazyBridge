# Envelope

Every agent call returns an `Envelope`. You almost never build one
yourself; engines do that. Your job is to read the three fields you care
about — `payload` for the result, `metadata` for cost/tokens/latency,
`error` when something went wrong.

Think of it as an HTTP response object for agent runs: one type that
carries either a successful payload or structured error info, with
metadata attached. That uniformity is why chaining, wrapping, and
verifying agents stays simple — there is no "agent returns a string here
but a model there" inconsistency.

`Envelope` is generic over its payload. If you know you set
`output=Summary`, writing `env: Envelope[Summary] = agent(task)` gives
you autocomplete on `env.payload.title` without any runtime cost.

## Example

The snippet below shows the three things you always do with an
Envelope — check `ok`, read `payload`, read `metadata` — plus the
one optional static-typing benefit of `Envelope[T]`.  Every agent
call returns exactly this shape, whether the underlying engine is an
LLM, a Plan, a Supervisor, or a pure-Python `MockAgent`.

```python
from lazybridge import Agent
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    body: str

# output=Article makes payload a validated Article instance on success.
env = Agent("claude-opus-4-7", output=Article)("write a one-paragraph article on bees")

# Step 1 — always branch on env.ok before dereferencing env.payload.
# Errors (rate limit, schema failure, guard block, timeout) surface
# here as env.error rather than raising from .run().
if env.ok:
    print(env.payload.title)
    print(env.payload.body)
else:
    # env.error.type is the exception class name; retryable hints at
    # whether a naive retry would help. message is the human string.
    print(f"failed ({env.error.type}): {env.error.message}")

# Step 2 — metadata is ALWAYS populated, even on error. This is how
# you get cost/latency/model without setting up a Session.
m = env.metadata
print(f"cost=${m.cost_usd:.4f}  in={m.input_tokens}  out={m.output_tokens}")

# Step 3 — Envelope[T] narrows payload for static checkers. Runtime
# shape is unchanged; mypy/pyright now know `env.payload` is an Article
# so you get autocomplete on .title without an assert.
def process(env: "Envelope[Article]") -> str:
    return env.payload.title
```

What you've seen: one data type carries either a success payload OR an
error channel, plus cost/latency metadata, plus optional type
narrowing.  There is no "agent returns a string here, a Pydantic
object there" inconsistency — every engine emits `Envelope`.

## Pitfalls

- ``payload`` can legitimately be ``None`` (e.g. when ``error`` is set or
  when the engine produced no content). Use ``env.ok`` or ``env.text()``
  if you want a safe string.
- ``Envelope.from_task(task)`` sets ``payload=task`` for convenience so
  the very first agent in a chain sees the input as both ``task`` and
  ``payload``. Downstream steps see the preceding step's ``payload``.
- ``nested_*`` fields in metadata are plumbed but not always populated
  yet; for accurate cross-agent cost, query ``session.usage_summary()``.

!!! note "API reference"

    class Envelope(BaseModel, Generic[T]):
        task: str | None = None
        context: str | None = None
        payload: T | None = None
        metadata: EnvelopeMetadata = ...
        error: ErrorInfo | None = None
    
        @property
        def ok: bool
        def text() -> str
        @classmethod
        def from_task(task: str, context: str | None = None) -> Envelope
        @classmethod
        def error_envelope(exc: Exception, retryable: bool = False) -> Envelope
    
    class EnvelopeMetadata(BaseModel):
        input_tokens: int = 0
        output_tokens: int = 0
        cost_usd: float = 0.0
        latency_ms: float = 0.0
        model: str | None = None
        provider: str | None = None
        run_id: str | None = None
        # Aggregation buckets filled when this agent called nested agents as tools
        nested_input_tokens: int = 0
        nested_output_tokens: int = 0
        nested_cost_usd: float = 0.0
    
    class ErrorInfo(BaseModel):
        type: str
        message: str
        retryable: bool = False

!!! warning "Rules & invariants"

    - ``Envelope`` is the single data type flowing between engines. Every
      engine receives an Envelope and returns an Envelope.
    - ``text()`` returns ``payload`` as a string (``str`` verbatim, Pydantic
      models as JSON, other types via ``json.dumps``). Use it when you want a
      plain string regardless of the payload shape.
    - ``Envelope[T]`` narrows the payload type for mypy / pyright. Untyped
      ``Envelope`` is equivalent to ``Envelope[Any]`` and stays the default.
    - ``ok`` is ``True`` iff ``error is None``. Always check ``ok`` before
      reading ``payload`` in production code.
    - ``Envelope.error_envelope(exc)`` is the canonical way for engines to
      convert an exception into an envelope without raising up the stack.

## See also

[agent](agent.md), [session](session.md), [sentinels](sentinels.md)
