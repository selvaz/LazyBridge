## signature
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

## rules
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

## example
```python
from lazybridge import Agent
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    body: str

# Agent(...) constructs; ("task") invokes → always returns an Envelope.
env = Agent("claude-opus-4-7", output=Article)("write a one-paragraph article on bees")

# Branch on success / failure.
if env.ok:
    print(env.payload.title)
    print(env.payload.body)
else:
    print(f"failed ({env.error.type}): {env.error.message}")

# Observability without a Session — metadata is always populated.
m = env.metadata
print(f"cost=${m.cost_usd:.4f}  in={m.input_tokens}  out={m.output_tokens}")

# Typed: the static checker knows env.payload is an Article.
def process(env: "Envelope[Article]") -> str:
    return env.payload.title
```

## pitfalls
- ``Envelope.from_task(task)`` sets ``payload=task`` for convenience so
  the very first agent in a chain sees the input as both ``task`` and
  ``payload``. Downstream steps see the preceding step's ``payload``.
- ``nested_*`` fields in metadata are plumbed but not always populated;
  for accurate cross-agent cost, query ``session.usage_summary()``.
