# Envelope

The single typed object that flows between every engine and agent. You
never construct one manually — every `agent(task)` call returns one,
and every step in a `Plan` reads one and produces another.

## Signature

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    task: str | None = None        # the input task / prompt
    context: str | None = None     # additional context (e.g. previous step output)
    images: list | None = None     # multimodal: list[ImageContent]
    audio: object | None = None    # multimodal: a single AudioContent clip
    payload: T | None = None       # the typed result (str by default; Pydantic with output=)
    metadata: EnvelopeMetadata     # token / cost / latency / provider info
    error: ErrorInfo | None = None # populated when the run failed

    @property
    def ok(self) -> bool: ...      # True iff error is None

    def text(self) -> str: ...     # payload as a string (str verbatim, BaseModel as JSON)

    @classmethod
    def from_task(cls, task, context=None) -> Envelope: ...

    @classmethod
    def error_envelope(cls, exc, *, retryable=False) -> Envelope: ...


class EnvelopeMetadata(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str | None = None
    provider: str | None = None
    run_id: str | None = None
    # Aggregated from sub-agent calls (agent-as-tool / Plan steps).
    nested_input_tokens: int = 0
    nested_output_tokens: int = 0
    nested_cost_usd: float = 0.0


class ErrorInfo(BaseModel):
    type: str                      # exception class name
    message: str                   # human-readable message
    retryable: bool = False        # whether the resilience layer may retry
```

## Synopsis

`Envelope` is the universal request / response object. It carries:

- **The result** — a string by default, a typed Pydantic instance when
  the agent was constructed with `output=SomeModel`.
- **Metadata** — token counts, cost in USD, latency in milliseconds,
  the model and provider that produced it, and a `run_id`. Also
  `nested_*` aggregation buckets that fill up when the agent called
  nested agents as tools, so the top-level envelope reflects total
  pipeline cost without any extra plumbing.
- **An error, if anything went wrong** — `error.type` is the
  exception class name, `error.message` is the message, and
  `error.retryable` tells the resilience layer whether a retry might
  succeed.

Generic typing (`Envelope[Article]`) narrows the payload type for
mypy / pyright without changing runtime behaviour. Untyped
`Envelope` is `Envelope[Any]` and stays the zero-friction default.

## When you'll see one

- **Every `agent(task)` call** returns one. That's the canonical
  point of contact.
- **Every step in a `Plan`** receives one (the previous step's
  envelope) and produces one. Sentinels like `from_prev`,
  `from_step("name")`, `from_parallel_all("name")` resolve to fields
  of those envelopes at run time.
- **Every `Agent` wrapped as a tool** also returns one — its
  metadata is folded into the parent's `nested_*` buckets so cost
  rollup is transitive.

## When NOT to construct one

- **Almost never directly.** The framework builds envelopes for you
  on every entry and exit. Manual construction is reserved for two
  cases:
    - Test fixtures (`Envelope.from_task("test prompt")` creates a
      ready-to-feed input).
    - Custom engines that need to surface an error path
      (`Envelope.error_envelope(exc)` is the canonical builder).
- **Don't mutate one in flight.** Envelopes are Pydantic models; if
  you need to derive one with a changed field, use
  `env.model_copy(update={"context": new_context})`.

## Example

```python
from lazybridge import Agent, LLMEngine
from pydantic import BaseModel


class Article(BaseModel):
    title: str
    body: str


# Constructing an Agent with structured output narrows Envelope.payload.
writer = Agent(
    engine=LLMEngine("gemini-3-flash-preview"),
    output=Article,
)

result = writer("write a one-paragraph article on bees")

# 1) Always check .ok before reading .payload in production code.
if result.ok:
    print(result.payload.title)
    print(result.payload.body)
else:
    print(f"failed ({result.error.type}): {result.error.message}")
    if result.error.retryable:
        print("(this error is retryable — the resilience layer may try again)")


# 2) Observability without a Session — metadata is always populated.
m = result.metadata
print(f"cost=${m.cost_usd:.4f}  in={m.input_tokens}  out={m.output_tokens}")
print(f"model={m.model}  provider={m.provider}  latency={m.latency_ms:.0f} ms")


# 3) text() — string regardless of payload shape (str verbatim, BaseModel as JSON).
print(result.text())   # JSON dump of the Article


# 4) Static typing — the checker knows env.payload is an Article.
def first_word(env: "Envelope[Article]") -> str:
    return env.payload.title.split()[0]
```

## Pitfalls

- **`output=SomeModel` + `.text()`** returns the JSON dump of the
  payload, not the human-readable text. With structured output, read
  `.payload` directly.
- **`Envelope.from_task(task)` sets `payload=task` for convenience**
  so the very first agent in a chain sees the input as both `task`
  and `payload`. Downstream steps see the *preceding step's*
  `payload`, not the original task — use `from_start` if you need
  the original input later.
- **`nested_*` metadata is plumbed but not always populated.** For
  authoritative cross-agent cost numbers in a multi-agent pipeline,
  query `session.usage_summary()` rather than
  `envelope.metadata.nested_cost_usd`. The envelope's nested
  buckets reflect what flowed through *this* envelope's lineage,
  not the entire run.
- **`error.retryable=False` does not mean "give up forever"** — it
  means "the resilience layer should not auto-retry this one". A
  caller's `fallback=` agent is still tried, and you can always
  re-run the agent yourself.
- **Multimodal attachments only ride on step 0** of a `Plan`.
  Downstream steps receive upstream output (text), not the original
  `images=` / `audio=` payload. Pass attachments to the first step.
- **`__str__` falls through to `text()`** — when an `Agent` is used
  as a tool, the LLM's tool-result block stringifies the envelope
  via `str(...)`; without this, every nested call would produce
  `"task=…  context=…"` garbage instead of the real answer. Don't
  override `__str__` on subclasses.

## See also

- [Agent](agent.md) — the producer of envelopes.
- [Tool](tool.md) — `returns_envelope=True` is the hint that lets
  engines roll up nested cost / token metadata correctly.
- [Mental model](../../concepts/mental-model.md) — `Envelope` is
  the only piece of "state" that's always present, regardless of
  whether you opt into `Memory`, `Session`, or `Store`.
