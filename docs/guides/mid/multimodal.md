# Multimodal (image / audio)

Pass images and audio clips alongside the text task. The agent forwards
them to the underlying provider as native content blocks; the provider
returns text (or a structured payload) the same way it does for plain
text turns. Capability checking is automatic — you can either silently
drop unsupported attachments (default) or raise.

## Signature

```python
result = agent(
    task,                              # str — the textual prompt
    images=[...],                      # list[str | Path | bytes | dict | ImageContent] | None
    audio=...,                         # str | Path | bytes | dict | AudioContent | None
)

# Async + streaming forms accept the same kwargs
result = await agent.run(task, images=[...], audio=...)
async for chunk in agent.stream(task, images=[...], audio=...):
    ...

# Strongly-typed content blocks (when you need control over media_type)
from lazybridge import ImageContent, AudioContent

ImageContent.from_url("https://example.com/cat.jpg")
ImageContent.from_path("/tmp/diagram.png")          # auto-detects media_type
ImageContent.from_bytes(buf, media_type="image/jpeg")
ImageContent.from_data_uri("data:image/png;base64,iVBOR…")

AudioContent.from_url("https://example.com/clip.mp3")
AudioContent.from_path("/tmp/voice.wav")
AudioContent.from_bytes(buf, media_type="audio/wav")
```

`Envelope.images` and `Envelope.audio` carry the (coerced) attachments
from input through the run. In a `Plan`, attachments only ride on the
**first step**; downstream steps see text and structured payloads, not
the original media.

## Synopsis

LazyBridge handles three things automatically:

- **Coercion.** A bare URL string, a `Path`, raw `bytes`, or a `dict`
  ({"url": …} / {"base64_data": …, "media_type": …}) is turned into the
  appropriate `ImageContent` or `AudioContent` for you. Pass typed
  blocks only when you need to override the auto-detected MIME type.
- **Capability gating.** Each provider knows which of its models accept
  vision / audio. By default an unsupported attachment is dropped with
  a single `UserWarning`; pass `LLMEngine(strict_multimodal=True)` to
  raise `UnsupportedFeatureError` instead.
- **Wire mapping.** Anthropic, OpenAI and Google each have a different
  content-block format; the framework converts once per request. You
  see one uniform API regardless of provider.

### Provider capability matrix (current)

| Provider | Vision-capable models | Audio-capable models |
|---|---|---|
| **Anthropic** | `claude-3*`, `claude-4*`, `claude-opus*`, `claude-sonnet*`, `claude-haiku*` | `claude-3-7*`, `claude-4*` (3.0 / 3.5 are vision-only) |
| **OpenAI** | `gpt-4-turbo`, `gpt-4o*`, `gpt-4.1*`, `gpt-5*`, reasoning `o1` / `o3` / `o4` | `gpt-4o-audio*`, `gpt-4o-realtime*`, `gpt-4o-mini-audio*`, `gpt-4o-mini-realtime*` |
| **Google** | `gemini-1.5*`, `gemini-2*`, `gemini-3*` | same — Gemini 1.5+ is uniformly multimodal |
| **DeepSeek**, **LMStudio**, **LiteLLM** | provider-default (none assumed) | provider-default (none assumed) |

`provider.supports_vision(model)` and `provider.supports_audio(model)`
expose the same check programmatically; both are class methods so you
can ask without instantiating a client.

## When to use it

- **Vision tasks** — describe an image, OCR a document scan, classify
  a UI screenshot, audit a chart, locate elements in a photo.
- **Audio tasks** — transcribe, summarise a meeting clip, score
  pronunciation, extract entities from a voice note (provider-dependent;
  text-out is universal, audio-out is OpenAI's realtime models only).
- **Mixed input** — pass `task="…"` plus N images plus one audio clip;
  the LLM sees the whole bundle as one turn.

## When NOT to use it

- **Document parsing where you control the pipeline.** Run OCR /
  extraction yourself and feed text — far cheaper, more deterministic,
  no per-image token cost.
- **Streaming media.** The framework batches a finite list of attachments
  per request. For real-time audio in/out you want a provider's realtime
  / WebSocket API directly, not LazyBridge.
- **Provider-hosted file search.** That's `NativeTool.FILE_SEARCH` (see
  [Native tools](../basic/native-tools.md)) — different surface; the
  model doesn't see the file content as an attachment, it queries an
  index server-side.

## Example

```python
from pathlib import Path

from lazybridge import Agent, AudioContent, ImageContent, LLMEngine


# 1) URL — bare string is coerced to ImageContent at call time.
agent = Agent(engine=LLMEngine("claude-opus-4-7"))
result = agent(
    "Describe what's in this picture in two sentences.",
    images=["https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"],
)
print(result.text())


# 2) Local files — Path is coerced to base64 + correct media_type.
result = agent(
    "Summarise the chart and call out the outlier.",
    images=[Path("/tmp/quarterly_revenue.png")],
)
print(result.text())


# 3) Multiple images in one turn — the model sees them all together.
result = agent(
    "Which of these screenshots shows the broken layout?",
    images=[
        Path("/tmp/before.png"),
        Path("/tmp/after.png"),
        Path("/tmp/expected.png"),
    ],
)


# 4) Audio — same coercion rules.
result = agent(
    "Transcribe this clip; highlight any product names mentioned.",
    audio=Path("/tmp/standup.wav"),
)
print(result.text())


# 5) Strongly-typed block when you need to override media_type or
#    embed bytes you already have in memory.
img = ImageContent.from_bytes(open("/tmp/screenshot.bin", "rb").read(), media_type="image/png")
agent("What's the error in this screenshot?", images=[img])


# 6) Strict mode — raise instead of dropping when the model can't
#    handle the modality. Useful in production, where a silent drop
#    would change semantics undetected.
strict_agent = Agent(
    engine=LLMEngine("claude-opus-4-7", strict_multimodal=True),
)
try:
    strict_agent(
        "Describe this audio.",
        audio=Path("/tmp/clip.mp3"),
    )
except Exception as exc:
    # UnsupportedFeatureError if the chosen model lacks audio support
    print(f"refused: {exc}")
```

Supported MIME types (auto-detected from extension by `from_path`):

- **Images:** `image/jpeg`, `image/png`, `image/gif`, `image/webp`,
  `image/bmp`, `image/tiff`
- **Audio:** `audio/wav`, `audio/mpeg`, `audio/mp3`, `audio/flac`,
  `audio/ogg`, `audio/webm`, `audio/aac`, `audio/mp4`

## Pitfalls

- **Default is "drop with warning", not "raise".** A plain
  `LLMEngine("gpt-4o-mini")` with `images=[...]` silently strips the
  images and emits one `UserWarning`. Pass `strict_multimodal=True`
  in production so a model swap that loses vision support breaks
  loudly instead of changing behaviour silently.
- **Attachments only ride on step 0 of a `Plan`.** Pass `images=` /
  `audio=` to `pipeline(task, images=...)` and only the first step
  sees them. Downstream steps receive text + structured payloads, never
  the original media. If a later step needs the bytes, embed a path or
  URL in the payload and re-attach in a `from_step(...)` predicate, or
  re-call the multimodal step with the bytes itself as input.
- **Bare `bytes` requires a typed block.** `agent(..., images=[buf])`
  works only because the coercer assumes `image/jpeg` for raw bytes.
  Use `ImageContent.from_bytes(buf, media_type="image/png")` whenever
  the format isn't JPEG; otherwise the provider may reject the request
  or render garbage.
- **`audio=` is a single value, `images=` is a list.** Most providers
  accept N images per turn but at most one audio clip. Pass a list of
  audio clips and the framework will refuse it.
- **Cost is per-image, per-tile.** Vision input is tokenised in tiles
  (Anthropic) or scaled blocks (OpenAI / Google). A 4K screenshot
  costs an order of magnitude more than a 512×512 thumbnail. Resize
  upstream when the model only needs a thumbnail.
- **`Envelope.images` / `Envelope.audio` after the run** carry the
  *input* attachments — providers that do *audio out* (OpenAI realtime)
  surface the response audio through provider-specific extensions, not
  this field. Read the run's text payload first; the field is for
  carry-through, not response media.
- **`UnsupportedFeatureError` subclasses `ValueError`.** Catch it
  precisely (`from lazybridge import UnsupportedFeatureError`) when
  you want to fall back to a vision-capable model rather than swallow
  every `ValueError` raised at construction time.

## See also

- [Envelope](../basic/envelope.md) — `Envelope.images` and
  `Envelope.audio` are the carry-through fields these calls populate.
- [Native tools](../basic/native-tools.md) —
  `NativeTool.FILE_SEARCH` and `NativeTool.IMAGE_GENERATION` are
  separate features (provider-hosted retrieval / generation), not
  attachments.
- [Engines](../../reference/engines.md) — `strict_multimodal` and
  the surrounding production knobs.
