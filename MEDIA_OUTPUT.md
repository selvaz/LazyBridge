# LazyBridge — Audio/Video output (design doc)

> **Status.** Proposal / design doc. Nothing ships yet.
>
> **Idea in one line.** Media *generation* (audio, video, images, speech)
> is **tools that return a `MediaRef`** — not new core output plumbing.
> The only core addition is the `MediaRef` handle itself.

## Current state (the surface we extend)

LazyBridge models multimodal **input** but is **text-only on output**:

- Input media exists: `ImageContent` / `AudioContent`
  (`core/types.py:122-239`) with `url | base64_data | media_type`.
- `Envelope.payload` is text-only; `Envelope.text()` raises on binary
  (`envelope.py:75-96`). `Envelope.images` / `audio` are **input
  carry-through**, not output.
- `LLMEngine` has **no output-modality** param (`engines/llm.py:154-176`).
- `NativeTool` has `IMAGE_GENERATION` only — no audio/video
  (`core/types.py:104-114`).
- Provider capabilities track input only (`supports_vision` /
  `supports_audio`, `base.py:176-207`); `ProviderCapabilities`
  (`matrix.py:46-58`) has no output-modality field.
- Cost is token-based only (`base.py:527-559`).
- LazyTools has **no** generation tools.

A maximalist extension (output media on `Envelope` +
`CompletionResponse` + `StreamChunk` + `ContentType`, per-provider media
parsing, output-capability flags) touches half the core and fights the
"core minimal / everything is a tool" ethos. We don't do that.

## Principle: classify by *where the media comes from*

- **Video** — always a **separate async endpoint** (Veo, Sora). No model
  emits video inside a chat turn. → **entirely a tool, zero core.**
- **Images** — separate endpoint (Imagen, gpt-image) or the existing
  `NativeTool.IMAGE_GENERATION`. → tool / native tool.
- **Audio TTS/STT** — separate endpoint. → tool.
- **Inline audio-out from the chat turn** (gpt-4o-audio / realtime) — the
  *only* case that would touch the core. Already marked out-of-scope by
  `docs/guides/mid/multimodal.md` ("use the provider's realtime API").
  → **deferred** (see below).

## The one core primitive: `MediaRef`

A small asset handle, **reference-first** (never hold a video in
memory / prompt / checkpoint by default):

```python
@dataclass
class MediaRef:
    media_type: str                 # MIME: "video/mp4", "audio/wav", "image/png"
    uri: str | None = None          # remote URL or store:// key (preferred)
    store_key: str | None = None    # blob in the durable Store
    data: bytes | None = None       # inline — small assets only
    duration_ms: int | None = None
    size_bytes: int | None = None
    # from_url / from_store / from_bytes constructors; resolve() -> bytes
```

It is the **bridge** that closes the loop: a generation tool returns a
`MediaRef`, and the same `MediaRef` feeds back as input via
`ImageContent.from_ref(...)` / `AudioContent.from_ref(...)` — so
*generate → hand to the next agent* works with no glue.

## Generation lives in LazyTools (tools returning `MediaRef`)

Tool providers, dropped into `tools=[...]`, wrappable by a Skill/
SuperTool ("video producer"). **No core change beyond `MediaRef`.**

- `image_gen(prompt, ...) -> MediaRef`
- `speech(text, voice, ...) -> MediaRef`   (TTS)
- `transcribe(audio, ...) -> str`          (STT; text out)
- `video_gen(prompt, ...) -> VideoJob`     (async; see below)

### Video: job handle + check tool (locked decision)

Video gen is long-running, so it does **not** block a turn:

```python
job = video_gen("a cat surfing a wave", model="veo-3")   # returns a job handle
# ... agent does other work ...
ref = check_video(job)   # -> MediaRef when ready, or a "pending" status
```

The agent loops `check_video` (and can checkpoint / hand to `HumanEngine`
for long waits). A blocking poll-until-ready variant is intentionally
*not* the default.

### Capability gating in the tool

The tool knows its own model, so it validates support and **raises a
clear error** when the model can't do the requested modality (consistent
with "errors always raise, no silent fallback"). No need to inflate
`ProviderCapabilities` for the tool path.

### Cost

Media billing (per-image, per-second) rides as **tool-supplied
metadata**: generation tools use `Tool.returns_envelope=True` and report
cost in `Envelope.metadata`. The token-based `_compute_cost` is
untouched.

## Deferred: inline audio-out from the chat turn

Kept provider-specific / out-of-scope, as today. If a concrete need
appears, the minimal addition is an `output_modalities: list[str]` scalar
knob on `LLMEngine` — which slots straight into the `agents.md` OVERRIDE
ladder (`PROJECT_LAYOUT.md` §4) — plus `MediaRef` carriers on the
Envelope and per-provider parsing. **Not built now.**

## Relationship to the rest

- `output=` (Pydantic) stays **text**. Media is a `MediaRef`, carried
  separately. The two are **orthogonal** — a media-producing agent can
  still have a typed text payload describing what it made.
- Generation tools are ordinary tools, so they compose with `agents.md`
  and SuperTool/Skill exactly like any other tool — no new wiring.

## Locked decisions

1. **Tool-first**; inline chat-turn audio-out **deferred** (core stays
   minimal: only `MediaRef` is added).
2. **Video** = `video_gen` returns a job handle + `check_video` tool
   (non-blocking).
3. **`MediaRef` is reference-first** (URI / Store-key; inline bytes only
   for small assets).

## Next step

Add `MediaRef` (core) + `ImageContent.from_ref` / `AudioContent.from_ref`
bridges; then the LazyTools generation providers (`image_gen`, `speech`,
`transcribe`, `video_gen` + `check_video`) with capability gating and
cost-in-metadata. One worked example: generate → feed to a downstream
agent.
