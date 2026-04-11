# veo

Google Veo video generation tool for LazyBridge.

Generate videos from text prompts, images, or reference assets.
The tool handles polling, downloading, and saving the result — the agent
just calls it with a prompt.

---

## How it works

```
prompt / image(s)
       │
       ▼
┌──────────────────────────────────────────┐
│  client.models.generate_videos()         │
│  Veo long-running operation              │
│  polls every N seconds until done        │
└──────────────────┬───────────────────────┘
                   │  operation.response
                   ▼
          download + save .mp4
                   │
                   ▼
          { ok, output_path, … }
```

---

## Quick start

```python
from lazybridge.tools.veo import veo_tool
from lazybridge import LazyAgent

tool  = veo_tool()
agent = LazyAgent("google", model="gemini-2.5-pro")
resp  = agent.loop(
    "Generate an 8-second cinematic drone shot of the Grand Canyon at sunset.",
    tools=[tool],
)
print(resp)
```

Direct invocation (no agent):

```python
tool = veo_tool()
result = tool.invoke(
    prompt="A golden perfume bottle on black glass, dramatic lighting.",
    duration_seconds=8,
    resolution="720p",
    negative_prompt="blurry, shaky, low quality",
)
print(result["output_path"])
```

---

## Public API

### `veo_tool(...)`

Factory that creates a `LazyTool` wrapping the Veo API.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | Google API key; falls back to `GOOGLE_API_KEY` env var |
| `model` | `str` | `"veo-3.1-generate-001"` | Veo model ID (see table below) |
| `output_dir` | `str` | `"generated_videos"` | Directory for generated `.mp4` files |
| `poll_interval_seconds` | `int` | `10` | Seconds between operation polls |
| `timeout_seconds` | `int` | `900` | Max wait time (default 15 min) |

Returns a `LazyTool` named `generate_veo_video`.

---

### Tool parameters (`tool.invoke(...)`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | required | Text description of the video |
| `duration_seconds` | `int` | `8` | `4`, `6`, or `8` (Veo 3.x); `5`, `6`, or `8` (Veo 2.x) |
| `aspect_ratio` | `str` | `"16:9"` | `"16:9"` (landscape) or `"9:16"` (portrait/mobile) |
| `resolution` | `str` | `"720p"` | `"720p"`, `"1080p"`, or `"4k"` |
| `generate_audio` | `bool` | `True` | Generate audio track; requires a non-fast Veo 3.x model |
| `negative_prompt` | `str \| None` | `None` | What to exclude — use nouns, not negations |
| `seed` | `int \| None` | `None` | `0–4294967295` for reproducibility |
| `enhance_prompt` | `bool` | `True` | Let Veo auto-expand the prompt |
| `first_frame` | `str \| None` | `None` | Local path or `gs://` URI — animates this image |
| `last_frame` | `str \| None` | `None` | Local path or `gs://` URI — interpolates to this frame (requires `duration_seconds=8`) |
| `reference_images` | `list[str] \| None` | `None` | Up to 3 paths/URIs as asset references (requires `duration_seconds=8`) |
| `person_generation` | `str` | `"allow_all"` | `"allow_all"`, `"allow_adult"`, or `"dont_allow"` |

**Returns** `dict`:
```python
{
    "ok": True,
    "model": "veo-3.1-generate-001",
    "prompt": "...",
    "duration_seconds": 8,
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "output_path": "/absolute/path/to/veo_abc123.mp4",
    "output_filename": "veo_abc123.mp4",
}
```

---

## Models

| Model ID | Status | Audio | Max resolution | Duration |
|---|---|---|---|---|
| `veo-3.1-generate-001` | GA | Yes | 4K (8s only) | 4/6/8s |
| `veo-3.1-fast-generate-001` | GA | No | 4K (8s only) | 4/6/8s |
| `veo-3.1-lite-generate-001` | Preview | — | 1080p (8s only) | 4/6/8s |
| `veo-3.0-generate-001` | GA | Yes | 4K (8s only) | 4/6/8s |
| `veo-3.0-fast-generate-001` | GA | No | 4K (8s only) | 4/6/8s |
| `veo-2.0-generate-001` | GA | No | 720p | 5/6/8s |

**Audio rules:**
- `generate_audio=True` → requires a non-fast Veo 3.x model (e.g. `veo-3.1-generate-001`)
- `generate_audio=False` → requires a fast or Veo 2.x model (e.g. `veo-3.1-fast-generate-001`)
- Passing the wrong combination raises a clear `ValueError`

---

## Image inputs

All image parameters (`first_frame`, `last_frame`, `reference_images`) accept:

- **Local file path** — file is read and sent as bytes. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- **GCS URI** (`gs://bucket/file.png`) — forwarded directly (requires GCS access)

**Constraints:**
- `last_frame` requires `first_frame` to also be set
- `last_frame`, `reference_images`, resolution `"1080p"` or `"4k"` all require `duration_seconds=8`
- `reference_images` accepts at most 3 asset images
- `reference_images` may be unreliable via the Gemini API in some SDK versions; Vertex AI is more robust

---

## Caveats

- Generated videos are stored on Google servers for **2 days** — download promptly (`output_path` is the local file after `tool.invoke()`)
- Generation latency: ~30s minimum, up to ~6 minutes at peak load
- `number_of_videos` is fixed at 1 in the Gemini API (Vertex AI supports 1–4)
- Veo 3.x text-to-video only supports `person_generation="allow_all"`

---

## Production / Vertex AI (future)

This tool currently uses the **Gemini API** (`GOOGLE_API_KEY` from AI Studio).
For stable production deployments, the recommended path is **Vertex AI**, which
unlocks features the Gemini API does not expose:

| Feature | Gemini API | Vertex AI |
|---|---|---|
| `seed` (reproducibility) | ❌ not supported | ✅ uint32 — anchors random state; helps keep subject/composition stable across runs with the same prompt |
| `reference_images` type `"style"` | ❌ | ✅ aesthetic style transfer |
| `number_of_videos` | ❌ fixed at 1 | ✅ 1–4 per request |
| `output_gcs_uri` | ❌ | ✅ write output directly to GCS |
| Reliability of `reference_images` | ⚠️ SDK issues in some versions | ✅ stable |

Vertex AI uses a **different authentication** model — it does **not** use
`GOOGLE_API_KEY`. It requires a Google Cloud project with billing enabled and
`gcloud auth application-default login` (once), or a service account JSON key.

When Vertex AI support is added, `veo_tool()` will gain:

```python
tool = veo_tool(
    vertexai=True,
    project="my-gcp-project",
    location="us-central1",
    # no api_key needed — uses ADC / service account
)
```

The rest of the API (`tool.run({...})`) will remain identical.

---

## Dependencies

```bash
pip install lazybridge[google]
# or
pip install google-genai>=1.10.0
```

The module raises `ImportError` with a helpful message if `google-genai` is not installed.

---

## Running the demo

```bash
# From the repo root
python tools/veo/demo_veo.py
```

Or open `demo_veo.py` in Spyder and press **F5**.
Requires `GOOGLE_API_KEY` in `.env`.
