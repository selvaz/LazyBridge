"""
lazybridge.tools.veo  —  Video generation via Google Veo API
=============================================================

Generate videos from text prompts, images, or reference assets using
Google's Veo models.  The tool handles polling the long-running operation,
downloading the result, and saving it to disk.

Public API
----------
    veo_tool(...)                           → LazyTool    wrap Veo as a LazyBridge tool
    VeoError                                              base exception for Veo failures

Quick start
-----------
    from lazybridge.tools.veo import veo_tool
    from lazybridge import LazyAgent

    tool  = veo_tool()
    agent = LazyAgent("google", model="gemini-2.5-pro")
    resp  = agent.loop(
        "Generate an 8-second cinematic video of a sunset over the ocean.",
        tools=[tool],
    )
    print(resp)

    # Or invoke the tool directly (no agent):
    tool.invoke(
        prompt="A golden perfume bottle on black glass with dramatic lighting.",
        duration_seconds=8,
        resolution="720p",
    )

Optional dependency
-------------------
    pip install lazybridge[google]
    # or: pip install google-genai>=1.10.0

Supported models
----------------
    veo-3.1-generate-001          GA  — best quality, audio, 4K, reference images
    veo-3.1-fast-generate-001     GA  — faster/cheaper, no audio
    veo-3.1-lite-generate-001     Preview — high-speed, 1080p max
    veo-3.0-generate-001          GA  — audio, 4K
    veo-3.0-fast-generate-001     GA  — no audio
    veo-2.0-generate-001          GA  — stable, 720p, no audio, style references

Image inputs (first_frame / last_frame / reference_images)
-----------------------------------------------------------
    Pass a local file path  →  read as bytes, mime type inferred from extension
    Pass a "gs://…" URI     →  forwarded as GCS URI (requires GCS access)

Caveats
-------
    - duration_seconds must be 8 when using last_frame, reference_images,
      or resolution "1080p"/"4k"
    - generate_audio=False is NOT supported by the Gemini API for Veo 3.x
      non-fast models; use a *-fast-* model for silent video
    - reference_images may be unreliable via the Gemini API in some SDK
      versions; Vertex AI is more robust for that feature
    - Generated videos are stored on Google servers for 2 days — download promptly

Production / Vertex AI (future)
--------------------------------
    This tool uses the Gemini API (GOOGLE_API_KEY).  For stable production use,
    migrate to Vertex AI which unlocks features not available in the Gemini API:

        Feature                     Gemini API      Vertex AI
        ──────────────────────────  ──────────────  ─────────────────────────
        seed (reproducibility)      ✗ not supported ✓ uint32 0–4 294 967 295
        reference_images "style"    ✗               ✓ aesthetic style transfer
        number_of_videos            ✗ fixed at 1    ✓ 1–4 per request
        output_gcs_uri              ✗               ✓ write directly to GCS

    Vertex AI uses a different client init and Google Cloud credentials (not
    GOOGLE_API_KEY); it requires a GCP project with billing enabled and
    `gcloud auth application-default login` (or a service account JSON key).

    When Vertex AI support is added, veo_tool() will gain:
        veo_tool(vertexai=True, project="my-gcp-project", location="us-central1")
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal, Optional

__all__ = ["veo_tool", "VeoError"]

# ── Optional dependency guard ─────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types as _gtypes
    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GENAI_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "veo-3.1-generate-001"

_VEO2_DURATIONS: frozenset[str] = frozenset({"5", "6", "8"})
_VEO3_DURATIONS: frozenset[str] = frozenset({"4", "6", "8"})

AspectRatio  = Literal["16:9", "9:16"]
Resolution   = Literal["720p", "1080p", "4k"]
PersonPolicy = Literal["allow_all", "allow_adult", "dont_allow"]

_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}


# ── Exceptions ────────────────────────────────────────────────────────────────

class VeoError(RuntimeError):
    """Raised when Veo returns an error or produces no output."""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require_genai() -> None:
    if not _GENAI_AVAILABLE:
        raise ImportError(
            "google-genai is required for lazybridge.tools.veo.\n"
            "Install it with:  pip install lazybridge[google]\n"
            "or:               pip install google-genai>=1.10.0"
        )


def _allowed_durations(model: str) -> frozenset[str]:
    return _VEO2_DURATIONS if "veo-2" in model else _VEO3_DURATIONS


def _model_has_audio(model: str) -> bool:
    return "veo-2" not in model and "fast" not in model


def _load_image(source: str) -> "_gtypes.Image":
    """Return a types.Image from a local path or gs:// URI."""
    if source.startswith("gs://"):
        return _gtypes.Image(gcs_uri=source, mime_type="image/png")
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {source}")
    mime = _MIME_MAP.get(path.suffix.lower(), "image/jpeg")
    return _gtypes.Image(image_bytes=path.read_bytes(), mime_type=mime)


# ── Public API ────────────────────────────────────────────────────────────────

def veo_tool(
    api_key:                Annotated[Optional[str], "Google API key; falls back to GOOGLE_API_KEY env var."] = None,
    model:                  Annotated[str,           "Veo model ID."] = DEFAULT_MODEL,
    output_dir:             Annotated[str,           "Directory for generated .mp4 files."] = "generated_videos",
    poll_interval_seconds:  Annotated[int,           "Seconds between operation status polls."] = 10,
    timeout_seconds:        Annotated[int,           "Maximum total wait time in seconds."] = 900,
) -> "LazyTool":
    """
    Create a LazyTool that generates videos with Google Veo.

    Parameters
    ----------
    api_key:
        Google API key.  Falls back to the GOOGLE_API_KEY environment variable.
    model:
        Veo model ID.  Default: "veo-3.1-generate-001" (best quality, GA, audio).
        Use "veo-3.1-fast-generate-001" for faster/cheaper generation without audio.
    output_dir:
        Local directory where generated .mp4 files are saved.  Created if absent.
    poll_interval_seconds:
        How often to check operation status (Veo generation typically takes 30s–6min).
    timeout_seconds:
        Raise TimeoutError if the operation does not complete within this many seconds.
        Default: 900 (15 min).

    Returns
    -------
    LazyTool
        A tool named "generate_veo_video" ready to be passed to any agent or pipeline.
    """
    _require_genai()
    from lazybridge import LazyTool  # imported here to avoid circular imports at module load

    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Missing Google API key.  Pass api_key= or set the GOOGLE_API_KEY "
            "environment variable."
        )

    client = genai.Client(api_key=key)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    def generate_veo_video(
        prompt:              Annotated[str,                  "Text description of the video to generate."],
        duration_seconds:    Annotated[int,                  "Video length in seconds.  Allowed: 4/6/8 (Veo 3.x) or 5/6/8 (Veo 2.x).  Must be 8 when using last_frame, reference_images, or resolution 1080p/4k."] = 8,
        aspect_ratio:        Annotated[AspectRatio,          "Aspect ratio: '16:9' (landscape, default) or '9:16' (portrait/mobile)."] = "16:9",
        resolution:          Annotated[Resolution,           "Output resolution: '720p' (default), '1080p', or '4k'.  Higher resolutions require duration_seconds=8."] = "720p",
        generate_audio:      Annotated[bool,                 "Generate an audio track with the video (default True).  Only supported by non-fast Veo 3.x models.  Setting True with a fast/Veo-2 model raises an error."] = True,
        negative_prompt:     Annotated[Optional[str],        "What to exclude from the video.  Use descriptive nouns, not negations: 'blur, noise, shaky camera' not 'no blur'."] = None,
        seed:                Annotated[Optional[int],        "NOT supported by the Gemini API — raises ValueError if set.  On Vertex AI (future) this would be a uint32 that anchors the random state, helping produce visually similar output across runs with the same prompt.  Useful for iterating on a prompt while keeping subject/composition stable."] = None,
        enhance_prompt:      Annotated[bool,                 "Let Veo auto-expand the prompt for richer results (default True)."] = True,
        first_frame:         Annotated[Optional[str],        "Local file path or gs:// URI for the first frame (image-to-video)."] = None,
        last_frame:          Annotated[Optional[str],        "Local file path or gs:// URI for the last frame (frame interpolation).  Requires first_frame and duration_seconds=8."] = None,
        reference_images:    Annotated[Optional[list[str]],  "List of up to 3 local paths or gs:// URIs used as asset references.  Requires duration_seconds=8."] = None,
        person_generation:   Annotated[PersonPolicy,         "Person content policy: 'allow_all' (default), 'allow_adult', or 'dont_allow'.  Veo 3.x text-to-video only supports 'allow_all'."] = "allow_all",
    ) -> dict:
        """
        Generate a video with Google Veo and save it to disk.

        Returns a dict with: ok, model, prompt, duration_seconds, aspect_ratio,
        resolution, output_path, output_filename.
        """
        # ── Validate ──────────────────────────────────────────────────────────

        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        dur_str = str(duration_seconds)
        allowed = _allowed_durations(model)
        if dur_str not in allowed:
            raise ValueError(
                f"duration_seconds={duration_seconds} is not supported by model "
                f"'{model}'.  Allowed values: {sorted(allowed, key=int)}"
            )

        has_audio = _model_has_audio(model)
        if generate_audio and not has_audio:
            raise ValueError(
                f"generate_audio=True is not supported by model '{model}'. "
                "Use a non-fast Veo 3.x model (e.g. 'veo-3.1-generate-001') "
                "to generate audio."
            )
        if not generate_audio and has_audio:
            raise ValueError(
                f"generate_audio=False is not supported by the Gemini API for "
                f"non-fast Veo 3.x models (got '{model}'). "
                "Use 'veo-3.1-fast-generate-001' for silent video."
            )

        if last_frame and not first_frame:
            raise ValueError("last_frame requires first_frame to also be provided")

        needs_8s = bool(last_frame or reference_images or resolution in ("1080p", "4k"))
        if needs_8s and dur_str != "8":
            raise ValueError(
                "duration_seconds must be 8 when using last_frame, reference_images, "
                f"or resolution '{resolution}'"
            )

        if reference_images and len(reference_images) > 3:
            raise ValueError("reference_images supports at most 3 asset images")

        # ── Build GenerateVideosConfig ────────────────────────────────────────
        # Only pass parameters explicitly — avoids 400 errors on SDK versions
        # that don't support every field (e.g. enhance_prompt, person_generation).

        cfg: dict = {
            "aspect_ratio":     aspect_ratio,
            "duration_seconds": dur_str,
            "resolution":       resolution,
        }
        if person_generation != "allow_all":
            # "allow_all" is the API default for Veo 3.x text-to-video; skip it
            cfg["person_generation"] = person_generation
        if enhance_prompt is not True:
            # True is the API default; only pass when user explicitly sets False
            cfg["enhance_prompt"] = enhance_prompt
        if negative_prompt:
            cfg["negative_prompt"] = negative_prompt
        if seed is not None:
            raise ValueError(
                "seed is not supported by the Gemini API — only on Vertex AI. "
                "Remove seed= from your call."
            )
        if last_frame:
            cfg["last_frame"] = _load_image(last_frame)
        if reference_images:
            cfg["reference_images"] = [
                _gtypes.VideoGenerationReferenceImage(
                    image=_load_image(p), reference_type="asset"
                )
                for p in reference_images
            ]

        config = _gtypes.GenerateVideosConfig(**cfg)

        # ── Call API ──────────────────────────────────────────────────────────

        gen_kwargs: dict = {"model": model, "prompt": prompt, "config": config}
        if first_frame:
            gen_kwargs["image"] = _load_image(first_frame)

        operation = client.models.generate_videos(**gen_kwargs)

        started = time.time()
        while not operation.done:
            if time.time() - started > timeout_seconds:
                raise TimeoutError(
                    f"Veo generation timed out after {timeout_seconds}s"
                )
            time.sleep(poll_interval_seconds)
            operation = client.operations.get(operation)

        if getattr(operation, "error", None):
            raise VeoError(f"Veo generation failed: {operation.error}")

        if not getattr(operation, "response", None) or \
                not operation.response.generated_videos:
            raise VeoError("Veo generation completed without producing a video")

        # ── Save ──────────────────────────────────────────────────────────────

        filename = outdir / f"veo_{uuid.uuid4().hex[:10]}.mp4"
        video = operation.response.generated_videos[0]
        client.files.download(file=video.video)
        video.video.save(str(filename))

        return {
            "ok":               True,
            "model":            model,
            "prompt":           prompt,
            "duration_seconds": duration_seconds,
            "aspect_ratio":     aspect_ratio,
            "resolution":       resolution,
            "output_path":      str(filename.resolve()),
            "output_filename":  filename.name,
        }

    return LazyTool.from_function(
        generate_veo_video,
        name="generate_veo_video",
        guidance=(
            "Use this tool to generate videos with Google Veo. "
            "Always pass a descriptive text prompt. "
            "Optional: duration_seconds (4/6/8), aspect_ratio ('16:9' or '9:16'), "
            "resolution ('720p'/'1080p'/'4k'), generate_audio (True/False), "
            "negative_prompt, seed, enhance_prompt, "
            "first_frame (path or gs:// URI for image-to-video), "
            "last_frame (path or gs:// URI, requires duration_seconds=8), "
            "reference_images (list of paths/URIs, max 3, requires duration_seconds=8). "
            "Default: 8s, 16:9, 720p, audio on, text-to-video."
        ),
    )
