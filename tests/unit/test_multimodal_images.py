"""Wave M1 — image input end-to-end across the framework.

Pins the multimodal-image contract:

* :class:`ImageContent` ships factory helpers for every shape callers
  reach for (URL, path, raw bytes, data URI), and ``_coerce_image``
  promotes the natural form they pass to :meth:`Agent.run`.
* :class:`Envelope` carries ``images=[...]`` end-to-end, surviving
  guard input rewrites, ``_inject_sources``, and the fallback re-run
  path.
* :class:`LLMEngine` builds a multimodal user :class:`Message`
  (``[TextContent, ImageContent...]``) when attachments are present
  and the resolved model supports vision; otherwise the plain-text
  path stays untouched.
* Per-model capability gating (substring match against the
  ``_VISION_CAPABLE_MODEL_PATTERNS`` table on each provider) drops
  attachments with a UserWarning by default — and raises
  :class:`UnsupportedFeatureError` when ``strict_multimodal=True``.
* :class:`Plan` and :class:`_ParallelAgent` propagate attachments to
  step 0 / every branch via :class:`Envelope`.
* HIL surfaces (HumanEngine, SupervisorEngine) display a short
  attachment hint instead of leaking raw base64 to the terminal.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from lazybridge import Agent, Step
from lazybridge.core.types import (
    AudioContent,
    ImageContent,
    TextContent,
    _coerce_image,
    _detect_image_mime,
)
from lazybridge.envelope import Envelope
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# ImageContent factory helpers
# ---------------------------------------------------------------------------


def test_image_from_url_guesses_media_type_from_path():
    img = ImageContent.from_url("https://example.com/cat.png")
    assert img.url == "https://example.com/cat.png"
    assert img.media_type == "image/png"


def test_image_from_url_explicit_media_type_wins():
    img = ImageContent.from_url("https://example.com/cat.gif", media_type="image/jpeg")
    assert img.media_type == "image/jpeg"


def test_image_from_bytes_detects_png_magic():
    png = b"\x89PNG\r\n\x1a\n" + b"X" * 50
    img = ImageContent.from_bytes(png)
    assert img.media_type == "image/png"
    assert img.base64_data and len(img.base64_data) > 0
    assert img.url is None


def test_image_from_bytes_detects_jpeg_magic():
    jpeg = b"\xff\xd8\xff\xe0" + b"X" * 50
    img = ImageContent.from_bytes(jpeg)
    assert img.media_type == "image/jpeg"


def test_image_from_bytes_detects_webp_magic():
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"X" * 50
    img = ImageContent.from_bytes(webp)
    assert img.media_type == "image/webp"


def test_image_from_bytes_unknown_magic_falls_back_to_jpeg():
    img = ImageContent.from_bytes(b"random-bytes-not-an-image")
    assert img.media_type == "image/jpeg"


def test_image_from_path_reads_and_encodes(tmp_path: Path):
    p = tmp_path / "cat.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"PIXEL")
    img = ImageContent.from_path(p)
    assert img.media_type == "image/png"
    assert img.base64_data and len(img.base64_data) > 0


def test_image_from_path_missing_raises():
    with pytest.raises(FileNotFoundError):
        ImageContent.from_path("/no/such/file.png")


def test_image_from_data_uri_parses_base64():
    img = ImageContent.from_data_uri("data:image/png;base64,iVBORw0KGgo=")
    assert img.media_type == "image/png"
    assert img.base64_data == "iVBORw0KGgo="


def test_detect_image_mime_recognises_gif():
    assert _detect_image_mime(b"GIF87a") == "image/gif"
    assert _detect_image_mime(b"GIF89a") == "image/gif"


def test_detect_image_mime_returns_none_for_garbage():
    assert _detect_image_mime(b"random") is None


# ---------------------------------------------------------------------------
# _coerce_image — accept the most natural form callers reach for
# ---------------------------------------------------------------------------


def test_coerce_image_url_string():
    img = _coerce_image("https://example.com/cat.png")
    assert isinstance(img, ImageContent)
    assert img.url == "https://example.com/cat.png"


def test_coerce_image_data_uri_string():
    img = _coerce_image("data:image/gif;base64,R0lGOD")
    assert img.media_type == "image/gif"
    assert img.base64_data == "R0lGOD"


def test_coerce_image_pathlib_path(tmp_path: Path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    img = _coerce_image(p)
    assert img.media_type == "image/png"


def test_coerce_image_bytes():
    img = _coerce_image(b"\x89PNG\r\n\x1a\n")
    assert img.media_type == "image/png"


def test_coerce_image_dict():
    img = _coerce_image({"url": "https://x.com/y.gif", "media_type": "image/gif"})
    assert img.url == "https://x.com/y.gif"
    assert img.media_type == "image/gif"


def test_coerce_image_passthrough_existing_instance():
    src = ImageContent.from_url("https://x.com/a.png")
    out = _coerce_image(src)
    assert out is src


def test_coerce_image_rejects_unsupported_type():
    with pytest.raises(TypeError, match="cannot coerce"):
        _coerce_image(42)


# ---------------------------------------------------------------------------
# Envelope — carry images / audio
# ---------------------------------------------------------------------------


def test_envelope_carries_images_field():
    env = Envelope(task="describe", images=[ImageContent.from_url("https://x.com/a.png")])
    assert env.images is not None
    assert len(env.images) == 1
    assert env.images[0].url == "https://x.com/a.png"


def test_envelope_default_images_is_none():
    env = Envelope(task="hello")
    assert env.images is None
    assert env.audio is None


def test_envelope_from_task_does_not_set_images():
    env = Envelope.from_task("hi")
    assert env.images is None


# ---------------------------------------------------------------------------
# Agent API — images= / audio= kwargs
# ---------------------------------------------------------------------------


def test_agent_run_images_kwarg_coerces_strings():
    env = Agent._to_envelope("describe", images=["https://x.com/a.png", "data:image/gif;base64,abc"])
    assert env.images is not None
    assert len(env.images) == 2
    assert env.images[0].url == "https://x.com/a.png"
    assert env.images[1].media_type == "image/gif"


def test_agent_run_images_kwarg_coerces_dicts():
    env = Agent._to_envelope("describe", images=[{"url": "https://x.com/y.png"}])
    assert env.images[0].url == "https://x.com/y.png"


def test_agent_run_envelope_with_images_passthrough():
    src = Envelope(task="x", images=[ImageContent.from_url("https://x.com/a.png")])
    out = Agent._to_envelope(src)
    assert out is src


def test_agent_run_conflict_envelope_and_kwarg_raises():
    src = Envelope(task="x", images=[ImageContent.from_url("https://x.com/a.png")])
    with pytest.raises(ValueError, match="exactly one channel"):
        Agent._to_envelope(src, images=["https://y.com/b.png"])


def test_agent_run_conflict_envelope_audio_and_kwarg_raises():
    src = Envelope(task="x", audio=AudioContent.from_url("https://x.com/a.wav"))
    with pytest.raises(ValueError, match="exactly one channel"):
        Agent._to_envelope(src, audio="https://y.com/b.wav")


# ---------------------------------------------------------------------------
# LLMEngine — multimodal user message construction
# ---------------------------------------------------------------------------


def test_llm_engine_builds_multimodal_user_message_when_vision_supported():
    """Vision-capable model: the user message becomes a content-block
    list ``[TextContent, ImageContent...]`` instead of a plain string."""
    from lazybridge.engines.llm import LLMEngine

    eng = LLMEngine("claude-opus-4-7")
    env = Envelope(task="what is this?", images=[ImageContent.from_url("https://x.com/cat.png")])
    content = eng._build_user_content("what is this?", env)
    assert isinstance(content, list)
    assert isinstance(content[0], TextContent)
    assert content[0].text == "what is this?"
    assert isinstance(content[1], ImageContent)
    assert content[1].url == "https://x.com/cat.png"


def test_llm_engine_drops_images_with_warning_when_vision_unsupported():
    """Text-only model (e.g. DeepSeek): attachments are stripped and
    a UserWarning is emitted; the message stays a plain string."""
    from lazybridge.engines.llm import LLMEngine

    eng = LLMEngine("deepseek-v4-pro", provider="deepseek")
    env = Envelope(task="hi", images=[ImageContent.from_url("https://x.com/a.png")])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        content = eng._build_user_content("hi", env)
    assert isinstance(content, str)
    assert content == "hi"
    assert any("does not support vision" in str(w.message) for w in record)


def test_llm_engine_strict_multimodal_raises_on_unsupported_vision():
    from lazybridge.core.providers.base import UnsupportedFeatureError
    from lazybridge.engines.llm import LLMEngine

    eng = LLMEngine("deepseek-v4-pro", provider="deepseek", strict_multimodal=True)
    env = Envelope(task="hi", images=[ImageContent.from_url("https://x.com/a.png")])
    with pytest.raises(UnsupportedFeatureError, match="vision"):
        eng._build_user_content("hi", env)


def test_llm_engine_no_attachments_returns_plain_string():
    from lazybridge.engines.llm import LLMEngine

    eng = LLMEngine("claude-opus-4-7")
    env = Envelope(task="hi")
    out = eng._build_user_content("hi", env)
    assert out == "hi"


def test_llm_engine_empty_images_list_treated_as_absent():
    from lazybridge.engines.llm import LLMEngine

    eng = LLMEngine("claude-opus-4-7")
    env = Envelope(task="hi", images=[])
    out = eng._build_user_content("hi", env)
    assert out == "hi"


# ---------------------------------------------------------------------------
# Provider capability matrix
# ---------------------------------------------------------------------------


def test_anthropic_vision_capability_includes_claude_3_plus():
    from lazybridge.core.providers.anthropic import AnthropicProvider

    assert AnthropicProvider.supports_vision("claude-opus-4-7")
    assert AnthropicProvider.supports_vision("claude-sonnet-4-6")
    assert AnthropicProvider.supports_vision("claude-3-5-sonnet")
    assert AnthropicProvider.supports_vision("claude-3-haiku")
    assert not AnthropicProvider.supports_vision("claude-2")


def test_openai_vision_capability_excludes_gpt_3_5():
    from lazybridge.core.providers.openai import OpenAIProvider

    assert OpenAIProvider.supports_vision("gpt-4o")
    assert OpenAIProvider.supports_vision("gpt-4-turbo")
    assert OpenAIProvider.supports_vision("gpt-5")
    assert not OpenAIProvider.supports_vision("gpt-3.5-turbo")


def test_google_vision_capability_includes_gemini_1_5_plus():
    from lazybridge.core.providers.google import GoogleProvider

    assert GoogleProvider.supports_vision("gemini-1.5-pro")
    assert GoogleProvider.supports_vision("gemini-2.5-pro")
    assert GoogleProvider.supports_vision("gemini-3.1-pro-preview")
    assert not GoogleProvider.supports_vision("gemini-1.0-pro")


def test_deepseek_explicitly_text_only():
    from lazybridge.core.providers.deepseek import DeepSeekProvider

    assert not DeepSeekProvider.supports_vision("deepseek-v4-pro")
    assert not DeepSeekProvider.supports_vision("deepseek-v4-flash")
    assert not DeepSeekProvider.supports_audio("deepseek-v4-pro")


def test_lmstudio_optimistic_capability():
    """LM Studio depends on the loaded model — we report True so the
    server can decide.  Strict mode + a known-text-only model id is
    the way to fail fast here."""
    from lazybridge.core.providers.lmstudio import LMStudioProvider

    assert LMStudioProvider.supports_vision("qwen2.5-vl-7b")
    assert LMStudioProvider.supports_vision("local-model")  # unknown defaults to True


def test_litellm_optimistic_capability():
    """LiteLLM forwards to the OpenAI shape — capability decided at
    the backend.  Optimistic ``True`` lets callers route any model
    through the bridge."""
    pytest.importorskip("litellm")
    from lazybridge.core.providers.litellm import LiteLLMProvider

    assert LiteLLMProvider.supports_vision("anything")


def test_capability_returns_false_for_none_model():
    """An unset model can't be classified — return False so callers
    don't accidentally send images to a provider that may or may not
    support them."""
    from lazybridge.core.providers.anthropic import AnthropicProvider

    assert not AnthropicProvider.supports_vision(None)
    assert not AnthropicProvider.supports_vision("")


# ---------------------------------------------------------------------------
# Plan — step-0 attachment propagation
# ---------------------------------------------------------------------------


def test_plan_step_0_receives_images():
    """Step 0's ``task=`` defaults to ``from_prev``, which for the
    very first step resolves to the original input envelope.  The
    attachments must reach the step's agent unchanged."""
    seen: dict = {}

    class Capturing(MockAgent):
        async def run(self, task):
            if isinstance(task, Envelope):
                seen["images"] = task.images
            return Envelope(task=str(task), payload="captured")

    a = Capturing("out", name="cap")
    plan = Agent.from_plan(Step(target=a, name="s0"))
    img = ImageContent.from_url("https://x.com/cat.png")
    plan(Envelope(task="describe", images=[img]))

    assert seen["images"] is not None
    assert len(seen["images"]) == 1
    assert seen["images"][0].url == "https://x.com/cat.png"


def test_plan_step_n_does_not_inherit_images_from_step_0():
    """Step N>0 receives upstream output (text), not the original
    images.  The propagation rule is intentional — multimodal input
    is per-call, not threaded through every step automatically.
    Re-attaching for downstream steps is the user's responsibility."""
    sees: list = []

    class Capturing(MockAgent):
        async def run(self, task):
            if isinstance(task, Envelope):
                sees.append(task.images)
            return Envelope(task=str(task), payload="result")

    s0 = Capturing("first", name="s0")
    s1 = Capturing("second", name="s1")
    plan = Agent.from_plan(
        Step(target=s0, name="s0"),
        Step(target=s1, name="s1"),
    )
    plan(Envelope(task="describe", images=[ImageContent.from_url("https://x.com/a.png")]))

    # s0 sees the original image; s1 inherits prev.images which the
    # output Envelope from s0 didn't populate, so it sees None.
    assert sees[0] is not None and len(sees[0]) == 1
    assert sees[1] is None


# ---------------------------------------------------------------------------
# _ParallelAgent — fan-out preserves attachments across branches
# ---------------------------------------------------------------------------


def test_parallel_branches_each_receive_the_same_image():
    sees: list = []

    class Capturing(MockAgent):
        async def run(self, task):
            if isinstance(task, Envelope):
                sees.append(task.images)
            return Envelope(task=str(task), payload="ok")

    fan = Agent.from_parallel(Capturing("A", name="a"), Capturing("B", name="b"))
    fan(Envelope(task="describe", images=[ImageContent.from_url("https://x.com/y.png")]))

    assert len(sees) == 2
    assert all(s is not None and len(s) == 1 for s in sees)
    assert all(s[0].url == "https://x.com/y.png" for s in sees)


# ---------------------------------------------------------------------------
# HIL — surface a human-readable attachment descriptor instead of base64
# ---------------------------------------------------------------------------


def test_hil_format_attachments_url_is_human_readable():
    from lazybridge.ext.hil.human import _format_attachments

    out = _format_attachments(
        [ImageContent.from_url("https://x.com/cat.png")],
        None,
    )
    assert "[attached images:" in out
    assert "image/png" in out
    assert "https://x.com/cat.png" in out


def test_hil_format_attachments_base64_shows_size_not_data():
    from lazybridge.ext.hil.human import _format_attachments

    big_b64 = "X" * 8000
    img = ImageContent(base64_data=big_b64, media_type="image/png")
    out = _format_attachments([img], None)
    assert "X" * 100 not in out  # raw base64 NEVER appears
    assert "bytes inline" in out


def test_hil_format_attachments_empty_returns_empty_string():
    from lazybridge.ext.hil.human import _format_attachments

    assert _format_attachments(None, None) == ""
    assert _format_attachments([], None) == ""


def test_hil_format_attachments_audio_branch():
    from lazybridge.ext.hil.human import _format_attachments

    out = _format_attachments(None, AudioContent.from_url("https://x.com/y.wav"))
    assert "[attached audio:" in out
    assert "https://x.com/y.wav" in out


# ---------------------------------------------------------------------------
# Session — opt-in base64 redaction helper
# ---------------------------------------------------------------------------


def test_redact_binary_attachments_strips_long_base64():
    from lazybridge.session import redact_binary_attachments

    payload = {
        "messages": [
            {
                "content": [
                    {"type": "image", "base64_data": "X" * 200, "media_type": "image/png"},
                ],
            },
        ],
    }
    out = redact_binary_attachments(payload)
    assert out["messages"][0]["content"][0]["base64_data"].startswith("<base64 redacted")
    assert "image/png" in out["messages"][0]["content"][0]["base64_data"]


def test_redact_preserves_short_strings_and_urls():
    from lazybridge.session import redact_binary_attachments

    payload = {"img": {"url": "https://x.com/a.png", "base64_data": "short"}}
    out = redact_binary_attachments(payload)
    assert out["img"]["url"] == "https://x.com/a.png"
    assert out["img"]["base64_data"] == "short"


def test_redact_handles_dataclass_instances():
    from lazybridge.session import redact_binary_attachments

    img = ImageContent.from_bytes(b"\x89PNG\r\n\x1a\n" + b"X" * 200)
    out = redact_binary_attachments({"image": img})
    # Walks dataclass into a dict, redacts the long base64.
    assert isinstance(out["image"], dict)
    assert "redacted" in out["image"]["base64_data"]
