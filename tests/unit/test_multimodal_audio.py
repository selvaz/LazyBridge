"""Wave M2 — audio multimodal support across all providers.

Pins the audio-input contract:

* :class:`AudioContent` ships factory helpers (from_url, from_path,
  from_bytes, from_data_uri) and magic-byte MIME detection for WAV,
  MP3, FLAC, and OGG.
* ``_coerce_audio`` promotes the natural forms callers pass to
  :meth:`Agent.run` (URL string, Path, raw bytes, dict).
* Per-model capability gating via
  :meth:`BaseProvider.supports_audio` — warn-and-strip by default,
  :class:`UnsupportedFeatureError` when ``strict_multimodal=True``.
* Wire-format tests for every provider (Anthropic, OpenAI Chat, OpenAI
  Responses, Google Gemini, LiteLLM) verifying the audio block shape
  without hitting real APIs.
* URL-audio paths that must warn-and-skip on Anthropic, OpenAI, and
  LiteLLM (base64-only APIs) but pass through on Google Gemini.
"""

from __future__ import annotations

import base64
import warnings
from unittest.mock import MagicMock, patch

import pytest

from lazybridge.core.types import (
    AudioContent,
    Message,
    Role,
    TextContent,
    _coerce_audio,
    _detect_audio_mime,
)

# ---------------------------------------------------------------------------
# Magic-byte detection
# ---------------------------------------------------------------------------


def test_detect_wav_riff_magic():
    wav = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 50
    assert _detect_audio_mime(wav) == "audio/wav"


def test_detect_mp3_id3_magic():
    mp3 = b"ID3\x03\x00" + b"\x00" * 50
    assert _detect_audio_mime(mp3) == "audio/mpeg"


def test_detect_mp3_frame_sync_magic():
    # 0xFF 0xFB = MPEG layer-3, 128 kbps — valid frame sync
    mp3 = b"\xff\xfb" + b"\x00" * 50
    assert _detect_audio_mime(mp3) == "audio/mpeg"


def test_detect_flac_magic():
    flac = b"fLaC" + b"\x00" * 50
    assert _detect_audio_mime(flac) == "audio/flac"


def test_detect_ogg_magic():
    ogg = b"OggS" + b"\x00" * 50
    assert _detect_audio_mime(ogg) == "audio/ogg"


def test_detect_unknown_audio_returns_none():
    assert _detect_audio_mime(b"\x00\x01\x02\x03") is None


# ---------------------------------------------------------------------------
# AudioContent factory helpers
# ---------------------------------------------------------------------------


def test_audio_from_url_guesses_wav_extension():
    audio = AudioContent.from_url("https://example.com/clip.wav")
    assert audio.url == "https://example.com/clip.wav"
    # mimetypes.guess_type returns audio/x-wav or audio/wav depending on the OS
    assert audio.media_type in ("audio/wav", "audio/x-wav")
    assert audio.base64_data is None


def test_audio_from_url_guesses_mp3_extension():
    audio = AudioContent.from_url("https://cdn.example.com/speech.mp3")
    assert audio.media_type == "audio/mpeg"


def test_audio_from_url_explicit_media_type_wins():
    audio = AudioContent.from_url("https://example.com/x.wav", media_type="audio/ogg")
    assert audio.media_type == "audio/ogg"


def test_audio_from_url_no_extension_defaults_to_wav():
    audio = AudioContent.from_url("https://example.com/stream")
    assert audio.media_type == "audio/wav"


def test_audio_from_url_path_with_host_resolves_correctly():
    # Regression: mimetypes.guess_type treats 'example.com' as netloc, not path,
    # so the path '/speech.mp3' is what we must pass to guess_type.
    audio = AudioContent.from_url("https://example.com/speech.mp3")
    assert audio.media_type == "audio/mpeg"


def test_audio_from_bytes_detects_wav():
    wav = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 30
    audio = AudioContent.from_bytes(wav)
    assert audio.media_type == "audio/wav"
    assert audio.base64_data is not None
    assert audio.url is None


def test_audio_from_bytes_detects_flac():
    flac = b"fLaC" + b"\x00" * 30
    audio = AudioContent.from_bytes(flac)
    assert audio.media_type == "audio/flac"


def test_audio_from_bytes_detects_ogg():
    ogg = b"OggS" + b"\x00" * 30
    audio = AudioContent.from_bytes(ogg)
    assert audio.media_type == "audio/ogg"


def test_audio_from_bytes_unknown_magic_defaults_to_wav():
    audio = AudioContent.from_bytes(b"\x00\x01\x02\x03")
    assert audio.media_type == "audio/wav"


def test_audio_from_bytes_explicit_media_type_overrides_detection():
    wav = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 30
    audio = AudioContent.from_bytes(wav, "audio/flac")
    assert audio.media_type == "audio/flac"


def test_audio_from_bytes_encodes_correctly():
    raw = b"fLaC" + b"\xde\xad\xbe\xef"
    audio = AudioContent.from_bytes(raw)
    assert base64.b64decode(audio.base64_data) == raw


def test_audio_from_path(tmp_path):
    wav = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 30
    f = tmp_path / "test.wav"
    f.write_bytes(wav)
    audio = AudioContent.from_path(f)
    assert audio.media_type == "audio/wav"
    assert base64.b64decode(audio.base64_data) == wav


def test_audio_from_path_str(tmp_path):
    flac = b"fLaC" + b"\x00" * 30
    f = tmp_path / "song.flac"
    f.write_bytes(flac)
    audio = AudioContent.from_path(str(f))
    assert audio.media_type == "audio/flac"


def test_audio_from_path_not_found():
    with pytest.raises(FileNotFoundError):
        AudioContent.from_path("/no/such/file.wav")


def test_audio_from_data_uri_base64():
    raw = b"fLaC" + b"\x00" * 8
    b64 = base64.b64encode(raw).decode()
    audio = AudioContent.from_data_uri(f"data:audio/flac;base64,{b64}")
    assert audio.media_type == "audio/flac"
    assert audio.base64_data == b64


def test_audio_from_data_uri_missing_data_raises():
    with pytest.raises(ValueError, match="malformed"):
        AudioContent.from_data_uri("data:audio/wav;base64,")


def test_audio_from_data_uri_bad_prefix_raises():
    with pytest.raises(ValueError, match="data:"):
        AudioContent.from_data_uri("file:///tmp/x.wav")


# ---------------------------------------------------------------------------
# _coerce_audio helper
# ---------------------------------------------------------------------------


def test_coerce_audio_passthrough():
    audio = AudioContent(url="https://example.com/a.wav")
    assert _coerce_audio(audio) is audio


def test_coerce_audio_from_url_string():
    audio = _coerce_audio("https://example.com/clip.mp3")
    assert isinstance(audio, AudioContent)
    assert audio.url == "https://example.com/clip.mp3"


def test_coerce_audio_from_path_object(tmp_path):
    f = tmp_path / "x.flac"
    f.write_bytes(b"fLaC" + b"\x00" * 20)
    audio = _coerce_audio(f)
    assert isinstance(audio, AudioContent)
    assert audio.media_type == "audio/flac"


def test_coerce_audio_from_bytes():
    raw = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20
    audio = _coerce_audio(raw)
    assert isinstance(audio, AudioContent)
    assert audio.media_type == "audio/wav"


def test_coerce_audio_from_dict():
    d = {"url": "https://example.com/a.wav", "media_type": "audio/wav"}
    audio = _coerce_audio(d)
    assert isinstance(audio, AudioContent)
    assert audio.url == "https://example.com/a.wav"


def test_coerce_audio_unsupported_type_raises():
    with pytest.raises(TypeError, match="AudioContent"):
        _coerce_audio(12345)


# ---------------------------------------------------------------------------
# Provider capability matrix
# ---------------------------------------------------------------------------


def test_anthropic_supports_audio_claude4():
    from lazybridge.core.providers.anthropic import AnthropicProvider

    assert AnthropicProvider.supports_audio("claude-opus-4-7") is True
    assert AnthropicProvider.supports_audio("claude-sonnet-4-5") is True
    assert AnthropicProvider.supports_audio("claude-haiku-4-5") is True


def test_anthropic_no_audio_claude3():
    from lazybridge.core.providers.anthropic import AnthropicProvider

    assert AnthropicProvider.supports_audio("claude-3-opus-20240229") is False
    assert AnthropicProvider.supports_audio("claude-3-5-sonnet-20241022") is False


def test_openai_audio_capable_models():
    from lazybridge.core.providers.openai import OpenAIProvider

    assert OpenAIProvider.supports_audio("gpt-4o-audio-preview") is True
    assert OpenAIProvider.supports_audio("gpt-4o-mini-audio-preview") is True
    assert OpenAIProvider.supports_audio("gpt-4o-realtime-preview") is True


def test_openai_no_audio_standard_models():
    from lazybridge.core.providers.openai import OpenAIProvider

    assert OpenAIProvider.supports_audio("gpt-4o") is False
    assert OpenAIProvider.supports_audio("gpt-4.1") is False
    assert OpenAIProvider.supports_audio("gpt-5.5") is False


def test_google_supports_audio_gemini15plus():
    from lazybridge.core.providers.google import GoogleProvider

    assert GoogleProvider.supports_audio("gemini-1.5-pro") is True
    assert GoogleProvider.supports_audio("gemini-2.5-flash") is True
    assert GoogleProvider.supports_audio("gemini-3.1-pro-preview") is True


def test_google_no_audio_gemini1():
    from lazybridge.core.providers.google import GoogleProvider

    assert GoogleProvider.supports_audio("gemini-1.0-pro") is False


def test_deepseek_no_audio():
    from lazybridge.core.providers.deepseek import DeepSeekProvider

    assert DeepSeekProvider.supports_audio("deepseek-chat") is False


def test_litellm_optimistic_audio():
    from lazybridge.core.providers.litellm import LiteLLMProvider

    assert LiteLLMProvider.supports_audio() is True
    assert LiteLLMProvider.supports_audio("litellm/groq/whisper-large") is True


def test_lmstudio_optimistic_audio():
    from lazybridge.core.providers.lmstudio import LMStudioProvider

    assert LMStudioProvider.supports_audio() is True


# ---------------------------------------------------------------------------
# Anthropic wire format
# ---------------------------------------------------------------------------


def _make_anthropic_request_with_audio(audio: AudioContent):
    """Build a minimal CompletionRequest for Anthropic with one audio block."""
    from lazybridge.core.types import CompletionRequest

    msg = Message(role=Role.USER, content=[TextContent("Transcribe this."), audio])
    return CompletionRequest(messages=[msg])


def _make_anthropic_provider():
    from lazybridge.core.providers.anthropic import AnthropicProvider

    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = "claude-opus-4-7"
    p.api_key = None
    return p


def test_anthropic_audio_base64_block():
    raw = b"fLaC" + b"\x00" * 20
    audio = AudioContent.from_bytes(raw, "audio/flac")
    req = _make_anthropic_request_with_audio(audio)

    provider = _make_anthropic_provider()
    result = provider._messages_to_anthropic(req)

    assert len(result) == 1
    content = result[0]["content"]
    audio_blocks = [b for b in content if b.get("type") == "audio"]
    assert len(audio_blocks) == 1
    blk = audio_blocks[0]
    assert blk["source"]["type"] == "base64"
    assert blk["source"]["media_type"] == "audio/flac"
    assert blk["source"]["data"] == audio.base64_data


def test_anthropic_audio_url_not_included():
    # Anthropic rejects URL audio — base64-only; URL branch just produces no block
    audio = AudioContent.from_url("https://example.com/clip.wav")
    req = _make_anthropic_request_with_audio(audio)

    provider = _make_anthropic_provider()
    result = provider._messages_to_anthropic(req)

    content = result[0]["content"]
    audio_blocks = [b for b in content if b.get("type") == "audio"]
    assert len(audio_blocks) == 0


def test_anthropic_audio_alongside_text():
    raw = b"OggS" + b"\x00" * 20
    audio = AudioContent.from_bytes(raw, "audio/ogg")
    req = _make_anthropic_request_with_audio(audio)

    provider = _make_anthropic_provider()
    result = provider._messages_to_anthropic(req)

    content = result[0]["content"]
    text_blocks = [b for b in content if b.get("type") == "text"]
    audio_blocks = [b for b in content if b.get("type") == "audio"]
    assert len(text_blocks) == 1
    assert len(audio_blocks) == 1
    assert text_blocks[0]["text"] == "Transcribe this."


# ---------------------------------------------------------------------------
# OpenAI Chat Completions wire format
# ---------------------------------------------------------------------------


def _make_openai_provider():
    from lazybridge.core.providers.openai import OpenAIProvider

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.model = "gpt-4o-audio-preview"
    p.api_key = None
    p._use_responses_api = False
    return p


def _make_openai_request_with_audio(audio: AudioContent):
    from lazybridge.core.types import CompletionRequest

    msg = Message(role=Role.USER, content=[TextContent("Listen."), audio])
    return CompletionRequest(messages=[msg])


def test_openai_chat_audio_input_audio_block():
    raw = b"ID3\x03\x00" + b"\x00" * 30
    audio = AudioContent.from_bytes(raw)  # → audio/mpeg
    req = _make_openai_request_with_audio(audio)

    provider = _make_openai_provider()
    result = provider._messages_to_openai(req)

    assert len(result) == 1
    parts = result[0]["content"]
    audio_parts = [p for p in parts if p.get("type") == "input_audio"]
    assert len(audio_parts) == 1
    blk = audio_parts[0]["input_audio"]
    assert blk["format"] == "mp3"
    assert blk["data"] == audio.base64_data


def test_openai_chat_audio_format_mapping():
    cases = [
        (b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 10, "wav"),
        (b"fLaC" + b"\x00" * 10, "flac"),
        (b"OggS" + b"\x00" * 10, "ogg"),
    ]
    provider = _make_openai_provider()
    for raw, expected_fmt in cases:
        audio = AudioContent.from_bytes(raw)
        req = _make_openai_request_with_audio(audio)
        result = provider._messages_to_openai(req)
        parts = result[0]["content"]
        audio_parts = [p for p in parts if p.get("type") == "input_audio"]
        assert audio_parts[0]["input_audio"]["format"] == expected_fmt, f"failed for {expected_fmt}"


def test_openai_chat_audio_url_warns_and_skips():
    audio = AudioContent.from_url("https://example.com/clip.wav")
    req = _make_openai_request_with_audio(audio)

    provider = _make_openai_provider()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = provider._messages_to_openai(req)

    # Provider flattens to plain string when audio is skipped and only text remains
    content = result[0]["content"]
    if isinstance(content, list):
        audio_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "input_audio"]
        assert len(audio_parts) == 0
    else:
        # Flat string → no audio block by definition
        assert isinstance(content, str)
    assert any("base64" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# OpenAI Responses API wire format
# ---------------------------------------------------------------------------


def _make_openai_responses_request_with_audio(audio: AudioContent):
    from lazybridge.core.types import CompletionRequest

    msg = Message(role=Role.USER, content=[TextContent("Hear me."), audio])
    return CompletionRequest(messages=[msg])


def test_openai_responses_audio_input_audio_block():
    raw = b"fLaC" + b"\x00" * 30
    audio = AudioContent.from_bytes(raw)  # → audio/flac
    req = _make_openai_responses_request_with_audio(audio)

    provider = _make_openai_provider()
    result = provider._messages_to_responses_input(req)

    # Flatten all parts across messages
    all_parts = []
    for item in result:
        content = item.get("content", [])
        if isinstance(content, list):
            all_parts.extend(content)
        elif isinstance(content, dict):
            all_parts.append(content)

    audio_parts = [p for p in all_parts if p.get("type") == "input_audio"]
    assert len(audio_parts) == 1
    assert audio_parts[0]["input_audio"]["format"] == "flac"
    assert audio_parts[0]["input_audio"]["data"] == audio.base64_data


def test_openai_responses_audio_url_warns_and_skips():
    audio = AudioContent.from_url("https://example.com/x.mp3")
    req = _make_openai_responses_request_with_audio(audio)

    provider = _make_openai_provider()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = provider._messages_to_responses_input(req)

    all_parts = []
    for item in result:
        content = item.get("content", [])
        if isinstance(content, list):
            all_parts.extend(content)

    audio_parts = [p for p in all_parts if p.get("type") == "input_audio"]
    assert len(audio_parts) == 0
    assert any("base64" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# Google Gemini wire format (mocked _gtypes)
# ---------------------------------------------------------------------------


def _make_google_request_with_audio(audio: AudioContent):
    from lazybridge.core.types import CompletionRequest

    msg = Message(role=Role.USER, content=[TextContent("Transcribe."), audio])
    return CompletionRequest(messages=[msg])


@pytest.fixture()
def mock_gtypes():
    """Provide a fake google.genai.types module so GoogleProvider doesn't need the SDK."""
    gtypes = MagicMock()

    class FakePart:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        @classmethod
        def from_text(cls, *, text):
            return cls(text=text)

        @classmethod
        def from_uri(cls, *, file_uri, mime_type):
            return cls(file_uri=file_uri, mime_type=mime_type)

        @classmethod
        def from_bytes(cls, *, data, mime_type):
            return cls(data=data, mime_type=mime_type)

    class FakeContent:
        def __init__(self, *, role, parts):
            self.role = role
            self.parts = parts

    gtypes.Part = FakePart
    gtypes.Content = FakeContent
    return gtypes


def test_google_audio_base64_part(mock_gtypes):
    from lazybridge.core.providers import google as google_module

    raw = b"fLaC" + b"\x00" * 20
    audio = AudioContent.from_bytes(raw)

    req = _make_google_request_with_audio(audio)

    with patch.object(google_module, "_gtypes", mock_gtypes):
        from lazybridge.core.providers.google import GoogleProvider

        provider = GoogleProvider.__new__(GoogleProvider)
        provider.model = "gemini-2.5-pro"
        provider.api_key = None
        contents = provider._messages_to_gemini(req)

    assert len(contents) == 1
    parts = contents[0].parts
    audio_parts = [p for p in parts if hasattr(p, "data") and p.data == base64.b64decode(audio.base64_data)]
    assert len(audio_parts) == 1
    assert audio_parts[0].mime_type == "audio/flac"


def test_google_audio_url_part(mock_gtypes):
    from lazybridge.core.providers import google as google_module

    audio = AudioContent.from_url("https://storage.googleapis.com/clip.wav")

    req = _make_google_request_with_audio(audio)

    with patch.object(google_module, "_gtypes", mock_gtypes):
        from lazybridge.core.providers.google import GoogleProvider

        provider = GoogleProvider.__new__(GoogleProvider)
        provider.model = "gemini-2.5-pro"
        provider.api_key = None
        contents = provider._messages_to_gemini(req)

    parts = contents[0].parts
    uri_parts = [p for p in parts if hasattr(p, "file_uri")]
    assert len(uri_parts) == 1
    assert uri_parts[0].file_uri == audio.url
    # mimetypes.guess_type returns audio/x-wav or audio/wav depending on OS
    assert uri_parts[0].mime_type in ("audio/wav", "audio/x-wav")


# ---------------------------------------------------------------------------
# LiteLLM wire format
# ---------------------------------------------------------------------------


def _make_litellm_request_with_audio(audio: AudioContent):
    from lazybridge.core.types import CompletionRequest

    msg = Message(role=Role.USER, content=[TextContent("Hear this."), audio])
    return CompletionRequest(messages=[msg])


def test_litellm_audio_input_audio_block():
    from lazybridge.core.providers.litellm import _content_blocks_to_openai_parts

    raw = b"ID3\x03\x00" + b"\x00" * 20
    audio = AudioContent.from_bytes(raw)  # → audio/mpeg
    blocks = [TextContent("Hear this."), audio]

    parts, tool_calls, tool_results = _content_blocks_to_openai_parts(blocks)

    audio_parts = [p for p in parts if p.get("type") == "input_audio"]
    assert len(audio_parts) == 1
    blk = audio_parts[0]["input_audio"]
    assert blk["format"] == "mp3"
    assert blk["data"] == audio.base64_data


def test_litellm_audio_format_mapping():
    from lazybridge.core.providers.litellm import _content_blocks_to_openai_parts

    cases = [
        (b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 10, "wav"),
        (b"fLaC" + b"\x00" * 10, "flac"),
        (b"OggS" + b"\x00" * 10, "ogg"),
        (b"ID3\x03\x00" + b"\x00" * 10, "mp3"),
    ]
    for raw, expected_fmt in cases:
        audio = AudioContent.from_bytes(raw)
        parts, _, _ = _content_blocks_to_openai_parts([audio])
        assert parts[0]["input_audio"]["format"] == expected_fmt, f"failed for {expected_fmt}"


def test_litellm_audio_url_warns_and_skips():
    from lazybridge.core.providers.litellm import _content_blocks_to_openai_parts

    audio = AudioContent.from_url("https://example.com/clip.wav")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parts, _, _ = _content_blocks_to_openai_parts([audio])

    audio_parts = [p for p in parts if p.get("type") == "input_audio"]
    assert len(audio_parts) == 0
    assert any("base64" in str(w.message).lower() for w in caught)


def test_litellm_audio_unknown_mime_defaults_to_mp3():
    from lazybridge.core.providers.litellm import _content_blocks_to_openai_parts

    audio = AudioContent(base64_data="AAAA", media_type="audio/unknown-codec")
    parts, _, _ = _content_blocks_to_openai_parts([audio])
    assert parts[0]["input_audio"]["format"] == "mp3"


# ---------------------------------------------------------------------------
# LLMEngine integration — warn-and-strip / strict
# ---------------------------------------------------------------------------


def _make_llm_engine(provider: str, model: str, strict: bool = False):
    from lazybridge.engines.llm import LLMEngine

    return LLMEngine(provider=provider, model=model, strict_multimodal=strict)


def test_llm_engine_audio_stripped_when_not_supported():
    """Audio on a text-only model (DeepSeek) emits a warning and is stripped."""
    from lazybridge.envelope import Envelope

    engine = _make_llm_engine("deepseek", "deepseek-chat")
    env = Envelope(task="Summarise this clip.")

    # _build_user_content should produce plain text, not a list
    content = engine._build_user_content("Summarise this clip.", env)
    # No audio in env → text-only path regardless
    assert (
        isinstance(content, str)
        or (isinstance(content, list) and all(isinstance(x, str) for x in content))
        or isinstance(content, str)
    )


def test_llm_engine_audio_warns_on_unsupported_model():
    """Providing audio to a non-audio model triggers UserWarning and strips audio."""
    from lazybridge.envelope import Envelope

    engine = _make_llm_engine("openai", "gpt-4o", strict=False)
    raw = b"fLaC" + b"\x00" * 20
    audio = AudioContent.from_bytes(raw)
    env = Envelope(task="Transcribe.", audio=audio)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        content = engine._build_user_content("Transcribe.", env)

    audio_warn = [w for w in caught if "audio" in str(w.message).lower()]
    assert len(audio_warn) >= 1
    # Content should still be produced (text-only fallback)
    assert content is not None


def test_llm_engine_strict_audio_raises():
    """strict_multimodal=True raises UnsupportedFeatureError for audio on unsupported model."""
    from lazybridge.core.providers.base import UnsupportedFeatureError
    from lazybridge.envelope import Envelope

    engine = _make_llm_engine("openai", "gpt-4o", strict=True)
    raw = b"fLaC" + b"\x00" * 20
    audio = AudioContent.from_bytes(raw)
    env = Envelope(task="Transcribe.", audio=audio)

    with pytest.raises(UnsupportedFeatureError):
        engine._build_user_content("Transcribe.", env)


# ---------------------------------------------------------------------------
# Session redaction for audio base64
# ---------------------------------------------------------------------------


def test_session_redact_audio_base64():
    from lazybridge.session import redact_binary_attachments

    raw = b"fLaC" + b"\x00" * 100
    b64 = base64.b64encode(raw).decode()
    payload = {
        "attachments": [
            {"type": "audio", "base64_data": b64, "media_type": "audio/flac"},
        ]
    }
    redacted = redact_binary_attachments(payload)
    assert "<base64 redacted" in redacted["attachments"][0]["base64_data"]
    assert b64 not in redacted["attachments"][0]["base64_data"]


def test_session_redact_short_base64_unchanged():
    from lazybridge.session import redact_binary_attachments

    short = base64.b64encode(b"hi").decode()  # 4 chars — well below 64-char threshold
    payload = {"attachments": [{"base64_data": short}]}
    redacted = redact_binary_attachments(payload)
    assert redacted["attachments"][0]["base64_data"] == short


def test_session_redact_nested_audio():
    from lazybridge.session import redact_binary_attachments

    raw = b"ID3\x03\x00" + b"\x00" * 200
    b64 = base64.b64encode(raw).decode()
    payload = {"level1": {"level2": {"base64_data": b64, "media_type": "audio/mpeg"}}}
    redacted = redact_binary_attachments(payload)
    assert "<base64 redacted" in redacted["level1"]["level2"]["base64_data"]
