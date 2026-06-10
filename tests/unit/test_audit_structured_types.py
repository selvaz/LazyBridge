"""Regression tests for core-audit fixes in structured.py / types.py."""

from __future__ import annotations

from lazybridge.core.structured import normalize_json_schema, parse_structured_output
from lazybridge.core.types import AudioContent, ImageContent


def test_parse_tolerates_prose_around_fenced_json():
    """Models often write 'Here is the JSON:' before the fence — that used
    to be an immediate parse error."""
    out = parse_structured_output('Here is the JSON:\n```json\n{"a": 1}\n```\nDone!', {"type": "object"})
    assert out == {"a": 1}


def test_normalize_closes_objects_without_explicit_type():
    s = normalize_json_schema({"properties": {"a": {"type": "string"}}})
    assert s["additionalProperties"] is False


def test_image_data_uri_with_extra_params_keeps_payload_verbatim():
    img = ImageContent.from_data_uri("data:image/png;charset=utf-8;base64,iVBORw0KGgo=")
    assert img.base64_data == "iVBORw0KGgo="
    assert img.media_type == "image/png"


def test_audio_data_uri_with_extra_params_keeps_payload_verbatim():
    a = AudioContent.from_data_uri("data:audio/wav;codec=pcm;base64,UklGRg==")
    assert a.base64_data == "UklGRg=="


def test_dict_value_type_reaches_schema():
    from lazybridge.core.tool_schema import _annotation_to_schema

    schema = _annotation_to_schema(dict[str, int])
    assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}
