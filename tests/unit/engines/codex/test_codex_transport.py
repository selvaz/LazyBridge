"""Codex model transport selection (HTTP override for the Responses WebSocket crash, D126)."""

from __future__ import annotations

import pytest

from lazybridge.engines.codex import engine
from lazybridge.engines.coding import CodexPolicy


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LAZYBRIDGE_CODEX_TRANSPORT", raising=False)
    monkeypatch.delenv("LAZYBRIDGE_CODEX_HTTP_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_default_leaves_codex_transport_alone():
    assert engine._config_overrides(CodexPolicy()) == ()


def test_http_policy_adds_provider_override():
    overrides = engine._config_overrides(CodexPolicy(transport="http"))
    pid = engine.HTTP_PROVIDER_ID
    assert overrides[0] == f"model_provider={pid}"
    assert f'model_providers.{pid}.name="OpenAI"' in overrides
    assert f'model_providers.{pid}.base_url="{engine.DEFAULT_HTTP_BASE_URL}"' in overrides
    assert f"model_providers.{pid}.supports_websockets=false" in overrides
    assert f"model_providers.{pid}.requires_openai_auth=true" in overrides
    assert f'model_providers.{pid}.wire_api="responses"' in overrides


def test_env_selects_http_and_policy_wins(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_TRANSPORT", "http")
    assert engine._config_overrides(CodexPolicy())
    assert engine._config_overrides(CodexPolicy(transport="websocket")) == ()


def test_base_url_override(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_HTTP_BASE_URL", "https://example.test/codex")
    overrides = engine._config_overrides(CodexPolicy(transport="http"))
    assert any(o.endswith('base_url="https://example.test/codex"') for o in overrides)


def test_invalid_env_rejected(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_TRANSPORT", "carrier-pigeon")
    with pytest.raises(ValueError):
        engine._config_overrides(CodexPolicy())


def test_other_overrides_are_kept():
    overrides = engine._config_overrides(CodexPolicy(transport="http", web_search="live"))
    assert overrides[0] == 'web_search="live"'


def test_api_key_auth_uses_public_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    overrides = engine._config_overrides(CodexPolicy(transport="http"))
    assert any(o.endswith(f'base_url="{engine.API_KEY_HTTP_BASE_URL}"') for o in overrides)


def test_base_url_cannot_inject_toml(monkeypatch):
    monkeypatch.setenv("LAZYBRIDGE_CODEX_HTTP_BASE_URL", 'https://x" injected="1')
    with pytest.raises(ValueError):
        engine._config_overrides(CodexPolicy(transport="http"))
