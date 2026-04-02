"""Shared helpers for opt-in live provider tests."""

from __future__ import annotations

import importlib.util
import os

import pytest


LIVE_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.5-flash-lite",
}

_SDK_MODULES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google.genai",
}


def live_model(provider: str) -> str:
    return LIVE_MODELS[provider]


def require_live_provider(provider: str) -> str:
    """Skip unless the provider credentials and SDK are available."""
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        env_name = "OPENAI_API_KEY"
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        env_name = "ANTHROPIC_API_KEY"
    elif provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        env_name = "GOOGLE_API_KEY or GEMINI_API_KEY"
    else:
        raise ValueError(f"Unsupported live test provider: {provider}")

    if not api_key:
        pytest.skip(f"live {provider} test requires {env_name}")

    module_name = _SDK_MODULES[provider]
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"live {provider} test requires installed SDK module '{module_name}'")

    return api_key
