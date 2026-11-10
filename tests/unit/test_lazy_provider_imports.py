"""`import lazybridge` must not import any provider SDK (it took ~6 s for the Anthropic SDK alone)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROVIDER_SDKS = ("anthropic", "openai", "google.genai", "litellm")


def test_import_lazybridge_loads_no_provider_sdk():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); import lazybridge; "
        f"print(','.join(m for m in {PROVIDER_SDKS!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


@pytest.mark.parametrize(
    "name", ["AnthropicProvider", "DeepSeekProvider", "GoogleProvider", "LMStudioProvider", "OpenAIProvider"]
)
def test_provider_classes_still_importable_by_name(name):
    import lazybridge.core.providers as providers

    cls = getattr(providers, name)
    assert cls.__name__ == name
    assert issubclass(cls, providers.BaseProvider)
    assert name in dir(providers)


def test_unknown_provider_name_raises_attribute_error():
    import lazybridge.core.providers as providers

    with pytest.raises(AttributeError):
        providers.NotAProvider  # noqa: B018
